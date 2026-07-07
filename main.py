from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import os
import json
import httpx
import traceback
from datetime import datetime, timedelta

# 加载环境变量
load_dotenv()

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 全局异常处理 ─────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": f"请求参数错误：{exc}"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": f"服务器内部错误：{str(exc)}"})


# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── 数据库层（Turso HTTP + SQLite 本地回退）─────────────────────────

TURSO_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)


class TursoDB:
    """通过 Turso HTTP Pipeline API 访问远程数据库"""

    def __init__(self, url, token):
        self.url = url.replace("libsql://", "https://").rstrip("/")
        self.token = token
        self._rows = []

    def execute(self, sql, params=None):
        """发送 SQL 到 Turso，返回 self（兼容 sqlite3.Cursor）"""
        if params is None:
            params = []

        # 将 SQLite 风格的 ? 占位符和参数转为 Turso HTTP API 格式
        args = []
        for p in params:
            if p is None:
                args.append({"type": "null", "value": ""})
            elif isinstance(p, bool):
                args.append({"type": "integer", "value": "1" if p else "0"})
            elif isinstance(p, int):
                args.append({"type": "integer", "value": str(p)})
            elif isinstance(p, float):
                args.append({"type": "real", "value": str(p)})
            else:
                args.append({"type": "text", "value": str(p)})

        body = {
            "requests": [
                {"type": "execute", "stmt": {"sql": sql, "args": args}},
                {"type": "close"}
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        try:
            r = httpx.post(f"{self.url}/v2/pipeline", json=body, headers=headers, timeout=15)
            r.raise_for_status()
            results = r.json()
            # 解析结果
            self._rows = []
            self._last_rowid = 0
            if results.get("results"):
                res = results["results"][0]
                if res.get("type") == "ok" and res.get("response", {}).get("type") == "results":
                    resp_data = res["response"]
                    cols = [c["name"] for c in resp_data.get("cols", [])]
                    raw_rows = resp_data.get("rows", [])
                    # 转换为 dict 列表
                    parsed_rows = []
                    for row in raw_rows:
                        parsed = {}
                        for i, col in enumerate(cols):
                            val = row[i]["value"] if i < len(row) and row[i].get("type") != "null" else None
                            parsed[col] = val
                        parsed_rows.append(parsed)
                    self._rows = [TursoRow(r) for r in parsed_rows]
                elif res.get("type") == "ok" and res.get("response", {}).get("type") == "execute":
                    self._last_rowid = res["response"].get("last_insert_rowid", 0)
        except Exception as e:
            raise Exception(f"Turso 请求失败：{e}")
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def commit(self):
        pass  # Turso 自动提交

    def close(self):
        pass

    @property
    def last_insert_rowid(self):
        return self._last_rowid


class TursoRow:
    """模拟 sqlite3.Row"""
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data.get(key)

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        return iter(self._data.values())


class LocalDB:
    """本地 SQLite（开发回退）"""

    def __init__(self, path="history.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")

    def execute(self, sql, params=None):
        if params is None:
            params = []
        return self.conn.execute(sql, params)

    def fetchone(self):
        raise NotImplementedError("use cursor directly")

    def fetchall(self):
        raise NotImplementedError("use cursor directly")

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    @property
    def last_insert_rowid(self):
        return None  # local uses SELECT last_insert_rowid() instead


def get_db():
    """返回数据库连接（Turso 或 SQLite）"""
    if USE_TURSO:
        return TursoDB(TURSO_URL, TURSO_TOKEN)
    else:
        return LocalDB("history.db")


def init_db():
    """创建表（如果不存在）"""
    db = get_db()
    is_local = isinstance(db, LocalDB)

    db.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    DEFAULT '',
            tasks       TEXT    DEFAULT '',
            template    TEXT    DEFAULT 'standard',
            custom_prompt TEXT   DEFAULT '',
            tags        TEXT    DEFAULT '',
            content     TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT '',
            updated_at  TEXT    DEFAULT ''
        )
    """)

    # 迁移：旧数据库可能没有 tags 列
    try:
        db.execute("ALTER TABLE reports ADD COLUMN tags TEXT DEFAULT ''")
    except:
        pass

    if is_local:
        db.commit()
    db.close()


def db_insert_id(db):
    """获取最后插入的 ID"""
    if isinstance(db, LocalDB):
        row = db.execute("SELECT last_insert_rowid()").fetchone()
        return row[0]
    else:
        return int(db.last_insert_rowid)


def db_fetchone(db, sql, params=None):
    """执行查询并返回一行"""
    result = db.execute(sql, params)
    if isinstance(db, LocalDB):
        return result.fetchone()
    else:
        return result.fetchone()


def db_fetchall(db, sql, params=None):
    """执行查询并返回所有行"""
    result = db.execute(sql, params)
    if isinstance(db, LocalDB):
        return result.fetchall()
    else:
        return result.fetchall()


def db_row_to_dict(row):
    """将行对象转为 dict"""
    if isinstance(row, TursoRow):
        return row._data
    else:
        return dict(row)


init_db()


# ── 周报模板定义 ─────────────────────────────────────────────────────

TEMPLATES = {
    "standard": {
        "name": "📋 标准周报",
        "description": "分为本周完成、下周计划、问题与建议三部分",
        "prompt": "请分为【本周完成】、【下周计划】、【遇到的问题与建议】三个部分，措辞专业、有逻辑、态度积极。"
    },
    "okr": {
        "name": "🎯 OKR 对齐周报",
        "description": "将工作内容与 OKR 目标对齐，突出关键成果",
        "prompt": "请将工作内容与 OKR 目标对齐，分为【关键成果进展】、【目标完成度】、【下周重点目标】、【风险与阻塞项】四个部分，突出量化成果和关键数据。"
    },
    "concise": {
        "name": "⚡ 简洁周报",
        "description": "只保留核心要点，适合快速汇报",
        "prompt": "请精简为要点式周报，分为【本周完成（3-5条）】、【下周计划（3-5条）】两部分，每条不超过两行，语言简洁有力。"
    },
    "detailed": {
        "name": "📝 详细周报",
        "description": "包含工作详情、数据指标、反思总结",
        "prompt": "请生成详细的周报，分为【本周工作详情】、【关键数据与指标】、【项目进展与里程碑】、【下周工作计划】、【反思与改进】、【需要的支持与资源】六个部分，内容详实，数据充分。"
    }
}


# ── Pydantic 模型 ────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    tasks: str
    template: str = "standard"
    custom_prompt: str = ""
    tags: str = ""

class SaveRequest(BaseModel):
    title: str = ""
    tasks: str
    template: str = "standard"
    custom_prompt: str = ""
    tags: str = ""
    content: str

class UpdateRequest(BaseModel):
    title: str = None
    tasks: str = None
    template: str = None
    custom_prompt: str = None
    tags: str = None
    content: str = None

class PolishRequest(BaseModel):
    text: str
    instruction: str = "让这段文字更专业、更流畅"


# ── 页面路由 ─────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return FileResponse("index.html")


# ── 模板 API ─────────────────────────────────────────────────────────

@app.get("/api/templates")
def get_templates():
    return {
        "templates": [
            {"id": tid, "name": t["name"], "description": t["description"]}
            for tid, t in TEMPLATES.items()
        ]
    }


# ── 生成 API ─────────────────────────────────────────────────────────

def _build_prompt(req: GenerateRequest):
    if req.custom_prompt.strip():
        prompt_instruction = req.custom_prompt.strip()
        template_id = "custom"
    else:
        template = TEMPLATES.get(req.template, TEMPLATES["standard"])
        prompt_instruction = template["prompt"]
        template_id = req.template
    prompt = f"""
    你是一个专业的职场助理。请根据以下本周完成的工作内容，帮我扩写成一篇专业的周报。
    要求：
    {prompt_instruction}
    使用 Markdown 格式。

    本周完成的工作：
    {req.tasks}
    """
    return prompt, template_id


def _save_report(tasks, template_id, custom_prompt, tags, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = content.strip().split("\n")[0].strip("# ").strip()[:80] if content.strip() else "周报"
    db = get_db()
    try:
        db.execute(
            "INSERT INTO reports (title, tasks, template, custom_prompt, tags, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, tasks, template_id, custom_prompt, tags, content, now, now)
        )
        db.commit()
        new_id = db_insert_id(db)
        return new_id
    finally:
        db.close()


@app.post("/api/generate")
async def generate_weekly(req: GenerateRequest):
    prompt, template_id = _build_prompt(req)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        weekly_report = response.choices[0].message.content
        new_id = _save_report(req.tasks, template_id, req.custom_prompt, req.tags, weekly_report)
        return {"weekly_report": weekly_report, "id": new_id}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"生成失败：{str(e)}"})


# ── 流式生成 API（SSE）────────────────────────────────────────────────

@app.post("/api/generate-stream")
async def generate_stream(req: GenerateRequest):
    prompt, template_id = _build_prompt(req)

    async def event_stream():
        full_text = ""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    yield f"data: {json.dumps({'chunk': text}, ensure_ascii=False)}\n\n"

            new_id = _save_report(req.tasks, template_id, req.custom_prompt, req.tags, full_text)
            yield f"data: {json.dumps({'done': True, 'id': new_id, 'full_text': full_text}, ensure_ascii=False)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ── 统计数据 API ────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    db = None
    try:
        db = get_db()
        total_reports = db_fetchone(db, "SELECT COUNT(*) FROM reports") or [0]; total_reports = total_reports[0]
        template_rows = db_fetchall(db, "SELECT template, COUNT(*) as count FROM reports GROUP BY template ORDER BY count DESC") or []
        template_usage = {db_row_to_dict(t)["template"]: db_row_to_dict(t)["count"] for t in template_rows}
        monthly_rows = db_fetchall(db, "SELECT substr(created_at, 1, 7) as month, COUNT(*) as count FROM reports GROUP BY month ORDER BY month DESC LIMIT 12") or []
        monthly_counts = [{"month": db_row_to_dict(m)["month"], "count": db_row_to_dict(m)["count"]} for m in reversed(monthly_rows)]
        this_month = datetime.now().strftime("%Y-%m")
        mc = db_fetchone(db, "SELECT COUNT(*) FROM reports WHERE substr(created_at, 1, 7) = ?", (this_month,)) or [0]; month_count = mc[0]
        tt = db_fetchone(db, "SELECT COUNT(*) FROM reports WHERE tasks != ''") or [0]; total_tasks = tt[0]
        recent_rows = db_fetchall(db, "SELECT id, title, template, created_at FROM reports ORDER BY created_at DESC LIMIT 5") or []
        recent_activity = [{"id": db_row_to_dict(r)["id"], "title": db_row_to_dict(r)["title"], "template": db_row_to_dict(r)["template"], "date": db_row_to_dict(r)["created_at"]} for r in recent_rows]
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        wc = db_fetchone(db, "SELECT COUNT(*) FROM reports WHERE created_at >= ?", (seven_days_ago,)) or [0]; week_count = wc[0]

        tpl_names = {"standard": "📋 标准", "okr": "🎯 OKR", "concise": "⚡ 简洁", "detailed": "📝 详细", "custom": "✏️ 自定义"}

        return {
            "total_reports": total_reports,
            "month_count": month_count,
            "week_count": week_count,
            "template_usage": [{"name": tpl_names.get(k, k), "key": k, "count": v} for k, v in template_usage.items()],
            "monthly_counts": monthly_counts,
            "recent_activity": recent_activity,
            "total_tasks_count": total_tasks
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"统计查询失败：{str(e)}"})
    finally:
        if db:
            db.close()


# ── 历史记录 CRUD API ────────────────────────────────────────────────

@app.get("/api/history")
def list_history(q: str = "", tag: str = ""):
    db = None
    try:
        db = get_db()
        query = "SELECT id, title, tasks, template, custom_prompt, tags, created_at, updated_at FROM reports WHERE 1=1"
        params = []
        if q.strip():
            query += " AND (title LIKE ? OR tasks LIKE ?)"
            like = f"%{q.strip()}%"
            params.extend([like, like])
        if tag.strip():
            query += " AND tags LIKE ?"
            params.append(f"%{tag.strip()}%")
        query += " ORDER BY updated_at DESC"
        rows = db_fetchall(db, query, params)
        return {"items": [db_row_to_dict(r) for r in rows]}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"加载历史记录失败：{str(e)}"})
    finally:
        if db:
            db.close()


@app.get("/api/history/{report_id}")
def get_history(report_id: int):
    db = None
    try:
        db = get_db()
        row = db_fetchone(db, "SELECT * FROM reports WHERE id = ?", (report_id,))
        if not row:
            raise HTTPException(status_code=404, detail="记录不存在")
        return db_row_to_dict(row)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"加载记录失败：{str(e)}"})
    finally:
        if db:
            db.close()


@app.post("/api/history")
def save_history(req: SaveRequest):
    db = None
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = req.title or req.content.strip().split("\n")[0].strip("# ").strip()[:80]
        db = get_db()
        db.execute(
            "INSERT INTO reports (title, tasks, template, custom_prompt, tags, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, req.tasks, req.template, req.custom_prompt, req.tags, req.content, now, now)
        )
        db.commit()
        new_id = db_insert_id(db)
        return {"id": new_id, "title": title}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"保存失败：{str(e)}"})
    finally:
        if db:
            db.close()


@app.put("/api/history/{report_id}")
def update_history(report_id: int, req: UpdateRequest):
    db = None
    try:
        db = get_db()
        existing = db_fetchone(db, "SELECT * FROM reports WHERE id = ?", (report_id,))
        if not existing:
            raise HTTPException(status_code=404, detail="记录不存在")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates = {}
        if req.title is not None: updates["title"] = req.title
        if req.tasks is not None: updates["tasks"] = req.tasks
        if req.template is not None: updates["template"] = req.template
        if req.custom_prompt is not None: updates["custom_prompt"] = req.custom_prompt
        if req.tags is not None: updates["tags"] = req.tags
        if req.content is not None: updates["content"] = req.content
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [report_id]
        db.execute(f"UPDATE reports SET {set_clause} WHERE id = ?", values)
        db.commit()
        return {"message": "更新成功", "id": report_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"更新失败：{str(e)}"})
    finally:
        if db:
            db.close()


@app.delete("/api/history/{report_id}")
def delete_history(report_id: int):
    db = None
    try:
        db = get_db()
        existing = db_fetchone(db, "SELECT id FROM reports WHERE id = ?", (report_id,))
        if not existing:
            raise HTTPException(status_code=404, detail="记录不存在")
        db.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        db.commit()
        return {"message": "删除成功", "id": report_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"删除失败：{str(e)}"})
    finally:
        if db:
            db.close()


# ── AI 润色 API ──────────────────────────────────────────────────────

@app.post("/api/polish")
async def polish_text(req: PolishRequest):
    prompt = f"""你是一个专业的文字润色助手。请根据以下要求对文本进行修改。

修改要求：{req.instruction}

原文：
{req.text}

请直接输出修改后的文本，不要加任何解释或前缀。"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        polished = response.choices[0].message.content
        return {"polished": polished}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"润色失败：{str(e)}"})
