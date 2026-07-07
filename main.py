from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import os
import json
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
# 确保所有未捕获异常都返回 JSON，而不是 HTML

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": f"请求参数错误：{exc}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": f"服务器内部错误：{str(exc)}"}
    )

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── SQLite 数据库初始化 ─────────────────────────────────────────────

DB_PATH = "history.db"


def get_db():
    """获取数据库连接（同一线程复用）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """创建表（如果不存在），并开启 WAL 模式避免并发锁"""
    conn = get_db()
    conn.execute("PRAGMA journal_mode=WAL")       # 写前日志：允许并发读写
    conn.execute("PRAGMA busy_timeout=5000")       # 忙等 5 秒而非直接报 SQLITE_BUSY
    conn.execute("""
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
        conn.execute("ALTER TABLE reports ADD COLUMN tags TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # 列已存在
    conn.commit()
    conn.close()


init_db()  # 启动时自动初始化


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


# ── 生成 API（支持自定义模板）─────────────────────────────────────────

@app.post("/api/generate")
async def generate_weekly(req: GenerateRequest):
    # 优先使用自定义模板，否则用预设模板
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

    conn = None
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        weekly_report = response.choices[0].message.content

        # 自动保存到历史记录
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = weekly_report.strip().split("\n")[0].strip("# ").strip()[:80]
        conn = get_db()
        conn.execute(
            "INSERT INTO reports (title, tasks, template, custom_prompt, tags, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, req.tasks, template_id, req.custom_prompt, req.tags, weekly_report, now, now)
        )
        conn.commit()
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        new_id = row[0]

        return {"weekly_report": weekly_report, "id": new_id}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"生成失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


# ── 流式生成 API（SSE）────────────────────────────────────────────────

@app.post("/api/generate-stream")
async def generate_stream(req: GenerateRequest):
    """流式生成周报，SSE 格式逐字推送"""
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

    async def event_stream():
        full_text = ""
        conn = None
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

            # 流结束后保存到历史记录
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            title = full_text.strip().split("\n")[0].strip("# ").strip()[:80] if full_text.strip() else "周报"
            conn = get_db()
            conn.execute(
                "INSERT INTO reports (title, tasks, template, custom_prompt, tags, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (title, req.tasks, template_id, req.custom_prompt, req.tags, full_text, now, now)
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
            new_id = row[0]
            conn.close()
            conn = None

            yield f"data: {json.dumps({'done': True, 'id': new_id, 'full_text': full_text}, ensure_ascii=False)}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if conn:
                conn.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── 统计数据 API ────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    conn = None
    try:
        conn = get_db()

        total_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

        template_rows = conn.execute(
            "SELECT template, COUNT(*) as count FROM reports GROUP BY template ORDER BY count DESC"
        ).fetchall()
        template_usage = {t["template"]: t["count"] for t in template_rows}

        monthly_rows = conn.execute(
            "SELECT substr(created_at, 1, 7) as month, COUNT(*) as count FROM reports GROUP BY month ORDER BY month DESC LIMIT 12"
        ).fetchall()
        monthly_counts = [{"month": m["month"], "count": m["count"]} for m in reversed(monthly_rows)]

        this_month = datetime.now().strftime("%Y-%m")
        month_count = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE substr(created_at, 1, 7) = ?", (this_month,)
        ).fetchone()[0]

        total_tasks = conn.execute("SELECT COUNT(*) FROM reports WHERE tasks != ''").fetchone()[0]

        recent_rows = conn.execute(
            "SELECT id, title, template, created_at FROM reports ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        recent_activity = [{"id": r["id"], "title": r["title"], "template": r["template"], "date": r["created_at"]} for r in recent_rows]

        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        week_count = conn.execute(
            "SELECT COUNT(*) FROM reports WHERE created_at >= ?", (seven_days_ago,)
        ).fetchone()[0]

        tpl_names = {
            "standard": "📋 标准", "okr": "🎯 OKR",
            "concise": "⚡ 简洁", "detailed": "📝 详细", "custom": "✏️ 自定义"
        }

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
        if conn:
            conn.close()


# ── 历史记录 CRUD API ────────────────────────────────────────────────

@app.get("/api/history")
def list_history(q: str = "", tag: str = ""):
    """获取所有历史记录（支持搜索和标签筛选）"""
    conn = None
    try:
        conn = get_db()
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
        rows = conn.execute(query, params).fetchall()
        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"加载历史记录失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


@app.get("/api/history/{report_id}")
def get_history(report_id: int):
    """获取单条历史记录详情（含完整内容）"""
    conn = None
    try:
        conn = get_db()
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="记录不存在")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"加载记录失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


@app.post("/api/history")
def save_history(req: SaveRequest):
    """手动保存一条记录"""
    conn = None
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = req.title or req.content.strip().split("\n")[0].strip("# ").strip()[:80]
        conn = get_db()
        conn.execute(
            "INSERT INTO reports (title, tasks, template, custom_prompt, tags, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (title, req.tasks, req.template, req.custom_prompt, req.tags, req.content, now, now)
        )
        conn.commit()
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        new_id = row[0]
        return {"id": new_id, "title": title}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"保存失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


@app.put("/api/history/{report_id}")
def update_history(report_id: int, req: UpdateRequest):
    """更新一条记录（编辑后保存）"""
    conn = None
    try:
        conn = get_db()
        existing = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="记录不存在")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates = {}
        if req.title is not None:
            updates["title"] = req.title
        if req.tasks is not None:
            updates["tasks"] = req.tasks
        if req.template is not None:
            updates["template"] = req.template
        if req.custom_prompt is not None:
            updates["custom_prompt"] = req.custom_prompt
        if req.tags is not None:
            updates["tags"] = req.tags
        if req.content is not None:
            updates["content"] = req.content
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [report_id]
        conn.execute(f"UPDATE reports SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return {"message": "更新成功", "id": report_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"更新失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


@app.delete("/api/history/{report_id}")
def delete_history(report_id: int):
    """删除一条记录"""
    conn = None
    try:
        conn = get_db()
        existing = conn.execute("SELECT id FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="记录不存在")
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        return {"message": "删除成功", "id": report_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"删除失败：{str(e)}"})
    finally:
        if conn:
            conn.close()


# ── AI 润色 API ──────────────────────────────────────────────────────

@app.post("/api/polish")
async def polish_text(req: PolishRequest):
    """对选中文本进行润色/改写"""
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


