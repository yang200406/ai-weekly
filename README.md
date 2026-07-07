# 🤖 AI 周报生成器

一键将零散的工作记录扩写成专业周报，支持流式生成、历史管理、AI 润色。

## ✨ 功能

- **AI 生成** — 输入本周工作内容，DeepSeek 自动扩写成结构化周报
- **流式输出** — 文字逐字出现，像 ChatGPT 一样实时看到生成过程
- **4 种预设模板** — 标准、OKR、简洁、详细，一键切换
- **自定义格式** — 自己描述想要的周报格式，AI 按要求写
- **历史记录** — SQLite 持久保存，搜索、标签筛选，随时回顾
- **编辑模式** — 生成后直接在页面上修改，改完再复制导出
- **AI 润色** — 选中任意文字，告诉 AI 怎么改（更专业/更精简/...），原地替换
- **深色模式** — 晚上写周报不刺眼
- **标签分类** — 给周报打标签，按标签筛选
- **数据统计** — 总篇数、模板分布、月度趋势
- **快捷键** — `Ctrl+Enter` 生成 / `Ctrl+S` 保存 / `Ctrl+Shift+R` 重新生成

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 3. 启动

```bash
uvicorn main:app --reload
```

打开 **http://localhost:8000** 即可使用。

## 📁 项目结构

```
ai-weekly/
├── main.py          # FastAPI 后端
├── index.html       # 前端（纯 HTML/CSS/JS，零框架）
├── requirements.txt # Python 依赖
├── .env.example     # 环境变量模板
└── history.db       # SQLite 数据库（自动生成）
```

## 🛠️ 技术栈

- **后端**: FastAPI + DeepSeek API (OpenAI SDK)
- **前端**: 原生 HTML/CSS/JS + Marked.js
- **数据库**: SQLite (WAL 模式)
- **流式输出**: Server-Sent Events (SSE)
