# 反思评估组 - 项目整合说明

## 项目概述

本项目整合了反思评估组四位成员的工作成果，实现了完整的论文评审反思评估系统。

This is the work of the reflection and evaluation team for our school's project, evaluation indicators for the quality of master's degree theses in software engineering.

## 团队分工

根据《反思评估组四人精细化周计划表》，四位成员的分工如下：

### 幻觉评估组
- **成员A**：冲突裁决、整体评分、项目进度调控
- **成员B**：重复过滤、幻觉过滤

### 对话/交互开发组
- **成员C**：导师对话生成
- **成员D**：优先级排序、人工复核标记、系统集成

## 项目结构

```
project/
├── src/                          # 源代码目录
│   ├── __init__.py              # 模块初始化
│   ├── models.py                # 统一数据模型定义
│   ├── main.py                  # 主入口和编排器
│   ├── conflict_resolver.py     # 成员A：冲突裁决模块
│   ├── llm_client.py            # 成员A：LLM客户端
│   ├── database_a.py            # 成员A：数据库客户端
│   ├── evidence.py              # 成员B：证据验证模块
│   ├── dedup.py                 # 成员B：重复过滤模块
│   ├── config_b.py              # 成员B：配置
│   ├── utils_b.py               # 成员B：工具函数
│   ├── database_b.py            # 成员B：数据库
│   ├── dialogue_engine.py       # 成员C：对话生成引擎
│   ├── config_c.py              # 成员C：配置
│   ├── database_c.py            # 成员C：数据库
│   └── review_engine.py         # 成员D：复核决策引擎
├── tests/                        # 测试目录
├── prompts/                      # 提示词模板
├── docs/                         # 文档目录
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

## 核心功能模块

### 1. 冲突裁决模块 (成员A)
- **文件**: `src/conflict_resolver.py`
- **功能**: 检测Agent评审结果冲突、LLM裁决、计算加权平均分、生成评审报告

### 2. 重复过滤和幻觉过滤模块 (成员B)
- **文件**: `src/dedup.py`, `src/evidence.py`
- **功能**: DBSCAN聚类去重、精确/语义匹配验证证据

### 3. 导师对话生成模块 (成员C)
- **文件**: `src/dialogue_engine.py`
- **功能**: 构建导师人设、生成指导性对话、对话质量评分

### 4. 优先级排序和复核标记模块 (成员D)
- **文件**: `src/review_engine.py`
- **功能**: 规则引擎排序、检测复核触发条件、生成人工复核标记

## 数据流程

```
Agent评审结果 → [成员D]排序标记 → [成员B]去重验证 → [成员A]冲突裁决 → [成员C]对话生成 → 最终报告
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```python
from src import ReflectionJudgeOrchestrator

orchestrator = ReflectionJudgeOrchestrator()
result = await orchestrator.process_paper_review(
    paper_id="paper_001",
    paper_title="论文标题",
    paper_content="论文内容...",
    agent_results=[...],
    field="软件工程"
)
```

## 技术栈

- **框架**: FastAPI, Pydantic
- **LLM**: LiteLLM
- **NLP**: Sentence-Transformers, scikit-learn
- **数据库**: asyncpg (PostgreSQL)
- **测试**: pytest

## 贡献者

- 成员A: 冲突裁决和整体评分
- 成员B: 重复过滤和幻觉过滤
- 成员C: 导师对话生成
- 成员D: 优先级排序和系统集成
