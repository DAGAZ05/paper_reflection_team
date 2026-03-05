# 反思评估组（组7）冲突裁决系统

## 概述

反思评估组是软件工程硕士论文评价系统的"主审稿人"，负责协调不同评审小组之间的意见冲突，生成最终的评审报告。本系统实现了以下核心功能：

1. **冲突检测**：识别不同Agent之间的意见分歧
2. **冲突裁决**：使用LLM-as-a-judge模式解决冲突
3. **报告生成**：生成Markdown格式的评审报告
4. **API服务**：提供RESTful API接口

## 系统架构

```
反思评估组（ReflectionJudge）
├── 冲突检测引擎（Conflict Detection Engine）
│   ├── 关键词矛盾检测
│   ├── 分数差异检测
│   └── 上下文依赖分析
├── 冲突裁决引擎（Conflict Resolution Engine）
│   ├── LLM-as-a-judge
│   ├── 降级方案（规则引擎）
│   └── 证据追溯
├── 报告生成器（Report Generator）
│   ├── Markdown报告
│   ├── 最终结论
│   └── 优先级排序
└── API服务（FastAPI）
    ├── 健康检查
    └── 冲突裁决端点
```

## 安装与配置

### 1. 环境要求

- Python 3.10+
- PostgreSQL 数据库（可选，用于持久化存储）
- LLM API 访问权限（OpenAI、Claude等）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 环境变量配置

创建 `.env` 文件并配置以下变量：

```env
# 数据库配置（可选）
DB_HOST=10.13.1.26
DB_PORT=5432
DB_USER=Guest
DB_PASSWORD=12345678
DB_NAME=thesis_review

# LLM配置
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1

$env:LLM_MODEL="deepseek/deepseek-chat"
$env:LLM_BASE_URL="https://api.deepseek.com"
$env:LLM_API_KEY="你的新key"

# 冲突检测阈值
CONFLICT_THRESHOLD=0.7
```

## 使用方法

### 1. 演示模式

运行演示，查看冲突裁决的完整流程：

```bash
python src/run_reflection_judge.py demo
```

这将展示两个场景：
- 有冲突的情况（代码审计组 vs 实验数据组）
- 无冲突的情况

### 2. API服务器模式

启动API服务器：

```bash
python src/run_reflection_judge.py server
```

服务器将在 `http://localhost:8000` 启动，可通过以下端点访问：

- 健康检查：`GET /health`
- API文档：`GET /docs`
- 冲突裁决：`POST /api/resolve_conflicts`

### 3. 生成模拟数据

生成模拟的Agent结果数据：

```bash
python src/run_reflection_judge.py generate-data
```

## API使用说明

### 冲突裁决端点

**端点**：`POST /api/resolve_conflicts`

**请求体**：

```json
{
  "request_id": "req_20231027_001",
  "metadata": {
    "paper_id": "uuid-string",
    "paper_title": "关于深度学习的研究"
  },
  "payload": {
    "agent_results": [
      {
        "request_id": "req_001",
        "agent_info": {
          "name": "代码审计组",
          "version": "v1.0"
        },
        "result": {
          "score": 85,
          "audit_level": "Info",
          "comment": "算法实现高效，时间复杂度为O(n log n)",
          "suggestion": "代码结构清晰，无明显优化空间",
          "tags": ["Performance_Issue"]
        },
        "usage": {
          "tokens": 100,
          "latency_ms": 500
        }
      },
      {
        "request_id": "req_002",
        "agent_info": {
          "name": "实验数据组",
          "version": "v1.0"
        },
        "result": {
          "score": 55,
          "audit_level": "Warning",
          "comment": "实验结果显示算法在大数据集上运行缓慢",
          "suggestion": "建议优化算法实现或考虑替代方案",
          "tags": ["Statistical_Weakness"]
        },
        "usage": {
          "tokens": 120,
          "latency_ms": 600
        }
      }
    ]
  },
  "config": {
    "temperature": 0.3,
    "max_tokens": 1000,
    "conflict_threshold": 0.7
  }
}
```

**响应体**：

```json
{
  "request_id": "req_20231027_001",
  "agent_info": {
    "name": "ReflectionJudge_ConflictResolver",
    "version": "v1.0"
  },
  "result": {
    "conflicts_resolved": true,
    "resolved_issues": [
      {
        "agent1_name": "代码审计组",
        "agent2_name": "实验数据组",
        "conflict_type": "direct_contradiction",
        "root_cause": "评估维度不同：代码审计关注实现效率，实验数据关注实际性能",
        "evidence_strength": 0.8,
        "confidence": 0.85,
        "resolved_comment": "算法在理论上是高效的，但在实际数据集上表现不佳，可能存在实现与理论不符的情况",
        "resolved_suggestion": "建议检查代码实现是否与理论算法一致，并在不同规模数据集上进行性能测试对比",
        "final_level": "Warning",
        "needs_human_review": false
      }
    ],
    "confidence_score": 0.85,
    "tags": ["Executive_Summary", "Critical_Fix_List", "Score_Calibration"],
    "final_verdict": {
      "average_score": 70.0,
      "level_distribution": {"Info": 0, "Warning": 1, "Critical": 0},
      "verdict": "存在少量问题，建议小修后录用（Minor Revision）",
      "total_conflicts": 1,
      "needs_human_review_count": 0
    },
    "markdown_report": "# 论文评审冲突裁决报告\n\n...",
    "result_json": {
      "conflicts_resolved": true,
      "resolved_issues": [...],
      "confidence_score": 0.85,
      "tags": [...],
      "final_verdict": {...}
    }
  },
  "usage": {
    "tokens": 200,
    "latency_ms": 1500
  }
}
```

## 代码执行步骤

### 1. 冲突检测

1. **输入标准化**：将不同格式的Agent结果转换为统一格式
2. **关键词分析**：检测反义词对（如"高效"vs"低效"）
3. **分数差异**：检测分数差异超过阈值的冲突
4. **上下文依赖**：识别基于不同上下文的意见分歧

### 2. 冲突裁决

1. **LLM裁决**：使用LLM-as-a-judge模式解决冲突
2. **降级方案**：当LLM不可用时，使用规则引擎
3. **证据追溯**：引用论文具体内容作为裁决依据
4. **置信度评估**：评估裁决结果的可信度

### 3. 报告生成

1. **去重处理**：合并相似的冲突裁决结果
2. **优先级排序**：按照严重程度和置信度排序
3. **最终结论**：基于所有裁决结果生成最终评审意见
4. **Markdown报告**：生成格式化的评审报告

## 测试

运行单元测试：

```bash
pytest tests/test_conflict_resolver.py -v
```

## 项目结构

```
project/
├── src/
│   ├── __init__.py
│   ├── conflict_resolver.py      # 主要冲突裁决逻辑
│   ├── database.py               # 数据库客户端
│   ├── llm_client.py             # LLM客户端
│   ├── schemas.py                # 数据模型定义
│   ├── mock_data_generator.py    # 模拟数据生成器
│   └── run_reflection_judge.py   # 执行脚本
├── prompts/
│   └── conflict_resolution/
│       ├── system_prompt.txt      # 系统提示词
│       └── user_prompt.txt       # 用户提示词
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # 测试配置
│   └── test_conflict_resolver.py # 单元测试
├── requirements.txt              # 依赖列表
└── README_ReflectionJudge.md     # 本文档
```

## 技术栈

- **后端框架**：FastAPI
- **数据模型**：Pydantic V2
- **数据库**：PostgreSQL + asyncpg
- **LLM集成**：LiteLLM
- **测试框架**：pytest + pytest-asyncio
- **并发处理**：asyncio

## 注意事项

1. **LLM配置**：确保正确配置LLM API密钥和基础URL
2. **数据库连接**：如果使用数据库，确保数据库服务正在运行
3. **冲突阈值**：根据实际需求调整冲突检测阈值
4. **Prompt优化**：可根据实际效果调整系统提示词和用户提示词

## 故障排除

### 常见问题

1. **LLM调用失败**：检查API密钥和网络连接
2. **数据库连接失败**：检查数据库配置和服务状态
3. **冲突检测不准确**：调整关键词模式和阈值
4. **性能问题**：考虑使用更快的LLM模型或优化提示词

### 日志调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展与定制

### 添加新的冲突类型

在 `conflict_resolver.py` 中的 `_analyze_comment_conflict` 方法中添加新的冲突检测逻辑。

### 自定义Prompt模板

修改 `prompts/conflict_resolution/` 目录下的提示词文件。

### 集成外部数据库

实现 `DatabaseClient` 类中的方法，适配您的数据库架构。

## 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。