# 反思评估组 - 论文评审反思评估系统

## 项目概述

本项目是反思评估组的完整实现，整合了四位成员的工作成果，实现了完整的论文评审反思评估系统。

This is the work of the reflection and evaluation team for our school's project, evaluation indicators for the quality of master's degree theses in software engineering.

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境
```bash
# 设置DeepSeek API密钥
export DEEPSEEK_API_KEY="your_api_key_here"
```

### 3. 生成测试数据
```bash
python tests/generate_test_data.py --num-papers 3
```

### 4. 运行程序
```bash
# 交互式运行
python run.py

# 从数据库读取
python run.py --mode database

# 从文件读取
python run.py --mode file --prompts-dir prompts
```

### 5. 查看评审报告
程序运行后会在`reports/`文件夹生成Markdown格式的评审报告：
```bash
ls reports/
# 输出: review_report_paper_001_20260305_143022.md
```

## 核心功能

### 支持两种运行方案

#### 方案1: 从PostgreSQL数据库读取
- 连接数据库: 10.13.1.26:5432
- 用户名: admin, 密码: ABCabc@123
- 从`agent_audits`表读取审计结果
- 结果保存到`reflection_results`表

#### 方案2: 从JSON文件读取
- 从`prompts`文件夹读取JSON文件
- 每篇论文需要5个审计组的结果（group_id: 2-6）
- 结果保存为`result_{paper_id}.json`

### 数据流程

```
审计结果输入 (数据库/文件)
    ↓
[成员D] 优先级排序 + 复核标记
    ↓
[成员A] 冲突裁决 + 整体评分
    ↓
[成员B] 证据验证 + 去重过滤
    ↓
[成员C] 导师对话生成
    ↓
最终评审报告 (数据库/文件)
```

## 项目结构

```
project/
├── run.py                          # 主运行程序 ⭐
├── src/
│   ├── db/                         # 统一数据库模块
│   │   └── database.py             # PostgreSQL连接管理
│   ├── api/                        # 统一API模块
│   │   └── deepseek_client.py      # DeepSeek API客户端
│   ├── common/                     # 公共模块
│   │   └── models.py               # 统一数据模型
│   ├── conflict_resolution/        # 成员A：冲突裁决
│   ├── deduplication/              # 成员B：重复过滤
│   ├── evidence_validation/        # 成员B：证据验证
│   ├── dialogue_generation/        # 成员C：对话生成
│   └── priority_sorting/           # 成员D：优先级排序
├── tests/
│   └── generate_test_data.py       # 测试数据生成器 ⭐
├── prompts/                        # JSON文件目录
├── reports/                        # Markdown评审报告目录 📄
└── docs/                           # 文档目录
    ├── 快速启动指南.md              # 快速开始
    ├── 项目结构说明.md              # 详细文档
    ├── 功能验证清单.md              # 功能验证
    └── work_week2.txt              # 数据格式说明
```

## 核心模块

### 1. 冲突裁决模块 (成员A)
- 检测Agent评审结果冲突
- 使用DeepSeek进行LLM裁决
- 加权投票机制（不同审计组权重不同）
- 计算加权平均分
- 生成Markdown评审报告

### 2. 重复过滤模块 (成员B)
- DBSCAN聚类去重
- 识别相似审计意见

### 3. 证据验证模块 (成员B)
- 从paper_sections表读取论文原文
- 精确匹配验证证据引用
- 语义匹配验证
- 幻觉过滤：检测虚假证据
- 强制证据关联：Warning/Critical必须有证据

### 4. 导师对话生成模块 (成员C)
- 构建导师人设
- 生成指导性对话
- 对话质量评分

### 5. 优先级排序模块 (成员D)
- 规则引擎排序
- 检测复核触发条件
- 生成人工复核标记

## 数据格式

### 输入格式 (审计结果)
```json
{
  "group_id": 2,
  "paper_id": "paper_001",
  "audit_results": [
    {
      "id": "item-001",
      "point": "审核点",
      "score": 85,
      "level": "Warning",
      "description": "问题描述",
      "evidence_quote": "原文引用",
      "location": {"section": "2.1", "line_start": 45},
      "suggestion": "改进建议"
    }
  ]
}
```

### 输出格式 (反思评估结果)
```json
{
  "paper_id": "paper_001",
  "final_score": 82.5,
  "verdict": "论文质量良好，建议小修后录用",
  "critical_issues": [],
  "major_issues": [...],
  "minor_issues": [...],
  "needs_human_review": false,
  "mentor_dialogue": {...}
}
```

## 技术栈

- **框架**: FastAPI, Pydantic
- **LLM**: DeepSeek API
- **NLP**: Sentence-Transformers, scikit-learn
- **数据库**: asyncpg (PostgreSQL)
- **HTTP**: httpx
- **测试**: pytest

## 配置说明

### 数据库配置
- 主机: 10.13.1.26
- 端口: 5432
- 用户名: admin
- 密码: ABCabc@123
- 数据库: paper_review

**数据库表**:
- `agent_audits`: 存储5个审计组的评审结果
- `paper_sections`: 存储论文章节内容（用于证据验证）
- `reflection_results`: 存储反思评估最终结果

### API配置
- 统一使用DeepSeek API
- 需要设置环境变量: `DEEPSEEK_API_KEY`

## 输出说明

### Markdown评审报告
每次评审完成后，系统会在`reports/`文件夹生成详细的Markdown报告，包含：

1. **执行摘要**: 综合评分、评审等级、问题统计
2. **审计详情**: 各审计组评分分布、权重说明
3. **冲突裁决**: 冲突检测结果、加权投票机制
4. **证据验证**: 幻觉过滤结果、无效证据统计
5. **问题列表**: 按Critical/Major/Minor分级
6. **导师意见**: 指导性对话内容
7. **复核建议**: 是否需要人工复核
8. **最终建议**: 评审结论和具体建议

报告文件命名格式: `review_report_{paper_id}_{timestamp}.md`

## 文档

- [快速启动指南](docs/快速启动指南.md) - 新手入门
- [项目结构说明](docs/项目结构说明.md) - 详细文档
- [功能验证清单](docs/功能验证清单.md) - work_week2.txt要求验证
- [数据格式说明](docs/work_week2.txt) - 接口规范
- [代码合并说明](docs/合并说明.md) - 开发历史

## 团队分工

### 幻觉评估组
- **成员A**: 冲突裁决、整体评分、项目进度调控
- **成员B**: 重复过滤、幻觉过滤

### 对话/交互开发组
- **成员C**: 导师对话生成
- **成员D**: 优先级排序、人工复核标记、系统集成

## 常见问题

### Q: 如何生成测试数据？
```bash
python tests/generate_test_data.py --num-papers 5
```

### Q: 数据库连接失败？
检查网络是否能访问10.13.1.26，或修改`src/db/database.py`中的配置。

### Q: DeepSeek API调用失败？
确保设置了环境变量`DEEPSEEK_API_KEY`。

### Q: 输入数据格式错误？
参考`docs/work_week2.txt`中的格式说明。

## 许可证

本项目用于学校课程项目，仅供学习和研究使用。

---

**更多信息请查看 [快速启动指南](docs/快速启动指南.md)** 📖
