# 反思评估组 - 论文评审反思评估系统

## 项目概述

本项目是反思评估组的完整实现，整合了四位成员的工作成果，实现了完整的论文评审反思评估系统。系统支持从数据库或文件读取5个审计组的评审结果，通过冲突裁决、幻觉过滤、重复过滤、优先级排序等模块，生成最终的评审报告和导师指导意见。

This is the work of the reflection and evaluation team for our school's project, evaluation indicators for the quality of master's degree theses in software engineering.

## ⚡ 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
创建 `.env` 文件并添加DeepSeek API密钥：
```bash
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行评审程序
```bash
# 从文件读取（推荐用于测试）
python run.py --mode file --prompts-dir prompts

# 启用导师对话生成
python run.py --mode file --prompts-dir prompts --enable-dialogue
```

### 4. 查看评审报告
```bash
# 查看生成的Markdown报告
ls reports/review_report_*.md

# 查看JSON结果
ls results/result_*.json
```

---

## 核心功能

### ✅ 已实现的功能模块

1. **冲突裁决 (Conflict Resolution)**
   - 检测分数差异、语义冲突、级别冲突
   - 使用DeepSeek API进行智能裁决
   - 加权投票机制（逻辑组1.2、代码组1.1、实验组1.1、文献组1.0、格式组0.8）
   - 自动补全截断的JSON响应

2. **幻觉过滤 (Hallucination Filtering)**
   - 证据真实性验证：检查evidence_quote是否在原文中存在
   - 强制证据关联：Warning/Critical级别问题必须包含有效证据
   - 自动剔除无效证据
   - 基于证据验证分数调整最终评分

3. **重复过滤 (Deduplication)**
   - 使用Sentence-Transformers或TF-IDF进行文本向量化
   - DBSCAN聚类识别相似问题
   - 自动选择代表性问题

4. **优先级排序 (Priority Sorting)**
   - 多维度排序：问题等级、置信度、影响范围
   - 可配置的权重系统
   - 问题分级：Critical、Major、Minor

5. **整体评分 (Overall Scoring)**
   - 加权平均分计算
   - 证据验证分数调整
   - 级别分布统计
   - 最终结论生成（Accept/Minor Revision/Major Revision）

6. **导师对话生成 (Mentor Dialogue)**
   - 根据领域和问题严重性构建导师人设
   - 使用DeepSeek API生成指导性对话
   - 对话质量自动评估
   - 可选功能（通过--enable-dialogue启用）

7. **人工复核标记 (Human Review Marking)**
   - 置信度低于阈值触发
   - Agent意见冲突触发
   - 证据缺失触发
   - 优先级分级：High、Medium、Low

8. **Markdown报告生成**
   - 完整的评审报告结构
   - 包含执行摘要、审计详情、冲突裁决、证据验证、问题列表、导师指导、人工复核建议、最终建议
   - 符合导师审阅习惯

## 运行模式

### 模式1: 从文件读取（推荐用于测试）
```bash
# 基本模式
python run.py --mode file --prompts-dir prompts

# 启用导师对话
python run.py --mode file --prompts-dir prompts --enable-dialogue
```

**输入**: `prompts/` 文件夹中的JSON文件（每篇论文需要5个审计组的结果）
**输出**:
- `results/result_{paper_id}.json` - JSON格式结果
- `reports/review_report_{paper_id}_{timestamp}.md` - Markdown报告

### 模式2: 从数据库读取
```bash
python run.py --mode database --paper-id paper_001
```

**输入**: PostgreSQL数据库中的`agent_audits`表
**输出**: 保存到`reflection_results`表 + Markdown报告

### 模式3: 交互式模式
```bash
python run.py --mode interactive
```

## 数据格式

### 输入格式 (审计结果)
```json
{
  "group_id": 6,
  "group_name": "文献真实性组",
  "paper_id": "test_paper_001",
  "audit_results": [
    {
      "id": "item-6-001",
      "point": "文献年份合理性",
      "score": 60,
      "level": "Critical",
      "description": "发现严重问题：文献年份合理性",
      "evidence_quote": "原文第7.1节提到：'文献年份合理性相关内容...'",
      "location": {
        "section": "6.5",
        "line_start": 215
      },
      "suggestion": "必须立即修正文献年份合理性"
    }
  ]
}
```

**字段说明**:
- `group_id`: 审计组ID（2=格式组, 3=逻辑组, 4=代码组, 5=实验组, 6=文献组）
- `point`: 审核点名称
- `score`: 评分（0-100）
- `level`: 问题级别（Critical/Warning/Info）
- `evidence_quote`: 原文证据引用（必需）
- `location`: 问题位置（章节、行号）

### 输出格式 (反思评估结果)
```json
{
  "paper_id": "test_paper_001",
  "final_score": 68.9,
  "verdict": "存在关键问题，建议大修后重新提交（Major Revision）",
  "critical_issues": [...],
  "major_issues": [...],
  "minor_issues": [...],
  "needs_human_review": true,
  "human_review_reason": "置信度低于阈值; Agent意见冲突",
  "mentor_dialogue": {
    "role": "导师",
    "field": "软件工程",
    "conversation": [...],
    "quality_score": 4.5
  }
}
```

## 项目结构

```
project/
├── run.py                          # 主运行程序 ⭐
├── requirements.txt                # 依赖列表
├── .env                            # API密钥配置（需自行创建）
├── src/
│   ├── db/                         # 统一数据库模块
│   │   └── database.py             # PostgreSQL连接管理
│   ├── api/                        # 统一API模块
│   │   └── deepseek_client.py      # DeepSeek API客户端（自动加载.env）
│   ├── common/                     # 公共模块
│   │   ├── models.py               # 统一数据模型
│   │   ├── config_c.py             # 配置管理
│   │   └── report_generator.py     # Markdown报告生成器
│   ├── conflict_resolution/        # 冲突裁决模块
│   │   └── conflict_resolver.py    # 冲突检测、LLM裁决、JSON修复
│   ├── deduplication/              # 重复过滤模块
│   │   └── dedup.py                # DBSCAN聚类去重
│   ├── evidence_validation/        # 证据验证模块
│   │   └── evidence.py             # 证据真实性验证
│   ├── dialogue_generation/        # 对话生成模块
│   │   └── dialogue_engine.py      # 导师对话生成
│   └── priority_sorting/           # 优先级排序模块
│       └── review_engine.py        # 规则引擎、复核标记
├── config/
│   └── rule_config.json            # 规则配置文件
├── prompts/                        # JSON输入文件目录
│   ├── test_paper_001_group_2.json
│   ├── test_paper_001_group_3.json
│   └── ...
├── results/                        # JSON结果输出目录
│   └── result_test_paper_001.json
├── reports/                        # Markdown报告输出目录 📄
│   └── review_report_test_paper_001_20260306_223332.md
└── docs/                           # 文档目录
    ├── work_week2.txt              # 数据格式说明
    ├── group_recommendations.txt   # 功能需求说明
    └── group_seperate_work.txt     # 模块分工说明
```

## 技术栈

- **框架**: FastAPI, Pydantic
- **LLM**: DeepSeek API（直接调用，不使用litellm）
- **NLP**: Sentence-Transformers, scikit-learn
- **数据库**: asyncpg (PostgreSQL)
- **HTTP**: httpx
- **配置**: python-dotenv, pyyaml
- **测试**: pytest, pytest-asyncio

## 配置说明

### API配置
创建 `.env` 文件：
```bash
DEEPSEEK_API_KEY=your_api_key_here
```
或者在PowerShell中设置环境变量：
```bash
$env:DEEPSEEK_API_KEY="your_api_key_here"
```
或者在git bash中设置环境变量：
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

系统会自动加载 `.env` 文件中的API密钥。

### 数据库配置（可选）
如果使用数据库模式，需要配置PostgreSQL连接：
- 主机: 10.13.1.26
- 端口: 5432
- 用户名: admin
- 密码: ABCabc@123
- 数据库: postgres

## 输出说明

### Markdown评审报告
每次评审完成后，系统会在`reports/`文件夹生成详细的Markdown报告，包含：

1. **执行摘要**: 综合评分、评审等级、问题统计
2. **审计详情**: 各审计组评分分布、权重说明
3. **冲突裁决**: 冲突检测结果、裁决详情、加权投票机制
4. **证据验证**: 幻觉过滤结果、有效/无效证据统计
5. **问题列表**: 按Critical/Major/Minor分级展示
6. **导师意见**: 指导性对话内容（如果启用）
7. **复核建议**: 是否需要人工复核及原因
8. **最终建议**: 评审结论和具体修改建议
9. **附录**: 系统信息、评审流程、数据统计

报告文件命名格式: `review_report_{paper_id}_{timestamp}.md`

### 数据统计
报告中的数据统计包括：
- **审核项总数**: 所有审计组提交的审核项数量
- **复核标记数**: 触发人工复核的项目数量
- **冲突裁决数**: 检测到并裁决的冲突数量

## 特色功能

### 1. JSON截断自动修复
当DeepSeek API响应被截断时（达到max_tokens限制），系统会：
- 自动检测截断（finish_reason == "length"）
- 统计未闭合的括号
- 自动补全缺失的 `]` 和 `}`
- 4层JSON修复策略确保解析成功

### 2. 证据验证与幻觉过滤
- 验证所有evidence_quote是否在原文中存在
- Warning/Critical级别问题必须包含有效证据
- 自动剔除无证据或证据无效的问题
- 基于证据验证分数调整最终评分（<0.7扣分，>0.9加分）

### 3. 智能冲突裁决
- 多维度冲突检测：分数差异、语义冲突、级别冲突
- DeepSeek主考官Agent进行智能裁决
- 加权投票机制（不同审计组权重不同）
- 详细的裁决理由和建议

### 4. 导师对话生成
- 根据领域和问题严重性构建导师人设
- 生成温和但专业的指导性对话
- 包含总体评价、具体问题分析、鼓励性结尾
- 对话质量自动评估

## 常见问题

### Q: 如何设置API密钥？
创建 `.env` 文件并添加：
```bash
DEEPSEEK_API_KEY=your_api_key_here
```

### Q: 为什么导师对话没有生成？
需要添加 `--enable-dialogue` 参数：
```bash
python run.py --mode file --prompts-dir prompts --enable-dialogue
```

### Q: 报告中的统计数据为什么是0？
这个问题已修复。系统现在正确读取 `plugin_metadata` 字段中的统计数据。

### Q: JSON解析失败怎么办？
系统已实现4层JSON修复策略和自动截断补全，大多数情况下会自动修复。如果仍然失败，检查日志中的详细错误信息。

### Q: 如何生成测试数据？
使用测试数据生成器：
```bash
python tests/generate_test_data.py --num-papers 3
```

### Q: 数据库连接失败？
检查网络是否能访问数据库服务器，或使用文件模式（`--mode file`）。

## 团队分工

### 幻觉评估组
- **成员A**: 冲突裁决、整体评分、JSON修复、项目集成
- **成员B**: 重复过滤、证据验证、幻觉过滤

### 对话/交互开发组
- **成员C**: 导师对话生成、对话质量评估
- **成员D**: 优先级排序、人工复核标记、规则引擎

## 文档

- [数据格式说明](docs/work_week2.txt) - 接口规范
- [功能需求说明](docs/group_recommendations.txt) - 各组任务
- [模块分工说明](docs/group_seperate_work.txt) - 技术实现

## 许可证

本项目用于学校课程项目，仅供学习和研究使用。

---

**最后更新**: 2026-03-06
**版本**: v1.0
**状态**: ✅ 所有核心功能已实现并测试通过
