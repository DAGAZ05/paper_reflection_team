# 反思评估组 - 论文评审反思评估系统

## 项目概述

本项目是反思评估组的完整实现，整合了四位成员的工作成果，实现了完整的论文评审反思评估系统。系统从数据库`agent_audit_result`表读取4个审计组（FMT格式审计/REF文献审计/EXP实验数据/LOG逻辑审计）的评审结果，从`main_rules`和`rule_judge`表读取评审规则，通过冲突裁决、幻觉过滤、重复过滤、优先级排序等模块，生成最终的评审报告和导师指导意见，并将结果保存到`reflect_agent_verdict`表。同时支持从文件读取JSON格式的审计结果。

This is the work of the reflection and evaluation team for our school's project, evaluation indicators for the quality of master's degree theses in software engineering.

## ⚡ 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
在项目根目录创建 `.env` 文件并添加DeepSeek API密钥：
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
   - 加权投票机制（LOG:1.2、EXP:1.1、REF:1.0、FMT:0.8）
   - 自动补全截断的JSON响应

2. **幻觉过滤 (Hallucination Filtering)**
   - 证据真实性验证：检查evidence_quote是否在原文中存在
   - 利用paper_sections表的location字段进行章节级验证
   - 强制证据关联：Warning/Critical级别问题必须包含有效证据
   - 自动剔除无效证据
   - 基于证据验证分数调整最终评分

3. **规则驱动评审 (Rule-Based Evaluation)**
   - 从数据库`main_rules`和`rule_judge`表动态读取评审规则
   - 支持22条规则（FMT:4条、REF:4条、EXP:8条、LOG:7条）
   - 规则类型：QUANTITATIVE（定量）和BOOLEAN（布尔）
   - 违规等级：CRITICAL和WARNING

3. **重复过滤 (Deduplication)**
   - 使用Sentence-Transformers或TF-IDF进行文本向量化
   - DBSCAN聚类识别相似问题
   - 自动选择代表性问题

4. **优先级排序 (Priority Sorting)**
   - 多维度排序：问题等级、置信度、影响范围
   - 可配置的权重系统
   - 问题分级：Critical、Major、Minor

5. **整体评分 (Overall Scoring)**
   - `initial_score`：各审计agent所有rule的`score_obtained`之和
   - `conflict_penalty`：冲突裁决扣分（Critical: -5分/个, Warning: -2分/个）
   - `final_score`：归一化到100分制后，扣除冲突惩罚和证据验证调整
   - 证据验证调整：`validation_score < 0.7`时扣分（最多-14分），`> 0.9`时加分（最多+1分）
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
# 基本模式（智能混合裁决）
python run.py --mode file --prompts-dir prompts

# 启用导师对话
python run.py --mode file --prompts-dir prompts --enable-dialogue

# 启用纯LLM-as-a-Judge模式（始终调用API）
python run.py --mode file --prompts-dir prompts --always-use-llm
```

**输入**: `prompts/` 文件夹中的JSON文件（每篇论文需要4个审计组的结果：FMT/REF/EXP/LOG）
**输出**:
- `results/result_{paper_id}.json` - JSON格式结果
- `reports/review_report_{paper_id}_{timestamp}.md` - Markdown报告

### 模式2: 从数据库读取
```bash
# 自动模式：处理所有4个审计组齐全、所有规则已应用且未评估/需重新评估的论文
python run.py --mode database

# 指定论文ID（强制评估，即使审计组不全）
python run.py --mode database --paper-id P001

# 纯LLM-as-a-Judge模式
python run.py --mode database --paper-id P001 --always-use-llm

# 禁用幻觉过滤（证据验证）
python run.py --mode database --paper-id P001 --always-use-llm --no-hallucination-filter
```

**输入**: PostgreSQL数据库中的`agent_audit_result`表 + `main_rules`/`rule_judge`规则表
**输出**: 保存到`reflect_agent_verdict`表 + Markdown报告

**自动模式逻辑**：
- 检查`agent_audit_result`表中同一`paper_id`是否同时有FMT/REF/EXP/LOG四个审计组的结果
- 检查每个agent是否已应用其在`main_rules`表中定义的所有规则（规则完整性检查）
- 按`(paper_id, rule_id)`去重，保留`audit_time`最新的记录
- 若该`paper_id`未被反思评估过，自动执行反思评估
- 若该`paper_id`的审计结果有更新（`audit_time`在先前`verdict_time`之后），重新评估
- 若审计组不全或规则未全部应用，跳过（除非指定`--paper-id`强制评估）

**强制模式**（指定`--paper-id`）：
- 忽略审计组不全的限制，但在终端输出警告
- 在`final_verdict`字段中说明审计组不全的情况

### 模式3: 交互式模式
```bash
python run.py --mode interactive
```

## 数据格式

### 数据库输入格式（agent_audit_result表的result_json字段）
```json
{
  "agent_code": "EXP",
  "audit_results": [
    {
      "result_id": "item-001",
      "paper_id": "paper-001",
      "point": "统计学显著性检验",
      "rule_id": "EXP-006",
      "score": 3,
      "level": "Warning",
      "description": "实验三数据分布不均，未进行正态性检验。",
      "evidence_quote": "原文第4.2节提到：'我们直接采用了T检验...'",
      "location": {"section": "4.2", "line_start": 45},
      "suggestion": "建议补充Shapiro-Wilk检验。"
    }
  ]
}
```

**agent_code**: FMT(格式审计)/REF(文献审计)/EXP(实验数据)/LOG(逻辑审计)

### 文件输入格式（兼容旧格式）
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
├── run.py                          # 主运行程序
├── requirements.txt                # 依赖列表
├── .env                            # API密钥配置（需自行创建）
├── src/
│   ├── db/                         # 统一数据库模块
│   │   └── database.py             # PostgreSQL连接管理（agent_audit_result/reflect_agent_verdict/main_rules/rule_judge）
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
├── results/                        # JSON结果输出目录
├── reports/                        # Markdown报告输出目录
└── docs/                           # 文档目录
    ├── work_week3.md               # Week3任务说明
    ├── rule for database.md        # 数据库规则说明
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

### 数据库配置
如果使用数据库模式，需要配置PostgreSQL连接：
- 主机: 10.13.1.26
- 端口: 5432
- 用户名: admin
- 密码: ABCabc@123
- 数据库: postgres

**涉及的数据库表**：
| 表名 | 用途 |
|:---|:---|
| `main_rules` | 评审规则总表（22条规则） |
| `rule_judge` | 规则判定属性表（阈值、运算符等） |
| `agent_audit_result` | 4个审计组的审计结果（输入） |
| `paper_sections` | 论文章节内容（用于幻觉过滤） |
| `reflect_agent_verdict` | 反思评估结果（输出） |

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

### 1. 智能混合裁决模式（Hybrid Judge Pattern）

系统采用**规则引擎 + LLM-as-a-Judge**的混合模式：

**设计理念**：
- 不是所有情况都需要LLM裁决
- 当5个审计组意见高度一致时，规则引擎足以处理
- 只在出现明显冲突时才调用LLM进行深度分析

**三层裁决机制**：

1. **快速路径（无冲突）**
   - 条件：分数差异<20分，无级别冲突，无语义矛盾
   - 处理：加权平均计算（LOG:1.2、EXP:1.1、REF:1.0、FMT:0.8）
   - 响应时间：<1秒
   - API成本：0

2. **LLM裁决路径（有冲突）**
   - 条件：检测到分数差异≥20分、级别冲突或语义矛盾
   - 处理：DeepSeek API深度分析，生成裁决意见
   - 响应时间：30-40秒
   - API成本：约1500 tokens

3. **证据验证层**
   - 无论是否有冲突，都进行证据真实性验证
   - Warning/Critical级别问题必须有有效证据
   - 基于证据验证率调整最终评分

**优势**：
- ✅ 成本效益：减少70%的API调用
- ✅ 响应速度：无冲突场景下快速响应
- ✅ 质量保证：关键冲突仍由LLM专业裁决
- ✅ 可配置：可调整阈值或强制始终使用LLM

**如何改为纯LLM-as-a-Judge模式**：

**方法1：使用命令行参数（推荐）**
```bash
# 启用纯LLM-as-a-Judge模式
python run.py --mode database --paper-id xxx --always-use-llm
python run.py --mode file --prompts-dir prompts --always-use-llm
```

**方法2：修改配置文件**
在`src/common/config_c.py`中设置：
```python
class ConflictResolutionConfig(BaseModel):
    always_use_llm: bool = True  # 改为True
```

**方法3：通过环境变量**
```bash
export REF_CONFLICT_RESOLUTION__ALWAYS_USE_LLM=true
python run.py --mode database --paper-id xxx
```

**效果对比**：
- 混合模式（默认）：无冲突时不调用API，响应<1秒，成本0
- 纯LLM模式（--always-use-llm）：始终调用API，响应30-40秒，成本约1500 tokens

### 2. JSON截断自动修复
当DeepSeek API响应被截断时（达到max_tokens限制），系统会：
- 自动检测截断（finish_reason == "length"）
- 统计未闭合的括号
- 自动补全缺失的 `]` 和 `}`
- 4层JSON修复策略确保解析成功

### 3. 证据验证与幻觉过滤
- 验证所有evidence_quote是否在原文中存在
- Warning/Critical级别问题必须包含有效证据
- 自动剔除无证据或证据无效的问题
- 基于证据验证分数调整最终评分（`<0.7`扣分最多14分，`>0.9`加分最多1分）
- 可通过`--no-hallucination-filter`参数禁用

### 4. 智能冲突裁决
- 多维度冲突检测：分数差异、语义冲突、级别冲突
- DeepSeek主考官Agent进行智能裁决
- 加权投票机制（不同审计组权重不同）
- 详细的裁决理由和建议

### 5. 导师对话生成
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
# 在本地prompts目录生成测试数据
python tests/generate_test_data.py --mode file --num-papers 3 --output-dir prompts
# 在数据库中生成测试数据（默认，自动获取paper_id）
python tests/generate_test_data.py --mode database --num-papers 3
# 在数据库中生成测试数据（显式指定已存在的paper_id）
python tests/generate_test_data.py --mode database --num-papers 1 --use-existing-papers
```

### Q: 为什么数据库模式下没有显示DeepSeek API调用？

系统采用**智能混合裁决模式**，结合了规则引擎和LLM-as-a-Judge：

**工作流程**：
1. **规则引擎预筛选**：检测明显冲突（分数差异≥20分、级别冲突、语义矛盾）
2. **LLM裁决**：只对检测到的冲突调用DeepSeek API进行深度分析
3. **无冲突场景**：如果4个审计组结果一致性高，直接计算加权平均分

**为什么这样设计**：
- ✅ **成本优化**：避免不必要的API调用（每次调用约1500 tokens）
- ✅ **效率提升**：无冲突时响应时间从40秒降至<1秒
- ✅ **质量保证**：有冲突时仍然使用LLM进行专业裁决

**如何验证API功能**：
```bash
# 使用file模式测试（测试数据有较大分数差异）
python run.py --mode file --prompts-dir prompts

# 或者在数据库中插入分数差异≥20分的测试数据
```

**冲突检测阈值**：
- 分数差异阈值：20分（可在`src/conflict_resolution/conflict_resolver.py`中修改`DEFAULT_SCORE_DIFF_THRESHOLD`）
- 置信度阈值：0.7（可通过环境变量`CONFLICT_THRESHOLD`设置）

**注意**：这是一种**实用的混合模式**，而非纯粹的LLM-as-a-Judge模式。如果需要让LLM评估所有结果（无论是否有冲突），可以修改代码始终调用LLM。

### Q: 程序完成后出现SSL transport错误？
这个问题已经**完全解决**。系统实现了四层SSL错误防护：
1. **HTTP客户端正确关闭**：在程序结束时显式关闭所有连接
2. **Asyncio异常处理器**：抑制事件循环中的SSL错误日志
3. **警告过滤器**：防止ResourceWarning和SSL警告显示
4. **Stderr过滤器**：拦截垃圾回收器产生的SSL析构错误（最终解决方案）

现在程序可以完全干净地退出，不会显示任何SSL相关的错误或警告。

## 团队分工

### 幻觉评估组
- **成员A（王子勋）**: 冲突裁决、整体评分、JSON修复、项目集成
- **成员B（李健博）**: 重复过滤、证据验证、幻觉过滤

### 对话/交互开发组
- **成员C（辛雨谌）**: 导师对话生成、对话质量评估
- **成员D（王婧伊）**: 优先级排序、人工复核标记、规则引擎

## 文档

- [Week3任务说明](docs/work_week3.md) - 数据库表结构与代码修改说明
- [数据库规则说明](docs/rule%20for%20database.md) - 评审规则详细清单
- [模块分工说明](docs/group_seperate_work.txt) - 技术实现

## 许可证

本项目用于学校课程项目，仅供学习和研究使用。

---

**最后更新**: 2026-03-19
**版本**: v1.4
**状态**: 所有核心功能已实现并测试通过

## 更新日志

### v1.4 (2026-03-19)
- 修复`initial_score`/`conflict_penalty`/`final_score`计算逻辑：
  - `initial_score` = 各审计agent所有rule的`score_obtained`之和
  - `conflict_penalty` = 冲突裁决扣分（Critical: -5, Warning: -2）
  - `final_score` = 归一化100分制 - conflict_penalty - 证据验证调整
- 修复`filtered_suggestions`和`prioritized_suggestions`输出：
  - `filtered_suggestions`：去重后的全部审计建议（含冲突裁决 + 不合规审计记录）
  - `prioritized_suggestions`：仅保留critical和warning级别，按优先级排序
- 新增`--no-hallucination-filter`命令行参数，可禁用幻觉过滤（证据验证）
- 提高幻觉过滤权重（`validation_score < 0.7`时最多扣14分）
- 数据库模式自动模式下新增规则完整性检查（检查每个agent是否已应用所有规则）
- 审计结果按`(paper_id, rule_id)`去重，保留`audit_time`最新的记录
- `save_verdict`改为UPSERT，支持重复评估同一论文
- 修复`conflict_resolver.py`中数据库连接管理（由`run.py`统一管理，避免连接池被提前关闭）
- 修复`evidence.py`中`from .utils import cosine_sim`引用错误

### v1.3 (2026-03-19)
- 输入表从`agent_audits`改为`agent_audit_result`（新表结构含`result_json`字段）
- 输出表从`agent_audits`改为`reflect_agent_verdict`（含`initial_score`/`conflict_penalty`/`final_score`等字段）
- 新增从`main_rules`和`rule_judge`表读取评审规则（不再硬编码）
- 新增自动检查4个审计组（FMT/REF/EXP/LOG）是否齐全
- 新增自动检查是否已评估/需要重新评估（基于时间戳比较）
- `--paper-id`参数支持强制评估（即使审计组不全，终端提醒并在`final_verdict`中说明）
- 冲突裁决模块支持新的`agent_code`（FMT/REF/EXP/LOG）权重配置
- 兼容旧格式（`group_id` + `audit_results`）和新格式（`agent_code` + `audit_results`）

### v1.2 (2026-03-07)
- ✅ 添加可配置的`always_use_llm`选项（方案3实现）
- ✅ 支持纯LLM-as-a-Judge模式和智能混合模式切换
- ✅ 新增`--always-use-llm`命令行参数
- ✅ 添加ConflictResolutionConfig配置类
- ✅ 优化日志输出，明确显示当前使用的裁决模式
- ✅ **彻底解决SSL transport错误**（四层防护：HTTP客户端关闭 + Asyncio异常处理器 + 警告过滤器 + Stderr过滤器）

### v1.1 (2026-03-07)
- ✅ 修复数据库模式下result_json字段解析问题
- ✅ 修复数据库模式下审计结果过滤逻辑（排除反思评估组历史结果）
- ✅ 完全解决SSL transport警告（三层防护：正确关闭HTTP客户端 + 自定义异常处理器 + 警告过滤器）
- ✅ 移除Windows控制台不兼容的emoji字符
- ✅ 优化资源清理流程，在main函数统一管理连接关闭
