# 成员B模块

本模块实现：
- 证据验证（精确匹配 + 语义匹配兜底）
- 重复过滤（Sentence-Transformers + DBSCAN）
- 数据库段落拉取（asyncpg）

## 目录结构
- `evidence.py`：证据验证逻辑
- `dedup.py`：重复过滤聚类
- `db.py`：数据库访问（paper_sections）
- `models.py`：请求/响应数据结构
- `config.py`：阈值与参数配置
- `member_b_module.py`：对外入口函数

## 最小使用示例

```python
from member_b.member_b_module import run_evidence_validation
from member_b.models import EvidenceItem, EvidenceValidationRequest, Section

sections = [
    Section(section_id="3.2", title="方法", paragraphs=["我们在数据集Y上采用方法X提升12%。"])
]
items = [
    EvidenceItem(item_id="item-001", agent="method_agent", claim_text="方法X在数据集Y上提升12%", quote=None)
]
request = EvidenceValidationRequest(
    paper_id="paper-001",
    paper_field="软件工程",
    sections=sections,
    items=items,
    config={"top_k": 5, "semantic_threshold": 0.86},
)
results = run_evidence_validation(request)
```
