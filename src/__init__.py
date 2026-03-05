"""
反思评估组 - 统一模块
整合了四位成员的工作成果
"""

from .models import (
    # 枚举类型
    AuditLevel,
    ConflictType,
    AgentType,
    # 成员B的模型
    Section,
    EvidenceItem,
    EvidenceSpan,
    EvidenceValidationResult,
    DedupItem,
    DedupCluster,
    DedupResult,
    # 成员A的模型
    AgentResultData,
    AgentResult,
    ConflictResolutionRequest,
    ConflictResolutionResponse,
    ResolvedIssue,
    # 成员D的模型
    AuditResult,
    ReviewMarkResult,
    SortedAuditResult,
    # 成员C的模型
    PrioritizedIssue,
    MentorDialogue,
    ReflectionResult,
)

from .conflict_resolver import ConflictResolver
from .dedup import Deduplicator
from .evidence import EvidenceValidator
from .dialogue_engine import DialogueEngine
from .review_engine import ReviewDecisionEngine
from .main import ReflectionJudgeOrchestrator

__version__ = "1.0.0"
__all__ = [
    # 主编排器
    "ReflectionJudgeOrchestrator",
    # 核心模块
    "ConflictResolver",
    "Deduplicator",
    "EvidenceValidator",
    "DialogueEngine",
    "ReviewDecisionEngine",
    # 数据模型
    "AuditLevel",
    "ConflictType",
    "AgentType",
    "Section",
    "EvidenceItem",
    "ReflectionResult",
]
