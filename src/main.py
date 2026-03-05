"""
反思评估组主入口文件
整合了四位成员的工作：
- 成员A：冲突裁决和整体评分
- 成员B：重复过滤和幻觉过滤
- 成员C：导师对话生成
- 成员D：优先级排序和人工复核标记
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional

from src.conflict_resolver import ConflictResolver
from src.dedup import Deduplicator
from src.evidence import EvidenceValidator
from src.dialogue_engine import DialogueEngine
from src.review_engine import ReviewDecisionEngine
from src.models import (
    ConflictResolutionRequest,
    ReflectionResult,
    PrioritizedIssue,
    MentorDialogue,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReflectionJudgeOrchestrator:
    """反思评估组编排器，协调各模块工作"""

    def __init__(self):
        self.conflict_resolver = ConflictResolver()
        self.deduplicator = Deduplicator()
        self.evidence_validator = EvidenceValidator()
        self.dialogue_engine = DialogueEngine()
        self.review_engine = ReviewDecisionEngine()

    async def process_paper_review(
        self,
        paper_id: str,
        paper_title: str,
        paper_content: str,
        agent_results: List[Dict[str, Any]],
        field: str = "软件工程"
    ) -> ReflectionResult:
        """
        处理论文评审的完整流程

        Args:
            paper_id: 论文ID
            paper_title: 论文标题
            paper_content: 论文内容
            agent_results: 各审计组的评审结果
            field: 论文领域

        Returns:
            ReflectionResult: 反思评估最终结果
        """
        logger.info(f"开始处理论文评审: {paper_id} - {paper_title}")

        # 步骤1：成员D - 优先级排序和复核标记
        logger.info("步骤1: 优先级排序和复核标记")
        sorted_results, review_marks = self.review_engine.process_audit_results(agent_results)

        # 步骤2：成员A - 冲突裁决
        logger.info("步骤2: 冲突裁决")
        conflict_request = ConflictResolutionRequest(
            metadata={"paper_id": paper_id, "paper_title": paper_title},
            payload={"agent_results": agent_results}
        )
        conflict_response = await self.conflict_resolver.resolve_conflicts(conflict_request)

        # 步骤3：提取优先级问题
        logger.info("步骤3: 提取优先级问题")
        critical_issues = []
        major_issues = []
        minor_issues = []

        for issue in conflict_response.result.get("resolved_issues", []):
            priority_issue = PrioritizedIssue(
                description=issue.get("resolved_comment", ""),
                priority=issue.get("final_level", "Info").lower(),
                agents=[issue.get("agent1_name", ""), issue.get("agent2_name", "")],
                evidence=issue.get("root_cause", "")
            )

            if issue.get("final_level") == "Critical":
                critical_issues.append(priority_issue)
            elif issue.get("final_level") == "Warning":
                major_issues.append(priority_issue)
            else:
                minor_issues.append(priority_issue)

        # 步骤4：成员C - 生成导师对话
        logger.info("步骤4: 生成导师对话")
        all_issues = critical_issues + major_issues + minor_issues
        mentor_dialogue = None
        if all_issues:
            mentor_dialogue = await self.dialogue_engine.generate_dialogue(field, all_issues)

        # 步骤5：确定是否需要人工复核
        needs_human_review = len(review_marks) > 0
        human_review_reason = None
        if needs_human_review:
            reasons = [mark.trigger_reason for mark in review_marks[:3]]
            human_review_reason = "; ".join(reasons)

        # 步骤6：构建最终结果
        final_verdict = conflict_response.result.get("final_verdict", {})
        final_score = final_verdict.get("average_score", 70.0)
        verdict = final_verdict.get("verdict", "待定")

        result = ReflectionResult(
            paper_id=paper_id,
            final_score=final_score,
            verdict=verdict,
            critical_issues=critical_issues,
            major_issues=major_issues,
            minor_issues=minor_issues,
            needs_human_review=needs_human_review,
            human_review_reason=human_review_reason,
            mentor_dialogue=mentor_dialogue,
            plugin_metadata={
                "conflict_resolution": conflict_response.result,
                "review_marks": [mark.dict() for mark in review_marks],
                "sorted_results": [res.dict() for res in sorted_results]
            }
        )

        logger.info(f"论文评审完成: 最终得分={final_score}, 结论={verdict}")
        return result


async def main():
    """主函数示例"""
    orchestrator = ReflectionJudgeOrchestrator()

    # 示例数据
    paper_id = "test_paper_001"
    paper_title = "基于深度学习的软件缺陷预测研究"
    paper_content = "这是论文内容..."
    agent_results = [
        {
            "agent_info": {"name": "逻辑审计组", "version": "v1.0"},
            "result": {
                "score": 85,
                "audit_level": "Info",
                "comment": "论文逻辑清晰",
                "suggestion": "建议补充更多实验数据"
            }
        }
    ]

    result = await orchestrator.process_paper_review(
        paper_id=paper_id,
        paper_title=paper_title,
        paper_content=paper_content,
        agent_results=agent_results
    )

    print(f"评审结果: {result.verdict}")
    print(f"最终得分: {result.final_score}")


if __name__ == "__main__":
    asyncio.run(main())
