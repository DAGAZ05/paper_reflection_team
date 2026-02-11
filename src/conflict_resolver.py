import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from .database import DatabaseClient
from .llm_client import LLMClient
from .schemas import ConflictResolutionRequest, ConflictResolutionResponse, ConflictType

logger = logging.getLogger(__name__)

app = FastAPI(title="Reflection Judge - Conflict Resolver")

DEFAULT_SCORE_DIFF_THRESHOLD = 20
LEVEL_PRIORITY = {"Info": 0, "Warning": 1, "Critical": 2}


class ConflictResolver:
    def __init__(self):
        self.llm_client = LLMClient()
        self.db_client = DatabaseClient()
        self.conflict_patterns = self._load_conflict_patterns()

    def _load_conflict_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load keyword-based conflict patterns used by the pre-filter rule engine."""
        return {
            "efficiency": {
                "keywords": [
                    "高效", "性能好", "快速", "优化",
                    "低效", "性能差", "缓慢", "耗时"
                ],
                "pattern": r"(高效|性能好|快速|优化|低效|性能差|缓慢|耗时)",
            },
            "quality": {
                "keywords": [
                    "质量高", "精确", "准确", "质量低", "不精确", "错误"
                ],
                "pattern": r"(质量高|精确|准确|质量低|不精确|错误)",
            },
            "completeness": {
                "keywords": ["完整", "全面", "缺失", "不足", "缺少"],
                "pattern": r"(完整|全面|缺失|不足|缺少)",
            },
        }

    @staticmethod
    def _coerce_score(value: Any, default: int = 70) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _normalize_agent_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Normalize agent results from list/object/JSON-string into the canonical list schema."""
        parsed = raw_results

        if isinstance(parsed, str):
            text = parsed.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"agent_results JSON字符串解析失败: {exc.msg}") from exc

        if isinstance(parsed, dict):
            if "agent_results" in parsed:
                return self._normalize_agent_results(parsed["agent_results"])
            parsed = [parsed]

        if not isinstance(parsed, list):
            raise ValueError("agent_results必须是列表、JSON字符串或包含agent_results的对象")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(parsed):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    logger.warning("Skip unparsable agent result string at index=%d", idx)
                    continue

            if not isinstance(item, dict):
                logger.warning("Skip non-object agent result at index=%d", idx)
                continue

            result_data = item.get("result")
            if not isinstance(result_data, dict):
                # Compatibility: some groups may put result fields directly in result_json.
                result_data = item.get("result_json", {})
            if not isinstance(result_data, dict):
                result_data = {}

            agent_info = item.get("agent_info")
            if not isinstance(agent_info, dict):
                agent_info = {
                    "name": item.get("agent_name") or f"agent_{idx + 1}",
                    "version": item.get("agent_version", "unknown"),
                }

            normalized.append(
                {
                    "request_id": item.get("request_id", ""),
                    "agent_info": {
                        "name": agent_info.get("name", f"agent_{idx + 1}"),
                        "version": agent_info.get("version", "unknown"),
                    },
                    "result": {
                        "score": self._coerce_score(result_data.get("score", item.get("score", 70))),
                        "audit_level": result_data.get("audit_level", item.get("audit_level", "Info")),
                        "comment": result_data.get("comment", ""),
                        "suggestion": result_data.get("suggestion", ""),
                        "tags": result_data.get("tags", []),
                    },
                    "usage": item.get("usage", {"tokens": 0, "latency_ms": 0}),
                }
            )

        return normalized

    @staticmethod
    def _normalize_resolution_data(resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(resolution_data or {})
        normalized.setdefault("conflicts_resolved", bool(normalized.get("resolved_issues")))
        normalized.setdefault("resolved_issues", [])
        normalized.setdefault("confidence_score", 0.5)
        return normalized

    @staticmethod
    def _attach_result_json(result_data: Dict[str, Any]) -> Dict[str, Any]:
        core_keys = [
            "conflicts_resolved",
            "resolved_issues",
            "confidence_score",
            "tags",
            "final_verdict",
            "paper_id",
        ]
        result_data["result_json"] = {key: result_data.get(key) for key in core_keys if key in result_data}
        return result_data

    def detect_conflicts(
        self,
        agent_results: List[Dict[str, Any]],
        conflict_threshold: Optional[float] = None,
        score_diff_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not agent_results or len(agent_results) < 2:
            logger.info("Agent result count < 2, skip conflict detection")
            return []

        comments: List[Dict[str, Any]] = []
        for result in agent_results:
            agent_info = result.get("agent_info")
            if not agent_info or "name" not in agent_info:
                logger.warning("Skip malformed agent result: request_id=%s", result.get("request_id", "unknown"))
                continue

            result_data = result.get("result", {})
            if isinstance(result_data, dict) and "comment" in result_data:
                comments.append(
                    {
                        "agent": agent_info["name"],
                        "comment": result_data.get("comment", ""),
                        "level": result_data.get("audit_level", "Info"),
                        "score": self._coerce_score(result_data.get("score", 70)),
                        "tags": result_data.get("tags", []),
                        "suggestion": result_data.get("suggestion", ""),
                    }
                )

        conflicts: List[Dict[str, Any]] = []
        score_threshold = DEFAULT_SCORE_DIFF_THRESHOLD if score_diff_threshold is None else score_diff_threshold

        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                c1, c2 = comments[i], comments[j]
                conflict_info = self._analyze_comment_conflict(c1, c2)

                score_diff = abs(c1["score"] - c2["score"])
                if score_diff >= score_threshold and not conflict_info:
                    conflict_info = {
                        "type": ConflictType.MEASUREMENT_DIFFERENCE,
                        "confidence": min(0.6 + (score_diff - score_threshold) * 0.01, 0.9),
                    }

                if conflict_info:
                    conflicts.append(
                        {
                            "agent1": c1["agent"],
                            "agent2": c2["agent"],
                            "comment1": c1["comment"],
                            "comment2": c2["comment"],
                            "level1": c1["level"],
                            "level2": c2["level"],
                            "score1": c1["score"],
                            "score2": c2["score"],
                            "conflict_type": conflict_info["type"],
                            "confidence": conflict_info["confidence"],
                        }
                    )

        threshold = float(os.getenv("CONFLICT_THRESHOLD", "0.7")) if conflict_threshold is None else float(conflict_threshold)
        conflicts.sort(key=lambda x: x["confidence"], reverse=True)
        filtered = [c for c in conflicts if c["confidence"] >= threshold]
        logger.info("Conflict detection done: %d found, %d kept", len(conflicts), len(filtered))
        return filtered

    def _analyze_comment_conflict(self, comment1: Dict[str, Any], comment2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text1 = comment1.get("comment", "").lower()
        text2 = comment2.get("comment", "").lower()

        for pattern_info in self.conflict_patterns.values():
            pattern = pattern_info["pattern"]
            matches1 = re.findall(pattern, text1)
            matches2 = re.findall(pattern, text2)
            if matches1 and matches2:
                for m1 in matches1:
                    for m2 in matches2:
                        if self._are_opposites(m1, m2):
                            level_diff = abs(
                                LEVEL_PRIORITY.get(comment1.get("level", "Info"), 0)
                                - LEVEL_PRIORITY.get(comment2.get("level", "Info"), 0)
                            )
                            return {
                                "type": ConflictType.DIRECT_CONTRADICTION,
                                "confidence": min(0.7 + level_diff * 0.1, 0.95),
                            }

        if self._has_contextual_dependency(text1, text2):
            return {"type": ConflictType.CONTEXT_DEPENDENT, "confidence": 0.65}

        return None

    def _are_opposites(self, word1: str, word2: str) -> bool:
        opposites = [
            ("高效", "低效"),
            ("性能好", "性能差"),
            ("快速", "缓慢"),
            ("完整", "缺失"),
            ("准确", "错误"),
            ("优化", "恶化"),
            ("质量高", "质量低"),
            ("精确", "不精确"),
        ]
        return (word1, word2) in opposites or (word2, word1) in opposites

    def _has_contextual_dependency(self, text1: str, text2: str) -> bool:
        contextual_triggers = [
            ("不同", "相同"),
            ("部分", "整体"),
            ("短期", "长期"),
            ("理论", "实践"),
            ("假设", "验证"),
        ]
        for t1, t2 in contextual_triggers:
            if (t1 in text1 and t2 in text2) or (t2 in text1 and t1 in text2):
                return True
        return False

    def deduplicate_issues(self, resolved_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not resolved_issues:
            return []

        seen: Dict[Any, Dict[str, Any]] = {}
        deduped: List[Dict[str, Any]] = []

        for issue in resolved_issues:
            key = (
                issue.get("conflict_type", ""),
                issue.get("final_level", ""),
                frozenset({issue.get("agent1_name", ""), issue.get("agent2_name", "")}),
            )
            if key in seen:
                existing = seen[key]
                if issue.get("confidence", 0) > existing.get("confidence", 0):
                    seen[key] = issue
                    deduped = [issue if d is existing else d for d in deduped]
            else:
                seen[key] = issue
                deduped.append(issue)

        return deduped

    def sort_by_priority(self, resolved_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def priority_key(issue: Dict[str, Any]) -> Any:
            level_score = LEVEL_PRIORITY.get(issue.get("final_level", "Info"), 0)
            confidence = issue.get("confidence", 0)
            needs_review = 1 if issue.get("needs_human_review", False) else 0
            return (-level_score, -confidence, -needs_review)

        return sorted(resolved_issues, key=priority_key)

    def compute_final_verdict(self, resolved_issues: List[Dict[str, Any]], agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_scores = []
        for result in agent_results:
            result_data = result.get("result", {})
            if isinstance(result_data, dict):
                all_scores.append(self._coerce_score(result_data.get("score", 70)))

        avg_score = sum(all_scores) / max(1, len(all_scores))

        level_counts = {"Info": 0, "Warning": 0, "Critical": 0}
        for issue in resolved_issues:
            level = issue.get("final_level", "Info")
            level_counts[level] = level_counts.get(level, 0) + 1

        if level_counts["Critical"] > 0:
            verdict = "存在关键问题，建议大修后重新提交（Major Revision）"
        elif level_counts["Warning"] >= 3:
            verdict = "存在多处需要关注的问题，建议修改后录用（Minor Revision with conditions）"
        elif level_counts["Warning"] > 0:
            verdict = "存在少量问题，建议小修后录用（Minor Revision）"
        elif avg_score >= 85:
            verdict = "论文质量良好，建议直接录用（Accept）"
        else:
            verdict = "论文基本合格，建议小修后录用（Minor Revision）"

        return {
            "average_score": round(avg_score, 1),
            "level_distribution": level_counts,
            "verdict": verdict,
            "total_conflicts": len(resolved_issues),
            "needs_human_review_count": sum(1 for issue in resolved_issues if issue.get("needs_human_review", False)),
        }

    def generate_markdown_report(self, resolution_data: Dict[str, Any], paper_title: str = "未知论文") -> str:
        final_verdict = resolution_data.get("final_verdict", {})
        resolved_issues = resolution_data.get("resolved_issues", [])
        confidence = resolution_data.get("confidence_score", 0)

        lines = [
            "# 论文评审冲突裁决报告",
            "",
            f"**论文标题**: {paper_title}",
            f"**综合得分**: {final_verdict.get('average_score', 'N/A')}",
            f"**裁决置信度**: {confidence:.2f}",
            f"**最终结论**: {final_verdict.get('verdict', '待定')}",
            "",
            "---",
            "",
            "## 一、总体评价",
            "",
        ]

        level_dist = final_verdict.get("level_distribution", {})
        lines.append(
            f"本次评审共检测到 **{len(resolved_issues)}** 个冲突，"
            f"其中 Critical {level_dist.get('Critical', 0)} 个、"
            f"Warning {level_dist.get('Warning', 0)} 个、"
            f"Info {level_dist.get('Info', 0)} 个。"
        )

        review_count = final_verdict.get("needs_human_review_count", 0)
        if review_count > 0:
            lines.append(f"有 **{review_count}** 个问题建议人工复核。")

        lines.append("")

        if resolved_issues:
            lines.append("## 二、冲突裁决详情")
            lines.append("")
            for idx, issue in enumerate(resolved_issues, 1):
                level = issue.get("final_level", "Info")
                level_icon = {"Critical": "[!]", "Warning": "[?]", "Info": "[i]"}.get(level, "[i]")
                lines.append(f"### {idx}. {level_icon} {issue.get('conflict_type', '未知类型')} ({level})")
                lines.append("")
                lines.append(f"- **冲突双方**: {issue.get('agent1_name', '?')} vs {issue.get('agent2_name', '?')}")
                if issue.get("root_cause"):
                    lines.append(f"- **根本原因**: {issue['root_cause']}")
                lines.append(f"- **裁决意见**: {issue.get('resolved_comment', '')}")
                lines.append(f"- **改进建议**: {issue.get('resolved_suggestion', '')}")
                if issue.get("needs_human_review"):
                    lines.append("- **需要人工复核**")
                lines.append("")

        lines.extend(
            [
                "## 三、最终建议",
                "",
                f"{final_verdict.get('verdict', '待定')}",
                "",
                "---",
                "*由 ReflectionJudge_ConflictResolver v1.0 自动生成*",
            ]
        )

        return "\n".join(lines)

    async def resolve_conflicts(self, request: ConflictResolutionRequest) -> ConflictResolutionResponse:
        try:
            if not request.payload:
                raise ValueError("payload不能为空")

            agent_results = request.payload.get("agent_results", [])
            paper_id = request.metadata.get("paper_id")
            paper_title = request.metadata.get("paper_title", "未知论文")
            paper_context = ""

            if not agent_results and paper_id:
                await self.db_client.connect()
                agent_results = await self.db_client.get_agent_results(paper_id)
                paper_context = await self.db_client.get_paper_content(paper_id)

            agent_results = self._normalize_agent_results(agent_results)

            if not agent_results:
                result_data = {
                    "conflicts_resolved": False,
                    "resolved_issues": [],
                    "confidence_score": 1.0,
                    "tags": ["Executive_Summary"],
                    "final_verdict": {
                        "average_score": 0,
                        "level_distribution": {},
                        "verdict": "无Agent结果可供裁决",
                        "total_conflicts": 0,
                        "needs_human_review_count": 0,
                    },
                    "message": "无Agent结果可供裁决",
                }
                return ConflictResolutionResponse(
                    request_id=request.request_id,
                    result=self._attach_result_json(result_data),
                    usage={"tokens": 0, "latency_ms": 0},
                )

            conflicts = self.detect_conflicts(
                agent_results,
                conflict_threshold=request.config.get("conflict_threshold"),
                score_diff_threshold=request.config.get("score_diff_threshold"),
            )

            if not conflicts:
                result_data = {
                    "conflicts_resolved": False,
                    "resolved_issues": [],
                    "confidence_score": 0.95,
                    "tags": ["Executive_Summary", "Score_Calibration"],
                    "final_verdict": self.compute_final_verdict([], agent_results),
                    "paper_id": paper_id,
                }
                result_data["markdown_report"] = self.generate_markdown_report(result_data, paper_title)
                return ConflictResolutionResponse(
                    request_id=request.request_id,
                    result=self._attach_result_json(result_data),
                    usage={"tokens": 0, "latency_ms": 50},
                )

            resolution_data, usage = await self.llm_client.resolve_conflicts_with_llm(request, conflicts, paper_context)
            resolution_data = self._normalize_resolution_data(resolution_data)

            resolved_issues = self.deduplicate_issues(resolution_data.get("resolved_issues", []))
            resolution_data["resolved_issues"] = self.sort_by_priority(resolved_issues)
            resolution_data["final_verdict"] = self.compute_final_verdict(resolution_data["resolved_issues"], agent_results)
            resolution_data["tags"] = ["Executive_Summary", "Critical_Fix_List", "Score_Calibration"]
            resolution_data["markdown_report"] = self.generate_markdown_report(resolution_data, paper_title)

            if paper_id:
                resolution_data["paper_id"] = paper_id
                await self.db_client.save_resolution_result(request.request_id, resolution_data, usage)

            return ConflictResolutionResponse(
                request_id=request.request_id,
                result=self._attach_result_json(resolution_data),
                usage=usage,
            )

        except ValueError as exc:
            logger.error("冲突裁决失败: %s", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("冲突裁决失败: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"内部服务错误: {exc}") from exc
        finally:
            await self.db_client.disconnect()


resolver = ConflictResolver()


@app.post("/api/resolve_conflicts", response_model=ConflictResolutionResponse)
async def resolve_conflicts_endpoint(request: ConflictResolutionRequest):
    return await resolver.resolve_conflicts(request)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "conflict_resolver"}
