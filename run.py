"""
反思评估组主运行程序
支持两种运行方案：
1. 从PostgreSQL数据库读取agent_audits表
2. 从prompts文件夹读取JSON文件
"""
import asyncio
import json
import logging
import sys
import warnings
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# 抑制SSL相关的警告
warnings.filterwarnings('ignore', category=ResourceWarning)
warnings.filterwarnings('ignore', message='.*SSL.*')

# 重定向stderr以抑制SSL错误（在导入其他模块之前）
class SuppressSSLErrors:
    """抑制SSL相关的stderr输出"""
    def __init__(self):
        self.original_stderr = sys.stderr
        self.null = open(os.devnull, 'w')

    def __enter__(self):
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        self.null.close()

    def write(self, text):
        # 只过滤SSL相关的错误
        if 'SSL' in text or 'ssl' in text.lower() or '_SSLProtocolTransport' in text:
            return
        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import db_manager
from src.common.models import ReflectionResult
from src.common.report_generator import report_generator
from src.conflict_resolution import ConflictResolver
from src.deduplication import Deduplicator
from src.evidence_validation import EvidenceValidator
from src.dialogue_generation import DialogueEngine
from src.priority_sorting import ReviewDecisionEngine
from src.api.deepseek_client import deepseek_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制asyncio的SSL错误日志
logging.getLogger('asyncio').setLevel(logging.CRITICAL)


def custom_exception_handler(loop, context):
    """自定义异常处理器，抑制SSL transport错误"""
    exception = context.get('exception')
    message = context.get('message', '')

    # 忽略SSL相关的错误
    if exception:
        if 'SSL' in str(exception) or 'ssl' in str(exception).lower():
            return
    if 'SSL' in message or 'ssl' in message.lower():
        return

    # 其他错误正常记录
    if exception:
        logger.error(f"Asyncio exception: {exception}", exc_info=exception)
    else:
        logger.error(f"Asyncio error: {message}")


class ReflectionOrchestrator:
    """反思评估编排器"""

    def __init__(self, mode: str = "database", enable_dialogue: bool = False, always_use_llm: bool = False):
        self.conflict_resolver = None
        self.deduplicator = None
        self.evidence_validator = None
        self.dialogue_engine = None
        self.review_engine = None
        self.mode = mode  # "database" or "file"
        self.enable_dialogue = enable_dialogue  # 是否启用导师对话生成
        self.always_use_llm = always_use_llm  # 是否始终使用LLM裁决

    def initialize_modules(self):
        """初始化各模块（延迟初始化以避免导入错误）"""
        try:
            self.conflict_resolver = ConflictResolver(mode=self.mode, always_use_llm=self.always_use_llm)
            self.deduplicator = Deduplicator()
            self.evidence_validator = EvidenceValidator()
            self.dialogue_engine = DialogueEngine()
            self.review_engine = ReviewDecisionEngine()
            logger.info("所有模块初始化成功")
            if self.always_use_llm:
                logger.info("⚠️ 已启用纯LLM-as-a-Judge模式（always_use_llm=True），所有评审都将调用DeepSeek API")
        except Exception as e:
            logger.error(f"模块初始化失败: {e}")
            raise

    async def process_paper(
        self,
        paper_id: str,
        audit_results: List[Dict[str, Any]],
        paper_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理单篇论文的评审

        Args:
            paper_id: 论文ID
            audit_results: 5个审计组的结果列表
            paper_content: 论文内容（用于证据验证）

        Returns:
            反思评估结果
        """
        logger.info(f"开始处理论文: {paper_id}")

        try:
            # 验证输入：确保有5个审计组的结果
            if len(audit_results) != 5:
                logger.warning(f"论文{paper_id}的审计结果数量不足5个，实际为{len(audit_results)}个")

            # 步骤1：优先级排序和复核标记
            logger.info("步骤1: 优先级排序和复核标记")
            sorted_results, review_marks = self.review_engine.process_audit_results(audit_results)

            # 步骤2：冲突裁决
            logger.info("步骤2: 冲突裁决")
            from src.common.models import ConflictResolutionRequest

            conflict_request = ConflictResolutionRequest(
                metadata={"paper_id": paper_id, "paper_title": f"Paper_{paper_id}"},
                payload={"agent_results": audit_results}
            )
            conflict_response = await self.conflict_resolver.resolve_conflicts(conflict_request)

            # 步骤3: 提取优先级问题
            logger.info("步骤3: 提取优先级问题")
            critical_issues = []
            major_issues = []
            minor_issues = []

            for issue in conflict_response.result.get("resolved_issues", []):
                from src.common.models import PrioritizedIssue
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

            # 步骤4: 生成导师对话（可选）
            mentor_dialogue = None
            if self.enable_dialogue:
                logger.info("步骤4: 生成导师对话")
                all_issues = critical_issues + major_issues + minor_issues
                if all_issues:
                    try:
                        mentor_dialogue = await self.dialogue_engine.generate_dialogue("软件工程", all_issues)
                    except Exception as e:
                        logger.warning(f"导师对话生成失败: {e}")
            else:
                logger.info("步骤4: 跳过导师对话生成（未启用）")

            # 步骤5: 确定是否需要人工复核
            needs_human_review = len(review_marks) > 0
            human_review_reason = None
            if needs_human_review:
                reasons = [mark.trigger_reason for mark in review_marks[:3]]
                human_review_reason = "; ".join(reasons)

            # 步骤6: 构建最终结果
            final_verdict = conflict_response.result.get("final_verdict", {})
            final_score = final_verdict.get("average_score", 70.0)
            verdict = final_verdict.get("verdict", "待定")

            # 使用 ReflectionResult 模型
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
                    "paper_title": f"Paper_{paper_id}",
                    "conflict_resolution": conflict_response.result,
                    "review_marks_count": len(review_marks),
                    "sorted_results_count": len(sorted_results),
                    "usage_tokens": conflict_response.usage.get("tokens", 0),
                    "latency_ms": conflict_response.usage.get("latency_ms", 0)
                }
            )

            # 步骤7: 生成Markdown报告
            logger.info("步骤7: 生成Markdown报告")
            markdown_report_path = None
            try:
                # 将 ReflectionResult 转换为字典用于报告生成
                result_dict = result.model_dump()
                result_dict["paper_title"] = result.plugin_metadata.get("paper_title", f"Paper_{paper_id}")

                report_path = report_generator.generate_report(
                    paper_id=paper_id,
                    paper_title=result_dict["paper_title"],
                    result=result_dict
                )
                markdown_report_path = report_path
                logger.info(f"Markdown报告已保存: {report_path}")
            except Exception as e:
                logger.error(f"生成Markdown报告失败: {e}")

            # 将报告路径添加到 plugin_metadata
            if markdown_report_path:
                result.plugin_metadata["markdown_report_path"] = markdown_report_path

            logger.info(f"论文{paper_id}处理完成: 最终得分={final_score}, 结论={verdict}")
            return result

        except Exception as e:
            logger.error(f"处理论文{paper_id}时发生错误: {e}", exc_info=True)
            return ReflectionResult(
                paper_id=paper_id,
                final_score=0.0,
                verdict="处理失败",
                plugin_metadata={"error": str(e)}
            )

    async def run_from_database(self, paper_id: Optional[str] = None):
        """
        方案1: 从数据库读取并处理

        Args:
            paper_id: 指定论文ID，如果为None则处理所有论文
        """
        logger.info("=== 方案1: 从数据库读取 ===")

        try:
            # 连接数据库
            await db_manager.connect()

            # 获取待处理的论文ID列表
            if paper_id:
                paper_ids = [paper_id]
            else:
                paper_ids = await db_manager.get_paper_ids()

            logger.info(f"找到{len(paper_ids)}篇待处理论文")

            # 处理每篇论文
            for pid in paper_ids:
                logger.info(f"\n{'='*60}")
                logger.info(f"处理论文: {pid}")
                logger.info(f"{'='*60}")

                # 读取该论文的所有审计结果
                audit_records = await db_manager.fetch_agent_audits(pid)

                # 过滤掉反思评估组的结果，只保留5个审计组的结果
                audit_records = [
                    record for record in audit_records
                    if record.get("agent_name") != "反思评估组"
                ]

                logger.info(f"过滤后剩余{len(audit_records)}条审计组结果")

                # 提取result_json作为审计结果列表
                audit_results = [record["result_json"] for record in audit_records]

                # 提取task_id（使用第一个记录的task_id）
                task_id = audit_records[0].get("task_id") if audit_records else None

                # 如果没有找到task_id，生成一个新的UUID格式
                if task_id is None:
                    import uuid
                    task_id = str(uuid.uuid4())

                # 获取论文内容
                paper_content = await db_manager.get_paper_content(pid)

                # 处理论文
                result = await self.process_paper(pid, audit_results, paper_content)

                # 保存结果到数据库
                if "error" not in result.plugin_metadata:
                    await db_manager.save_reflection_result(
                        paper_id=pid,
                        task_id=task_id,
                        final_score=result.final_score,
                        verdict=result.verdict,
                        result_json=result.model_dump(),
                        usage_tokens=result.plugin_metadata.get("usage_tokens", 0),
                        latency_ms=result.plugin_metadata.get("latency_ms", 0)
                    )

                # 打印结果摘要
                print(f"\n论文ID: {pid}")
                print(f"最终得分: {result.final_score}")
                print(f"评审结论: {result.verdict}")
                print(f"是否需要人工复核: {result.needs_human_review}")
                if result.human_review_reason:
                    print(f"复核原因: {result.human_review_reason}")
                markdown_path = result.plugin_metadata.get("markdown_report_path")
                if markdown_path:
                    print(f"Markdown报告: {markdown_path}")

        except Exception as e:
            logger.error(f"从数据库读取处理失败: {e}", exc_info=True)
        finally:
            # 关闭数据库连接
            await db_manager.disconnect()

    async def run_from_files(self, prompts_dir: str = "prompts"):
        """
        方案2: 从文件读取并处理

        Args:
            prompts_dir: JSON文件所在目录
        """
        logger.info("=== 方案2: 从文件读取 ===")

        try:
            prompts_path = Path(prompts_dir)
            if not prompts_path.exists():
                logger.error(f"目录不存在: {prompts_path}")
                return

            # 查找所有JSON文件
            json_files = list(prompts_path.glob("*.json"))
            if not json_files:
                logger.warning(f"在{prompts_path}目录下未找到JSON文件")
                return

            logger.info(f"找到{len(json_files)}个JSON文件")

            # 按paper_id分组
            paper_audits = {}
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取paper_id（从文件名或JSON内容）
                    paper_id = data.get("paper_id", json_file.stem)

                    if paper_id not in paper_audits:
                        paper_audits[paper_id] = []

                    paper_audits[paper_id].append(data)
                    logger.info(f"读取文件: {json_file.name} -> paper_id: {paper_id}")

                except Exception as e:
                    logger.error(f"读取文件{json_file}失败: {e}")

            # 处理每篇论文
            for paper_id, audits in paper_audits.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"处理论文: {paper_id}")
                logger.info(f"{'='*60}")

                result = await self.process_paper(paper_id, audits)

                # 保存结果到results文件夹
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                output_file = results_dir / f"result_{paper_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存到: {output_file}")

                # 打印结果摘要
                print(f"\n论文ID: {paper_id}")
                print(f"最终得分: {result.final_score}")
                print(f"评审结论: {result.verdict}")
                print(f"是否需要人工复核: {result.needs_human_review}")
                markdown_path = result.plugin_metadata.get("markdown_report_path")
                if markdown_path:
                    print(f"Markdown报告: {markdown_path}")

        finally:
            # 不需要在这里关闭deepseek_client，会在main函数的finally中统一关闭
            pass


async def main():
    """主函数"""
    # 设置自定义异常处理器以抑制SSL错误
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(custom_exception_handler)

    parser = argparse.ArgumentParser(description="反思评估组主程序")
    parser.add_argument(
        "--mode",
        choices=["database", "file", "interactive"],
        default="interactive",
        help="运行模式: database(从数据库读取), file(从文件读取), interactive(交互式选择)"
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        help="指定论文ID（仅在database模式下有效）"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="prompts",
        help="JSON文件目录（仅在file模式下有效）"
    )
    parser.add_argument(
        "--enable-dialogue",
        action="store_true",
        help="启用导师对话生成（需要DeepSeek API）"
    )
    parser.add_argument(
        "--always-use-llm",
        action="store_true",
        help="启用纯LLM-as-a-Judge模式（始终调用DeepSeek API进行裁决，即使无冲突）"
    )

    args = parser.parse_args()

    try:
        # 根据模式运行
        if args.mode == "interactive":
            print("\n" + "="*60)
            print("反思评估组 - 主运行程序")
            print("="*60)
            print("\n请选择运行方案:")
            print("1. 从PostgreSQL数据库读取 (agent_audits表)")
            print("2. 从prompts文件夹读取JSON文件")
            print("0. 退出")

            choice = input("\n请输入选项 (0-2): ").strip()

            if choice == "1":
                # 创建database模式的编排器
                orchestrator = ReflectionOrchestrator(mode="database", enable_dialogue=args.enable_dialogue, always_use_llm=args.always_use_llm)
                orchestrator.initialize_modules()
                paper_id = input("请输入论文ID (留空处理所有论文): ").strip() or None
                await orchestrator.run_from_database(paper_id)
            elif choice == "2":
                # 创建file模式的编排器
                orchestrator = ReflectionOrchestrator(mode="file", enable_dialogue=args.enable_dialogue, always_use_llm=args.always_use_llm)
                orchestrator.initialize_modules()
                prompts_dir = input(f"请输入JSON文件目录 (默认: prompts): ").strip() or "prompts"
                await orchestrator.run_from_files(prompts_dir)
            elif choice == "0":
                print("退出程序")
            else:
                print("无效选项")

        elif args.mode == "database":
            # 创建database模式的编排器
            orchestrator = ReflectionOrchestrator(mode="database", enable_dialogue=args.enable_dialogue, always_use_llm=args.always_use_llm)
            orchestrator.initialize_modules()
            await orchestrator.run_from_database(args.paper_id)

        elif args.mode == "file":
            # 创建file模式的编排器
            orchestrator = ReflectionOrchestrator(mode="file", enable_dialogue=args.enable_dialogue, always_use_llm=args.always_use_llm)
            orchestrator.initialize_modules()
            await orchestrator.run_from_files(args.prompts_dir)

    finally:
        # 确保在main函数结束前关闭所有HTTP连接
        await deepseek_client.close()
        # 给一点时间让所有异步任务完成清理
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    # 启用SSL错误抑制
    ssl_suppressor = SuppressSSLErrors()

    try:
        with ssl_suppressor:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
