"""
测试数据生成器
模拟5个审计组对论文的评审结果
支持两种模式：
1. file模式：生成JSON文件到prompts目录
2. database模式：插入数据到PostgreSQL的agent_audits表
"""
import json
import random
import uuid
import asyncio
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.database import DatabaseManager

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """测试数据生成器"""

    # 5个审计组的配置
    AUDIT_GROUPS = {
        2: "格式审计组",
        3: "逻辑审计组",
        4: "代码审计组",
        5: "实验数据组",
        6: "文献真实性组"
    }

    # 审核点示例
    AUDIT_POINTS = {
        2: ["标题层级规范", "图表位置", "公式对齐", "参考文献格式", "页边距设置"],
        3: ["摘要与结论一致性", "论证逻辑严密性", "章节衔接", "研究问题明确性", "方法论合理性"],
        4: ["代码规范性", "算法实现正确性", "代码与描述一致性", "README完整性", "依赖管理"],
        5: ["统计学显著性检验", "样本量充足性", "数据分布合理性", "图表数据一致性", "实验可重复性"],
        6: ["参考文献真实性", "引用格式规范", "文献年份合理性", "作者信息完整性", "DOI有效性"]
    }

    # 问题描述模板
    DESCRIPTIONS = {
        "Critical": [
            "发现严重问题：{}",
            "存在关键缺陷：{}",
            "致命错误：{}"
        ],
        "Warning": [
            "需要注意：{}",
            "建议改进：{}",
            "存在问题：{}"
        ],
        "Info": [
            "符合规范：{}",
            "表现良好：{}",
            "基本合格：{}"
        ]
    }

    # 建议模板
    SUGGESTIONS = {
        "Critical": [
            "必须立即修正{}",
            "强烈建议重新审查{}",
            "需要彻底修改{}"
        ],
        "Warning": [
            "建议补充{}",
            "建议优化{}",
            "建议完善{}"
        ],
        "Info": [
            "可以进一步提升{}",
            "保持当前水平",
            "继续保持"
        ]
    }

    @staticmethod
    def generate_audit_result(
        paper_id: str,
        group_id: int,
        num_items: int = 3
    ) -> Dict[str, Any]:
        """
        生成单个审计组的结果

        Args:
            paper_id: 论文ID
            group_id: 审计组ID (2-6)
            num_items: 生成的审核项数量

        Returns:
            审计结果JSON
        """
        audit_results = []

        for i in range(num_items):
            # 随机选择问题级别
            level = random.choices(
                ["Critical", "Warning", "Info"],
                weights=[0.1, 0.3, 0.6]  # Critical少，Info多
            )[0]

            # 根据级别确定分数范围
            if level == "Critical":
                score = random.randint(40, 60)
            elif level == "Warning":
                score = random.randint(60, 80)
            else:
                score = random.randint(80, 95)

            # 选择审核点
            point = random.choice(TestDataGenerator.AUDIT_POINTS[group_id])

            # 生成描述和建议
            desc_template = random.choice(TestDataGenerator.DESCRIPTIONS[level])
            sugg_template = random.choice(TestDataGenerator.SUGGESTIONS[level])

            description = desc_template.format(point)
            suggestion = sugg_template.format(point)

            # 生成证据引用
            evidence_quote = f"原文第{random.randint(1, 10)}.{random.randint(1, 5)}节提到：'{point}相关内容...'"

            audit_results.append({
                "id": f"item-{group_id}-{i+1:03d}",
                "point": point,
                "score": score,
                "level": level,
                "description": description,
                "evidence_quote": evidence_quote,
                "location": {
                    "section": f"{random.randint(1, 10)}.{random.randint(1, 5)}",
                    "line_start": random.randint(1, 500)
                },
                "suggestion": suggestion
            })

        return {
            "group_id": group_id,
            "group_name": TestDataGenerator.AUDIT_GROUPS[group_id],
            "paper_id": paper_id,
            "audit_results": audit_results,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def generate_paper_audits(
        paper_id: Optional[str] = None,
        num_items_per_group: int = 3
    ) -> List[Dict[str, Any]]:
        """
        生成一篇论文的5个审计组结果

        Args:
            paper_id: 论文ID，如果为None则自动生成
            num_items_per_group: 每个审计组生成的审核项数量

        Returns:
            5个审计组的结果列表
        """
        if paper_id is None:
            paper_id = f"paper_{uuid.uuid4().hex[:8]}"

        results = []
        for group_id in TestDataGenerator.AUDIT_GROUPS.keys():
            result = TestDataGenerator.generate_audit_result(
                paper_id=paper_id,
                group_id=group_id,
                num_items=num_items_per_group
            )
            results.append(result)

        return results

    @staticmethod
    def save_to_files(
        paper_audits: List[Dict[str, Any]],
        output_dir: str = "prompts"
    ):
        """
        将审计结果保存为JSON文件

        Args:
            paper_audits: 审计结果列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paper_id = paper_audits[0]["paper_id"]

        for audit in paper_audits:
            group_id = audit["group_id"]
            filename = f"{paper_id}_group_{group_id}.json"
            filepath = output_path / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(audit, f, ensure_ascii=False, indent=2)

            print(f"已生成: {filepath}")

    @staticmethod
    async def save_to_database(
        paper_audits: List[Dict[str, Any]],
        db_manager: DatabaseManager,
        task_id: Optional[str] = None
    ):
        """
        将审计结果保存到PostgreSQL数据库

        数据库表结构：agent_audits
        - id: 自增主键（SERIAL类型，由数据库自动生成）
        - task_id: UUID类型（必填）
        - paper_id: UUID类型（必填）
        - agent_name: 文本类型（必填）
        - agent_version: 文本类型（必填）
        - status: 枚举类型（PENDING/RUNNING/SUCCESS/FAILED/TIMEOUT），默认PENDING
        - 可选字段：chunk_id, score, audit_level, result_json, error_msg,
                   usage_tokens, latency_ms, created_at, updated_at

        注意：需要数据库用户有USAGE权限访问agent_audits_id_seq序列

        Args:
            paper_audits: 审计结果列表
            db_manager: 数据库管理器实例
            task_id: 任务ID（UUID格式），如果为None则自动生成
        """
        paper_id_str = paper_audits[0]["paper_id"]

        # 将paper_id转换为UUID格式（如果不是UUID格式）
        try:
            # 尝试解析为UUID
            paper_id_uuid = uuid.UUID(paper_id_str)
            paper_id = str(paper_id_uuid)
        except (ValueError, AttributeError):
            # 如果不是有效的UUID，生成一个新的UUID
            # 但保留原始paper_id在result_json中
            paper_id = str(uuid.uuid4())
            logger.warning(f"paper_id '{paper_id_str}' 不是有效的UUID格式，已生成新UUID: {paper_id}")

        if task_id is None:
            # 生成标准UUID格式（基础task_id）
            base_task_id = str(uuid.uuid4())
        else:
            base_task_id = task_id

        try:
            async with db_manager.acquire() as conn:
                for audit in paper_audits:
                    group_id = audit["group_id"]
                    group_name = audit["group_name"]

                    # 为每个审计组生成唯一的task_id（避免unique_task_per_paper约束冲突）
                    unique_task_id = str(uuid.uuid4())

                    # 计算平均分数
                    audit_results = audit.get("audit_results", [])
                    avg_score = sum(item["score"] for item in audit_results) / len(audit_results) if audit_results else 0

                    # 确定审核级别（根据最严重的问题）
                    levels = [item["level"] for item in audit_results]
                    if "Critical" in levels:
                        audit_level = "Critical"
                    elif "Warning" in levels:
                        audit_level = "Warning"
                    else:
                        audit_level = "Info"

                    # 生成唯一的整数ID（绕过序列权限问题）
                    # 使用时间戳+随机数确保唯一性
                    import time
                    record_id = int(time.time() * 1000000) + random.randint(0, 999999)

                    # 插入数据库（手动指定id以绕过序列权限问题）
                    query = """
                        INSERT INTO agent_audits (
                            id, task_id, paper_id, chunk_id, agent_name, agent_version,
                            status, score, audit_level, result_json, error_msg,
                            usage_tokens, latency_ms, created_at, updated_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
                    """

                    await conn.execute(
                        query,
                        record_id,                              # id (手动生成)
                        unique_task_id,                         # task_id (每个审计组唯一)
                        paper_id,                               # paper_id (必填)
                        None,                                   # chunk_id (可选)
                        group_name,                             # agent_name (必填，使用组名)
                        "1.0.0",                                # agent_version (必填)
                        "PENDING",                              # status (枚举类型，默认PENDING)
                        round(avg_score, 2),                    # score
                        audit_level,                            # audit_level
                        json.dumps(audit, ensure_ascii=False),  # result_json
                        None,                                   # error_msg
                        0,                                      # usage_tokens
                        0                                       # latency_ms
                    )

                    print(f"已插入数据库: {paper_id}, {group_name} (group_{group_id}), 平均分: {avg_score:.2f}")

        except Exception as e:
            print(f"数据库插入失败: {e}")
            raise

    @staticmethod
    def generate_multiple_papers(
        num_papers: int = 3,
        output_dir: str = "prompts"
    ):
        """
        生成多篇论文的测试数据（文件模式）

        Args:
            num_papers: 论文数量
            output_dir: 输出目录
        """
        print(f"开始生成{num_papers}篇论文的测试数据...")

        for i in range(num_papers):
            paper_id = f"test_paper_{i+1:03d}"
            print(f"\n生成论文 {i+1}/{num_papers}: {paper_id}")

            paper_audits = TestDataGenerator.generate_paper_audits(
                paper_id=paper_id,
                num_items_per_group=random.randint(2, 5)
            )

            TestDataGenerator.save_to_files(paper_audits, output_dir)

        print(f"\n完成！共生成{num_papers}篇论文，每篇5个审计组结果")
        print(f"文件保存在: {output_dir}/")

    @staticmethod
    async def generate_multiple_papers_to_db(
        num_papers: int = 3,
        db_manager: DatabaseManager = None,
        use_existing_papers: bool = False
    ):
        """
        生成多篇论文的测试数据（数据库模式）

        Args:
            num_papers: 论文数量
            db_manager: 数据库管理器实例
            use_existing_papers: 是否使用数据库中已存在的paper_id（默认False，生成新的）
        """
        if db_manager is None:
            db_manager = DatabaseManager()

        await db_manager.connect()

        print(f"开始生成{num_papers}篇论文的测试数据并插入数据库...")

        try:
            # 如果使用已存在的paper_id，先从papers表查询
            if use_existing_papers:
                async with db_manager.acquire() as conn:
                    # 尝试不同的列名（paper_id或id）
                    try:
                        query = "SELECT paper_id FROM papers LIMIT $1"
                        rows = await conn.fetch(query, num_papers)
                        existing_paper_ids = [str(row["paper_id"]) for row in rows]
                    except Exception:
                        # 如果paper_id列不存在，尝试id列
                        try:
                            query = "SELECT id FROM papers LIMIT $1"
                            rows = await conn.fetch(query, num_papers)
                            existing_paper_ids = [str(row["id"]) for row in rows]
                        except Exception as e:
                            print(f"错误：无法从papers表读取数据: {e}")
                            print("将尝试创建新的paper记录")
                            existing_paper_ids = []

                    if existing_paper_ids and len(existing_paper_ids) < num_papers:
                        print(f"警告：papers表中只有{len(existing_paper_ids)}篇论文，将只生成{len(existing_paper_ids)}篇的审计数据")
                        num_papers = len(existing_paper_ids)
            else:
                existing_paper_ids = []

            for i in range(num_papers):
                # 使用已存在的paper_id或生成新的UUID
                if use_existing_papers and i < len(existing_paper_ids):
                    paper_id = existing_paper_ids[i]
                    print(f"\n生成论文 {i+1}/{num_papers}: 使用已存在的paper_id")
                else:
                    # 不创建新paper记录，直接使用已存在的paper_id
                    # 因为papers表结构未知，且可能没有插入权限
                    print(f"\n生成论文 {i+1}/{num_papers}: 使用已存在的paper_id")
                    try:
                        async with db_manager.acquire() as conn:
                            # 尝试使用paper_id列名
                            query = "SELECT paper_id FROM papers LIMIT 1 OFFSET $1"
                            row = await conn.fetchrow(query, i)
                            if row:
                                paper_id = str(row["paper_id"])
                                print(f"  从papers表获取: {paper_id}")
                            else:
                                # 如果没有足够的paper记录，生成UUID但警告用户
                                paper_id = str(uuid.uuid4())
                                print(f"  警告：papers表中没有足够的记录，生成新UUID: {paper_id}")
                                print(f"  注意：此UUID可能不满足外键约束，插入可能失败")
                    except Exception as e:
                        print(f"  错误：无法从papers表读取数据: {e}")
                        print(f"  跳过此论文")
                        continue

                task_id = str(uuid.uuid4())
                print(f"  paper_id (UUID): {paper_id}")
                print(f"  task_id (UUID): {task_id}")

                paper_audits = TestDataGenerator.generate_paper_audits(
                    paper_id=paper_id,  # 使用UUID格式
                    num_items_per_group=random.randint(2, 5)
                )

                await TestDataGenerator.save_to_database(paper_audits, db_manager, task_id)

            print(f"\n完成！共生成{num_papers}篇论文，每篇5个审计组结果")
            print(f"数据已插入到agent_audits表")

        finally:
            await db_manager.disconnect()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试数据生成器")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["file", "database"],
        default="file",
        help="生成模式：file=生成JSON文件，database=插入数据库"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=3,
        help="生成的论文数量"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prompts",
        help="输出目录（仅file模式有效）"
    )
    parser.add_argument(
        "--use-existing-papers",
        action="store_true",
        help="使用数据库中已存在的paper_id（仅database模式有效）"
    )

    args = parser.parse_args()

    if args.mode == "file":
        # 文件模式：生成JSON文件
        TestDataGenerator.generate_multiple_papers(
            num_papers=args.num_papers,
            output_dir=args.output_dir
        )
    elif args.mode == "database":
        # 数据库模式：插入到PostgreSQL
        asyncio.run(
            TestDataGenerator.generate_multiple_papers_to_db(
                num_papers=args.num_papers,
                use_existing_papers=args.use_existing_papers
            )
        )


if __name__ == "__main__":
    main()
