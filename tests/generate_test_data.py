"""
测试数据生成器
模拟5个审计组对论文的评审结果
"""
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


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
    def generate_multiple_papers(
        num_papers: int = 3,
        output_dir: str = "prompts"
    ):
        """
        生成多篇论文的测试数据

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


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试数据生成器")
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
        help="输出目录"
    )

    args = parser.parse_args()

    TestDataGenerator.generate_multiple_papers(
        num_papers=args.num_papers,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
