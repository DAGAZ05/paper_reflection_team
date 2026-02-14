import asyncio
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conflict_resolver import ConflictResolver
from src.mock_data_generator import MockDataGenerator
from src.schemas import ConflictResolutionRequest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_conflict_resolution_demo():
    """运行冲突裁决演示"""
    print("=== 反思评估组冲突裁决演示 ===\n")

    # 创建模拟数据生成器
    generator = MockDataGenerator()

    # 创建冲突裁决器
    resolver = ConflictResolver()

    # 场景1: 有冲突的情况
    print("\n--- 场景1: 有冲突的情况 ---")
    conflict_request_data = generator.generate_conflict_resolution_request(
        paper_title="基于深度学习的图像识别算法优化研究",
        with_conflict=True
    )

    print("\n输入数据 (Agent结果):")
    for i, agent_result in enumerate(conflict_request_data["payload"]["agent_results"], 1):
        agent_name = agent_result["agent_info"]["name"]
        score = agent_result["result"]["score"]
        audit_level = agent_result["result"]["audit_level"]
        comment = agent_result["result"]["comment"]
        print(f"{i}. {agent_name} (评分:{score}, 级别:{audit_level}): {comment}")

    # 转换为请求对象
    conflict_request = ConflictResolutionRequest(**conflict_request_data)

    # 执行冲突裁决
    print("\n执行冲突裁决...")
    try:
        response = await resolver.resolve_conflicts(conflict_request)

        print("\n裁决结果:")
        print(f"- 冲突是否已解决: {response.result['conflicts_resolved']}")
        print(f"- 置信度: {response.result.get('confidence_score', 'N/A')}")

        if response.result.get('final_verdict'):
            verdict = response.result['final_verdict']
            print(f"- 最终结论: {verdict.get('verdict', 'N/A')}")
            print(f"- 平均分: {verdict.get('average_score', 'N/A')}")

        if response.result.get('resolved_issues'):
            print("\n解决的冲突详情:")
            for i, issue in enumerate(response.result['resolved_issues'], 1):
                agent1 = issue.get('agent1_name', 'N/A')
                agent2 = issue.get('agent2_name', 'N/A')
                conflict_type = issue.get('conflict_type', 'N/A')
                final_level = issue.get('final_level', 'N/A')
                resolved_comment = issue.get('resolved_comment', 'N/A')
                resolved_suggestion = issue.get('resolved_suggestion', 'N/A')

                print(f"\n  冲突 {i}:")
                print(f"    - 冲突双方: {agent1} vs {agent2}")
                print(f"    - 冲突类型: {conflict_type}")
                print(f"    - 最终级别: {final_level}")
                print(f"    - 裁决意见: {resolved_comment}")
                print(f"    - 改进建议: {resolved_suggestion}")

        if response.result.get('markdown_report'):
            print("\n--- Markdown报告 ---")
            print(response.result['markdown_report'])

    except Exception as e:
        logger.error(f"冲突裁决失败: {str(e)}")
        print(f"错误: {str(e)}")

    # 场景2: 无冲突的情况
    print("\n\n--- 场景2: 无冲突的情况 ---")
    no_conflict_request_data = generator.generate_conflict_resolution_request(
        paper_title="微服务架构在大型企业中的应用研究",
        with_conflict=False
    )

    print("\n输入数据 (Agent结果):")
    for i, agent_result in enumerate(no_conflict_request_data["payload"]["agent_results"], 1):
        agent_name = agent_result["agent_info"]["name"]
        score = agent_result["result"]["score"]
        audit_level = agent_result["result"]["audit_level"]
        comment = agent_result["result"]["comment"]
        print(f"{i}. {agent_name} (评分:{score}, 级别:{audit_level}): {comment}")

    # 转换为请求对象
    no_conflict_request = ConflictResolutionRequest(**no_conflict_request_data)

    # 执行冲突裁决
    print("\n执行冲突裁决...")
    try:
        response = await resolver.resolve_conflicts(no_conflict_request)

        print("\n裁决结果:")
        print(f"- 冲突是否已解决: {response.result['conflicts_resolved']}")
        print(f"- 置信度: {response.result.get('confidence_score', 'N/A')}")

        if response.result.get('final_verdict'):
            verdict = response.result['final_verdict']
            print(f"- 最终结论: {verdict.get('verdict', 'N/A')}")
            print(f"- 平均分: {verdict.get('average_score', 'N/A')}")

    except Exception as e:
        logger.error(f"冲突裁决失败: {str(e)}")
        print(f"错误: {str(e)}")


async def run_api_server():
    """启动API服务器"""
    import uvicorn
    from src.conflict_resolver import app

    print("\n=== 启动反思评估组API服务器 ===")
    print("API文档将在 http://localhost:8000/docs 可用")
    print("健康检查: http://localhost:8000/health")
    print("冲突裁决端点: http://localhost:8000/api/resolve_conflicts")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """主函数"""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="反思评估组冲突裁决系统")
    parser.add_argument(
        "mode",
        choices=["demo", "server", "generate-data"],
        default="demo",
        help="运行模式: demo(演示冲突裁决), server(启动API服务器), generate-data(生成模拟数据)"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(run_conflict_resolution_demo())
    elif args.mode == "server":
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(run_api_server())
    elif args.mode == "generate-data":
        generator = MockDataGenerator()

        # 生成有冲突的Agent结果JSON字符串
        conflict_json = generator.generate_json_string()
        print("=== 有冲突的Agent结果JSON字符串 ===")
        print(conflict_json)

        # 生成无冲突的Agent结果JSON字符串
        no_conflict_json = generator.generate_json_string(generator.generate_agent_results(with_conflict=False))
        print("\n=== 无冲突的Agent结果JSON字符串 ===")
        print(no_conflict_json)

        # 生成完整的冲突裁决请求
        conflict_request = generator.generate_conflict_resolution_request()
        print("\n=== 完整的冲突裁决请求 ===")
        print(json.dumps(conflict_request, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
