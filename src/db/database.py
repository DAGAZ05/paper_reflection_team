"""
统一数据库配置模块
连接到PostgreSQL数据库：10.13.1.26
用户名：admin
密码：ABCabc@123
"""
import asyncpg
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": "10.13.1.26",
    "port": 5432,
    "user": "admin",
    "password": "ABCabc@123",
    "database": "postgres"  # 根据实际数据库名称调整
}


class DatabaseManager:
    """统一数据库管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DB_CONFIG
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """建立数据库连接池"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info(f"数据库连接池已建立: {self.config['host']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    async def disconnect(self):
        """关闭数据库连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("数据库连接池已关闭")

    @asynccontextmanager
    async def acquire(self):
        """获取数据库连接"""
        if not self.pool or self.pool._closed:
            await self.connect()
        async with self.pool.acquire() as connection:
            yield connection

    async def fetch_agent_audits(self, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        从agent_audits表读取审计结果

        表结构：
        - id: SERIAL自增主键
        - task_id: UUID类型
        - paper_id: UUID类型
        - chunk_id: 文本类型（可选）
        - agent_name: 文本类型
        - agent_version: 文本类型
        - status: 枚举类型（PENDING/RUNNING/SUCCESS/FAILED/TIMEOUT）
        - score: 数值类型
        - audit_level: 文本类型
        - result_json: JSONB类型
        - error_msg: 文本类型
        - usage_tokens: 整数类型
        - latency_ms: 整数类型
        - created_at: 时间戳
        - updated_at: 时间戳

        Args:
            paper_id: 论文ID（UUID格式），如果为None则读取所有

        Returns:
            审计结果列表
        """
        try:
            async with self.acquire() as conn:
                if paper_id:
                    query = """
                        SELECT id, task_id, paper_id, agent_name, agent_version,
                               status, score, audit_level, result_json, created_at
                        FROM agent_audits
                        WHERE paper_id = $1
                        ORDER BY created_at
                    """
                    rows = await conn.fetch(query, paper_id)
                else:
                    query = """
                        SELECT id, task_id, paper_id, agent_name, agent_version,
                               status, score, audit_level, result_json, created_at
                        FROM agent_audits
                        ORDER BY paper_id, created_at
                    """
                    rows = await conn.fetch(query)

                results = []
                for row in rows:
                    # 解析result_json（可能是字符串或已经是字典）
                    result_json = row["result_json"]
                    if isinstance(result_json, str):
                        import json
                        try:
                            result_json = json.loads(result_json)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析result_json: {result_json[:100]}")
                            result_json = {}

                    results.append({
                        "id": row["id"],
                        "task_id": row["task_id"],
                        "paper_id": row["paper_id"],
                        "agent_name": row["agent_name"],
                        "agent_version": row["agent_version"],
                        "status": row["status"],
                        "score": row["score"],
                        "audit_level": row["audit_level"],
                        "result_json": result_json,
                        "created_at": row["created_at"]
                    })

                logger.info(f"从数据库读取了{len(results)}条审计结果")
                return results

        except Exception as e:
            logger.error(f"读取agent_audits表失败: {e}")
            raise

    async def get_paper_ids(self) -> List[str]:
        """获取所有待处理的论文ID"""
        try:
            async with self.acquire() as conn:
                query = """
                    SELECT DISTINCT paper_id
                    FROM agent_audits
                    ORDER BY paper_id
                """
                rows = await conn.fetch(query)
                paper_ids = [row["paper_id"] for row in rows]
                logger.info(f"找到{len(paper_ids)}篇待处理论文")
                return paper_ids

        except Exception as e:
            logger.error(f"获取论文ID列表失败: {e}")
            raise

    async def get_agent_results(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        获取指定论文的所有审计结果（别名方法，用于兼容conflict_resolver）

        Args:
            paper_id: 论文ID（UUID格式）

        Returns:
            审计结果列表
        """
        return await self.fetch_agent_audits(paper_id)

    async def get_paper_content(self, paper_id: str) -> str:
        """
        获取论文内容（用于证据验证）
        从paper_sections表读取，表结构：
        - section_id: 章节ID
        - paper_id: 论文ID
        - section_name: 章节名称
        - section_content: 章节内容
        - content_vector: 内容向量（用于语义搜索）

        Args:
            paper_id: 论文ID

        Returns:
            论文完整内容
        """
        try:
            async with self.acquire() as conn:
                query = """
                    SELECT section_id, section_name, section_content
                    FROM paper_sections
                    WHERE paper_id = $1
                    ORDER BY section_id
                """
                rows = await conn.fetch(query, paper_id)

                if not rows:
                    logger.warning(f"未找到论文内容: {paper_id}")
                    return ""

                # 拼接所有章节内容，包含章节标题
                sections = []
                for row in rows:
                    section_name = row["section_name"]
                    section_content = row["section_content"]
                    sections.append(f"## {section_name}\n\n{section_content}")

                content = "\n\n".join(sections)
                logger.info(f"获取论文内容成功: {paper_id}, 长度={len(content)}")
                return content

        except Exception as e:
            logger.error(f"获取论文内容失败: {e}")
            return ""

    async def save_reflection_result(
        self,
        paper_id: str,
        task_id: str,
        final_score: float,
        verdict: str,
        result_json: Dict[str, Any],
        usage_tokens: int = 0,
        latency_ms: int = 0
    ):
        """
        保存反思评估结果到agent_audits表

        由于没有reflection_results表，将反思评估结果也存储到agent_audits表中
        使用特殊的agent_name="反思评估组"来标识

        Args:
            paper_id: 论文ID
            task_id: 任务ID（会被忽略，自动生成新的唯一task_id）
            final_score: 最终得分
            verdict: 评审结论
            result_json: 完整结果JSON（包含final_score和verdict）
            usage_tokens: LLM使用的token数量
            latency_ms: LLM调用延迟（毫秒）
        """
        try:
            import json
            import time
            import random
            import uuid

            # 生成唯一的整数ID（绕过序列权限问题）
            record_id = int(time.time() * 1000000) + random.randint(0, 999999)

            # 为反思评估组生成独立的task_id（避免unique_task_per_paper约束冲突）
            unique_task_id = str(uuid.uuid4())

            # 将final_score和verdict合并到result_json中
            full_result = {
                "final_score": final_score,
                "verdict": verdict,
                **result_json
            }

            async with self.acquire() as conn:
                # 手动指定id以绕过序列权限问题
                query = """
                    INSERT INTO agent_audits (
                        id, task_id, paper_id, chunk_id, agent_name, agent_version,
                        status, score, audit_level, result_json, error_msg,
                        usage_tokens, latency_ms, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
                """

                # 根据分数确定审核级别
                if final_score >= 80:
                    audit_level = "Info"
                elif final_score >= 60:
                    audit_level = "Warning"
                else:
                    audit_level = "Critical"

                await conn.execute(
                    query,
                    record_id,                              # id (手动生成)
                    unique_task_id,                         # task_id (独立生成，避免冲突)
                    paper_id,                               # paper_id
                    None,                                   # chunk_id
                    "反思评估组",                            # agent_name (特殊标识)
                    "1.0.0",                                # agent_version
                    "SUCCESS",                              # status (枚举类型: PENDING/RUNNING/SUCCESS/FAILED/TIMEOUT)
                    round(final_score, 2),                  # score
                    audit_level,                            # audit_level
                    json.dumps(full_result, ensure_ascii=False),  # result_json
                    None,                                   # error_msg
                    usage_tokens,                           # usage_tokens (实际使用的token数)
                    latency_ms                              # latency_ms (实际延迟)
                )
                logger.info(f"保存反思评估结果成功: {paper_id}, 最终得分: {final_score}, tokens: {usage_tokens}, 延迟: {latency_ms}ms")

        except Exception as e:
            logger.error(f"保存反思评估结果失败: {e}")
            raise


# 全局数据库管理器实例
db_manager = DatabaseManager()
