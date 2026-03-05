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
    "database": "paper_review"  # 根据实际数据库名称调整
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
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            yield connection

    async def fetch_agent_audits(self, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        从agent_audits表读取审计结果

        Args:
            paper_id: 论文ID，如果为None则读取所有

        Returns:
            审计结果列表，每个元素包含paper_id和result_json
        """
        try:
            async with self.acquire() as conn:
                if paper_id:
                    query = """
                        SELECT paper_id, group_id, result_json, created_at
                        FROM agent_audits
                        WHERE paper_id = $1
                        ORDER BY group_id
                    """
                    rows = await conn.fetch(query, paper_id)
                else:
                    query = """
                        SELECT paper_id, group_id, result_json, created_at
                        FROM agent_audits
                        ORDER BY paper_id, group_id
                    """
                    rows = await conn.fetch(query)

                results = []
                for row in rows:
                    results.append({
                        "paper_id": row["paper_id"],
                        "group_id": row["group_id"],
                        "result_json": row["result_json"],
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
        final_score: float,
        verdict: str,
        result_json: Dict[str, Any]
    ):
        """
        保存反思评估结果到数据库

        Args:
            paper_id: 论文ID
            final_score: 最终得分
            verdict: 评审结论
            result_json: 完整结果JSON
        """
        try:
            async with self.acquire() as conn:
                query = """
                    INSERT INTO reflection_results
                    (paper_id, final_score, verdict, result_json, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (paper_id)
                    DO UPDATE SET
                        final_score = EXCLUDED.final_score,
                        verdict = EXCLUDED.verdict,
                        result_json = EXCLUDED.result_json,
                        updated_at = NOW()
                """
                await conn.execute(
                    query,
                    paper_id,
                    final_score,
                    verdict,
                    result_json
                )
                logger.info(f"保存反思评估结果成功: {paper_id}")

        except Exception as e:
            logger.error(f"保存反思评估结果失败: {e}")
            raise


# 全局数据库管理器实例
db_manager = DatabaseManager()
