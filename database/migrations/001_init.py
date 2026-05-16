"""
database/migrations/001_init.py - Alembic 初始迁移
运行: alembic upgrade head
"""

from alembic import op
import sqlalchemy as sa

revision = '001_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """
    Alembic 迁移说明：
    
    实际项目使用 Alembic 管理数据库版本，
    schema_postgresql.sql 用于首次初始化，
    后续变更通过迁移文件追踪。
    
    生成迁移：
      alembic revision --autogenerate -m "描述"
    
    执行迁移：
      alembic upgrade head
    
    回滚：
      alembic downgrade -1
    """
    # 首次初始化通过 schema_postgresql.sql 完成
    # 此文件作为迁移历史起点
    pass


def downgrade():
    pass
