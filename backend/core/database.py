"""
core/database.py - 异步数据库连接（SQLAlchemy 2.0 + asyncpg）
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from core.config import settings

_engine_kwargs = {}
if settings.DATABASE_TYPE == "sqlite":
    _engine_kwargs = {"connect_args": {"check_same_thread": False}}
else:
    _engine_kwargs = {"pool_size": 20, "max_overflow": 10, "pool_pre_ping": True}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "development",
    **_engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI 依赖注入：获取数据库 Session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
