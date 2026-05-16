import hmac
"""
core/redis.py - Redis 连接管理（验证码缓存 & 限流）
"""

import json
import asyncio
from typing import Optional, Any
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# 尝试导入 redis，若未安装则使用内存降级方案
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 未安装，使用内存缓存降级（仅适合开发环境）")

# ── 内存降级方案（开发/测试用）────────────────────────────
_memory_store: dict[str, tuple[Any, float]] = {}


class MemoryCache:
    """Redis 不可用时的内存降级实现"""

    async def get(self, key: str) -> Optional[str]:
        import time
        if key in _memory_store:
            val, expire_at = _memory_store[key]
            if expire_at == 0 or time.time() < expire_at:
                return val
            del _memory_store[key]
        return None

    async def setex(self, key: str, seconds: int, value: Any) -> None:
        import time
        _memory_store[key] = (str(value), time.time() + seconds)

    async def delete(self, key: str) -> None:
        _memory_store.pop(key, None)

    async def incr(self, key: str) -> int:
        import time
        val, exp = _memory_store.get(key, ("0", 0))
        new_val = int(val) + 1
        _memory_store[key] = (str(new_val), exp)
        return new_val

    async def expire(self, key: str, seconds: int) -> None:
        import time
        if key in _memory_store:
            val, _ = _memory_store[key]
            _memory_store[key] = (val, time.time() + seconds)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# ── Redis 客户端单例 ──────────────────────────────────────
_redis_client = None


async def get_redis():
    """获取 Redis 客户端（懒加载单例）"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if not REDIS_AVAILABLE:
        _redis_client = MemoryCache()
        return _redis_client

    try:
        client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        await client.ping()
        _redis_client = client
        logger.info("✅ Redis 连接成功")
    except Exception as e:
        logger.warning(f"Redis 连接失败，降级使用内存缓存: {e}")
        _redis_client = MemoryCache()

    return _redis_client


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# ── 验证码 helper ─────────────────────────────────────────
class SmsCodeService:

    PREFIX = "sms_code:"
    LIMIT_PREFIX = "sms_limit:"

    @classmethod
    async def save_code(cls, phone_hash: str, code: str, expire: int = 300) -> None:
        """保存验证码，默认 5 分钟过期"""
        redis = await get_redis()
        await redis.setex(f"{cls.PREFIX}{phone_hash}", expire, code)

    @classmethod
    async def verify_code(cls, phone_hash: str, code: str) -> bool:
        """验证并消费验证码（验证成功后立即删除）"""
        redis = await get_redis()
        stored = await redis.get(f"{cls.PREFIX}{phone_hash}")
        if stored and hmac.compare_digest(stored, code):
            await redis.delete(f"{cls.PREFIX}{phone_hash}")
            return True
        return False

    @classmethod
    async def check_rate_limit(cls, phone_hash: str, max_per_minute: int = 1) -> bool:
        """
        发送频率限制：1分钟内最多发 max_per_minute 次
        返回 True 表示允许发送，False 表示超过限制
        """
        redis = await get_redis()
        key = f"{cls.LIMIT_PREFIX}{phone_hash}"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
        return count <= max_per_minute


# ── 通用缓存 helper ───────────────────────────────────────
class CacheService:

    @staticmethod
    async def get_json(key: str) -> Optional[Any]:
        redis = await get_redis()
        val = await redis.get(key)
        return json.loads(val) if val else None

    @staticmethod
    async def set_json(key: str, value: Any, expire: int = 300) -> None:
        redis = await get_redis()
        await redis.setex(key, expire, json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    async def delete(key: str) -> None:
        redis = await get_redis()
        await redis.delete(key)

    @staticmethod
    async def incr_with_expire(key: str, expire: int = 60) -> int:
        """原子自增（用于限流计数）"""
        redis = await get_redis()
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, expire)
        return count
