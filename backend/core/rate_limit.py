"""
core/rate_limit.py - 基于 Redis 的 API 限流中间件 (v4.1 修复版)
"""

from fastapi import Request, HTTPException
from core.redis import CacheService
import logging

logger = logging.getLogger(__name__)

# （v4.1 安全修复后不再区分敏感/非敏感，Redis 故障时所有端点返回 503）


async def rate_limit(request: Request, max_calls: int = 100, window: int = 60) -> None:
    """
    通用限流依赖（FastAPI Depends 使用）

    安全修复（2026-05）:
    - M1: Redis 故障时，所有端点均拒绝请求（不再降级非敏感端点）
    - M2: 支持 X-Forwarded-For 反代 IP 检测
    """
    # 优先用 JWT uid，降级用 IP（支持 X-Forwarded-For 反代）
    client_id = getattr(request.state, "user_id", None)
    if not client_id:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_id = forwarded.split(",")[0].strip()
        else:
            client_id = request.client.host if request.client else "unknown"
    route = request.url.path.replace("/", "_")
    key = f"rl:{route}:{client_id}"

    try:
        count = await CacheService.incr_with_expire(key, expire=window)
        if count > max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请 {window} 秒后重试",
                headers={"Retry-After": str(window)},
            )
    except HTTPException:
        raise
    except Exception as e:
        # M1: Redis 故障时所有端点统一拒绝（不再区分敏感/非敏感）
        logger.error(f"Redis 故障，拒绝请求: {request.url.path} — {e}")
        raise HTTPException(
            status_code=503,
            detail="服务暂时不可用，请稍后重试",
            headers={"Retry-After": "30"},
        )


async def strict_rate_limit(request: Request) -> None:
    """严格限流：5次/分钟，用于短信发送等敏感接口"""
    await rate_limit(request, max_calls=5, window=60)


async def auth_rate_limit(request: Request) -> None:
    """认证限流：20次/分钟"""
    await rate_limit(request, max_calls=20, window=60)

