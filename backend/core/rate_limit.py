"""
core/rate_limit.py - 基于 Redis 的 API 限流中间件
"""

from fastapi import Request, HTTPException
from core.redis import CacheService
import logging

logger = logging.getLogger(__name__)


async def rate_limit(request: Request, max_calls: int = 100, window: int = 60) -> None:
    """
    通用限流依赖（FastAPI Depends 使用）
    
    用法:
        from core.rate_limit import rate_limit
        from functools import partial
        
        @router.post("/sensitive")
        async def endpoint(_=Depends(partial(rate_limit, max_calls=5, window=60))):
            ...
    """
    # 优先用 JWT uid，降级用 IP
    client_id = getattr(request.state, "user_id", None) or (
        request.client.host if request.client else "unknown"
    )
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
        # Redis 故障时放行，避免影响正常业务
        logger.warning(f"限流检查异常（已放行）: {e}")


async def strict_rate_limit(request: Request) -> None:
    """严格限流：5次/分钟，用于短信发送等敏感接口"""
    await rate_limit(request, max_calls=5, window=60)


async def auth_rate_limit(request: Request) -> None:
    """认证限流：20次/分钟"""
    await rate_limit(request, max_calls=20, window=60)
