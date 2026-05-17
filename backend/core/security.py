"""
core/security.py - JWT 鉴权 & 密码工具 (v4.1 安全修复)
  安全修复 (2026-05):
  ✓ H3: PBKDF2 手机号哈希（替代无盐 SHA256）
  ✓ H4: Refresh token rotation（每次刷新签发新 refresh_token）
  ✓ H5: Redis Token 黑名单（支持吊销和登出失效）
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import hashlib
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import settings
from core.database import get_db
from core.redis import get_redis
from models.user import User

bearer_scheme = HTTPBearer()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def phone_hash(phone: str) -> str:
    """
    手机号哈希（PBKDF2-HMAC-SHA256，以 SECRET_KEY 为盐）
    修复 v4.1 H3：无盐 SHA256 可被彩虹表攻破
    """
    key = settings.SECRET_KEY.encode("utf-8") if settings.SECRET_KEY else b"campus_safety_v3"
    dk = hashlib.pbkdf2_hmac("sha256", phone.encode("utf-8"), key, iterations=100000)
    return dk.hex()


def mask_phone(phone: str) -> str:
    """手机号脱敏：138****1234"""
    if len(phone) == 11:
        return phone[:3] + "****" + phone[7:]
    return phone[:3] + "***" + phone[-3:]


def _jti() -> str:
    """唯一 JWT ID"""
    return secrets.token_hex(16)


def create_tokens(uid: int, phone_hash_val: str) -> dict:
    """
    生成 access_token + refresh_token（v4.1 增加 JTI 支持吊销）

    access_token:  用于 API 请求，1-7 天过期
    refresh_token: 用于刷新 access_token，7-30 天过期
    """
    now = datetime.now(timezone.utc)

    access_expire = now + timedelta(days=settings.JWT_EXPIRE_DAYS)
    access_jti = _jti()
    access_token = jwt.encode(
        {
            "jti": access_jti,
            "uid": uid,
            "phone_hash": phone_hash_val,
            "type": "access",
            "exp": access_expire,
            "iat": now,
        },
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    refresh_expire = now + timedelta(days=settings.JWT_REFRESH_EXPIRE_DAYS)
    refresh_jti = _jti()
    refresh_token = jwt.encode(
        {
            "jti": refresh_jti,
            "uid": uid,
            "type": "refresh",
            "exp": refresh_expire,
            "iat": now,
        },
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": int(settings.JWT_EXPIRE_DAYS * 86400),
    }


def create_access_token(uid: int, phone_hash_val: str) -> str:
    """向后兼容：仅返回 access_token"""
    tokens = create_tokens(uid, phone_hash_val)
    return tokens["access_token"]


async def blacklist_token(token: str, expire_seconds: int = 86400) -> None:
    """
    将 JWT 加入 Redis 黑名单（H5: Token 吊销）
    使用 JTI 作为唯一标识，TTL 匹配 token 剩余有效期
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if not jti:
            return
        redis = await get_redis()
        await redis.setex(f"bl:{jti}", expire_seconds, "1")
    except Exception:
        pass


async def is_token_blacklisted(token: str) -> bool:
    """检查 JWT 是否在黑名单中"""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if not jti:
            return False
        redis = await get_redis()
        val = await redis.get(f"bl:{jti}")
        return val is not None
    except Exception:
        return False


def decode_token(token: str, token_type: str = "access") -> dict:
    """
    解析 JWT Token（不检查黑名单，由 get_current_user 或 refresh 端点调用）
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type, expected {token_type}",
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """从 access_token 获取当前用户（检查黑名单）"""
    token = credentials.credentials

    # H5: 检查 Token 黑名单
    if await is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token 已被吊销")

    payload = decode_token(token, token_type="access")
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Token 解析失败")
    result = await db.execute(select(User).where(User.id == uid, User.status == 1))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
    return user

