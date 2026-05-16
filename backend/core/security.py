"""
core/security.py - JWT 鉴权 & 密码工具
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
from models.user import User

bearer_scheme = HTTPBearer()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def mask_phone(phone: str) -> str:
    """手机号脱敏：138****1234"""
    if len(phone) == 11:
        return phone[:3] + "****" + phone[7:]
    return phone[:3] + "***" + phone[-3:]


def create_access_token(uid: int, phone_hash: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS)
    payload = {"uid": uid, "phone_hash": phone_hash, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效或已过期")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_token(credentials.credentials)
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Token 解析失败")
    result = await db.execute(select(User).where(User.id == uid, User.status == 1))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
    return user
