"""
api/v1/auth.py - 认证路由
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import secrets, hashlib, datetime

from core.database import get_db
from core.security import sha256, mask_phone, create_access_token
from core.config import settings
from models.user import User
from schemas.schemas import SendCodeRequest, LoginRequest, TokenResponse

router = APIRouter()
logger = __import__("logging").getLogger(__name__)


@router.post("/send-code", summary="发送短信验证码")
async def send_code(body: SendCodeRequest, request: Request):
    code = str(secrets.randbelow(900000) + 100000)
    # 实际项目：将 code 存入 Redis，过期 300s；调用阿里云/腾讯云短信 API
    # await redis.setex(f"sms_code:{sha256(body.phone)}", settings.SMS_CODE_EXPIRE_SECONDS, code)
    logger.debug("[DEV] 验证码已生成（手机号已脱敏，code 仅开发模式可见）")
    return {"code": 200, "message": "验证码已发送，请注意查收"}


@router.post("/login", response_model=dict, summary="登录 / 注册")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    # 生产环境从 Redis 校验验证码：
    # stored = await redis.get(f"sms_code:{sha256(body.phone)}")
    # if stored != body.code: raise HTTPException(400, "验证码错误或已过期")

    phone_hash = sha256(body.phone)
    result = await db.execute(select(User).where(User.phone_hash == phone_hash))
    user = result.scalar_one_or_none()

    if not user:
        user = User(phone=mask_phone(body.phone), phone_hash=phone_hash)
        db.add(user)
        await db.flush()

    user.last_login_at = datetime.datetime.now(datetime.timezone.utc)
    token = create_access_token(user.id, phone_hash)
    return {
        "code": 200,
        "data": {
            "token": token,
            "user": {"id": user.id, "nickname": user.nickname, "protection_score": user.protection_score}
        }
    }
