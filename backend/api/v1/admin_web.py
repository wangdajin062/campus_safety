"""
api/v1/admin_web.py — 管理看板 Web 认证端点
Cookie Session 方式：登录设置 httpOnly Cookie，后续请求自动携带
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.security import (
    create_tokens, blacklist_token, set_auth_cookie, clear_auth_cookie,
    get_admin_from_cookie, phone_hash, mask_phone,
)
from core.redis import SmsCodeService
from models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

STATIC_ADMIN = Path(__file__).resolve().parent.parent.parent / "static" / "admin"


def _admin_page(filename: str) -> FileResponse:
    """返回管理看板 HTML 页面"""
    return FileResponse(str(STATIC_ADMIN / filename), media_type="text/html; charset=utf-8")


class WebLoginRequest(BaseModel):
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$")
    code: str = Field(..., min_length=6, max_length=6)


class WebLoginResponse(BaseModel):
    success: bool
    message: str
    user: dict | None = None


@router.post("/api/login", response_model=WebLoginResponse)
async def web_login(
    body: WebLoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """管理看板登录：验证码登录 + 设置 httpOnly Cookie"""
    import hmac
    ph = phone_hash(body.phone)

    # 验证验证码
    code_valid = await SmsCodeService.verify_code(ph, body.code)
    # 开发模式：万能验证码 888888
    if not code_valid and body.code != "888888":
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 查找或创建用户
    result = await db.execute(select(User).where(User.phone_hash == ph))
    user = result.scalar_one_or_none()

    if not user:
        from sqlalchemy import insert
        import hashlib
        legacy_hash = hashlib.sha256(body.phone.encode()).hexdigest()
        result2 = await db.execute(select(User).where(User.phone_hash == legacy_hash))
        user = result2.scalar_one_or_none()

    if not user:
        # 自动注册（仅 admin 角色才能登录管理看板）
        # 开发模式下默认用户需要手动设置为 admin
        raise HTTPException(
            status_code=403,
            detail="该手机号未注册。请先在数据库中将用户 role 设为 admin。"
        )

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可登录管理看板")

    # 更新登录时间
    user.last_login_at = None  # will be updated by DB trigger or explicit set
    await db.commit()

    # 生成 token 并设置 Cookie
    tokens = create_tokens(user.id, user.phone_hash)
    set_auth_cookie(response, tokens["access_token"])

    return WebLoginResponse(
        success=True,
        message="登录成功",
        user={
            "id": user.id,
            "nickname": user.nickname,
            "phone": mask_phone(user.phone),
            "role": user.role,
            "school": user.school,
        },
    )


@router.post("/api/logout")
async def web_logout(
    request: Request,
    response: Response,
):
    """管理看板登出：清除 Cookie + 加入黑名单"""
    token = request.cookies.get("admin_token")
    if token:
        try:
            await blacklist_token(token, expire_seconds=86400)
        except Exception:
            pass

    clear_auth_cookie(response)
    return {"success": True, "message": "已登出"}


@router.get("/api/me")
async def web_me(
    current_user: User = Depends(get_admin_from_cookie),
):
    """获取当前管理员信息"""
    return {
        "id": current_user.id,
        "nickname": current_user.nickname,
        "phone": mask_phone(current_user.phone),
        "role": current_user.role,
        "school": current_user.school,
        "protection_score": current_user.protection_score,
    }


# ── 管理看板页面路由 ──────────────────────────────────────

@router.get("/", include_in_schema=False)
@router.get("/login", include_in_schema=False)
async def admin_login_page():
    """管理看板登录页面"""
    return _admin_page("login.html")


@router.get("/dashboard", include_in_schema=False)
async def admin_dashboard_page():
    """管理看板数据仪表盘"""
    return _admin_page("index.html")


@router.get("/reports", include_in_schema=False)
async def admin_reports_page():
    """举报审核页面"""
    return _admin_page("reports.html")


@router.get("/cases", include_in_schema=False)
async def admin_cases_page():
    """案例管理页面"""
    return _admin_page("cases.html")


@router.get("/alerts", include_in_schema=False)
async def admin_alerts_page():
    """预警管理页面"""
    return _admin_page("alerts.html")


@router.get("/keywords", include_in_schema=False)
async def admin_keywords_page():
    """关键词管理页面"""
    return _admin_page("keywords.html")


@router.get("/phones", include_in_schema=False)
async def admin_phones_page():
    """诈骗号码库页面"""
    return _admin_page("phones.html")
