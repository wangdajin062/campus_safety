"""
api/v1/users.py - 用户信息路由
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.fraud import FraudPhone, FraudCase, UserDevice
from schemas.schemas import UserStatsResponse, DeviceRegisterRequest

router = APIRouter()


@router.get("/stats", summary="用户防护统计")
async def user_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    score = current_user.protection_score
    level = (
        "防骗专家 🏆" if score >= 80
        else "优秀守卫者 ⭐⭐⭐" if score >= 50
        else "安全学员 ⭐⭐" if score >= 20
        else "新手防护 ⭐"
    )
    return {
        "code": 200,
        "data": UserStatsResponse(
            blocked_calls=current_user.blocked_calls,
            alerted_sms=current_user.alerted_sms,
            total_reports=current_user.total_reports,
            cases_read=current_user.cases_read,
            protection_score=score,
            protection_level=level,
        ).model_dump()
    }


@router.post("/device", summary="注册设备（用于推送通知）")
async def register_device(
    body: DeviceRegisterRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserDevice).where(UserDevice.user_id == current_user.id, UserDevice.device_id == body.device_id)
    )
    device = result.scalar_one_or_none()
    if device:
        device.fcm_token = body.fcm_token
        device.app_version = body.app_version
        device.is_active = True
    else:
        db.add(UserDevice(
            user_id=current_user.id,
            device_id=body.device_id,
            platform=body.platform,
            fcm_token=body.fcm_token,
            app_version=body.app_version,
            os_version=body.os_version,
        ))
    return {"code": 200, "data": {"message": "设备注册成功"}}


@router.get("/home", summary="首页聚合数据")
async def home_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    phone_count = (await db.execute(
        select(func.count()).select_from(FraudPhone).where(FraudPhone.is_active == True)
    )).scalar()
    case_count = (await db.execute(
        select(func.count()).select_from(FraudCase).where(FraudCase.status == "published")
    )).scalar()
    return {"code": 200, "data": {
        "blocked_today":      current_user.blocked_calls,
        "alerted_sms":        current_user.alerted_sms,
        "total_protected":    current_user.blocked_calls,
        "fraud_phone_count":  phone_count,
        "case_count":         case_count,
    }}
