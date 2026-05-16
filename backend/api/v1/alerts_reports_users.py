"""
api/v1/alerts.py - 电诈预警路由
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.fraud import FraudAlert
from schemas.schemas import FraudAlertResponse, FraudAlertSummary

router = APIRouter()


@router.get("", summary="预警列表")
async def list_alerts(
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limit = 10
    offset = (page - 1) * limit
    result = await db.execute(
        select(FraudAlert)
        .where(FraudAlert.status == "published")
        .order_by(FraudAlert.is_urgent.desc(), FraudAlert.published_at.desc())
        .limit(limit).offset(offset)
    )
    rows = result.scalars().all()
    return {"code": 200, "data": [FraudAlertResponse.model_validate(r).model_dump() for r in rows]}


@router.get("/latest", summary="最新2条预警（首页用）")
async def latest_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(FraudAlert).where(FraudAlert.status == "published")
        .order_by(FraudAlert.published_at.desc()).limit(2)
    )
    rows = result.scalars().all()
    return {"code": 200, "data": [FraudAlertSummary.model_validate(r).model_dump() for r in rows]}


"""
api/v1/reports.py - 用户举报路由
"""
from fastapi import APIRouter as _R2, Depends as _D2, Request
from sqlalchemy.ext.asyncio import AsyncSession as _AS2
from sqlalchemy import select as _sel2
import hashlib as _hl, datetime as _dt

from core.database import get_db as _gdb2
from core.security import get_current_user as _gcu2, sha256 as _sha2
from models.user import User as _U2
from models.fraud import UserReport, FraudPhone
from schemas.schemas import ReportRequest

reports_router = _R2()


@reports_router.post("", summary="提交举报")
async def submit_report(
    body: ReportRequest,
    request: Request,
    db: _AS2 = _D2(_gdb2),
    current_user: _U2 = _D2(_gcu2),
):
    target_hash = _sha2(body.target.strip().lower())
    report = UserReport(
        user_id=current_user.id,
        report_type=body.report_type,
        target=body.target[:500],
        target_hash=target_hash,
        description=body.description,
        school=body.school,
        ip_address=request.client.host if request.client else None,
    )
    db.add(report)
    current_user.total_reports += 1
    return {"code": 200, "data": {"message": "举报成功，感谢您的贡献！审核后将入库。"}}


"""
api/v1/users.py - 用户信息路由
"""
from fastapi import APIRouter as _R3, Depends as _D3
from sqlalchemy.ext.asyncio import AsyncSession as _AS3

from core.database import get_db as _gdb3
from core.security import get_current_user as _gcu3
from models.user import User as _U3
from schemas.schemas import UserStatsResponse, DeviceRegisterRequest
from models.fraud import UserDevice

users_router = _R3()


@users_router.get("/stats", summary="用户防护统计")
async def user_stats(
    db: _AS3 = _D3(_gdb3),
    current_user: _U3 = _D3(_gcu3),
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


@users_router.post("/device", summary="注册设备（用于推送通知）")
async def register_device(
    body: DeviceRegisterRequest,
    db: _AS3 = _D3(_gdb3),
    current_user: _U3 = _D3(_gcu3),
):
    from sqlalchemy import select as _s3
    result = await db.execute(
        _s3(UserDevice).where(UserDevice.user_id == current_user.id, UserDevice.device_id == body.device_id)
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


@users_router.get("/home", summary="首页聚合数据")
async def home_summary(
    db: _AS3 = _D3(_gdb3),
    current_user: _U3 = _D3(_gcu3),
):
    from sqlalchemy import select as _s4, func as _f4
    from models.fraud import FraudPhone as _FP, FraudCase as _FC
    phone_count = (await db.execute(_s4(_f4.count()).select_from(_FP).where(_FP.is_active == True))).scalar()
    case_count  = (await db.execute(_s4(_f4.count()).select_from(_FC).where(_FC.status == "published"))).scalar()
    return {"code": 200, "data": {
        "blocked_today":      current_user.blocked_calls,
        "alerted_sms":        current_user.alerted_sms,
        "total_protected":    current_user.blocked_calls,
        "fraud_phone_count":  phone_count,
        "case_count":         case_count,
    }}
