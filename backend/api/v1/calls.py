"""
api/v1/calls.py - 来电检测路由
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func as sqlfunc
import datetime

from core.database import get_db
from core.security import get_current_user, sha256, mask_phone
from models.user import User
from models.fraud import FraudPhone, CallDetectionLog
from schemas.schemas import PhoneCheckResponse, CallLogResponse

router = APIRouter()


@router.get("/check", response_model=dict, summary="查询号码风险")
async def check_phone(
    phone: str = Query(..., description="手机号或座机号"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    clean = phone.replace(" ", "").replace("-", "")
    phone_hash = sha256(clean)
    result = await db.execute(
        select(FraudPhone).where(FraudPhone.phone_hash == phone_hash, FraudPhone.is_active == True)
    )
    fraud = result.scalar_one_or_none()

    # 写检测日志
    log = CallDetectionLog(
        user_id=current_user.id,
        phone_number=mask_phone(clean) if len(clean) == 11 else clean[:3] + "***",
        fraud_phone_id=fraud.id if fraud else None,
        risk_level=fraud.risk_level if fraud else "safe",
        detection_type="manual_query",
    )
    db.add(log)

    if fraud:
        fraud.query_count += 1

    return {
        "code": 200,
        "data": PhoneCheckResponse(
            id=fraud.id if fraud else None,
            risk_level=fraud.risk_level if fraud else "safe",
            risk_type=fraud.risk_type if fraud else None,
            risk_score=fraud.risk_score if fraud else 0,
            report_count=fraud.report_count if fraud else 0,
            location=fraud.location if fraud else "未知",
            is_verified=fraud.is_verified if fraud else False,
        ).model_dump()
    }


@router.get("/history", summary="来电检测历史记录")
async def call_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    offset = (page - 1) * limit
    result = await db.execute(
        select(CallDetectionLog)
        .where(CallDetectionLog.user_id == current_user.id)
        .order_by(CallDetectionLog.detected_at.desc())
        .limit(limit).offset(offset)
    )
    rows = result.scalars().all()
    total_r = await db.execute(
        select(sqlfunc.count()).select_from(CallDetectionLog).where(CallDetectionLog.user_id == current_user.id)
    )
    total = total_r.scalar()
    return {"code": 200, "data": [CallLogResponse.model_validate(r).model_dump() for r in rows],
            "meta": {"total": total, "page": page, "limit": limit}}


@router.post("/report", summary="举报诈骗电话")
async def report_phone(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    phone  = str(body.get("phone", "")).strip()[:30]   # 长度截断
    reason = str(body.get("reason", ""))[:200]        # 长度截断
    if not phone:
        return {"code": 400, "message": "号码不能为空"}
    phone_hash = sha256(phone)
    result = await db.execute(select(FraudPhone).where(FraudPhone.phone_hash == phone_hash))
    existing = result.scalar_one_or_none()
    if existing:
        existing.report_count += 1
        existing.last_reported_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        db.add(FraudPhone(phone_number=mask_phone(phone), phone_hash=phone_hash, risk_level="medium", source="user_report", risk_type=reason))
    return {"code": 200, "data": {"message": "举报成功，感谢您的贡献"}}
