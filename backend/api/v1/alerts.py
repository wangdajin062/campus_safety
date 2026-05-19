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
