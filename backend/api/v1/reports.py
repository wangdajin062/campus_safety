"""
api/v1/reports.py - 用户举报路由
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.security import get_current_user, sha256
from models.user import User
from models.fraud import UserReport, FraudPhone
from schemas.schemas import ReportRequest

router = APIRouter()


@router.post("", summary="提交举报")
async def submit_report(
    body: ReportRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    target_hash = sha256(body.target.strip().lower())
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
