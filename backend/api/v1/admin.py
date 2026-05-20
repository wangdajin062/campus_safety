"""
api/v1/admin.py - 管理后台 API
功能：举报审核、案例管理、预警发布、数据统计
需要管理员权限（role=admin）
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, desc
from datetime import datetime, timezone
from typing import Optional

from core.database import get_db
from core.security import get_admin_from_cookie
from models.user import User
from models.fraud import (
    UserReport, FraudPhone, FraudCase, FraudAlert,
    CallDetectionLog, SmsAnalysisLog, SmsKeyword,
)
from schemas.schemas import RiskLevelEnum

logger = __import__("logging").getLogger(__name__)
router = APIRouter()


# ── 管理员鉴权 ────────────────────────────────────────────
async def require_admin(current_user: User = Depends(get_admin_from_cookie)) -> User:
    """管理员鉴权：支持 Cookie (httpOnly) 和 Bearer header 双重认证"""
    logger.info(f"Admin access by user_id={current_user.id}")
    return current_user


# ── 数据总览 ─────────────────────────────────────────────
@router.get("/dashboard", summary="管理后台数据总览")
async def dashboard(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    user_count    = (await db.execute(select(func.count()).select_from(User))).scalar()
    phone_count   = (await db.execute(select(func.count()).select_from(FraudPhone).where(FraudPhone.is_active == True))).scalar()
    case_count    = (await db.execute(select(func.count()).select_from(FraudCase).where(FraudCase.status == "published"))).scalar()
    pending_count = (await db.execute(select(func.count()).select_from(UserReport).where(UserReport.status == "pending"))).scalar()
    alert_count   = (await db.execute(select(func.count()).select_from(FraudAlert).where(FraudAlert.status == "published"))).scalar()

    # 本月新增举报
    from datetime import date
    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_reports = (await db.execute(
        select(func.count()).select_from(UserReport).where(UserReport.created_at >= month_start)
    )).scalar()

    # 高危号码 Top 5
    top_phones = (await db.execute(
        select(FraudPhone.phone_number, FraudPhone.risk_type, FraudPhone.report_count)
        .where(FraudPhone.is_active == True, FraudPhone.risk_level == "high")
        .order_by(desc(FraudPhone.report_count)).limit(5)
    )).fetchall()

    return {
        "code": 200,
        "data": {
            "total_users":       user_count,
            "fraud_phones":      phone_count,
            "published_cases":   case_count,
            "pending_reports":   pending_count,
            "published_alerts":  alert_count,
            "month_new_reports": month_reports,
            "top_fraud_phones":  [
                {"phone": r[0], "risk_type": r[1], "report_count": r[2]}
                for r in top_phones
            ],
        }
    }


# ── 举报审核 ─────────────────────────────────────────────
@router.get("/reports", summary="待审核举报列表")
async def list_reports(
    status: str = Query("pending", pattern="^(pending|approved|rejected|all)$"),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    limit = 20
    offset = (page - 1) * limit
    q = select(UserReport).order_by(desc(UserReport.created_at))
    if status != "all":
        q = q.where(UserReport.status == status)
    result = await db.execute(q.limit(limit).offset(offset))
    rows = result.scalars().all()

    total = (await db.execute(
        select(func.count()).select_from(UserReport)
        .where(UserReport.status == status if status != "all" else True)
    )).scalar()

    return {
        "code": 200,
        "data": [
            {
                "id": r.id, "user_id": r.user_id, "report_type": r.report_type,
                "target": r.target, "description": r.description, "school": r.school,
                "status": r.status, "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
        "meta": {"total": total, "page": page, "limit": limit},
    }


@router.post("/reports/{report_id}/approve", summary="审核通过举报")
async def approve_report(
    report_id: int,
    body: dict = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    body = body or {}
    result = await db.execute(select(UserReport).where(UserReport.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "举报记录不存在")
    if report.status != "pending":
        raise HTTPException(400, f"举报已处于 {report.status} 状态，无法重复审核")

    # 写入诈骗号码库（仅 phone 类型）
    fraud_phone_id = None
    if report.report_type == "phone":
        existing = await db.execute(
            select(FraudPhone).where(FraudPhone.phone_hash == report.target_hash)
        )
        fp = existing.scalar_one_or_none()
        if fp:
            fp.report_count += 1
            fp.risk_level = "high"
        else:
            fp = FraudPhone(
                phone_number=report.target[:30],
                phone_hash=report.target_hash,
                risk_level="high",
                source="user_report",
                risk_type=body.get("risk_type"),
            )
            db.add(fp)
            await db.flush()
        fraud_phone_id = fp.id

    report.status = "approved"
    report.reviewer_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.review_note = body.get("note", "审核通过")
    report.fraud_phone_id = fraud_phone_id

    return {"code": 200, "data": {"message": "审核通过", "fraud_phone_id": fraud_phone_id}}


@router.post("/reports/{report_id}/reject", summary="拒绝举报")
async def reject_report(
    report_id: int,
    body: dict = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    body = body or {}
    result = await db.execute(select(UserReport).where(UserReport.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(404, "举报记录不存在")

    report.status = "rejected"
    report.reviewer_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.review_note = body.get("note", "信息不实")

    return {"code": 200, "data": {"message": "已拒绝"}}


# ── 案例管理 ─────────────────────────────────────────────
@router.get("/cases", summary="管理端案例列表")
async def admin_list_cases(
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    limit = 15
    offset = (page - 1) * limit
    q = select(FraudCase).where(FraudCase.status.in_(["published", "draft"])).order_by(desc(FraudCase.published_at))
    result = await db.execute(q.limit(limit).offset(offset))
    rows = result.scalars().all()
    total = (await db.execute(select(func.count()).select_from(FraudCase).where(FraudCase.status.in_(["published", "draft"])))).scalar()
    return {
        "code": 200,
        "data": [
            {"id": r.id, "title": r.title, "summary": r.summary, "content": r.content,
             "category": r.category, "risk_level": r.risk_level, "emoji": r.emoji,
             "view_count": r.view_count, "status": r.status, "published_at": r.published_at.isoformat() if r.published_at else None}
            for r in rows
        ],
        "meta": {"total": total, "page": page, "limit": limit},
    }


@router.post("/cases", summary="新建诈骗案例")
async def create_case(
    body: dict,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    required = ["title", "summary", "content", "category", "risk_level"]
    for field in required:
        if not body.get(field):
            raise HTTPException(400, f"缺少必填字段: {field}")

    case = FraudCase(
        title=body["title"],
        summary=body["summary"],
        content=body["content"],
        category=body["category"],
        risk_level=body["risk_level"],
        emoji=body.get("emoji", "📋"),
        tags=body.get("tags", []),
        author_id=admin.id,
        status=body.get("status", "published"),
        is_featured=body.get("is_featured", False),
    )
    db.add(case)
    await db.flush()
    return {"code": 200, "data": {"id": case.id, "message": "案例创建成功"}}


@router.put("/cases/{case_id}", summary="编辑案例")
async def update_case(
    case_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    result = await db.execute(select(FraudCase).where(FraudCase.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(404, "案例不存在")

    updatable = {"title", "summary", "content", "category", "risk_level", "emoji", "tags", "status", "is_featured"}
    string_fields = {"title", "summary", "content", "category", "risk_level", "emoji", "status"}
    for field in updatable:
        if field not in body:
            continue
        val = body[field]
        # 类型校验
        if field in string_fields and not isinstance(val, str):
            raise HTTPException(400, f"字段 {field} 必须为字符串")
        if field == "is_featured" and not isinstance(val, bool):
            raise HTTPException(400, "is_featured 必须为布尔值")
        setattr(case, field, val)

    return {"code": 200, "data": {"message": "案例已更新"}}


@router.delete("/cases/{case_id}", summary="下架案例")
async def archive_case(
    case_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    await db.execute(
        update(FraudCase).where(FraudCase.id == case_id).values(status="archived")
    )
    return {"code": 200, "data": {"message": "案例已下架"}}


# ── 预警管理 ─────────────────────────────────────────────
@router.get("/alerts", summary="管理端预警列表")
async def admin_list_alerts(
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    limit = 15
    offset = (page - 1) * limit
    q = select(FraudAlert).order_by(desc(FraudAlert.published_at))
    result = await db.execute(q.limit(limit).offset(offset))
    rows = result.scalars().all()
    total = (await db.execute(select(func.count()).select_from(FraudAlert))).scalar()
    return {
        "code": 200,
        "data": [
            {"id": r.id, "title": r.title, "emoji": r.emoji, "risk_level": r.risk_level,
             "is_urgent": r.is_urgent, "push_count": r.push_count, "read_count": r.read_count,
             "tags": r.tags or [], "published_at": r.published_at.isoformat() if r.published_at else None}
            for r in rows
        ],
        "meta": {"total": total, "page": page, "limit": limit},
    }


@router.post("/alerts", summary="发布电诈预警")
async def create_alert(
    body: dict,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if not body.get("title") or not body.get("content"):
        raise HTTPException(400, "标题和内容不能为空")

    alert = FraudAlert(
        title=body["title"],
        content=body["content"],
        risk_level=body.get("risk_level", "medium"),
        emoji=body.get("emoji", "📢"),
        tags=body.get("tags", []),
        is_urgent=body.get("is_urgent", False),
        target_schools=body.get("target_schools"),
        author_id=admin.id,
        status="published",
    )
    db.add(alert)
    await db.flush()

    # 触发即时推送（异步，不阻塞响应）
    import asyncio
    from services.scheduler import push_pending_alerts
    asyncio.create_task(push_pending_alerts())

    return {"code": 200, "data": {"id": alert.id, "message": "预警已发布，正在推送"}}


# ── 关键词管理 ───────────────────────────────────────────
@router.get("/keywords", summary="短信关键词列表")
async def list_keywords(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    result = await db.execute(
        select(SmsKeyword).where(SmsKeyword.is_active == True).order_by(desc(SmsKeyword.risk_weight))
    )
    rows = result.scalars().all()
    return {
        "code": 200,
        "data": [
            {"id": r.id, "keyword": r.keyword, "risk_weight": r.risk_weight,
             "category": r.category, "hit_count": r.hit_count}
            for r in rows
        ]
    }


@router.post("/keywords", summary="添加风险关键词")
async def add_keyword(
    body: dict,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    kw = str(body.get("keyword", "")).strip()[:100]  # 长度截断防注入
    weight = min(100, max(1, int(body.get("risk_weight", 50) or 50)))  # 范围约束
    if not kw:
        raise HTTPException(400, "关键词不能为空")
    if not (1 <= weight <= 100):
        raise HTTPException(400, "风险权重须在 1-100 之间")

    db.add(SmsKeyword(keyword=kw, risk_weight=weight, category=body.get("category")))
    return {"code": 200, "data": {"message": f"关键词 [{kw}] 已添加"}}


@router.delete("/keywords/{kw_id}", summary="停用关键词")
async def disable_keyword(
    kw_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    await db.execute(update(SmsKeyword).where(SmsKeyword.id == kw_id).values(is_active=False))
    return {"code": 200, "data": {"message": "关键词已停用"}}


# ── 诈骗号码管理 ─────────────────────────────────────────
@router.get("/fraud-phones", summary="诈骗号码库")
async def list_fraud_phones(
    risk_level: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    limit = 20
    offset = (page - 1) * limit
    q = select(FraudPhone).where(FraudPhone.is_active == True).order_by(desc(FraudPhone.report_count))
    if risk_level:
        q = q.where(FraudPhone.risk_level == risk_level)

    result = await db.execute(q.limit(limit).offset(offset))
    rows = result.scalars().all()
    return {
        "code": 200,
        "data": [
            {
                "id": r.id, "phone_number": r.phone_number, "risk_level": r.risk_level,
                "risk_type": r.risk_type, "report_count": r.report_count,
                "is_verified": r.is_verified, "location": r.location,
                "last_reported_at": r.last_reported_at.isoformat(),
            }
            for r in rows
        ]
    }


@router.post("/fraud-phones/{phone_id}/verify", summary="人工核实诈骗号码")
async def verify_phone(
    phone_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    await db.execute(
        update(FraudPhone).where(FraudPhone.id == phone_id).values(is_verified=True, risk_level="high")
    )
    return {"code": 200, "data": {"message": "已标记为人工核实"}}
