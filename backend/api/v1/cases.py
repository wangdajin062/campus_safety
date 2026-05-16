"""
api/v1/cases.py - 案例库路由
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func as sqlfunc, or_

from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.fraud import FraudCase, UserCaseFavorite
from schemas.schemas import FraudCaseListItem, FraudCaseDetail

router = APIRouter()

SORT_MAP = {
    "latest":  FraudCase.published_at.desc(),
    "popular": FraudCase.view_count.desc(),
    "risk":    FraudCase.risk_level.desc(),
}


@router.get("", summary="案例列表")
async def list_cases(
    category: str | None = Query(None),
    keyword:  str | None = Query(None),          # ← Android端关键词搜索
    sort: str = Query("latest", pattern="^(latest|popular|risk)$"),
    page: int  = Query(1,  ge=1),
    limit: int = Query(15, ge=1, le=50),          # ← Android默认请求15条
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    offset = (page - 1) * limit

    # ── 构建查询 ──────────────────────────────────────────────
    q = select(FraudCase).where(FraudCase.status == "published")

    if category:
        q = q.where(FraudCase.category == category)

    if keyword:                                    # 标题 + 摘要模糊匹配
        kw = f"%{keyword}%"
        q = q.where(or_(
            FraudCase.title.ilike(kw),
            FraudCase.summary.ilike(kw),
        ))

    q = q.order_by(SORT_MAP.get(sort, FraudCase.published_at.desc())) \
         .limit(limit).offset(offset)

    result = await db.execute(q)
    rows   = result.scalars().all()

    # ── 总数 ─────────────────────────────────────────────────
    count_q = select(sqlfunc.count()).select_from(FraudCase) \
                  .where(FraudCase.status == "published")
    if category:
        count_q = count_q.where(FraudCase.category == category)
    if keyword:
        kw = f"%{keyword}%"
        count_q = count_q.where(or_(
            FraudCase.title.ilike(kw),
            FraudCase.summary.ilike(kw),
        ))
    total = (await db.execute(count_q)).scalar()

    # ── 返回 PageResult 结构（与 Android PageResult<T> 字段对齐）──
    # Android 期待: { code, data: { items, total, page, limit } }
    return {
        "code": 200,
        "data": {
            "items": [FraudCaseListItem.model_validate(r).model_dump() for r in rows],
            "total": total,
            "page":  page,
            "limit": limit,
        }
    }


@router.get("/{case_id}", summary="案例详情")
async def get_case(
    case_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(FraudCase).where(
            FraudCase.id == case_id,
            FraudCase.status == "published"
        )
    )
    case = result.scalar_one_or_none()
    if not case:
        return {"code": 404, "message": "案例不存在"}
    case.view_count += 1
    current_user.cases_read += 1
    return {"code": 200, "data": FraudCaseDetail.model_validate(case).model_dump()}


@router.post("/{case_id}/favorite", summary="收藏/取消收藏案例")
async def toggle_favorite(
    case_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserCaseFavorite).where(
            UserCaseFavorite.user_id == current_user.id,
            UserCaseFavorite.case_id == case_id
        )
    )
    fav = result.scalar_one_or_none()
    if fav:
        await db.delete(fav)
        return {"code": 200, "data": {"favorited": False}}
    else:
        db.add(UserCaseFavorite(user_id=current_user.id, case_id=case_id))
        return {"code": 200, "data": {"favorited": True}}
