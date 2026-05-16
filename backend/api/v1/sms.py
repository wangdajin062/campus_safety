"""
api/v1/sms.py - 短信预警路由
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import hashlib, datetime

from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.fraud import SmsKeyword, SmsAnalysisLog
from schemas.schemas import SmsAnalyzeRequest, SmsAnalyzeResponse

router = APIRouter()


@router.post("/analyze", summary="分析短信关键词（原文在客户端处理）")
async def analyze_sms(
    body: SmsAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    score = 0
    matched = []

    # 静态权重字典（数据库关键词库为空时的回退）
    STATIC_WEIGHTS = {
        "安全账户": 95, "立即转账": 90, "公安局": 90, "涉案资金": 92,
        "资产冻结": 88, "配合调查": 85, "刷单": 85, "验证码": 70,
        "贷款": 60, "兼职": 45, "恭喜中奖": 80, "账户异常": 75,
    }

    if body.keywords:
        # 优先从数据库加载关键词权重
        result = await db.execute(
            select(SmsKeyword).where(
                SmsKeyword.keyword.in_(body.keywords),
                SmsKeyword.is_active == True,
            )
        )
        kws = result.scalars().all()
        db_matched = set()
        for kw in kws:
            score += kw.risk_weight
            matched.append(kw.keyword)
            db_matched.add(kw.keyword)
            kw.hit_count += 1
        # 回退：未从数据库匹配到的关键词使用静态权重
        for kw in body.keywords:
            if kw not in db_matched and kw in STATIC_WEIGHTS:
                score += STATIC_WEIGHTS[kw]
                matched.append(kw)

    if body.has_url:
        score += 30

    score = min(100, score)
    risk_level = "high" if score >= 70 else "medium" if score >= 35 else "safe"

    content_hash = hashlib.sha256(f"{body.sender}:{','.join(body.keywords)}:{datetime.datetime.now().isoformat()}".encode()).hexdigest()
    db.add(SmsAnalysisLog(
        user_id=current_user.id,
        sender=body.sender,
        content_hash=content_hash,
        content_length=body.content_length,
        risk_level=risk_level,
        risk_score=score,
        matched_keywords=matched,
        has_suspicious_url=body.has_url,
    ))

    if risk_level in ("medium", "high"):
        current_user.alerted_sms += 1

    return {"code": 200, "data": SmsAnalyzeResponse(risk_level=risk_level, risk_score=score, matched_keywords=matched).model_dump()}
