"""
services/scheduler.py - 后台定时任务
- 每小时同步黑名单数据库
- 每天统计用户防护积分
- 预警发布时批量推送
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, func
from core.database import AsyncSessionLocal
from models.fraud import FraudAlert, UserReport, FraudPhone
from models.user import User
from services.push_service import push_service

logger = logging.getLogger(__name__)


# ── 任务：批量推送未推送的预警 ────────────────────────────
async def push_pending_alerts() -> None:
    """
    找出最近 1 小时内发布但 push_count=0 的预警，
    取所有活跃设备 FCM Token，批量推送
    """
    async with AsyncSessionLocal() as db:
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=1)
            result = await db.execute(
                select(FraudAlert)
                .where(
                    FraudAlert.status == "published",
                    FraudAlert.push_count == 0,
                    FraudAlert.published_at >= since,
                )
                .order_by(FraudAlert.is_urgent.desc())
            )
            alerts = result.scalars().all()

            if not alerts:
                return

            # 获取所有活跃设备 token
            from models.fraud import UserDevice
            tokens_result = await db.execute(
                select(UserDevice.fcm_token)
                .where(UserDevice.is_active == True, UserDevice.fcm_token.isnot(None))
            )
            tokens = [r[0] for r in tokens_result.fetchall()]

            if not tokens:
                logger.info("无活跃设备，跳过推送")
                return

            for alert in alerts:
                result = await push_service.send_fraud_alert(
                    tokens=tokens,
                    alert_title=alert.title,
                    alert_id=alert.id,
                    is_urgent=alert.is_urgent,
                )
                # 更新推送计数
                await db.execute(
                    update(FraudAlert)
                    .where(FraudAlert.id == alert.id)
                    .values(
                        push_count=result["success"],
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                logger.info(
                    "预警推送完成 alert_id=%d success=%d failure=%d",
                    alert.id, result["success"], result["failure"],
                )

                # 清理失效 token
                if result["invalid_tokens"]:
                    from sqlalchemy import delete
                    from models.fraud import UserDevice
                    await db.execute(
                        update(UserDevice)
                        .where(UserDevice.fcm_token.in_(result["invalid_tokens"]))
                        .values(is_active=False)
                    )
                    logger.info("清理失效 FCM Token %d 个（token已脱敏不打印）", len(result["invalid_tokens"]))

            await db.commit()

        except Exception as e:
            await db.rollback()
            logger.error("推送预警任务异常: %s", e)


# ── 任务：自动审核高确信举报 ──────────────────────────────
async def auto_approve_high_confidence_reports() -> None:
    """
    同一号码举报次数 >= 5 次且均为 phone 类型时，
    自动审核通过并加入诈骗数据库
    """
    async with AsyncSessionLocal() as db:
        try:
            # 找出被多次举报的号码
            subq = (
                select(UserReport.target_hash, func.count().label("cnt"))
                .where(
                    UserReport.report_type == "phone",
                    UserReport.status == "pending",
                )
                .group_by(UserReport.target_hash)
                .having(func.count() >= 5)
                .subquery()
            )

            result = await db.execute(select(subq))
            high_conf = result.fetchall()

            for row in high_conf:
                target_hash = row[0]

                # 取一条举报拿到号码原文
                sample = await db.execute(
                    select(UserReport)
                    .where(UserReport.target_hash == target_hash, UserReport.report_type == "phone")
                    .limit(1)
                )
                report = sample.scalar_one_or_none()
                if not report:
                    continue

                # 写入或更新诈骗号码库
                existing = await db.execute(
                    select(FraudPhone).where(FraudPhone.phone_hash == target_hash)
                )
                fraud_phone = existing.scalar_one_or_none()

                if not fraud_phone:
                    fraud_phone = FraudPhone(
                        phone_number=report.target,
                        phone_hash=target_hash,
                        risk_level="high",
                        source="user_report",
                        report_count=row[1],
                    )
                    db.add(fraud_phone)
                    await db.flush()
                else:
                    fraud_phone.report_count = row[1]
                    fraud_phone.risk_level = "high"

                # 批量审核通过相关举报
                await db.execute(
                    update(UserReport)
                    .where(
                        UserReport.target_hash == target_hash,
                        UserReport.status == "pending",
                    )
                    .values(
                        status="approved",
                        fraud_phone_id=fraud_phone.id,
                        reviewed_at=datetime.now(timezone.utc),
                        review_note="系统自动审核：举报次数≥5",
                    )
                )
                logger.info("自动审核通过: hash=%s report_count=%d", target_hash[:8], row[1])

            await db.commit()

        except Exception as e:
            await db.rollback()
            logger.error("自动审核任务异常: %s", e)


# ── 任务：每日更新用户防护积分 ────────────────────────────
async def update_protection_scores() -> None:
    """
    根据用户行为更新 protection_score：
    - 每拦截 1 个电话 +1 分（上限 30）
    - 每条预警短信 +2 分（上限 20）
    - 每次举报 +3 分（上限 30）
    - 每阅读 5 个案例 +1 分（上限 20）
    """
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(User).where(User.status == 1))
            users = result.scalars().all()

            for user in users:
                score = (
                    min(user.blocked_calls, 30)
                    + min(user.alerted_sms * 2, 20)
                    + min(user.total_reports * 3, 30)
                    + min(user.cases_read // 5, 20)
                )
                user.protection_score = min(100, score)

            await db.commit()
            logger.info("防护积分更新完成，共 %d 名用户", len(users))

        except Exception as e:
            await db.rollback()
            logger.error("防护积分更新异常: %s", e)


# ── 定时任务循环 ──────────────────────────────────────────
async def scheduler_loop() -> None:
    """后台任务主循环，随应用启动"""
    logger.info("🕐 后台任务调度器已启动")

    while True:
        now = datetime.now(timezone.utc)

        try:
            # 每 5 分钟：推送待发预警
            await push_pending_alerts()

            # 每小时整点：自动审核高确信举报
            if now.minute < 5:
                await auto_approve_high_confidence_reports()

            # 每天凌晨 2 点：更新防护积分
            if now.hour == 2 and now.minute < 5:
                await update_protection_scores()

        except Exception as e:
            logger.error("调度器主循环异常: %s", e)

        # 5 分钟间隔
        await asyncio.sleep(300)


def start_scheduler() -> asyncio.Task:
    """在应用 lifespan 中调用，启动后台调度器"""
    return asyncio.create_task(scheduler_loop())
