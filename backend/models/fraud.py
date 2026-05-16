"""
models/fraud.py - 诈骗相关数据模型
"""

import enum
from sqlalchemy import (
    BigInteger, Integer, String, SmallInteger, Integer, DateTime, Text,
    JSON, ForeignKey, Boolean, Index
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from core.database import Base


class RiskLevel(str, enum.Enum):
    safe   = "safe"
    medium = "medium"
    high   = "high"


class DataSource(str, enum.Enum):
    user_report = "user_report"
    police      = "police"
    system      = "system"
    manual      = "manual"


# ── 设备表 ────────────────────────────────────────────────
class UserDevice(Base):
    __tablename__ = "user_devices"

    id:          Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:     Mapped[int]       = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    device_id:   Mapped[str]       = mapped_column(String(128), nullable=False)
    platform:    Mapped[str]       = mapped_column(String(10), nullable=False)   # ios | android
    fcm_token:   Mapped[str|None]  = mapped_column(String(256))
    app_version: Mapped[str|None]  = mapped_column(String(20))
    os_version:  Mapped[str|None]  = mapped_column(String(30))
    is_active:   Mapped[bool]      = mapped_column(Boolean, default=True)
    created_at:  Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:  Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="devices")

    __table_args__ = (
        Index("idx_device_user", "user_id"),
        {"comment": "用户设备注册表"},
    )


# ── 诈骗号码库（核心）────────────────────────────────────
class FraudPhone(Base):
    __tablename__ = "fraud_phones"

    id:              Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone_number:    Mapped[str]      = mapped_column(String(30), nullable=False)
    phone_hash:      Mapped[str]      = mapped_column(String(64), unique=True, nullable=False)

    risk_level:      Mapped[str]      = mapped_column(String(10), default=RiskLevel.medium, index=True)
    risk_type:       Mapped[str|None] = mapped_column(String(50))
    risk_score:      Mapped[int]      = mapped_column(SmallInteger, default=50)

    source:          Mapped[str]      = mapped_column(String(20), default=DataSource.user_report)
    location:        Mapped[str|None] = mapped_column(String(50))
    carrier:         Mapped[str|None] = mapped_column(String(30))

    report_count:    Mapped[int]      = mapped_column(Integer, default=1)
    confirmed_count: Mapped[int]      = mapped_column(Integer, default=0)
    query_count:     Mapped[int]      = mapped_column(Integer, default=0)

    is_verified:     Mapped[bool]     = mapped_column(Boolean, default=False)
    is_active:       Mapped[bool]     = mapped_column(Boolean, default=True, index=True)

    first_reported_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_reported_at:  Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_fraud_phone_score", "risk_score"),
        Index("idx_fraud_phone_report", "report_count"),
        {"comment": "诈骗电话号码库"},
    )


# ── 来电检测日志 ─────────────────────────────────────────
class CallDetectionLog(Base):
    __tablename__ = "call_detection_logs"

    id:              Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:         Mapped[int]       = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    phone_number:    Mapped[str]       = mapped_column(String(30), nullable=False)
    fraud_phone_id:  Mapped[int|None]  = mapped_column(BigInteger, ForeignKey("fraud_phones.id", ondelete="SET NULL"))
    risk_level:      Mapped[str]       = mapped_column(String(10), default="safe")
    detection_type:  Mapped[str]       = mapped_column(String(20), default="manual_query")
    user_action:     Mapped[str|None]  = mapped_column(String(20))
    detected_at:     Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("idx_call_log_user", "user_id"), {"comment": "来电检测记录"})


# ── 短信分析日志 ─────────────────────────────────────────
class SmsAnalysisLog(Base):
    __tablename__ = "sms_analysis_logs"

    id:               Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:          Mapped[int]      = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    sender:           Mapped[str]      = mapped_column(String(50), nullable=False)
    content_hash:     Mapped[str]      = mapped_column(String(64), nullable=False, comment="原文哈希，不存原文")
    content_length:   Mapped[int]      = mapped_column(Integer, default=0)
    risk_level:       Mapped[str]      = mapped_column(String(10), default="safe")
    risk_score:       Mapped[int]      = mapped_column(SmallInteger, default=0)
    matched_keywords: Mapped[list|None]= mapped_column(JSON)
    has_suspicious_url: Mapped[bool]   = mapped_column(Boolean, default=False)
    analyzed_at:      Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("idx_sms_log_user", "user_id"), {"comment": "短信分析日志（不存原文）"})


# ── 短信关键词规则库 ─────────────────────────────────────
class SmsKeyword(Base):
    __tablename__ = "sms_keywords"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    keyword:     Mapped[str]      = mapped_column(String(100), unique=True, nullable=False)
    risk_weight: Mapped[int]      = mapped_column(SmallInteger, default=10, comment="风险权重 1-100")
    category:    Mapped[str|None] = mapped_column(String(50), index=True)
    is_active:   Mapped[bool]     = mapped_column(Boolean, default=True)
    hit_count:   Mapped[int]      = mapped_column(Integer, default=0)
    created_at:  Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = ({"comment": "短信关键词风险规则库"},)


# ── 诈骗案例库 ───────────────────────────────────────────
class FraudCase(Base):
    __tablename__ = "fraud_cases"

    id:          Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title:       Mapped[str]      = mapped_column(String(200), nullable=False)
    summary:     Mapped[str]      = mapped_column(String(500), nullable=False)
    content:     Mapped[str]      = mapped_column(Text, nullable=False, comment="Markdown 正文")
    category:    Mapped[str]      = mapped_column(String(20), nullable=False, index=True)
    risk_level:  Mapped[str]      = mapped_column(String(10), default=RiskLevel.medium, index=True)
    emoji:       Mapped[str]      = mapped_column(String(10), default="📋")
    view_count:  Mapped[int]      = mapped_column(Integer, default=0)
    like_count:  Mapped[int]      = mapped_column(Integer, default=0)
    share_count: Mapped[int]      = mapped_column(Integer, default=0)
    tags:        Mapped[list|None]= mapped_column(JSON)
    related_ids: Mapped[list|None]= mapped_column(JSON)
    author_id:   Mapped[int|None] = mapped_column(BigInteger)
    status:      Mapped[str]      = mapped_column(String(20), default="published", index=True)
    is_featured: Mapped[bool]     = mapped_column(Boolean, default=False, index=True)
    published_at:Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at:  Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:  Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = ({"comment": "诈骗案例库"},)


# ── 用户收藏 ────────────────────────────────────────────
class UserCaseFavorite(Base):
    __tablename__ = "user_case_favorites"

    id:         Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:    Mapped[int]      = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    case_id:    Mapped[int]      = mapped_column(BigInteger, ForeignKey("fraud_cases.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_fav_user_case", "user_id", "case_id", unique=True),
        {"comment": "用户收藏案例"},
    )


# ── 电诈预警 ────────────────────────────────────────────
class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id:              Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title:           Mapped[str]       = mapped_column(String(200), nullable=False)
    content:         Mapped[str]       = mapped_column(Text, nullable=False)
    risk_level:      Mapped[str]       = mapped_column(String(10), default=RiskLevel.medium, index=True)
    emoji:           Mapped[str]       = mapped_column(String(10), default="📢")
    tags:            Mapped[list|None] = mapped_column(JSON)
    is_urgent:       Mapped[bool]      = mapped_column(Boolean, default=False, index=True)
    push_count:      Mapped[int]       = mapped_column(Integer, default=0)
    read_count:      Mapped[int]       = mapped_column(Integer, default=0)
    report_count:    Mapped[int]       = mapped_column(Integer, default=0)
    target_schools:  Mapped[list|None] = mapped_column(JSON, comment="定向学校，None=全量推送")
    status:          Mapped[str]       = mapped_column(String(20), default="published", index=True)
    author_id:       Mapped[int|None]  = mapped_column(BigInteger)
    published_at:    Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at:      Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:      Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = ({"comment": "电诈预警推送"},)


# ── 用户举报 ────────────────────────────────────────────
class UserReport(Base):
    __tablename__ = "user_reports"

    id:             Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:        Mapped[int]       = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    report_type:    Mapped[str]       = mapped_column(String(20), nullable=False)  # phone|sms|link|other
    target:         Mapped[str]       = mapped_column(String(500), nullable=False)
    target_hash:    Mapped[str]       = mapped_column(String(64), nullable=False, index=True)
    description:    Mapped[str|None]  = mapped_column(Text)
    school:         Mapped[str|None]  = mapped_column(String(100))
    status:         Mapped[str]       = mapped_column(String(20), default="pending", index=True)  # pending|approved|rejected
    reviewer_id:    Mapped[int|None]  = mapped_column(BigInteger)
    review_note:    Mapped[str|None]  = mapped_column(String(200))
    reviewed_at:    Mapped[DateTime|None] = mapped_column(DateTime(timezone=True))
    fraud_phone_id: Mapped[int|None]  = mapped_column(BigInteger, ForeignKey("fraud_phones.id", ondelete="SET NULL"))
    ip_address:     Mapped[str|None]  = mapped_column(String(45))
    created_at:     Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("idx_report_user", "user_id"), {"comment": "用户举报记录"})


# ── 推送通知日志 ─────────────────────────────────────────
class NotificationLog(Base):
    __tablename__ = "notification_logs"

    id:        Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:   Mapped[int]       = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type:      Mapped[str]       = mapped_column(String(30), nullable=False)   # call_alert|sms_alert|fraud_alert|...
    title:     Mapped[str]       = mapped_column(String(200), nullable=False)
    body:      Mapped[str]       = mapped_column(Text, nullable=False)
    data:      Mapped[dict|None] = mapped_column(JSON)
    is_read:   Mapped[bool]      = mapped_column(Boolean, default=False, index=True)
    sent_at:   Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    read_at:   Mapped[DateTime|None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("idx_notif_user", "user_id"), {"comment": "用户通知日志"})


# ── 风险评估题库 ─────────────────────────────────────────
class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id:          Mapped[int]       = mapped_column(Integer, primary_key=True, autoincrement=True)
    question:    Mapped[str]       = mapped_column(Text, nullable=False)
    options:     Mapped[list]      = mapped_column(JSON, nullable=False, comment='[{"label":"A","text":"...","is_correct":true}]')
    explanation: Mapped[str]       = mapped_column(Text, nullable=False)
    category:    Mapped[str|None]  = mapped_column(String(50), index=True)
    difficulty:  Mapped[int]       = mapped_column(SmallInteger, default=1, comment="1=简单 2=中等 3=困难")
    is_active:   Mapped[bool]      = mapped_column(Boolean, default=True)
    created_at:  Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = ({"comment": "风险评估题库"},)


# ── 用户测试记录 ─────────────────────────────────────────
class UserQuizRecord(Base):
    __tablename__ = "user_quiz_records"

    id:         Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:    Mapped[int]       = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    score:      Mapped[int]       = mapped_column(SmallInteger, nullable=False, comment="得分")
    total:      Mapped[int]       = mapped_column(SmallInteger, nullable=False, comment="总分")
    answers:    Mapped[list]      = mapped_column(JSON, nullable=False, comment="答题详情")
    weak_areas: Mapped[list|None] = mapped_column(JSON, comment="薄弱环节标签")
    taken_at:   Mapped[DateTime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_quiz_records_user", "user_id"), {"comment": "用户测试记录"})
