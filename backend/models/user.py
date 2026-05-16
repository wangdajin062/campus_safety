"""
models/user.py - 用户相关模型
"""

from sqlalchemy import BigInteger, String, SmallInteger, Integer, DateTime, Enum, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from core.database import Base
import enum


class UserStatus(int, enum.Enum):
    disabled = 0
    active = 1


class User(Base):
    __tablename__ = "users"

    id:               Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone:            Mapped[str]          = mapped_column(String(20),  unique=True, nullable=False, comment="脱敏手机号")
    phone_hash:       Mapped[str]          = mapped_column(String(64),  unique=True, nullable=False, index=True)
    nickname:         Mapped[str]          = mapped_column(String(50),  default="校园守护者")
    school:           Mapped[str | None]   = mapped_column(String(100))
    avatar_url:       Mapped[str | None]   = mapped_column(String(255))

    blocked_calls:    Mapped[int]          = mapped_column(Integer, default=0)
    alerted_sms:      Mapped[int]          = mapped_column(Integer, default=0)
    total_reports:    Mapped[int]          = mapped_column(Integer, default=0)
    cases_read:       Mapped[int]          = mapped_column(Integer, default=0)
    protection_score: Mapped[int]          = mapped_column(SmallInteger, default=0)

    role:             Mapped[str]          = mapped_column(String(20), default="user", comment="user|admin")
    status:           Mapped[int]          = mapped_column(SmallInteger, default=1)
    last_login_at:    Mapped[DateTime|None]= mapped_column(DateTime(timezone=True))
    created_at:       Mapped[DateTime]     = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:       Mapped[DateTime]     = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    devices:          Mapped[list["UserDevice"]] = relationship(back_populates="user", cascade="all, delete-orphan")
