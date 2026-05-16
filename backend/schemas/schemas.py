"""
schemas/schemas.py - Pydantic 请求/响应 Schema
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# ── 通用 ─────────────────────────────────────────────────
class RiskLevelEnum(str, Enum):
    safe   = "safe"
    medium = "medium"
    high   = "high"


class Resp(BaseModel):
    """统一响应包装"""
    code: int = 200
    data: Any = None
    message: str = "ok"


class PageMeta(BaseModel):
    total: int
    page: int = Field(..., ge=1)
    limit: int = Field(..., ge=1, le=50)


# ── 认证 ─────────────────────────────────────────────────
class SendCodeRequest(BaseModel):
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$", description="11位手机号")

class LoginRequest(BaseModel):
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$")
    code:  str = Field(..., min_length=6, max_length=6)

class TokenResponse(BaseModel):
    model_config = {"from_attributes": True}
    token: str
    user:  dict


# ── 来电检测 ─────────────────────────────────────────────
class PhoneCheckResponse(BaseModel):
    model_config = {"from_attributes": True}
    id:           Optional[int]   = None
    risk_level:   RiskLevelEnum   = RiskLevelEnum.safe
    risk_type:    Optional[str]   = None
    risk_score:   int             = 0
    report_count: int             = 0
    location:     Optional[str]   = "未知"
    is_verified:  bool            = False

class CallLogResponse(BaseModel):
    id:             int
    phone_number:   str
    risk_level:     RiskLevelEnum
    detection_type: str
    detected_at:    datetime

    model_config = {"from_attributes": True}


# ── 短信预警 ─────────────────────────────────────────────
class SmsAnalyzeRequest(BaseModel):
    sender:   str                 = Field(..., max_length=50)
    keywords: List[str]          = Field(default=[])
    has_url:  bool               = False
    content_length: int          = Field(default=0, ge=0)

class SmsAnalyzeResponse(BaseModel):
    model_config = {"from_attributes": True}
    risk_level:       RiskLevelEnum
    risk_score:       int
    matched_keywords: List[str]


# ── 案例库 ───────────────────────────────────────────────
class FraudCaseListItem(BaseModel):
    id:           int
    title:        str
    summary:      str
    category:     str
    risk_level:   RiskLevelEnum
    emoji:        str
    view_count:   int
    like_count:   int
    tags:         List[str] = []
    published_at: datetime

    model_config = {"from_attributes": True}

class FraudCaseDetail(FraudCaseListItem):
    content:      str
    share_count:  int
    related_ids:  List[int] = []


# ── 电诈预警 ─────────────────────────────────────────────
class FraudAlertResponse(BaseModel):
    id:           int
    title:        str
    content:      str
    risk_level:   RiskLevelEnum
    emoji:        str
    tags:         List[str] = []
    is_urgent:    bool
    report_count: int
    published_at: datetime

    model_config = {"from_attributes": True}

class FraudAlertSummary(BaseModel):
    id:           int
    title:        str
    risk_level:   RiskLevelEnum
    emoji:        str
    is_urgent:    bool
    published_at: datetime

    model_config = {"from_attributes": True}


# ── 举报 ─────────────────────────────────────────────────
class ReportRequest(BaseModel):
    report_type: str    = Field(..., pattern=r"^(phone|sms|link|other)$")
    target:      str    = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=1000)
    school:      Optional[str] = Field(None, max_length=100)


# ── 用户 ─────────────────────────────────────────────────
class UserStatsResponse(BaseModel):
    model_config = {"from_attributes": True}
    blocked_calls:     int
    alerted_sms:       int
    total_reports:     int
    cases_read:        int
    protection_score:  int
    protection_level:  str


class DeviceRegisterRequest(BaseModel):
    device_id:   str = Field(..., max_length=128)
    platform:    str = Field(..., pattern=r"^(ios|android)$")
    fcm_token:   Optional[str] = Field(None, max_length=256)
    app_version: Optional[str] = Field(None, max_length=20)
    os_version:  Optional[str] = Field(None, max_length=30)
