"""
core/config.py - 全局配置（从环境变量读取）
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "校园安全"
    APP_ENV: str = "development"
    SECRET_KEY: str = Field(
        default="",
        min_length=32,
        description="JWT 签名密钥，生产环境必须设置，长度 >= 32 字符"
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = Field(default=1, ge=1, le=7, description="access_token 过期时间（推荐 1-7 天）")
    JWT_REFRESH_EXPIRE_DAYS: int = Field(default=7, ge=1, le=30, description="refresh_token 过期时间")

    # 数据库类型: "sqlite" (开发) / "postgresql" (生产)
    DATABASE_TYPE: str = "sqlite"

    # 数据库连接 URL（根据 DATABASE_TYPE 自动选择）
    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_TYPE == "sqlite":
            return "sqlite+aiosqlite:///campus_safety.db"
        return "postgresql+asyncpg://postgres:password@localhost:5432/campus_safety"

    # Redis（验证码缓存 & 限流）
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "https://campus-safety.com"]

    # 短信服务（阿里云/腾讯云）
    SMS_PROVIDER: str = "aliyun"
    SMS_ACCESS_KEY: str = ""
    SMS_SECRET_KEY: str = ""
    SMS_SIGN_NAME: str = "校园安全"
    SMS_TEMPLATE_CODE: str = "SMS_000000"

    # Firebase Cloud Messaging
    FCM_SERVER_KEY: str = ""


    # 运行配置
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # ML 推理（可选）
    DRAFT_MODEL_PATH: str = "/models/fraud_draft_q4km.gguf"
    VERIFY_MODEL_URL: str = "http://llm-server:8001"
    GAMMA: int = 5
    # 限流
    RATE_LIMIT_PER_MINUTE: int = 100
    SMS_CODE_EXPIRE_SECONDS: int = 300

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}
    
    def __init__(self, **data):
        """初始化时验证 SECRET_KEY"""
        super().__init__(**data)
        # 生产环境强制检查 SECRET_KEY
        if self.APP_ENV == "production":
            if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "❌ 生产环境必须在 .env 中设置 SECRET_KEY，"
                    "最少 32 字符。示例: "
                    "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
                )
            if self.SECRET_KEY == "campus_safety_secret_change_in_prod":
                raise ValueError("❌ 不能使用默认 SECRET_KEY，请生成新的密钥")


settings = Settings()
