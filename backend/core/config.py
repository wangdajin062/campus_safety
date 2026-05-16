"""
core/config.py - 全局配置（从环境变量读取）
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "校园安全"
    APP_ENV: str = "development"
    SECRET_KEY: str = "campus_safety_secret_change_in_prod"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 30

    # 数据库 PostgreSQL
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/campus_safety"

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


settings = Settings()
