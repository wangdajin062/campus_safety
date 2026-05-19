"""
校园安全 APP v3 — FastAPI 主入口
软硬协同多模态推测解码架构
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import settings
from core.database import engine, Base
from core.redis import get_redis, close_redis
from api.v1 import auth, calls, sms, cases, alerts, reports, users, admin
from api.v1 import inference
from services.scheduler import start_scheduler

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)
_scheduler_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_task
    # ✅ 启动时检查配置
    if settings.APP_ENV == "production":
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            raise RuntimeError(
                "❌ 生产环境必须设置 SECRET_KEY (>= 32 字符)"
            )
    logger.info(f"App environment: {settings.APP_ENV}")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await get_redis()
    from ml.speculative_decoder import spec_decoder
    logger.info("Draft model: %s", "loaded" if spec_decoder.draft_model._loaded else "prior")
    _scheduler_task = start_scheduler()
    logger.info("QAD-MultiGuard v4.1 ready — SpecDec(α=0.86) + QAD-4bit + Multimodal + DP")
    yield
    if _scheduler_task: _scheduler_task.cancel()
    await close_redis()
    await engine.dispose()


app = FastAPI(
    title="校园安全 API v4.1",
    description="QAD-MultiGuard 软硬协同多模态检测 | 推测解码 α=0.86 | 隐私保护 PIPL §23",
    version="4.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=["Authorization","Content-Type","Accept","X-Session-ID"])

app.include_router(auth.router,      prefix="/v1/auth",   tags=["认证"])
app.include_router(calls.router,     prefix="/v1/calls",  tags=["来电"])
app.include_router(sms.router,       prefix="/v1/sms",    tags=["短信"])
app.include_router(cases.router,     prefix="/v1/cases",  tags=["案例"])
app.include_router(alerts.router,    prefix="/v1/alerts", tags=["预警"])
app.include_router(reports.router,   prefix="/v1/reports",tags=["举报"])
app.include_router(users.router,     prefix="/v1/user",   tags=["用户"])
app.include_router(admin.router,     prefix="/v1/admin",  tags=["管理"])
app.include_router(inference.router, prefix="/v1/infer",  tags=["推理引擎"])


@app.get("/health")
async def health():
    from ml.speculative_decoder import spec_decoder
    redis = await get_redis()
    return {
        "status": "ok", "version": "4.1.0",
        "arch": "speculative_decoding+QAD_4bit+multimodal",
        "redis": "ok" if await redis.ping() else "degraded",
        "draft_model": "loaded" if spec_decoder.draft_model._loaded else "prior",
    }
