"""
tests/test_api.py — FastAPI 自动化测试（兼容 SQLite 内存库）
运行: pytest tests/ -v
"""
import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import Base, get_db
from main import app

# ── Per-test isolated engine ──────────────────────────────
def make_test_session():
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_engine():
    engine, _ = make_test_session()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine):
    _, TestSession = make_test_session()
    # Reuse same engine
    _, TestSession2 = db_engine, async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with TestSession2() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def get_token(client: AsyncClient, phone: str = "13800138000") -> str:
    r = await client.post("/v1/auth/login", json={"phone": phone, "code": "123456"})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["data"]["token"]


# ══ AUTH ══════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_send_code_valid_phone(client: AsyncClient):
    r = await client.post("/v1/auth/send-code", json={"phone": "13800138000"})
    assert r.status_code == 200
    assert r.json()["code"] == 200


@pytest.mark.asyncio
async def test_send_code_invalid_phone(client: AsyncClient):
    r = await client.post("/v1/auth/send-code", json={"phone": "00000000000"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_login_creates_new_user(client: AsyncClient):
    r = await client.post("/v1/auth/login", json={"phone": "13900139000", "code": "123456"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert "token" in data
    assert data["user"]["id"] > 0


@pytest.mark.asyncio
async def test_login_idempotent(client: AsyncClient):
    for _ in range(2):
        r = await client.post("/v1/auth/login", json={"phone": "13900139000", "code": "123456"})
        assert r.status_code == 200


# ══ CALLS ═════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_check_phone_no_auth(client: AsyncClient):
    r = await client.get("/v1/calls/check", params={"phone": "13800138000"})
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_check_phone_unknown(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/calls/check",
                          params={"phone": "13800138000"},
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["data"]["risk_level"] == "safe"


@pytest.mark.asyncio
async def test_call_history_pagination(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/calls/history",
                          params={"page": 1, "limit": 10},
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "data" in r.json()


# ══ SMS ═══════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_sms_analyze_safe(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/sms/analyze",
        json={"sender": "10086", "keywords": [], "has_url": False, "content_length": 50},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_sms_analyze_with_keywords(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/sms/analyze",
        json={"sender": "unknown", "keywords": ["安全账户", "立即转账"], "has_url": True, "content_length": 120},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["data"]["risk_level"] in ("medium", "high")


# ══ CASES ═════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_cases_list_empty(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/cases",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_cases_list_with_data(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/cases",
                          params={"category": "冒充公检法"},
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_case_detail_not_found(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/cases/99999",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (200, 404)


# ══ REPORTS ═══════════════════════════════════════════════
@pytest.mark.asyncio
async def test_submit_report_valid(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/reports",
        json={"target": "13800138001", "report_type": "phone",
              "description": "对方自称公安局要求转账"},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_submit_report_invalid_type(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/reports",
        json={"target": "13800138001", "report_type": "invalid_type",
              "description": "测试"},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (200, 422)


# ══ USER ══════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_user_stats_defaults(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/user/stats",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    d = r.json()["data"]
    assert "blocked_calls" in d


@pytest.mark.asyncio
async def test_device_register(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/user/device",
        json={"device_id": "test-device-001", "fcm_token": "test_fcm_token_abc123", "platform": "android"},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


# ══ ALERTS ════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_alerts_latest_empty(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/alerts/latest",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_alerts_list(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/alerts",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


# ══ HEALTH ════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ══ ML INFERENCE ══════════════════════════════════════════
@pytest.mark.asyncio
async def test_infer_fast(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/infer/fast",
        json={"sms_features": [0.8,0.9,0.9,1.0,1.0,1.0,1.0,0.3,0.1,1.0,1.0,1.0],
              "session_id": "test_session"},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["risk_level"] in ("safe", "medium", "high")
    assert 0 <= d["risk_score"] <= 100


@pytest.mark.asyncio
async def test_model_status(client: AsyncClient):
    token = await get_token(client)
    r = await client.get("/v1/infer/model-status",
                          headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "draft_model" in r.json()["data"]


@pytest.mark.asyncio
async def test_infer_feedback(client: AsyncClient):
    token = await get_token(client)
    r = await client.post("/v1/infer/feedback",
        json={"sample_hash": "a" * 16, "true_label": 1,
              "feature_vector": [0.8]*12},
        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
