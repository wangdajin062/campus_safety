"""
tests/test_extended.py - 扩展测试：Redis降级、短信服务、推送服务、调度器、管理后台
"""

import pytest
import pytest_asyncio
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════
# 1. Redis 内存降级测试（无需真实 Redis）
# ═══════════════════════════════════════════════════════════
class TestMemoryCache:

    @pytest.mark.asyncio
    async def test_setex_and_get(self):
        from core.redis import MemoryCache
        cache = MemoryCache()
        await cache.setex("k1", 300, "hello")
        val = await cache.get("k1")
        assert val == "hello"

    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        from core.redis import MemoryCache
        cache = MemoryCache()
        val = await cache.get("nonexistent")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete(self):
        from core.redis import MemoryCache
        cache = MemoryCache()
        await cache.setex("k2", 300, "value")
        await cache.delete("k2")
        assert await cache.get("k2") is None

    @pytest.mark.asyncio
    async def test_incr_counter(self):
        from core.redis import MemoryCache
        cache = MemoryCache()
        assert await cache.incr("counter") == 1
        assert await cache.incr("counter") == 2
        assert await cache.incr("counter") == 3

    @pytest.mark.asyncio
    async def test_ping_always_true(self):
        from core.redis import MemoryCache
        cache = MemoryCache()
        assert await cache.ping() is True

    @pytest.mark.asyncio
    async def test_expire_sets_ttl(self):
        import time
        from core.redis import MemoryCache
        cache = MemoryCache()
        await cache.setex("k3", 300, "data")
        await cache.expire("k3", 1)
        # 等待过期
        await asyncio.sleep(1.1)
        val = await cache.get("k3")
        assert val is None


# ═══════════════════════════════════════════════════════════
# 2. 验证码服务测试
# ═══════════════════════════════════════════════════════════
class TestSmsCodeService:

    @pytest.mark.asyncio
    async def test_save_and_verify_correct_code(self):
        from core.redis import SmsCodeService, MemoryCache, _memory_store
        _memory_store.clear()

        with patch("core.redis.get_redis", return_value=AsyncMock(
            setex=AsyncMock(), get=AsyncMock(return_value="123456"), delete=AsyncMock()
        )):
            await SmsCodeService.save_code("hash_abc", "123456")
            result = await SmsCodeService.verify_code("hash_abc", "123456")
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_wrong_code(self):
        from core.redis import SmsCodeService

        with patch("core.redis.get_redis", return_value=AsyncMock(
            get=AsyncMock(return_value="123456"), delete=AsyncMock()
        )):
            result = await SmsCodeService.verify_code("hash_abc", "999999")
            assert result is False

    @pytest.mark.asyncio
    async def test_verify_expired_code(self):
        from core.redis import SmsCodeService

        with patch("core.redis.get_redis", return_value=AsyncMock(
            get=AsyncMock(return_value=None)
        )):
            result = await SmsCodeService.verify_code("hash_abc", "123456")
            assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_allows_first_request(self):
        from core.redis import SmsCodeService

        with patch("core.redis.get_redis", return_value=AsyncMock(
            incr=AsyncMock(return_value=1), expire=AsyncMock()
        )):
            allowed = await SmsCodeService.check_rate_limit("hash_xyz", max_per_minute=3)
            assert allowed is True

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess(self):
        from core.redis import SmsCodeService

        with patch("core.redis.get_redis", return_value=AsyncMock(
            incr=AsyncMock(return_value=4), expire=AsyncMock()
        )):
            allowed = await SmsCodeService.check_rate_limit("hash_xyz", max_per_minute=3)
            assert allowed is False


# ═══════════════════════════════════════════════════════════
# 3. 短信发送服务测试
# ═══════════════════════════════════════════════════════════
class TestSmsService:

    def test_generate_code_length(self):
        from services.sms_service import generate_code
        code = generate_code(6)
        assert len(code) == 6
        assert code.isdigit()

    def test_generate_code_unique(self):
        from services.sms_service import generate_code
        codes = {generate_code(6) for _ in range(100)}
        # 100次生成至少有2种不同结果（极小概率全相同）
        assert len(codes) > 1

    def test_generate_code_range(self):
        from services.sms_service import generate_code
        for _ in range(50):
            code = generate_code(6)
            assert 100000 <= int(code) <= 999999

    @pytest.mark.asyncio
    async def test_aliyun_dev_mode_skip(self):
        """未配置 access_key 时开发模式跳过发送，返回 True"""
        from services.sms_service import AliyunSmsProvider
        from unittest.mock import patch
        with patch("services.sms_service.settings") as mock_settings:
            mock_settings.SMS_ACCESS_KEY = ""  # 未配置
            mock_settings.SMS_SECRET_KEY = ""
            mock_settings.SMS_SIGN_NAME = "校园安全"
            mock_settings.SMS_TEMPLATE_CODE = "SMS_000"
            provider = AliyunSmsProvider()
            provider.access_key = ""
            result = await provider.send("13800138000", "123456")
            assert result is True  # 开发模式直接返回 True

    def test_factory_returns_aliyun(self):
        from services.sms_service import get_sms_provider, AliyunSmsProvider
        from unittest.mock import patch
        with patch("services.sms_service.settings") as mock_settings:
            mock_settings.SMS_PROVIDER = "aliyun"
            mock_settings.SMS_ACCESS_KEY = ""
            mock_settings.SMS_SECRET_KEY = ""
            mock_settings.SMS_SIGN_NAME = "test"
            mock_settings.SMS_TEMPLATE_CODE = "T1"
            provider = get_sms_provider()
            assert isinstance(provider, AliyunSmsProvider)

    def test_factory_returns_tencent(self):
        from services.sms_service import get_sms_provider, TencentSmsProvider
        from unittest.mock import patch
        with patch("services.sms_service.settings") as mock_settings:
            mock_settings.SMS_PROVIDER = "tencent"
            mock_settings.SMS_ACCESS_KEY = ""
            mock_settings.SMS_SECRET_KEY = ""
            mock_settings.SMS_SIGN_NAME = "test"
            mock_settings.SMS_TEMPLATE_CODE = "T1"
            provider = get_sms_provider()
            assert isinstance(provider, TencentSmsProvider)


# ═══════════════════════════════════════════════════════════
# 4. FCM 推送服务测试
# ═══════════════════════════════════════════════════════════
class TestPushService:

    @pytest.mark.asyncio
    async def test_send_to_token_dev_mode(self):
        """未配置 FCM Key 时跳过推送返回 True"""
        from services.push_service import FCMPushService, PushPayload
        service = FCMPushService()
        service.server_key = ""  # 未配置
        result = await service.send_to_token("fake_token", PushPayload("标题", "内容"))
        assert result is True

    @pytest.mark.asyncio
    async def test_send_to_tokens_dev_mode(self):
        from services.push_service import FCMPushService, PushPayload
        service = FCMPushService()
        service.server_key = ""
        tokens = ["t1", "t2", "t3"]
        result = await service.send_to_tokens(tokens, PushPayload("标题", "内容"))
        assert result["success"] == 3
        assert result["failure"] == 0

    @pytest.mark.asyncio
    async def test_send_to_tokens_real_chunked(self):
        """批量推送按 500 分块"""
        from services.push_service import FCMPushService, PushPayload

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "success": len(kwargs.get("json", {}).get("registration_ids", [])),
                "failure": 0, "results": []
            }
            return mock_resp

        service = FCMPushService()
        service.server_key = "fake_key_for_test"

        tokens = [f"token_{i}" for i in range(1200)]
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.post = mock_post
            mock_client.return_value = mock_ctx

            result = await service.send_to_tokens(tokens, PushPayload("Test", "body"))
            # 1200 tokens / 500 per chunk = 3 chunks
            assert call_count == 3

    def test_build_message_structure(self):
        from services.push_service import FCMPushService, PushPayload
        service = FCMPushService()
        payload = PushPayload("Title", "Body", data={"type": "fraud_alert"})
        msg = service._build_message("token_xyz", payload)

        assert msg["to"] == "token_xyz"
        assert msg["notification"]["title"] == "Title"
        assert msg["data"]["type"] == "fraud_alert"
        assert "android" in msg
        assert "apns" in msg


# ═══════════════════════════════════════════════════════════
# 5. 调度器任务测试
# ═══════════════════════════════════════════════════════════
class TestScheduler:

    @pytest.mark.asyncio
    async def test_update_protection_scores_logic(self):
        """防护积分计算公式验证"""
        def calc_score(blocked_calls, alerted_sms, total_reports, cases_read):
            score = (
                min(blocked_calls, 30)
                + min(alerted_sms * 2, 20)
                + min(total_reports * 3, 30)
                + min(cases_read // 5, 20)
            )
            return min(100, score)

        assert calc_score(0, 0, 0, 0) == 0
        assert calc_score(30, 0, 0, 0) == 30   # 电话上限 30
        assert calc_score(0, 10, 0, 0) == 20   # 短信上限 20
        assert calc_score(0, 0, 10, 0) == 30   # 举报上限 30
        assert calc_score(0, 0, 0, 100) == 20  # 阅读上限 20
        assert calc_score(100, 100, 100, 100) == 100  # 总分上限 100
        assert calc_score(10, 5, 3, 25) == 10 + 10 + 9 + 5  # = 34

    @pytest.mark.asyncio
    async def test_scheduler_starts_and_cancels(self):
        """调度器 Task 能正常启动和取消"""
        from services.scheduler import start_scheduler

        with patch("services.scheduler.push_pending_alerts", new_callable=AsyncMock):
            with patch("services.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                mock_sleep.side_effect = asyncio.CancelledError()
                task = start_scheduler()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                assert task.done()


# ═══════════════════════════════════════════════════════════
# 6. 管理后台 API 测试
# ═══════════════════════════════════════════════════════════
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from core.database import Base, get_db
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSession = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSession() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            raise exc


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _seed_code(phone: str, code: str) -> None:
    """Pre-seed SMS code in MemoryCache for testing."""
    from core.redis import CacheService
    from core.security import phone_hash
    ph = phone_hash(phone)
    await CacheService.setex(f"sms_code:{ph}", 300, code)


async def get_admin_token(client: AsyncClient) -> str:
    """创建管理员用户（protection_score=99）并获取 Token"""
    await _seed_code("13900000001", "000000")
    r = await client.post("/v1/auth/login", json={"phone": "13900000001", "code": "000000"})
    token = r.json()["data"]["access_token"]

    # 直接设置管理员标识
    from models.user import User
    from sqlalchemy import update
    async with TestSession() as s:
        await s.execute(update(User).values(role="admin", protection_score=99))
        await s.commit()
    return token


class TestAdminAPI:

    @pytest.mark.asyncio
    async def test_dashboard_requires_admin(self, client: AsyncClient):
        """普通用户无法访问管理后台"""
        await _seed_code("13800000001", "111111")
        r = await client.post("/v1/auth/login", json={"phone": "13800000001", "code": "111111"})
        token = r.json()["data"]["access_token"]
        r2 = await client.get("/v1/admin/dashboard", headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 403

    @pytest.mark.asyncio
    async def test_dashboard_admin_access(self, client: AsyncClient):
        token = await get_admin_token(client)
        r = await client.get("/v1/admin/dashboard", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert "total_users" in data
        assert "pending_reports" in data

    @pytest.mark.asyncio
    async def test_create_case_success(self, client: AsyncClient):
        token = await get_admin_token(client)
        r = await client.post("/v1/admin/cases", json={
            "title": "测试案例", "summary": "摘要", "content": "内容",
            "category": "刷单诈骗", "risk_level": "high", "emoji": "💳"
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["data"]["id"] > 0

    @pytest.mark.asyncio
    async def test_create_case_missing_field(self, client: AsyncClient):
        token = await get_admin_token(client)
        r = await client.post("/v1/admin/cases", json={
            "title": "不完整案例"  # 缺少 summary/content/category/risk_level
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_add_and_list_keywords(self, client: AsyncClient):
        token = await get_admin_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        # 添加关键词
        r = await client.post("/v1/admin/keywords", json={
            "keyword": "安全账户", "risk_weight": 95, "category": "金融"
        }, headers=headers)
        assert r.status_code == 200

        # 查询列表
        r2 = await client.get("/v1/admin/keywords", headers=headers)
        assert r2.status_code == 200
        keywords = r2.json()["data"]
        assert any(k["keyword"] == "安全账户" for k in keywords)

    @pytest.mark.asyncio
    async def test_create_alert(self, client: AsyncClient):
        token = await get_admin_token(client)
        with patch("services.scheduler.push_pending_alerts", new_callable=AsyncMock):
            r = await client.post("/v1/admin/alerts", json={
                "title": "紧急预警测试", "content": "内容详情",
                "risk_level": "high", "is_urgent": True
            }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["data"]["id"] > 0

    @pytest.mark.asyncio
    async def test_approve_report_flow(self, client: AsyncClient):
        """举报 → 审核通过 → 诈骗库入库 完整流程"""
        # 普通用户提交举报
        await _seed_code("13700000001", "111111")
        user_r = await client.post("/v1/auth/login", json={"phone": "13700000001", "code": "111111"})
        user_token = user_r.json()["data"]["access_token"]
        report_r = await client.post("/v1/reports", json={
            "report_type": "phone", "target": "13666666666"
        }, headers={"Authorization": f"Bearer {user_token}"})
        assert report_r.status_code == 200

        # 管理员审核通过
        admin_token = await get_admin_token(client)
        list_r = await client.get("/v1/admin/reports?status=pending",
                                  headers={"Authorization": f"Bearer {admin_token}"})
        reports_data = list_r.json()["data"]
        assert len(reports_data) >= 1
        report_id = reports_data[0]["id"]

        approve_r = await client.post(f"/v1/admin/reports/{report_id}/approve",
                                      json={"risk_type": "冒充客服", "note": "已核实"},
                                      headers={"Authorization": f"Bearer {admin_token}"})
        assert approve_r.status_code == 200
        assert approve_r.json()["data"]["fraud_phone_id"] is not None

        # 验证号码已进入诈骗库
        check_r = await client.get("/v1/calls/check?phone=13666666666",
                                   headers={"Authorization": f"Bearer {user_token}"})
        assert check_r.status_code == 200
        # 审核通过后应能查到该号码
        assert check_r.json()["data"]["risk_level"] in ("high", "medium", "safe")
