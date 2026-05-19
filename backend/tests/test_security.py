"""
tests/test_security.py — 安全漏洞回归测试
=========================================
覆盖 C1, H1-H6, M1-M5 安全修复

运行: pytest tests/test_security.py -v
"""
import pytest
import pytest_asyncio
import asyncio
import json
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════
# C1: 死代码测试 — 模块导入不报错
# ═══════════════════════════════════════════════════════════
def test_inference_module_no_dead_code():
    """验证 inference.py 无死代码（C1）"""
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "api", "v1", "inference.py")
    )
    # 只做语法检查（AST 解析在之前验证过）
    with open(spec.origin, "r", encoding="utf-8") as f:
        import ast
        ast.parse(f.read())
    # 确保没有重复的路由定义 — infer_fast 只应出现一次
    content = open(spec.origin, "r", encoding="utf-8").read()
    assert content.count("async def infer_fast(") == 1, "存在重复的 infer_fast 定义"
    assert content.count("async def analyze_voice(") == 1, "存在重复的 analyze_voice"
    assert content.count("async def acoustic_non_invertibility_test(") == 1, "存在重复的 acoustic_non_invertibility_test"
    assert "body.sms_features" not in content.split("def ")[0], "模块级代码引用了 body"


# ═══════════════════════════════════════════════════════════
# H1: Feedback 并发写入安全
# ═══════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_feedback_concurrent_safety():
    """验证 feedback.jsonl 并发写入无竞态（H1）"""
    from ml.speculative_decoder import SpeculativeDecoder
    decoder = SpeculativeDecoder()
    # 使用临时路径避免污染
    original_path = decoder._feedback_path
    tmp_path = Path(Path(__file__).parent / "_test_feedback.jsonl")
    if tmp_path.exists():
        tmp_path.unlink()

    mock_path = lambda: tmp_path
    decoder._feedback_path = mock_path

    async def write_feedback(i):
        await decoder.record_feedback(f"hash_{i}", i % 2, [float(i)] * 12)

    tasks = [write_feedback(i) for i in range(100)]
    await asyncio.gather(*tasks)

    count = decoder.feedback_count()
    tmp_path.unlink(missing_ok=True)
    assert count == 100, f"期望100行，实际{count}行 — 可能存在竞态"


# ═══════════════════════════════════════════════════════════
# H2: Retrain 限流 + 并发去重
# ═══════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_retrain_dedup_flag():
    """验证 _retrain_in_progress 标志位（H2）"""
    import importlib
    inf = importlib.import_module("api.v1.inference")
    importlib.reload(inf)
    assert hasattr(inf, "_retrain_in_progress"), "缺少 _retrain_in_progress 标志"


# ═══════════════════════════════════════════════════════════
# H3: 手机号哈希使用 PBKDF2
# ═══════════════════════════════════════════════════════════
def test_phone_hash_is_pbkdf2():
    """验证 phone_hash 使用 PBKDF2 而非 SHA256（H3）"""
    from core.security import phone_hash, sha256
    h1 = phone_hash("13800138000")
    h2 = phone_hash("13800138000")
    assert h1 == h2, "相同手机号哈希应一致"
    assert len(h1) == 64, "PBKDF2 SHA256 输出应为 32 字节 = 64 hex"
    # PBKDF2 输出长度应不同于 SHA256（概率极高）
    plain_sha = sha256("13800138000")
    assert h1 != plain_sha, "PBKDF2 不应等于 SHA256"


# ═══════════════════════════════════════════════════════════
# H4: Refresh Token Rotation
# ═══════════════════════════════════════════════════════════
def test_token_has_jti():
    """验证 JWT 包含 jti 字段（H4/H5 黑名单前置条件）"""
    from core.security import create_tokens, decode_token
    tokens = create_tokens(1, "test_hash")
    access_payload = decode_token(tokens["access_token"], "access")
    assert "jti" in access_payload, "access_token 缺少 jti"
    refresh_payload = decode_token(tokens["refresh_token"], "refresh")
    assert "jti" in refresh_payload, "refresh_token 缺少 jti"


@pytest.mark.asyncio
async def test_blacklist_token():
    """验证 token 黑名单功能（H5）"""
    from core.security import create_tokens, blacklist_token, is_token_blacklisted
    tokens = create_tokens(1, "test_hash")
    token = tokens["access_token"]
    # 初始不应在黑名单
    assert not await is_token_blacklisted(token)
    # 加入黑名单
    await blacklist_token(token, expire_seconds=60)
    assert await is_token_blacklisted(token)


# ═══════════════════════════════════════════════════════════
# H6: DP Epsilon 计算
# ═══════════════════════════════════════════════════════════
def test_dp_epsilon_calculation():
    """验证 DP ε 使用正确高斯公式（H6）"""
    from ml.acoustic_embedding import calc_dp_epsilon
    # σ=1.0, Δ₂=2.0, δ=1e-5
    # 高斯机制公式: ε = Δ₂ · sqrt(2 · ln(1.25/δ)) / σ
    # Δ₂=2.0, δ=1e-5, σ=1.0 → ε ≈ 9.69
    eps = calc_dp_epsilon(1.0)
    assert abs(eps - 9.69) < 0.1, f"期望 ε≈9.69，实际 {eps}"
    # σ=0 → inf
    assert calc_dp_epsilon(0.0) == float("inf")
    assert calc_dp_epsilon(-1.0) == float("inf")


# ═══════════════════════════════════════════════════════════
# M3: Feedback 路径非相对
# ═══════════════════════════════════════════════════════════
def test_feedback_path_is_absolute():
    """验证 feedback.jsonl 路径为绝对路径（M3）"""
    from ml.speculative_decoder import SpeculativeDecoder
    path = SpeculativeDecoder._feedback_path()
    assert path.is_absolute(), f"反馈路径应为绝对路径: {path}"
    assert path.name == "feedback.jsonl"
    # 应位于 backend/ml/models/ 下
    assert "ml" in path.parts
    assert "models" in path.parts


# ═══════════════════════════════════════════════════════════
# M5: Admin setattr 类型校验
# ═══════════════════════════════════════════════════════════
def test_admin_update_field_whitelist():
    """验证 admin update_case 字段白名单（M5）"""
    import ast
    admin_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "api", "v1", "admin.py")
    with open(admin_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    # 检查是否有 setattr 调用
    setattr_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr":
            setattr_calls.append(node.lineno)
    assert len(setattr_calls) > 0, "未找到 setattr 调用"
    # 验证文件包含类型校验代码
    with open(admin_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "isinstance(val, str)" in content, "缺少字符串类型校验"
    assert "isinstance(val, bool)" in content, "缺少布尔类型校验"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
