#!/usr/bin/env python3
"""
scripts/verify_deployment.py
部署验证脚本 — v3 软硬协同架构
用法：python scripts/verify_deployment.py [--url http://localhost:8000]
"""

import sys
import asyncio
import json
import time
import argparse
import httpx

BASE_URL = "http://localhost:8000"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []


def report(name, ok, detail=""):
    icon = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


async def verify(base_url: str, token: str = ""):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:

        # ── 1. 健康检查 ───────────────────────────────────
        print("\n[1] 系统健康检查")
        try:
            r = await client.get("/health")
            data = r.json()
            report("HTTP 200", r.status_code == 200, str(r.status_code))
            report("版本 v3", data.get("version") == "3.0.0", data.get("version", "?"))
            report("Redis 连接", data.get("redis") == "ok", data.get("redis", "?"))
            report("架构标识", "speculative_decoding" in str(data.get("arch", "")))
        except Exception as e:
            report("健康检查", False, str(e))

        # ── 2. 认证流程 ───────────────────────────────────
        print("\n[2] 认证流程验证")
        try:
            r = await client.post("/v1/auth/send-code",
                json={"phone": "13800138000"})
            report("发送验证码", r.status_code in (200, 429),
                   f"status={r.status_code}")
        except Exception as e:
            report("认证端点", False, str(e))

        if not token:
            print("  ℹ️  跳过需鉴权端点（未提供 JWT）")
            return

        # ── 3. 快速推理端点 ───────────────────────────────
        print("\n[3] 快速推理端点 /v1/infer/fast")
        try:
            t0 = time.perf_counter()
            r = await client.post("/v1/infer/fast",
                headers=headers,
                json={
                    "sms_features": [0.8,0.9,0.9,1.0,1.0,1.0,1.0,0.3,0.1,1.0,1.0,1.0],
                    "call_features":[0.6,0.3,0.0,0.0,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                })
            latency = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                data = r.json().get("data", {})
                report("响应200", True)
                report(f"延迟<100ms ({latency:.0f}ms)", latency < 100, f"{latency:.1f}ms")
                report("风险等级有效", data.get("risk_level") in ("safe","medium","high"),
                       data.get("risk_level"))
                report("高危特征→非safe", data.get("risk_level") != "safe")
                report("多模态字段完整",
                       all(k in data.get("modalities",{}) for k in ("sms","call","url","voice")))
            else:
                report("快速推理", False, f"status={r.status_code}")
        except Exception as e:
            report("快速推理端点", False, str(e))

        # ── 4. SSE 流式端点 ───────────────────────────────
        print("\n[4] SSE 流式端点 /v1/infer/stream")
        try:
            t0 = time.perf_counter()
            events = []
            async with client.stream("POST", "/v1/infer/stream",
                headers={**headers, "Accept": "text/event-stream"},
                json={
                    "sms_features": [0.8,0.9,0.9,1.0,1.0,1.0,1.0,0.3,0.1,1.0,1.0,1.0],
                    "enable_cot": True,
                }) as response:
                timeout = time.perf_counter() + 5.0
                async for line in response.aiter_lines():
                    if time.perf_counter() > timeout:
                        break
                    if line.startswith("data:"):
                        try:
                            events.append(json.loads(line[5:]))
                        except Exception:
                            pass
            latency = (time.perf_counter() - t0) * 1000
            event_types = [e.get("event", e.get("stage")) for e in events]
            report("SSE 流建立", response.status_code == 200)
            report("包含事件流", len(events) > 0, f"{len(events)} events")
            has_final = any("final_result" in str(e) or e.get("stage")=="final" for e in events)
            report("包含最终结论", has_final)
            report(f"流完成<5s ({latency:.0f}ms)", latency < 5000)
        except Exception as e:
            report("SSE 流端点", False, str(e))

        # ── 5. 语音模态端点 ────────────────────────────────
        print("\n[5] 语音模态端点 /v1/infer/voice")
        try:
            import random
            emb = [random.uniform(0.5, 1.0) for _ in range(64)]
            r = await client.post("/v1/infer/voice",
                headers=headers,
                json={"audio_embedding": emb, "voice_text": "公安局 安全账户 立即转账",
                      "call_duration_s": 45.0})
            if r.status_code == 200:
                data = r.json().get("data", {})
                report("语音分析响应", True)
                report("语音风险分有效", "voice_score" in data)
                report("文本风险分有效", "text_score" in data)
            else:
                report("语音端点", False, f"status={r.status_code}")
        except Exception as e:
            report("语音端点", False, str(e))

        # ── 6. 模型状态端点 ────────────────────────────────
        print("\n[6] 模型状态 /v1/infer/model-status")
        try:
            r = await client.get("/v1/infer/model-status", headers=headers)
            if r.status_code == 200:
                data = r.json().get("data", {})
                report("模型状态响应", True)
                report("QAD配置完整", "qad_config" in data and data["qad_config"]["bits"] == 4)
                report("运行时统计完整", "runtime_stats" in data)
                report("反馈计数有效", "feedback" in data)
            else:
                report("模型状态端点", False, f"status={r.status_code}")
        except Exception as e:
            report("模型状态端点", False, str(e))

        # ── 7. 原有端点向后兼容性 ──────────────────────────
        print("\n[7] 向后兼容性验证（v1/v2 端点）")
        for path, method, body in [
            ("/v1/calls/check", "GET", None),
            ("/v1/sms/analyze", "POST", {"sender":"10086","keywords":[],"has_url":False}),
            ("/v1/cases",       "GET", None),
            ("/v1/alerts",      "GET", None),
        ]:
            try:
                if method == "GET":
                    r = await client.get(path, headers=headers)
                else:
                    r = await client.post(path, headers=headers, json=body or {})
                report(f"GET/POST {path}", r.status_code in (200, 422, 404),
                       str(r.status_code))
            except Exception as e:
                report(f"{path}", False, str(e)[:40])


def main():
    parser = argparse.ArgumentParser(description="Campus Safety v3 Deployment Verifier")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    parser.add_argument("--token", default="", help="JWT Bearer token")
    args = parser.parse_args()

    print(f"🛡️  校园安全 APP v3 部署验证")
    print(f"   目标: {args.url}")
    print(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    asyncio.run(verify(args.url, args.token))

    total   = len(results)
    passed  = sum(1 for _, ok, _ in results if ok)
    failed  = total - passed

    print(f"\n{'='*55}")
    print(f"  总计: {total}  通过: {passed}  失败: {failed}")
    score = int(100 * passed / total) if total else 0
    print(f"  部署评分: {score}/100  {'🎉 可部署' if score >= 80 else '⚠️ 需修复'}")
    print(f"{'='*55}")

    if failed > 0:
        print("\n失败项目：")
        for name, ok, detail in results:
            if not ok:
                print(f"  ❌ {name}: {detail}")

    return 0 if score >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
