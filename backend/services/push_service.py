"""
services/push_service.py - Firebase Cloud Messaging (FCM) 推送服务
支持 iOS (APNs via FCM) 和 Android
"""

import logging
import httpx
import json
from typing import Optional
from core.config import settings

logger = logging.getLogger(__name__)


class PushPayload:
    """推送消息结构"""
    def __init__(
        self,
        title: str,
        body: str,
        data: Optional[dict] = None,
        badge: int = 1,
        sound: str = "default",
    ):
        self.title = title
        self.body = body
        self.data = data or {}
        self.badge = badge
        self.sound = sound


class FCMPushService:
    """
    Firebase Cloud Messaging 推送服务
    支持单设备推送 & 批量推送（最多 500 个 token/次）
    文档: https://firebase.google.com/docs/cloud-messaging/http-server-ref
    """

    FCM_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    LEGACY_URL = "https://fcm.googleapis.com/fcm/send"

    def __init__(self):
        self.server_key = settings.FCM_SERVER_KEY

    def _build_message(self, token: str, payload: PushPayload) -> dict:
        return {
            "to": token,
            "notification": {
                "title": payload.title,
                "body": payload.body,
                "sound": payload.sound,
                "badge": payload.badge,
            },
            "data": {
                **payload.data,
                "click_action": "FLUTTER_NOTIFICATION_CLICK",
            },
            "android": {
                "priority": "high",
                "notification": {
                    "channel_id": "campus_safety_alerts",
                    "sound": payload.sound,
                    "default_vibrate_timings": True,
                },
            },
            "apns": {
                "headers": {"apns-priority": "10"},
                "payload": {
                    "aps": {
                        "alert": {"title": payload.title, "body": payload.body},
                        "badge": payload.badge,
                        "sound": payload.sound,
                    }
                },
            },
        }

    async def send_to_token(self, token: str, payload: PushPayload) -> bool:
        """向单个设备发送推送"""
        if not self.server_key:
            logger.warning("[DEV] FCM 未配置，跳过推送: %s", payload.title)
            return True

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self.LEGACY_URL,
                    json=self._build_message(token, payload),
                    headers={
                        "Authorization": f"key={self.server_key}",
                        "Content-Type": "application/json",
                    },
                )
                data = resp.json()
                success = data.get("success", 0) == 1
                if not success:
                    logger.warning("FCM 推送失败: %s", data.get("results", [{}])[0])
                return success
        except Exception as e:
            logger.error("FCM 推送异常: %s", e)
            return False

    async def send_to_tokens(self, tokens: list[str], payload: PushPayload) -> dict:
        """
        批量推送（最多 500 个 token/次）
        返回: {"success": N, "failure": M, "invalid_tokens": [...]}
        """
        if not self.server_key:
            logger.warning("[DEV] FCM 批量推送跳过，tokens=%d", len(tokens))
            return {"success": len(tokens), "failure": 0, "invalid_tokens": []}

        # FCM 限制每次最多 500 个
        CHUNK = 500
        total_success = 0
        total_failure = 0
        invalid_tokens = []

        for i in range(0, len(tokens), CHUNK):
            chunk = tokens[i:i + CHUNK]
            message = {
                "registration_ids": chunk,
                "notification": {
                    "title": payload.title,
                    "body": payload.body,
                    "sound": payload.sound,
                },
                "data": payload.data,
                "android": {"priority": "high"},
            }

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        self.LEGACY_URL,
                        json=message,
                        headers={
                            "Authorization": f"key={self.server_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    data = resp.json()
                    total_success += data.get("success", 0)
                    total_failure += data.get("failure", 0)

                    # 收集失效 token
                    for idx, result in enumerate(data.get("results", [])):
                        if result.get("error") in ("InvalidRegistration", "NotRegistered"):
                            invalid_tokens.append(chunk[idx])

            except Exception as e:
                logger.error("FCM 批量推送异常（chunk %d）: %s", i // CHUNK, e)
                total_failure += len(chunk)

        logger.info(
            "FCM 批量推送完成: success=%d failure=%d invalid=%d",
            total_success, total_failure, len(invalid_tokens),
        )
        return {
            "success": total_success,
            "failure": total_failure,
            "invalid_tokens": invalid_tokens,
        }

    async def send_fraud_alert(self, tokens: list[str], alert_title: str, alert_id: int, is_urgent: bool = False) -> dict:
        """专用：发送电诈预警推送"""
        payload = PushPayload(
            title="🚨 电诈预警" if is_urgent else "📢 防骗提示",
            body=alert_title,
            data={"type": "fraud_alert", "alert_id": str(alert_id), "is_urgent": str(is_urgent)},
            sound="alert.wav" if is_urgent else "default",
        )
        return await self.send_to_tokens(tokens, payload)

    async def send_call_warning(self, token: str, phone_number: str, risk_type: str) -> bool:
        """专用：来电高危警告推送"""
        payload = PushPayload(
            title="🚨 高危来电警告",
            body=f"检测到疑似诈骗电话：{phone_number}，风险类型：{risk_type}",
            data={"type": "call_alert", "phone": phone_number},
            sound="alert.wav",
        )
        return await self.send_to_token(token, payload)


# 全局单例
push_service = FCMPushService()
