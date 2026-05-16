"""
services/sms_service.py - 短信发送服务
支持阿里云、腾讯云，可扩展其他供应商
"""

import secrets
import random
import logging
import httpx
import hmac
import hashlib
import time
import json
from abc import ABC, abstractmethod
from core.config import settings

logger = logging.getLogger(__name__)


def generate_code(length: int = 6) -> str:
    """生成 N 位数字验证码"""
    return str(secrets.randbelow(9 * 10**(length-1)) + 10**(length-1))


# ── 抽象基类 ─────────────────────────────────────────────
class BaseSmsProvider(ABC):
    @abstractmethod
    async def send(self, phone: str, code: str) -> bool:
        """发送验证码，返回是否成功"""
        ...


# ── 阿里云短信 ───────────────────────────────────────────
class AliyunSmsProvider(BaseSmsProvider):
    """
    阿里云短信服务 (SMS)
    文档: https://help.aliyun.com/document_detail/101414.html
    """

    API_URL = "https://dysmsapi.aliyuncs.com"

    def __init__(self):
        self.access_key = settings.SMS_ACCESS_KEY
        self.secret_key = settings.SMS_SECRET_KEY
        self.sign_name  = settings.SMS_SIGN_NAME
        self.template   = settings.SMS_TEMPLATE_CODE

    def _sign(self, params: dict) -> str:
        import urllib.parse, base64, hmac, hashlib
        sorted_params = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted(params.items()))
        str_to_sign = f"GET&%2F&{urllib.parse.quote(sorted_params, safe='')}"
        key = f"{self.secret_key}&"
        sig = hmac.new(key.encode(), str_to_sign.encode(), hashlib.sha1).digest()
        return base64.b64encode(sig).decode()

    async def send(self, phone: str, code: str) -> bool:
        if not self.access_key:
            logger.warning("[DEV] 阿里云未配置，跳过发送。验证码: %s -> %s", phone, code)
            return True  # 开发环境跳过

        params = {
            "Action": "SendSms",
            "Version": "2017-05-25",
            "RegionId": "cn-hangzhou",
            "PhoneNumbers": phone,
            "SignName": self.sign_name,
            "TemplateCode": self.template,
            "TemplateParam": json.dumps({"code": code}),
            "AccessKeyId": self.access_key,
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": str(secrets.randbelow(90000) + 10000),
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "Format": "JSON",
        }
        params["Signature"] = self._sign(params)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(self.API_URL, params=params)
                data = resp.json()
                ok = data.get("Code") == "OK"
                if not ok:
                    logger.error("阿里云短信发送失败: %s", data)
                return ok
        except Exception as e:
            logger.error("阿里云短信请求异常: %s", e)
            return False


# ── 腾讯云短信 ───────────────────────────────────────────
class TencentSmsProvider(BaseSmsProvider):
    """
    腾讯云短信服务
    文档: https://cloud.tencent.com/document/product/382/55981
    """

    API_URL = "https://sms.tencentcloudapi.com"

    def __init__(self):
        self.secret_id  = settings.SMS_ACCESS_KEY
        self.secret_key = settings.SMS_SECRET_KEY
        self.sdk_app_id = getattr(settings, "SMS_SDK_APP_ID", "")
        self.sign_name  = settings.SMS_SIGN_NAME
        self.template   = settings.SMS_TEMPLATE_CODE

    async def send(self, phone: str, code: str) -> bool:
        if not self.secret_id:
            logger.warning("[DEV] 腾讯云未配置，跳过发送。验证码: %s -> %s", phone, code)
            return True

        payload = {
            "PhoneNumberSet": [f"+86{phone}"],
            "SmsSdkAppId": self.sdk_app_id,
            "SignName": self.sign_name,
            "TemplateId": self.template,
            "TemplateParamSet": [code],
        }

        # 腾讯云签名 v3（TC3-HMAC-SHA256）
        timestamp = int(time.time())
        headers = {
            "Content-Type": "application/json",
            "Host": "sms.tencentcloudapi.com",
            "X-TC-Action": "SendSms",
            "X-TC-Version": "2021-01-11",
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Region": "ap-guangzhou",
        }
        # 完整签名逻辑需参考腾讯云 SDK，此处为结构示意
        headers["Authorization"] = f"TC3-HMAC-SHA256 Credential={self.secret_id}/..."

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.API_URL, json=payload, headers=headers)
                data = resp.json()
                errors = data.get("Response", {}).get("SendStatusSet", [{}])
                ok = all(s.get("Code") == "Ok" for s in errors)
                if not ok:
                    logger.error("腾讯云短信发送失败: %s", data)
                return ok
        except Exception as e:
            logger.error("腾讯云短信请求异常: %s", e)
            return False


# ── 工厂函数 ─────────────────────────────────────────────
def get_sms_provider() -> BaseSmsProvider:
    """根据配置返回对应的短信供应商"""
    provider_map = {
        "aliyun":  AliyunSmsProvider,
        "tencent": TencentSmsProvider,
    }
    cls = provider_map.get(settings.SMS_PROVIDER, AliyunSmsProvider)
    return cls()


# 全局单例
sms_provider: BaseSmsProvider = get_sms_provider()
