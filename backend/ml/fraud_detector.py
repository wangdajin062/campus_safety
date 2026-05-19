"""
ml/fraud_detector.py  (v3)
集成检测器 — 规则引擎 + 梯度提升 + 集成融合
与 v2 完全向后兼容，供 MultimodalDetector 调用
"""
import hmac, math, json, logging, pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PhoneFeatures:
    report_count:     int   = 0
    query_count:      int   = 0
    confirmed_count:  int   = 0
    days_since_first: float = 0.0
    source:           str   = "user_report"
    location_code:    int   = 0
    carrier_code:     int   = 0

@dataclass
class SmsFeatures:
    keyword_hits:    int   = 0
    keyword_weight:  float = 0.0
    urgency_score:   float = 0.0
    has_url:         bool  = False
    url_count:       int   = 0
    money_mentioned: bool  = False
    impersonation:   bool  = False
    char_count:      int   = 0
    digit_ratio:     float = 0.0
    sender_is_number:bool  = False

@dataclass
class DetectionResult:
    risk_level:     str   = "safe"
    risk_score:     int   = 0
    ml_probability: float = 0.0
    ml_risk_level:  str   = "safe"
    features_used:  list  = field(default_factory=list)
    rule_triggered: Optional[str] = None


class RuleEngine:
    HARD_HIGH_RULES = [
        ("安全账户",     "转账至安全账户"),
        ("公安冻结",     "公安冻结资产"),
        ("配合调查转账", "配合调查要求转账"),
        ("涉案资金",     "涉及刑事案件资金"),
    ]
    KEYWORD_WEIGHT_MAP = {
        "安全账户":100,"立即转账":90,"公安局":90,
        "涉案资金":92, "资产冻结":88,"配合调查":85,
        "刷单":85,     "刷好评":80,  "解冻":80,
        "验证码":70,   "贷款":60,    "兼职":45,
        "助学贷款":70, "内部名额":65,"恭喜中奖":80,
        "账户异常":75, "点击链接":70,
    }

    def check_sms(self, keywords, has_url, urgency_score, money_mentioned, impersonation):
        for kw, reason in self.HARD_HIGH_RULES:
            if kw in keywords:
                return "high", 100, reason
        score = sum(self.KEYWORD_WEIGHT_MAP.get(kw, 10) for kw in keywords)
        if has_url:        score += 40
        if money_mentioned: score += 20
        if impersonation:   score += 25
        score += int(urgency_score * 30)
        score = min(100, score)
        level = "high" if score >= 70 else "medium" if score >= 35 else "safe"
        return level, score, None

    def check_phone(self, features: PhoneFeatures):
        if features.report_count == 0 and features.confirmed_count == 0:
            return "safe", 0
        score = min(60, int(20 * math.log1p(features.report_count)))
        score += min(30, features.confirmed_count * 10)
        if features.source == "police":   score += 20
        elif features.source == "system": score += 10
        score = min(100, score)
        return ("high" if score >= 70 else "medium" if score >= 35 else "safe"), score


class GradientBoostingDetector:
    _BASE_DIR   = Path(__file__).resolve().parent
    MODEL_PATH  = _BASE_DIR / "models" / "fraud_detector.pkl"
    FEATURE_DIM = 12

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        if self.MODEL_PATH.exists():
            try:
                with open(self.MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("ML model loaded")
            except Exception as e:
                logger.warning("Model load failed: %s", e)

    def vectorize_sms(self, feat: SmsFeatures) -> np.ndarray:
        return np.array([
            min(feat.keyword_hits/5.0, 1.0),
            min(feat.keyword_weight/100.0, 1.0),
            feat.urgency_score,
            1.0 if feat.has_url else 0.0,
            min(feat.url_count/3.0, 1.0),
            1.0 if feat.money_mentioned else 0.0,
            1.0 if feat.impersonation else 0.0,
            min(feat.char_count/300.0, 1.0),
            feat.digit_ratio,
            1.0 if feat.sender_is_number else 0.0,
            float(feat.keyword_hits > 0 and feat.has_url),
            float(feat.impersonation and feat.money_mentioned),
        ], dtype=np.float32)

    def vectorize_phone(self, feat: PhoneFeatures) -> np.ndarray:
        se = {"user_report":0.3,"system":0.6,"police":1.0,"manual":0.8}
        return np.array([
            min(feat.report_count/50.0, 1.0),
            min(feat.confirmed_count/20.0, 1.0),
            min(feat.query_count/100.0, 1.0),
            min(feat.days_since_first/365.0, 1.0),
            se.get(feat.source, 0.3),
            float(feat.location_code == 1),
            float(feat.carrier_code == 0),
            float(feat.report_count >= 10),
            float(feat.confirmed_count >= 3),
            float(feat.report_count > 0 and feat.confirmed_count == 0),
            math.log1p(feat.report_count)/6.0,
            float(feat.source == "police"),
        ], dtype=np.float32)

    def predict(self, vec: np.ndarray) -> float:
        if self._model is not None:
            try:
                return float(self._model.predict_proba(vec.reshape(1,-1))[0][1])
            except Exception: pass
        if float(np.sum(np.abs(vec))) < 0.001:
            return 0.05
        w = np.array([0.35,0.25,0.20,0.30,0.15,0.20,0.40,0.05,0.10,0.25,0.45,0.55])
        logit = float(np.dot(vec, w) - 0.3)
        return 1.0 / (1.0 + math.exp(-logit * 3.5))


class EnsembleDetector:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.gb_model    = GradientBoostingDetector()

    def detect_sms(self, keywords, keyword_weight, has_url, url_count,
                   urgency_score, money_mentioned, impersonation,
                   char_count, sender) -> DetectionResult:
        feat = SmsFeatures(
            keyword_hits=len(keywords),
            keyword_weight=keyword_weight,
            urgency_score=urgency_score,
            has_url=has_url,
            url_count=url_count,
            money_mentioned=money_mentioned,
            impersonation=impersonation,
            char_count=char_count,
            digit_ratio=sum(c.isdigit() for c in sender)/max(len(sender),1),
            sender_is_number=sender.lstrip("+").isdigit() if sender else False,
        )
        rule_level, rule_score, rule_triggered = self.rule_engine.check_sms(
            keywords, has_url, urgency_score, money_mentioned, impersonation)
        vec     = self.gb_model.vectorize_sms(feat)
        ml_prob = self.gb_model.predict(vec)
        ml_score= int(ml_prob * 100)
        ml_level= "high" if ml_prob>=0.70 else "medium" if ml_prob>=0.35 else "safe"
        ORDER   = {"safe":0,"medium":1,"high":2}
        final_score = max(rule_score, ml_score)
        final_level = max(rule_level, ml_level, key=lambda l: ORDER[l])
        return DetectionResult(
            risk_level=final_level, risk_score=final_score,
            ml_probability=ml_prob, ml_risk_level=ml_level,
            features_used=keywords[:5], rule_triggered=rule_triggered)

    def detect_phone(self, features: PhoneFeatures) -> DetectionResult:
        if features.report_count == 0 and features.confirmed_count == 0:
            return DetectionResult(risk_level="safe", risk_score=0, ml_probability=0.0)
        rule_level, rule_score = self.rule_engine.check_phone(features)
        vec     = self.gb_model.vectorize_phone(features)
        ml_prob = self.gb_model.predict(vec)
        ml_score= int(ml_prob * 100)
        ml_level= "high" if ml_prob>=0.70 else "medium" if ml_prob>=0.35 else "safe"
        ORDER   = {"safe":0,"medium":1,"high":2}
        return DetectionResult(
            risk_level=max(rule_level, ml_level, key=lambda l: ORDER[l]),
            risk_score=max(rule_score, ml_score),
            ml_probability=ml_prob, ml_risk_level=ml_level)

    def record_feedback(self, sample_hash: str, true_label: int, features: np.ndarray):
        fb_path = self.gb_model.MODEL_PATH.parent / "feedback.jsonl"
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fb_path, "a") as f:
            f.write(json.dumps({"hash":sample_hash,"label":true_label,
                                "features":features.tolist()}) + "\n")

    def retrain_from_feedback(self):
        fb_path = self.gb_model.MODEL_PATH.parent / "feedback.jsonl"
        if not fb_path.exists(): return
        X, y = [], []
        with open(fb_path) as f:
            for line in f:
                rec = json.loads(line)
                X.append(rec["features"]); y.append(rec["label"])
        if len(X) >= 50:
            self.gb_model.train(np.array(X), np.array(y))

    def train(self, X, y):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            m = Pipeline([("scaler",StandardScaler()),
                          ("clf",GradientBoostingClassifier(n_estimators=200,max_depth=4,
                               learning_rate=0.08,subsample=0.8,random_state=42))])
            m.fit(X, y)
            self.gb_model._model = m
            self.gb_model.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.gb_model.MODEL_PATH, "wb") as f:
                pickle.dump(m, f)
            logger.info("Model retrained on %d samples", len(y))
        except ImportError:
            logger.warning("sklearn not available, skipping retrain")


    # ── 评估 ──────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        在测试集上评估 GBM 模型。

        返回:
            accuracy, precision, recall, f1, roc_auc, 以及按风险等级的细分指标。
        """
        result = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1_score": 0.0, "roc_auc": 0.0,
            "samples": int(len(y_test)), "positives": int(y_test.sum()),
        }
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, classification_report,
            )

            if self.gb_model._model is None:
                logger.warning("GBM 模型未加载，使用回退权重评估")
                preds = []
                for i in range(len(X_test)):
                    p = self.gb_model.predict(X_test[i])
                    preds.append(1 if p >= 0.5 else 0)
                y_pred = np.array(preds)
                y_prob = np.array([self.gb_model.predict(X_test[i])
                                    for i in range(len(X_test))])
            else:
                y_pred = self.gb_model._model.predict(X_test)
                y_prob = self.gb_model._model.predict_proba(X_test)[:, 1]

            result["accuracy"]  = round(float(accuracy_score(y_test, y_pred)), 4)
            result["precision"] = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
            result["recall"]    = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
            result["f1_score"]  = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            result["roc_auc"]   = round(float(roc_auc_score(y_test, y_prob)), 4)

            # 等级细分（风险评分映射）
            threshold_high  = np.percentile(y_prob, 70) if len(y_prob) > 0 else 0.7
            threshold_med   = np.percentile(y_prob, 35) if len(y_prob) > 0 else 0.35
            levels = []
            for p, t in zip(y_prob, y_test):
                if p >= threshold_high:
                    levels.append(("high", t))
                elif p >= threshold_med:
                    levels.append(("medium", t))
                else:
                    levels.append(("safe", t))

            for level_name in ("high", "medium", "safe"):
                group = [(p, t) for p, t in zip(y_prob, y_test)
                         if (p >= threshold_high and level_name == "high")
                         or (threshold_med <= p < threshold_high and level_name == "medium")
                         or (p < threshold_med and level_name == "safe")]
                if group:
                    _, gt = zip(*group)
                    result[f"{level_name}_count"] = len(group)
                    result[f"{level_name}_fraud_ratio"] = round(sum(gt) / len(gt), 3)

            result["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            logger.info(
                "评估完成: acc=%.4f, prec=%.4f, rec=%.4f, f1=%.4f, auc=%.4f (n=%d)",
                result["accuracy"], result["precision"], result["recall"],
                result["f1_score"], result["roc_auc"], len(y_test),
            )

        except ImportError as e:
            logger.warning("评估指标不可用: %s", e)
            result["error"] = str(e)
        except Exception as e:
            logger.error("评估失败: %s", e)
            result["error"] = str(e)

        return result

    # ── 从 TeleAntiFraud 数据训练 ─────────────────────────────
    def train_from_teleantifraud(
        self, max_samples: int = 4000
    ) -> dict:
        """
        使用 TeleAntiFraud 真实数据训练 GBM 模型。

        步骤:
          1. 通过 TeleAntiFraudLoader 加载数据
          2. 用训练集调用 self.train()
          3. 在测试集上调用 self.evaluate()
          4. 返回训练&评估结果

        返回:
            {"status", "train_samples", "evaluation": {...}}
        """
        try:
            from ml.data_loader import TeleAntiFraudLoader

            loader = TeleAntiFraudLoader()
            data = loader.load_train_test(max_samples=max_samples)

            if data is None:
                return {"status": "error", "message": "TeleAntiFraud 数据不可用"}

            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            logger.info(
                "开始 TeleAntiFraud 训练: 训练集=%d, 测试集=%d, 特征维=%d",
                len(X_train), len(X_test), X_train.shape[1],
            )

            self.train(X_train, y_train)

            eval_result = self.evaluate(X_test, y_test)

            result = {
                "status": "success",
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "feature_dim": int(X_train.shape[1]),
                "evaluation": eval_result,
            }
            logger.info("TeleAntiFraud 训练完成: %s", result["status"])
            return result

        except Exception as e:
            logger.exception("TeleAntiFraud 训练失败")
            return {"status": "error", "message": str(e)}


detector = EnsembleDetector()
