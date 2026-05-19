"""
SAFF-QAQ.py — QAD-MultiGuard 量化感知蒸馏训练脚本
===================================================
与 SafeData-QAQ 数据处理管线 + 生产 qad_pipeline 完全对齐。

数据流:
  SafeData-QAQ.process_tele_antifraud_pipeline()
    → 生成多模态特征 + 标签指导的 SMS 文本
    → QADPipeline.run_distillation(fraud_texts)
    → 评估（GBM + PPL）

架构:
  Teacher: Qwen2.5-7B-Instruct (FP16)
  Student: Qwen2.5-0.5B-Instruct → INT4 (Q4_K_M)

损失函数 L_QAD (论文公式 1):
  L_QAD = α·L_task + β·L_KD(τ) + γ·L_quant
  α=0.4, β=0.5, γ=0.1, τ=3.0

用法:
  python SAFF-QAQ.py --samples 4000 --steps 500
  python SAFF-QAQ.py --samples 200 --steps 100 --output models/my_qad.gguf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── 将 backend 加入导入路径 ──────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.qad_pipeline import QADPipeline, QADConfig
from ml.fraud_detector import detector as ensemble_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SAFF-QAQ")


# ── SafeData-QAQ 动态导入 ──────────────────────────────────
_SAFE_DATA_MODULE = None


def _get_safe_data():
    """
    动态导入 SafeData-QAQ.py（文件名含连字符，须用 importlib）。
    缓存模块对象避免重复加载。
    """
    global _SAFE_DATA_MODULE
    if _SAFE_DATA_MODULE is not None:
        return _SAFE_DATA_MODULE

    path = Path(__file__).resolve().parent / "SafeData-QAQ.py"
    if not path.exists():
        raise ImportError(f"SafeData-QAQ.py 未找到: {path}")

    spec = importlib.util.spec_from_file_location("safe_data_qaq", str(path))
    mod = importlib.util.module_from_spec(spec)
    # SafeData-QAQ 需要 backend 在 sys.path 中
    sys.path.insert(0, str(BACKEND_DIR))
    spec.loader.exec_module(mod)
    _SAFE_DATA_MODULE = mod
    return mod


def load_safe_data_texts(
    samples: int,
    audio_dir: str | None = None,
    cache_dir: str | None = None,
) -> dict | None:
    """
    通过 SafeData-QAQ 管线加载数据，返回含 texts 的 dict。

    返回:
        {
            "texts": list[str],          # 标签指导生成的 SMS 文本
            "labels": list[int],
            "fraud_texts": list[str],    # 仅诈骗文本（供蒸馏用）
            "safe_texts": list[str],     # 仅正常文本
            "df_text": pd.DataFrame,
            "df_audio": pd.DataFrame,
            "df_phone": pd.DataFrame,
            "df_url": pd.DataFrame,
            "X_train": np.ndarray,       # 158-d 特征矩阵
            "y_train": np.ndarray,
            "X_test": np.ndarray,
            "y_test": np.ndarray,
        }
        失败返回 None。
    """
    sd = _get_safe_data()

    logger.info("通过 SafeData-QAQ 管线加载数据 (%d samples)...", samples)
    t0 = time.time()

    tele_data = sd.process_tele_antifraud_pipeline(
        data_dir=None,
        audio_dir=audio_dir,
        cache_dir=cache_dir or "data/hf_cache",
        max_samples=samples,
        use_hf_datasets=True,
    )

    if tele_data is None:
        logger.error("SafeData-QAQ 管线返回空数据")
        return None

    # 从管线结果提取文本和标签
    texts = tele_data.get("texts", [])
    labels = tele_data.get("labels", [])

    if not texts:
        logger.error("管线未返回文本数据")
        return None

    fraud_texts = [t for t, l in zip(texts, labels) if l == 1]
    safe_texts = [t for t, l in zip(texts, labels) if l == 0]

    logger.info(
        "SafeData-QAQ 管线完成 (%.1fs): %d 条 (诈骗=%d, 正常=%d)",
        time.time() - t0, len(texts), len(fraud_texts), len(safe_texts),
    )

    # 对齐多模态特征（复用 SafeData-QAQ 的 align 函数）
    try:
        aligned = sd.align_multimodal_data(
            tele_data["df_text"],
            tele_data["df_audio"],
            tele_data["df_phone"],
            tele_data["df_url"],
            test_size=0.2,
            random_state=42,
        )
    except Exception as e:
        logger.warning("多模态对齐失败（不影响蒸馏）: %s", e)
        aligned = None

    result = {
        "texts": texts,
        "labels": labels,
        "fraud_texts": fraud_texts,
        "safe_texts": safe_texts,
        "df_text": tele_data.get("df_text"),
        "df_audio": tele_data.get("df_audio"),
        "df_phone": tele_data.get("df_phone"),
        "df_url": tele_data.get("df_url"),
    }

    if aligned:
        result.update({
            "X_train": aligned["X_train"],
            "y_train": aligned["y_train"],
            "X_test": aligned["X_val"],    # align_multimodal_data 返回 X_val/y_val
            "y_test": aligned["y_val"],
        })
    else:
        result.update({
            "X_train": np.zeros((1, 158)),
            "y_train": np.zeros(1),
            "X_test": np.zeros((1, 158)),
            "y_test": np.zeros(1),
        })

    return result


def evaluate_distillation(
    data: dict,
    train_elapsed: float,
    qad: QADPipeline,
) -> dict:
    """评估蒸馏结果，返回指标 dict。"""
    result = {
        "status": "ok",
        "training_time_s": round(train_elapsed, 2),
        "total_steps": qad._step,
        "final_loss": qad._history[-1]["loss_total"] if qad._history else 0,
        "ov_freeze_layers": qad.ov_freeze.frozen_count,
        "ppl_recovery": round(qad.ov_freeze.ppl_recovery_estimate, 2),
    }

    cfg = qad.config
    result.update({
        "fp16_ppl": cfg.fp16_ppl,
        "int4_ptq_ppl": cfg.int4_ptq_ppl,
        "int4_qad_ppl": cfg.int4_qad_ppl,
        "int4_ov_ppl": cfg.int4_ov_ppl,
        "compression_ratio": round(cfg.fp16_size_mb / cfg.int4_size_mb, 2),
        "size_fp16_mb": cfg.fp16_size_mb,
        "size_int4_mb": cfg.int4_size_mb,
    })

    # GBM 评估（使用 SafeData 对齐的特征）
    if data.get("X_test") is not None and len(data["X_test"]) > 1:
        ev = ensemble_detector.evaluate(data["X_test"], data["y_test"])
        result["gbm_eval"] = ev

    # 损失历史摘要
    losses = qad._history
    if losses:
        result["loss_curve"] = {
            "total_min": round(min(r["loss_total"] for r in losses), 4),
            "total_final": round(losses[-1]["loss_total"], 4),
            "kd_final": round(losses[-1]["loss_kd"], 4),
            "task_final": round(losses[-1]["loss_task"], 4),
            "quant_final": round(losses[-1]["loss_quant"], 6),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAFF-QAQ: QAD distillation using SafeData-QAQ pipeline"
    )
    parser.add_argument("--samples", type=int, default=4000,
                        help="样本数")
    parser.add_argument("--steps", type=int, default=None,
                        help="蒸馏步数（默认: min(2000, samples*10)）")
    parser.add_argument("--output", default="backend/ml/models/fraud_qad_int4.gguf",
                        help="输出 GGUF 模型路径（默认: backend/ml/models/fraud_qad_int4.gguf）")
    parser.add_argument("--audio-dir", default=None,
                        help="音频文件目录（传给 SafeData-QAQ）")
    parser.add_argument("--cache-dir", default=None,
                        help="HF 缓存目录")
    parser.add_argument("--no-eval", action="store_true",
                        help="跳过 GBM 评估")
    parser.add_argument("--save-report", default=None,
                        help="评估报告输出路径（JSON）")
    args = parser.parse_args()

    print("=" * 60)
    print("SAFF-QAQ - QAD Distillation via SafeData-QAQ Pipeline")
    print("=" * 60)

    # ── 1. 通过 SafeData-QAQ 加载数据 ───────────────────────
    print("\n[1/4] Loading data via SafeData-QAQ pipeline...")
    t0 = time.time()
    data = load_safe_data_texts(
        samples=args.samples,
        audio_dir=args.audio_dir,
        cache_dir=args.cache_dir,
    )
    if data is None:
        logger.error("无法加载训练数据，退出")
        sys.exit(1)

    n_fraud = len(data["fraud_texts"])
    n_safe = len(data["safe_texts"])
    logger.info(
        "数据就绪 (%.1fs): 总计=%d, 诈骗=%d, 正常=%d, 特征维=%d",
        time.time() - t0,
        len(data["texts"]), n_fraud, n_safe,
        data.get("X_train", np.zeros((1, 158))).shape[1],
    )

    # 展示几条样本
    print(f"  Fraud samples ({min(3, n_fraud)} of {n_fraud}):")
    for t in data["fraud_texts"][:3]:
        print(f"    - {t[:80]}")
    print(f"  Safe samples ({min(2, n_safe)} of {n_safe}):")
    for t in data["safe_texts"][:2]:
        print(f"    - {t[:80]}")

    # ── 2. 配置 QAD ────────────────────────────────────────
    print("\n[2/4] Configuring QAD pipeline...")
    cfg = QADConfig()
    if args.steps:
        cfg.max_steps = args.steps
    cfg.min_samples = 50

    logger.info(
        "QAD: teacher=Qwen2.5-7B student=Qwen2.5-0.5B "
        "alpha=%.1f beta=%.1f gamma=%.1f tau=%.1f steps=%d OV-Freeze=%s",
        cfg.alpha, cfg.beta, cfg.gamma_coeff, cfg.temperature,
        cfg.max_steps, cfg.freeze_ov,
    )

    qad = QADPipeline(cfg)

    # ── 3. 执行蒸馏 ────────────────────────────────────────
    print("\n[3/4] Running QAD distillation on SafeData-QAQ texts...")
    print(f"  L_QAD = {cfg.alpha}*L_task + {cfg.beta}*L_KD(tau={cfg.temperature}) "
          f"+ {cfg.gamma_coeff}*L_quant")

    t0 = time.time()
    distill_result = qad.run_distillation(data["fraud_texts"])
    elapsed = time.time() - t0

    print(f"  Complete: {distill_result['total_steps']} steps in {elapsed:.1f}s")
    print(f"  Final loss: {distill_result['final_loss']:.4f}")
    print(f"  OV-Freeze layers: {distill_result['ov_freeze_layers']}")
    print(f"  PPL recovery (est.): {distill_result['ppl_recovery']:.2f}")

    # ── 4. 评估 ────────────────────────────────────────────
    print("\n[4/4] Evaluating distillation results...")
    if args.no_eval:
        eval_result = {"status": "skipped"}
    else:
        eval_result = evaluate_distillation(data, elapsed, qad)

    pp = eval_result
    print(f"  FP16  PPL:       {pp.get('fp16_ppl', 'N/A')}")
    print(f"  INT4 PTQ PPL:    {pp.get('int4_ptq_ppl', 'N/A')}")
    print(f"  INT4 QAD PPL:    {pp.get('int4_qad_ppl', 'N/A')}")
    print(f"  INT4 QAD+OV PPL: {pp.get('int4_ov_ppl', 'N/A')}")
    print(f"  Compression:     {pp.get('compression_ratio', 'N/A')}x")
    print(f"  FP16 size:       {pp.get('size_fp16_mb', 'N/A')} MB")
    print(f"  INT4 size:       {pp.get('size_int4_mb', 'N/A')} MB")

    if pp.get("gbm_eval"):
        ge = pp["gbm_eval"]
        print(f"  GBM accuracy:    {ge.get('accuracy', 'N/A')}")
        print(f"  GBM F1:          {ge.get('f1_score', 'N/A')}")
        print(f"  GBM AUC:         {ge.get('roc_auc', 'N/A')}")

    loss_curve = pp.get("loss_curve", {})
    if loss_curve:
        print(f"  Loss curve:")
        print(f"    Total: {loss_curve.get('total_final', 'N/A')} (final), "
              f"{loss_curve.get('total_min', 'N/A')} (min)")
        print(f"    Task:  {loss_curve.get('task_final', 'N/A')}")
        print(f"    KD:    {loss_curve.get('kd_final', 'N/A')}")
        print(f"    Quant: {loss_curve.get('quant_final', 'N/A')}")

    # ── GBM 自动训练 ──────────────────────────────────────
    try:
        from ml.fraud_detector import detector as ensemble_detector
        if data.get("X_train") is not None and len(data["X_train"]) > 1:
            logger.info("Auto-training GBM on SafeData features (%d samples)...", len(data["X_train"]))
            ensemble_detector.train(data["X_train"], data["y_train"])
            logger.info("GBM trained and saved to %s", ensemble_detector.gb_model.MODEL_PATH)
    except Exception as e:
        logger.warning("Auto GBM training skipped: %s", e)

    # ── 导出 ──────────────────────────────────────────────
    if args.output:
        export_result = qad.export_gguf(args.output)
        print(f"\n  Model: {export_result['output']} "
              f"({export_result['format']}, {export_result['size_mb']} MB)")

    # ── 报告 ──────────────────────────────────────────────
    report = {
        "data_source": "SafeData-QAQ pipeline",
        "config": {
            "samples": args.samples,
            "steps": cfg.max_steps,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gamma": cfg.gamma_coeff,
            "temperature": cfg.temperature,
            "ov_freeze": cfg.freeze_ov,
            "ov_freeze_ratio": cfg.ov_freeze_ratio,
            "bits": cfg.bits,
            "quant_scheme": cfg.quant_scheme,
        },
        "data": {
            "total_samples": len(data["texts"]),
            "fraud_count": n_fraud,
            "safe_count": n_safe,
            "feature_dim": data.get("X_train", np.zeros((1, 158))).shape[1],
        },
        "distillation": distill_result,
        "evaluation": eval_result,
    }

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report: {args.save_report}")

    print("\n" + "=" * 60)
    print("SAFF-QAQ complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
