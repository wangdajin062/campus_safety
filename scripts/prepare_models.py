"""
scripts/prepare_models.py — QAD-MultiGuard 模型准备
===================================================
下载/准备部署所需的模型文件：
  1. GGUF 草稿模型 → backend/ml/models/fraud_draft_q4km.gguf
  2. GBM PKL       → backend/ml/models/fraud_detector.pkl

用法:
  python scripts/prepare_models.py             # 完整运行
  python scripts/prepare_models.py --check     # 仅检查状态
  python scripts/prepare_models.py --gguf-only # 仅下载 GGUF
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prepare_models")

# ── 路径 ──────────────────────────────────────────────────
BACKEND_ML_DIR = Path(__file__).resolve().parent.parent / "backend" / "ml"
MODELS_DIR = BACKEND_ML_DIR / "models"

GGUF_TARGET = MODELS_DIR / "fraud_draft_q4km.gguf"
PKL_TARGET  = MODELS_DIR / "fraud_detector.pkl"

# 备选 GBM PKL 位置（项目根 ml/models/）
PROJECT_ROOT_PKL = Path(__file__).resolve().parent.parent / "ml" / "models" / "fraud_detector.pkl"

# HuggingFace GGUF 源
GGUF_SOURCES = [
    {
        "repo": "unsloth/Qwen2.5-0.5B-Instruct-gguf",
        "file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "description": "Qwen2.5-0.5B Q4_K_M via unsloth",
    },
    {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "description": "Qwen2.5-0.5B Q4_K_M official",
    },
]


def check_status() -> dict:
    """检查各模型文件状态。"""
    return {
        "gguf_draft": {
            "exists": GGUF_TARGET.exists(),
            "size_mb": round(GGUF_TARGET.stat().st_size / 1024 / 1024, 1) if GGUF_TARGET.exists() else 0,
            "expected_min_mb": 100,
            "path": str(GGUF_TARGET),
        },
        "gbm_pkl": {
            "exists": PKL_TARGET.exists(),
            "path": str(PKL_TARGET),
        },
        "gbm_pkl_alt": {
            "exists": PROJECT_ROOT_PKL.exists(),
            "path": str(PROJECT_ROOT_PKL),
        },
        "models_dir": {
            "exists": MODELS_DIR.exists(),
            "path": str(MODELS_DIR),
        },
    }


def ensure_models_dir():
    """确保模型目录存在。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Models dir: %s", MODELS_DIR)


def download_gguf() -> bool:
    """从 HuggingFace 下载 GGUF 草稿模型。"""
    if GGUF_TARGET.exists() and GGUF_TARGET.stat().st_size > 100 * 1024 * 1024:
        logger.info("GGUF already exists: %s (%d MB), skipping",
                    GGUF_TARGET, GGUF_TARGET.stat().st_size / 1024 / 1024)
        return True

    ensure_models_dir()

    for source in GGUF_SOURCES:
        try:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading %s from %s...", source["file"], source["repo"])
            downloaded = hf_hub_download(
                repo_id=source["repo"],
                filename=source["file"],
                resume_download=True,
            )
            src = Path(downloaded)
            if src.exists() and src.stat().st_size > 100 * 1024 * 1024:
                shutil.copy2(src, GGUF_TARGET)
                logger.info("Copied to %s (%d MB)", GGUF_TARGET,
                            GGUF_TARGET.stat().st_size / 1024 / 1024)
                return True
            else:
                logger.warning("Downloaded file too small or missing: %s", src)
        except Exception as e:
            logger.warning("Failed to download from %s: %s", source["repo"], e)

    logger.error(
        "Could not download GGUF model from any source.\n"
        "  Place a Q4_K_M GGUF manually at: %s\n"
        "  Download from: https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-gguf",
        GGUF_TARGET,
    )
    return False


def copy_gbm_pkl() -> bool:
    """从项目根复制 GBM PKL（若存在）。"""
    if PKL_TARGET.exists():
        logger.info("GBM PKL already exists: %s", PKL_TARGET)
        return True

    if PROJECT_ROOT_PKL.exists():
        ensure_models_dir()
        shutil.copy2(PROJECT_ROOT_PKL, PKL_TARGET)
        logger.info("Copied GBM PKL from %s to %s", PROJECT_ROOT_PKL, PKL_TARGET)
        return True

    logger.info("GBM PKL not found at %s — will be auto-trained on first run", PROJECT_ROOT_PKL)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare QAD-MultiGuard model files"
    )
    parser.add_argument("--check", action="store_true",
                        help="仅检查模型状态，不下载")
    parser.add_argument("--gguf-only", action="store_true",
                        help="仅下载 GGUF 模型")
    parser.add_argument("--pkl-only", action="store_true",
                        help="仅复制 GBM PKL")
    args = parser.parse_args()

    print("=" * 60)
    print("QAD-MultiGuard Model Preparation")
    print("=" * 60)

    if args.check:
        status = check_status()
        print(f"\nModels directory: {status['models_dir']['path']}")
        print(f"  Exists: {status['models_dir']['exists']}")
        print()
        for key, info in status.items():
            if key == "models_dir":
                continue
            print(f"  {key}:")
            print(f"    Path:   {info['path']}")
            print(f"    Exists: {info['exists']}")
            if "size_mb" in info:
                print(f"    Size:   {info['size_mb']} MB (min {info['expected_min_mb']} MB)")
        return

    if args.pkl_only:
        copy_gbm_pkl()
        return

    if args.gguf_only:
        download_gguf()
        return

    # 完整流程
    ensure_models_dir()

    print("\n[1/2] GGUF Draft Model...")
    gguf_ok = download_gguf()

    print("\n[2/2] GBM PKL...")
    pkl_ok = copy_gbm_pkl()

    print("\n" + "=" * 60)
    if gguf_ok:
        logger.info("GGUF draft model: ready")
    else:
        logger.warning("GGUF draft model: NOT available (fallback to domain prior)")
    if pkl_ok:
        logger.info("GBM PKL: ready")
    else:
        logger.info("GBM PKL: will be auto-trained on first API call")
    print("=" * 60)


if __name__ == "__main__":
    main()
