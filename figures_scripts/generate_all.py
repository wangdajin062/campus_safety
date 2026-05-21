"""
generate_all.py — Generate all 10 figures for QAD-MultiGuard paper.

Usage:
    python generate_all.py               # Generate all figures
    python generate_all.py --fig 5       # Generate only fig 5

Output directory: ./output/
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGURES = [
    ("fig01_architecture",          "Three-tier edge-cloud architecture"),
    ("fig02_main_results",          "TAF-28k main results (15 baselines)"),
    ("fig03_ablation_loss_teacher", "Loss + teacher ablation"),
    ("fig04_ovf_ablation",          "OV-Freeze ablation"),
    ("fig05_speculative_decoding",  "Speculative decoding analysis"),
    ("fig06_fusion_analysis",       "Multimodal fusion analysis"),
    ("fig07_privacy_glo",           "Privacy GLO attack verification"),
    ("fig08_deployment",            "30-day IRB deployment results"),
    ("fig09_qad_pipeline",          "QAD training pipeline"),
    ("fig10_acoustic_embedding",    "128-d non-invertible acoustic F_v"),
]


def run_figure(name):
    """Run a figure script as a top-level program (so __name__='__main__' works)."""
    script_path = SCRIPT_DIR / f"{name}.py"
    if not script_path.exists():
        print(f"  [SKIP] {name}.py not found")
        return False
    try:
        original_cwd = os.getcwd()
        os.chdir(SCRIPT_DIR)
        t0 = time.time()
        # Use runpy to execute as __main__
        import runpy
        runpy.run_path(str(script_path), run_name="__main__")
        elapsed = time.time() - t0
        os.chdir(original_cwd)
        print(f"  [OK]   {name}  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        os.chdir(original_cwd)
        print(f"  [FAIL] {name}: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate single figure (1-10)")
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures to render: ", end="")

    if args.fig is not None:
        if 1 <= args.fig <= 10:
            target = FIGURES[args.fig - 1]
            print(f"fig{args.fig:02d}\n")
            run_figure(target[0])
        else:
            print(f"\nError: --fig must be 1-10")
            sys.exit(1)
    else:
        print(f"all 10\n")
        ok = 0
        for name, desc in FIGURES:
            if run_figure(name):
                ok += 1
        print(f"\n{ok}/{len(FIGURES)} figures generated successfully.")


if __name__ == "__main__":
    main()
