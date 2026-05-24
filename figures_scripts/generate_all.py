"""
generate_all.py — Master driver to regenerate all 10 figures.

Usage:
    pip install matplotlib numpy
    python generate_all.py

Outputs:
    output/fig01_architecture.png
    output/fig02_main_results.png
    ...
    output/fig10_acoustic_embedding.png
"""
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    ("fig01_architecture.py",         "Figure 1 — Three-tier edge-cloud architecture"),
    ("fig02_main_results.py",         "Figure 2 — Main results vs 11 baselines (Table II)"),
    ("fig03_ablation_loss_teacher.py","Figure 3 — Loss & teacher ablation (Tables IV-V)"),
    ("fig04_ovf_ablation.py",         "Figure 4 — OV-Freeze layer & step-ratio (Tables VI-VII)"),
    ("fig05_speculative.py",          "Figure 5 — Speculative decoding (Table IX)"),
    ("fig06_fusion_analysis.py",      "Figure 6 — Multimodal fusion (Tables X-XII)"),
    ("fig07_privacy_glo.py",          "Figure 7 — White-box + black-box GLO attack (Table XIII)"),
    ("fig08_deployment.py",           "Figure 8 — 30-day IRB deployment results"),
    ("fig09_qad_pipeline.py",         "Figure 9 — QAD training pipeline & schedule"),
    ("fig10_acoustic_embedding.py",   "Figure 10 — 128-d non-invertible F_v construction"),
]


def main():
    here = Path(__file__).parent.resolve()
    out_dir = here / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print(f"  QAD-MultiGuard figure generation — {len(SCRIPTS)} scripts")
    print(f"  Output directory: {out_dir}")
    print("=" * 72)

    t0 = time.time()
    ok, fail = [], []
    for script, desc in SCRIPTS:
        path = here / script
        if not path.exists():
            print(f"  ✗ MISSING: {script}")
            fail.append(script)
            continue
        print(f"\n→ {script}")
        print(f"  {desc}")
        try:
            # Run script in its own directory so it writes to output/
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=str(here),
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                # Move output to ./output/ if it landed elsewhere
                png_name = script.replace(".py", ".png")
                src = here / png_name
                dst = out_dir / png_name
                if src.exists() and src != dst:
                    src.rename(dst)
                print(f"  ✓ {png_name}")
                ok.append(script)
            else:
                print(f"  ✗ FAILED (exit {result.returncode})")
                print(f"    stderr: {result.stderr[:200]}")
                fail.append(script)
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (> 120s)")
            fail.append(script)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            fail.append(script)

    print("\n" + "=" * 72)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Success: {len(ok)}/{len(SCRIPTS)}")
    if fail:
        print(f"  Failed:  {fail}")
        sys.exit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
