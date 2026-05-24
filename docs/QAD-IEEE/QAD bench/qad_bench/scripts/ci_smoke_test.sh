#!/bin/bash
# QAD-Bench CI smoke test — runs in <30 s on any machine.
set -e

cd "$(dirname "$0")/.."

echo "═══ Step 1/4: Install package ═══════════════════════════════════════"
python3 -m pip install --quiet -e .

echo
echo "═══ Step 2/4: Run unit tests ════════════════════════════════════════"
python3 -m unittest tests.test_qad_bench -v

echo
echo "═══ Step 3/4: Run quickstart example ════════════════════════════════"
python3 examples/quickstart.py

echo
echo "═══ Step 4/4: Run CLI smoke test ════════════════════════════════════"
python3 -m qad_bench.runner \
    --model_path auto \
    --prefer_offline \
    --synthetic_n_per_class 20 \
    --no_progress \
    --output_dir /tmp/qad_bench_smoke \
    --log_level WARNING

echo
echo "═══ ALL TESTS PASSED ════════════════════════════════════════════════"
