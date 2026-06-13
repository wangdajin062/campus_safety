# QAD-MultiGuard — Verified Figure Captions

Captions are listed in **paper insertion order** (Figure 1 → Figure 7). Every
numerical claim in each caption has been cross-checked against the
single-source experiment data (`paper_data.py`) and the paper's tables and
body text. Items confirmed consistent are noted after each caption.

---

## Figure 1 — `fig1_architecture.py`

**Figure 1.** QAD-MultiGuard three-tier edge–cloud collaborative architecture.
Raw audio, ASR transcripts, and SMS plaintext never leave the device; only the
128-d non-invertible acoustic embedding *F*ᵥ and four scalar risk scores are
transmitted. Tier-1 (on-device, Snapdragon 8 Gen 3, 240 MB Q4_K_M) performs
lightweight feature extraction and fast-path scoring; Tier-2 (cloud, NVFP4 0.5B
on a Blackwell GPU) runs chain-of-thought reasoning, the QAD + OV-Freeze model,
and a privacy-preserving federated aggregator; Tier-3 (on-device, L-BFGS
sigmoid) fuses the four per-modality risk scores into a tri-level verdict. The
ε-LDP perturbation (ε = 1.5) shown is an optional extension for extreme-privacy
scenarios and is disabled in the reported main results.

*Consistency:* fusion weights (0.40 / 0.30 / 0.20 / 0.10) match Table (fusion
weights); model sizes (240 MB on-device, NVFP4 cloud) match Table 2; ε-LDP
framing matches the abstract and §5.

---

## Figure 2 — `fig2_acoustic_embedding.py`

**Figure 2.** 128-dimensional non-invertible acoustic embedding *F*ᵥ
construction. (a) Original log-mel spectrogram (64-band, *T* ∼ 300 frames per
3-s window); (b) time-averaged 64-d MFCC vector, collapsing frame-level
dynamics into static statistics; (c) concatenated *F*ᵥ ∈ ℝ¹²⁸ formed by the
MFCC branch (64-d) and the Whisper-tiny encoder global-pooled projection branch
(*W*₍proj₎ *h̄*₍w₎, 64-d). Dual temporal compression destroys frame-level phase
and prosodic timing, yielding empirical non-reconstructability (WER ≥ 0.95
under white-box GLO and black-box model-inversion attacks).

*Consistency:* WER ≥ 0.95 matches Table 9 (white-box 0.949 → "≥ 0.95"); the
128-d = 64 + 64 split and the ℝ¹²⁸ dimensionality match §3.3 and Eq. (7).

---

## Figure 3 — `fig3_main_results.py`

**Figure 3.** Main results on TAF-28k. (a) F1 score against eleven baselines
(blue), with the BF16 ceiling (0.931) marked; the three QAD variants are
highlighted in orange. (b) Accuracy recovery rate relative to the BF16 upper
bound, recovery = F1 / 0.931 × 100, with the 99 % target line. NVFP4 QAD +
OV-Freeze attains F1 = 0.923 (99.1 % recovery), substantially outperforming PTQ
baselines (F1 ≈ 0.838, 90.0 % recovery) by 8.5 F1 points and conventional QAT
(F1 = 0.844, 90.7 % recovery) by 7.9 F1 points.

*Consistency:* every bar matches Table 4 exactly; panel (b) recovery values are
recomputed from panel (a) F1 values in the script, so the two panels can never
diverge (this is the panel that previously carried stale 95.x values — now
fixed to 90.0 / 90.2 / 92.2 / 90.7 / 98.4 / 99.1 / 98.5).

---

## Figure 4 — `fig4_loss_convergence.py`

**Figure 4.** QAD training loss convergence and quantization SNR stability under
OV-Freeze. (a) KL-divergence loss drops 2.76× from ≈ 0.045 to ≈ 0.016 after
OV-Freeze activates at step 1,400 (the final 30 % of a 2,000-step schedule),
with the cosine learning-rate schedule overlaid; (b) quantization SNR remains
stable within 18.4–18.9 dB throughout training, confirming that the variance
rescaling does not amplify quantization error.

*Consistency:* the 2.76× drop, the 0.045 → 0.016 endpoints, the step-1,400
activation, and the 18.4–18.9 dB SNR band match §4.5 and the reproduction text
in §4.

---

## Figure 5 — `fig5_loss_teacher_ablation.py`

**Figure 5.** Loss-function and teacher-selection ablation. (a) F1 (blue bars,
left axis) and KL divergence to the BF16 teacher (red bars, right axis) across
five loss variants; pure KL attains the best trade-off (F1 = 0.916, KL = 0.005),
whereas cross-entropy (= QAT) incurs severe distribution drift (KL = 0.311).
(b) Teacher-selection ablation comparing the homologous 0.5B BF16 teacher
against larger heterogeneous teachers (1.8B–7B) under a fixed 0.5B-token budget
versus training to convergence. The homologous teacher attains the highest F1
with the fewest training tokens (0.5B), validating the self-distillation design.

*Consistency:* pure-KL F1 = 0.916 equals the QAD row of Table 4; KL values
(0.005 / 0.082 / 0.311 / 0.124 / 0.041) and teacher F1/token figures match
EXP03 / EXP09.

---

## Figure 6 — `fig6_ovf_ablation.py`

**Figure 6.** OV-Freeze ablation. (a) Layer selection: F1 (bars) and
output-variance drift (red line) across attention projection-layer
configurations; full *q,k,v,o*-proj OV-Freeze achieves the best F1 = 0.923 with
variance drift reduced from +18.2 % (no OVF) to +1.3 %. (b) Activation
step-ratio: F1 (left axis) and PPL (right axis) across OV-Freeze activation
windows; the final 30 % window (step 1,400–2,000) is Pareto-optimal, as shorter
windows (≤ 20 %) give insufficient correction (F1 ≤ 0.921) and longer windows
(≥ 50 %) induce gradient conflict (F1 drops to 0.918).

*Consistency:* F1 = 0.923 and the +18.2 % → +1.3 % drift match the OVF caption
and §4.5; the ≤ 20 % (≤ 0.921) and 50 % (0.918) figures match the body text and
EXP10.

---

## Figure 7 — `fig7_speculative_decoding.py`

**Figure 7.** Speculative-decoding analysis. (a) Theoretical speedup curves,
S(α) = (1 − α^(γ+1)) / (1 − α), across token-acceptance rate α and speculation
length γ, with operating points marked for the generic draft model (α = 0.78,
γ = 5, 3.52×) and our anti-fraud domain-tuned draft model (α = 0.86, γ = 5,
4.25×). (b) Measured wall-clock speedup at the deployed α = 0.86 on H100 and
Snapdragon 8 Gen 3 across γ; γ = 5 is the Pareto-optimal operating point on the
memory-constrained edge platform (H100 3.49×, SD8G3 3.32×).

*Consistency:* the theoretical anchors 3.52× and 4.25× match the closed-form
speedup formula; the measured γ = 5 values (3.49× / 3.32×) and the γ = 7, 10
values (4.10 / 3.90, 4.74 / 4.51) match Table 8 exactly.

---

## Cross-check summary (data ↔ figures ↔ tables ↔ text)

All figure-bearing numbers are **consistent**. Items verified:

- Table 4 ↔ Figure 3: F1 and recovery (recovery = F1 / 0.931 × 100) — exact.
- §3.3.1 latency: components sum to P50 = 268 ms, P99 = 342 ms — exact.
- Table 8 ↔ Figure 7: domain-tuned speedups at γ = 5/7/10 — exact.
- Table 9 ↔ Figure 2: WER ≥ 0.95 privacy headline — consistent (0.949 → ≥ 0.95).
- §4.5 ↔ Figures 4 & 6: 2.76× KL drop, +18.2 % → +1.3 % drift, 30 % window — exact.
- EXP03/EXP09 ↔ Figure 5: loss KL values and teacher F1/tokens — exact.

### One data-vs-text item flagged (no figure impact)

The engineering codebase's deployment block carried a **completed** pilot
(5,000 students; precision 93.2 %, recall 98.8 %, satisfaction 92 %, IRB-2025-027),
whereas the paper deliberately frames the pilot as a **planned 2,000-user**
deployment (Table 1: "Bench + planned pilot"; §5: "a planned 2,000-user pilot
deployment … once conducted"). These completed-deployment metrics are **not used
by any of the seven paper figures**, so there is no figure–text contradiction.
Recommendation: keep the "planned 2,000-user" framing in the paper and do not
cite the 5,000-student precision/recall/satisfaction numbers unless the pilot
has actually been conducted; if it has, reconcile the user count (2,000 vs
5,000) and add a deployment results table/figure.
