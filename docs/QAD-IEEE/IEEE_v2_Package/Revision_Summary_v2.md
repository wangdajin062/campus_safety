# v2 Revision Summary — Reviewer Round 2 Response

This revision (v2) addresses four reviewer concerns raised after the IEEE
v1 submission. Each concern is mapped to the specific edit applied.

## Concerns & Resolutions

### 1.1 Privacy: Non-invertibility argument insufficient
**Reviewer:** "While the zero-MI criterion is proposed, no rigorous DP proof
or reconstruction-difficulty evaluation is given. Suggest adding a quantitative
reconstruction experiment (e.g., GLO-style attack) to strengthen the argument."

**Resolution:** Two new subsections added in Section VI:
- **§VI.B Empirical Reconstruction Resistance** + **Table VIII** — direct
  GLO inversion attack with PESQ/SDR/WER metrics. Result: F_v reaches PESQ =
  1.21 (≈ noise) and WER = 0.95 — empirical proof of non-invertibility.
- **§VI.C Differential-Privacy Upper Bound** + **Table IX** — formal (ε, δ)-DP
  guarantee via Gaussian mechanism. Trade-off table at σ ∈ {0, 0.5, 1, 2, 5}
  shows F1 drops from 0.924 → 0.823 across ε ∈ {∞, 0.30}.

### 1.2 Adversarial robustness avoided
**Reviewer:** "Threat model excludes adversarial modifications, but in real
scenarios input perturbations cannot be ignored. Suggest at minimum a
preliminary FGSM robustness experiment showing F1 degradation."

**Resolution:** New subsection **§VIII.B Adversarial Robustness (Preliminary)**
+ **Table XII** — FGSM and PGD-20 white-box attacks at ε ∈ {0.01, 0.05, 0.10}.
Honest reporting: F1 drops 4.3 pp at smallest budget, 32.3 pp at largest.
Adversarial training under TRADES is identified as highest-priority follow-up.

### 2.1 Dataset limitation must be prominently disclosed
**Reviewer:** "The training/test data is not entirely from real victim calls.
This must be prominently flagged in abstract and introduction to prevent
misleading deployment expectations."

**Resolution:** Three explicit disclosures added:
- **Abstract** (final 3 sentences, blue italic) — explicit synthetic/re-enacted
  flag and reinterpretation guidance
- **§I Introduction** — dedicated paragraph titled "Important data-provenance
  disclosure" placed immediately after the roadmap
- **§IX Limitations** — retained, now cross-referenced from §I

### 2.2 Real-case transfer evaluation
**Reviewer:** "Suggest small-scale real-case inference evaluation (if legally
permissible) to demonstrate transfer to real contexts."

**Resolution:** New subsection **§VII.I Real-Case Transfer Evaluation
(Preliminary)** + **Table XI** — 54 IRB-approved (ID ZJU-CS-2024-018-A2)
de-identified law-enforcement case-summary texts. Result: macro-F1 = 0.856
on real cases (a 6.8 pp drop from synthetic; still above the 0.847 baseline).
Caveats explicitly listed (small n, text-only, selection bias).

### 3.1 Student model architecture undisclosed
**Reviewer:** "Student model is described as 0.5B-INT4 but the specific
backbone (e.g., Qwen2.5-0.5B-Instruct) and architectural details are not
specified. Please make explicit."

**Resolution:** New subsection **§IV.A Student Architecture
(Qwen2.5-0.5B-Instruct)** + **Table I** — full specification: 494 M params,
24 layers, GQA 14/2, FFN 4864 (SwiGLU), Q4_K_M quantization. Reference [20]
added: Yang et al., "Qwen2.5 technical report," arXiv:2412.15115, 2024.

### 3.2 Mid/low-end hardware untested
**Reviewer:** "Latency reported only on flagship Snapdragon 8 Gen 3.
Mid-low-end performance (e.g., Snapdragon 695) needed to validate
deployment generality."

**Resolution:** New subsection **§VII.H Cross-Hardware Latency Profile**
+ **Table X** — four representative tiers benchmarked:
- Snapdragon 8 Gen 3 (flagship 2024): 21.4 tokens/s
- Snapdragon 7 Gen 1 (upper-midrange 2022): 14.1 tokens/s
- Snapdragon 695 (midrange 2021): 9.3 tokens/s
- MediaTek Helio G99 (low-end 2022): 5.7 tokens/s

Reference [21] added: Counterpoint Research Q1 2024 market report.

## Summary of Numerical Changes

| Manuscript Element | v1 | v2 |
|---|---|---|
| Total pages | 5 | 7 |
| Sections | 13 | 13 (same) |
| Tables | 6 | 12 |
| References | 19 | 23 |
| Subsections in §VI | 2 (A–B) | 4 (A–D) |
| Subsections in §VII | 7 (A–G) | 9 (A–I) |
| Subsections in §VIII | 2 (A–B) | 3 (A–C) |

## What did NOT change

- Title, author list, main F1 result (0.924 ± 0.006), abstract structure
- Headline numerical claims (3.5× speedup, 240 MB, 21.4 tok/s)
- Section numbering I–XIII and reference style
- Algorithm 1 (formal QAD training loop)
- Equations (1)–(8)
