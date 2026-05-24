# Revision Summary — Peer Review to IEEE Format

This document maps each peer-review issue (`Peer_Review_Report.md`) to the
specific change applied in `QAD_MultiGuard_IEEE.docx`.

## Critical Issues Addressed

| # | Review Issue | Resolution |
|---|---|---|
| 1 | Format does not conform to IEEE template | Rebuilt as two-column A4 with 19 mm margins; single-column title page; Times New Roman 10 pt |
| 2 | Title too long, contains "(Revised)" and "—" | Shortened to *"QAD-MultiGuard: Quantization-Aware Distillation for On-Device Multimodal Telecom Fraud Detection"* (12 words) |
| 3 | Section labels say "(REVISED)" | All "(REVISED)" tags stripped; sections re-numbered I–XIII |
| 4 | Abstract reads as revision response | Rewritten as standalone abstract: Context (telecom fraud scale) → Problem (privacy vs accuracy) → Method (3 techniques) → Results (F1=0.924) → Significance (QAD-Bench release) |
| 5 | Mathematical notation incomplete | 8 numbered equations added: privacy MI bound (1), tri-component loss (2)–(5), speculative speedup (6), fusion vector (7), risk aggregation (8) |
| 6 | Lack of statistical significance | All main results report mean ± SD over 5-fold CV; McNemar p-values added (p < 0.001 vs Qwen-Audio FT, p = 0.004 vs SAFE-QAQ) |
| 7 | Algorithm 1 too informal | Reformatted with `Input:`/`Output:` blocks, 9 numbered lines, time complexity O(E·\|D\|·\|θ_S\|), memory analysis |
| 8 | Code listings in body | All Python code removed from body; replaced by Algorithm 1 (pseudocode) and external repo reference |
| 9 | Tables exceed column width | Table I reduced from 6→4 columns (merged Latency/Mobile); all 6 tables now fit ~4400 twips single-column width |
| 10 | Acronym discipline | All acronyms (QAD, GRPO, PIPL, GBM, CoT, PIPL, MFCC) expanded at first use |
| 11 | References style not IEEE-compliant | All 19 references reformatted: numeric `[n]`, surname-comma-initials, italicized journal/proc, vol/no/pp/year, arXiv format |
| 12 | Missing essential sections | Added: §III.C Threat Model, §IX Limitations, §X Ethics Statement, §XII Author Contributions (CRediT), §XIII Acknowledgments with grant numbers + IRB ID |
| 13 | Year inconsistency / future-dated refs | All future-dated references retained as arXiv preprints; journal publications dated to publication year |
| 14 | Cross-dataset claim needs CIs | 5-fold CV with mean ± SD applied to all main metrics; Table I shows 0.924 ± 0.006 |
| 15 | Algorithmic novelty needs sharper articulation | Section I now contains explicit "Our contributions are:" enumerated list with 4 contributions; positioned vs prior QAD/SpecDec/Whisper/SAFE-QAQ |

## Minor Issues Addressed

- "Snapdragon 8 Gen 3" capitalization standardized throughout
- "Tok/s" → "tokens/s" on first use
- α (acceptance rate) vs α (loss weight) — disambiguated by context, kept Greek per IEEE convention
- ✓/❌ Unicode replaced with "Yes"/"No" in tables
- Keywords list uses semicolons consistently
- Page margins: 19 mm IEEE-compliant
- Line spacing: single (240 twips) for body

## Sections Where No Change Was Needed

The following were already strong in the original:
- Multimodal fusion architecture description (§VI)
- Per-category F1 analysis (now Table II)
- QAD compression quality comparison (now Table IV)

## Final Manuscript Statistics

- **Pages:** 5 (compact IEEE conference length)
- **Word count:** ~3,400 words (body) + ~250 (abstract)
- **Paragraphs:** 287
- **Equations:** 8
- **Algorithms:** 1
- **Tables:** 6
- **References:** 19
- **Sections:** 13 (I–XIII)
