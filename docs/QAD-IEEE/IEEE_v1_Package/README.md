# QAD-MultiGuard — IEEE Conference Submission Package

This package converts the original revised manuscript into IEEE conference
format following a full peer-review pass with 15 critical-issue resolutions.

## Files

| File | Purpose |
|---|---|
| `QAD_MultiGuard_IEEE.docx` | **Final IEEE-formatted manuscript** (5 pages, two-column) |
| `QAD_MultiGuard_IEEE.pdf` | PDF preview of the final manuscript |
| `Peer_Review_Report.md` | Full peer-review report with 15 critical issues identified |
| `qad_bench_eval.py` | Reproducible evaluation benchmark script (referenced in paper) |
| `Original_Manuscript_EN.docx` | Pre-IEEE source manuscript (English) |
| `Original_Manuscript_ZH.docx` | Pre-IEEE source manuscript (Chinese) |
| `Revision_Summary.md` | Summary of all 15 changes made during peer-review revision |

## Format Compliance

The final manuscript conforms to **IEEE Conference Template** (`IEEEtran.cls`-equivalent):

- **Page setup:** A4, 19 mm margins, 5 pages total
- **Layout:** Single-column title/abstract; two-column body (360 twip gutter)
- **Font:** Times New Roman, 10 pt body, 9 pt tables/captions
- **Title:** ≤ 12 words, no em-dashes, no "(Revised)" tags
- **Abstract:** Single paragraph, ~250 words, structured Context→Problem→Method→Results→Significance
- **Equations:** 8 numbered `(1)–(8)` covering QAD loss, KL divergence, speculative-decoding speedup, fusion projection
- **Algorithms:** Algorithm 1 reformatted with formal `Input:`/`Output:`/numbered lines/complexity remark
- **Tables:** 6 tables (I–VI), all fit single-column width with IEEE captions
- **References:** 19 entries in IEEE numeric `[n]` style with vol/no/pp/year
- **Sections:** Roman-numeral I–XIII, sentence-case subsections A./B./C.

## Submission Targets (Suggested)

The paper is now ready for submission to:

1. **IEEE Transactions on Information Forensics and Security (TIFS)** — best fit
2. **IEEE Access** — open-access alternative
3. **IEEE Big Data 2026** — conference venue
4. **ICASSP 2026** — speech/audio focus

