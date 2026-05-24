# Peer-Review Report — QAD-MultiGuard (Revised) Manuscript

**Reviewer Role:** Senior reviewer for IEEE Transactions on Information Forensics and Security (TIFS)
**Manuscript:** QAD-MultiGuard: Quantization-Aware Distillation for Multimodal Telecom Fraud Detection via Large Language Model Reasoning (Revised)
**Recommendation:** Major Revision required for IEEE submission readiness

---

## Summary of Manuscript

The authors propose QAD-MultiGuard, a quantization-aware distillation framework integrated with fraud-tuned speculative decoding and audio-text multimodal fusion for on-device telecom fraud detection. The revision integrates training on the TeleAntiFraud-28k dataset and reports macro-F1 = 0.924, surpassing the SAFE-QAQ baseline (0.891) under PIPL-compliant constraints. A reproducible benchmark (QAD-Bench) is also released.

---

## Strengths

1. **Practically motivated.** The privacy-compliant on-device deployment angle is a real-world differentiator from cloud-based GPU baselines.
2. **Strong reproducibility commitment.** The release of QAD-Bench and full evaluation scripts is commendable.
3. **Clear quantitative gains.** A 7.7-point F1 improvement over Qwen-Audio FT on a public benchmark is substantive.
4. **Well-positioned related work.** TeleAntiFraud-28k and SAFE-QAQ are appropriate, current comparators.

---

## Critical Issues to Address Before IEEE Submission

### 1. Format does not conform to IEEE template
- **Issue:** The current manuscript uses single-column layout with running headers; IEEE conference / TIFS requires two-column format with strict spacing.
- **Action:** Reformat to IEEE Conference template (two-column, 10pt Times, balanced columns, IEEE reference style with `[n]` and bracketed citations in-text).

### 2. Title is too long and contains "(Revised)" and "—"
- **Issue:** IEEE titles must be concise (≤ 12 words preferred), no em-dashes, no subtitle markers like "Revised Version."
- **Action:** Shorten to: "QAD-MultiGuard: Quantization-Aware Distillation for On-Device Multimodal Telecom Fraud Detection."

### 3. Section labels say "(REVISED)"
- **Issue:** Marking sections "REVISED" makes the paper read as a revision response, not as a standalone publication.
- **Action:** Strip all "(REVISED)" tags; rewrite Introduction so it stands alone without referring to "the original paper."

### 4. Abstract structure is non-standard
- **Issue:** The abstract reads as a revision-response narrative ("This paper presents the revised...incorporating..."). IEEE requires: Context → Problem → Method → Results → Significance.
- **Action:** Rewrite abstract in 4-6 sentences with clear structure; remove "revised/extended" language; lead with the privacy-on-device problem, not with "we revise."

### 5. Mathematical notation incomplete
- **Issue:** The QAD loss `L = α·L_task + β·L_KD(τ) + γ·L_quant` is described but not fully specified (no equation-numbered formal definition). The audio fusion `F_v = [f_mfcc ; f_whisper]` is described in algorithmic pseudocode but not as an equation block.
- **Action:** Add formal numbered equations (1)-(5) covering: QAD loss components, KL with temperature, speculative-decoding acceptance probability α, audio-text fusion projection, final risk score aggregation.

### 6. Lack of statistical-significance protocol
- **Issue:** Table VI mentions "McNemar p = 0.12" once, but most claimed gains lack significance tests.
- **Action:** Add 5-fold cross-validation results with mean ± SD for all main metrics; report McNemar or paired-bootstrap p-values for the 7.7 pp gain over Qwen-Audio FT.

### 7. Algorithm 1 is too informal
- **Issue:** Pseudocode lacks complexity analysis, line numbers, and formal Input/Output declarations.
- **Action:** Reformat as IEEE-style Algorithm with `Require:` / `Ensure:` keywords, numbered lines, and a "Time/Space complexity" remark below.

### 8. Code listings do not belong in the body
- **Issue:** Sections V.B and V.C contain ~200 lines of Python code. IEEE papers must move code to appendix or reference an external repository.
- **Action:** Replace inline code with a short pseudocode summary + one paragraph; move full code to an "Appendix A: Reproducibility Artifacts" or cite the GitHub repo URL.

### 9. Tables exceed column width
- **Issue:** Table I has 6 columns and Table IV has 5 columns; in two-column IEEE format these must be `\begin{table*}` (full-width) or restructured.
- **Action:** Mark wide tables as full-width spanning both columns; trim non-essential columns where possible.

### 10. Acronym discipline
- **Issue:** "QAD," "GRPO," "PIPL," "GBM," "CoT" appear without first-use expansion in some places.
- **Action:** Audit every acronym; expand at first use, then use the abbreviation consistently. Add a notation/acronym list at end of Section I if useful.

### 11. References style not IEEE-compliant
- **Issue:** Refs have inconsistent author names ("J. Devlin, M.-W. Chang…" vs "X. Liu et al."), missing DOIs, missing publisher info for arXiv preprints, and use Chinese references mixed with English in the ZH version.
- **Action:** Convert all references to IEEE style: numeric in-text, fully-listed in alphabetical-by-citation-order, includes vol/no/pp/year/DOI (where available); arXiv entries use proper format `arXiv:YYMM.NNNNN`.

### 12. Missing essential sections for an IEEE journal/conference paper
- Add **threat model** subsection in problem formulation.
- Add **limitations** discussion before the conclusion.
- Add **ethics statement** (audio data, IRB, dataset licensing).
- Add **author contributions** statement (CRediT taxonomy).
- Add **acknowledgments** (funding sources, IRB approval ID).

### 13. Year inconsistency / future-dated references
- **Issue:** Papers cite 2026 references (Wang 2026, Liu 2026, Ding 2026). For a real submission, future-dated work cannot be a baseline. Either keep as preprints (arXiv with future date is OK) or drop.
- **Action:** Verify each ref publication date; for arXiv preprints with 2026 date, retain as such; for journals, replace with an actually-published equivalent or remove.

### 14. Cross-dataset claim needs additional safeguards
- **Issue:** The claim "98.8% retention on campus dataset" is from a single split of a small (1,283) sample; needs CI.
- **Action:** Run 5-fold CV on the campus subset and report mean ± SD; or transparently report this as a limitation.

### 15. Algorithmic novelty needs sharper articulation
- **Issue:** It is unclear what is technically novel vs prior QAD/specdec/Whisper work. Reviewers will ask: what is the *minimum* contribution that could not be obtained by combining existing techniques?
- **Action:** Add an explicit "Contributions over Prior Work" subsection in the Introduction with a comparison table positioning this paper against [QAD original], [SpecDec], [Whisper], [SAFE-QAQ].

---

## Minor Issues

- "Snapdragon 8 Gen 3" inconsistent capitalization; standardize.
- "Tok/s" → spell out "tokens/s" on first use.
- The Greek letters α/β/γ are used both for loss weights and acceptance rates — rename one to avoid clash.
- Figure references missing — paper has no figures at all; add at least one architectural diagram and one results bar chart.
- Page margins / line spacing currently not IEEE 25mm-margin compliant.
- No keywords list separator (use semicolons consistently).
- "✓" / "❌" Unicode in tables — replace with "Yes/No" for IEEE compliance.

---

## Recommended Action Plan

1. **Reformat to IEEE two-column conference template.**
2. **Tighten title/abstract; remove all revision-narrative language.**
3. **Add formal equations, statistical tests, and ablations.**
4. **Move code to appendix; replace with concise pseudocode in body.**
5. **Add limitations, ethics, author contributions, acknowledgments.**
6. **Polish references to IEEE style, fix date inconsistencies.**
7. **Add at least one system diagram (Figure 1) and one results figure (Figure 2).**

After these revisions, the paper will be in submittable shape for IEEE TIFS, IEEE Access, or as an IEEE conference (e.g., IEEE Big Data, ICASSP, INFOCOM-WS).
