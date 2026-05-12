# GAP=5 End-to-End Pipeline Audit and Documentation

This document describes the full GAP=5 pipeline used for the final ablation submission, including audit checks for mapping correctness, metric correctness, and leakage risks.

## 1) Scope and Final Verdict

- **Pipeline audited**: cache generation → GAP=5 inference → prediction formatting → fair evaluation → ablation comparison.
- **Submission subset**: **203 queries** (fixed denominator).
- **Final GAP=5 prediction status**:
  - Non-empty predictions: **181**
  - Empty predictions: **22**
- **Leakage audit**:
  - A future-frame issue was found in historical outputs (14 queries had boxes at/after query frame).
  - Outputs were sanitized and code was hardened.
  - Current audited outputs have **0 future-frame violations**.
- **Packaging constraint**: MP4 files are excluded in this package (verified).

---

## 2) Data and Inputs

- Validation annotations: `/DATA/Group25/data/v2/annoVQ2D/vq_val.json`
- SiamRCNN cache directory: `/DATA/Group25/data/ImprovedBaselineVectorSpace`
- Clip videos: `/DATA/Group25/data/clips`
- Priority list: `code/inference_gap5/priority_order.json`

Observed counts during audit:

- Queries with valid cache: **331**
- Priority queries: **331**
- Fixed evaluation subset: **203**

---

## 3) End-to-End Pipeline

## 3.1 Cache generation (SiamRCNN retrieval cache)

File: `code/cache_generation/extract_vq_detection_scores.py`

- Reads visual crop and query metadata from `vq_val.json`.
- Generates per-query `.pt` cache with:
  - `ret_bboxes`
  - `ret_scores`
- Retrieval window is restricted to frames before query (`range(0, query_frame)`), so no intentional future-frame use at cache stage.

## 3.2 GAP=5 inference

File: `code/inference_gap5/samuria_inference_ablation.py`

- Core setting: `MAX_GAP = 5`
- Loads cached `ret_bboxes`/`ret_scores`, selects peak frame, validates with SAM2 gate, tracks bidirectionally.
- Writes challenge-format predictions (`fno, x1, y1, x2, y2`) to `results/predictions.json`.
- Also writes filtered subset file: `results/predictions_filtered.json`.
- Produces per-query artifacts (similarity plots; MP4 intentionally not included in this package).

## 3.3 Evaluation (fair denominator)

File: `code/evaluation/fair_metrics_gap5.py`

- Uses fixed subset file: `results/eval_subset_keys_203.json`.
- Evaluates GAP=5, samuriaNew, samuriaGDino on the exact same 203-query denominator.
- Missing predictions are scored as zero (no cherry-picking).

## 3.4 Ablation table/plot

File: `code/ablation/ablation_report_gap5.py`

- Uses the same fixed subset logic.
- Produces comparison table and ablation bar plot.

---

## 4) Mapping and Format Correctness Checks

Checks performed:

1. Per-query label mapping (`{aid[:8]}_{qid}_{object_title}`) to validation annotations.
2. Prediction JSON structure and query-set indexing.
3. BBox validity (`x2>x1`, `y2>y1`).
4. Track ordering and duplicate frame indices.

Results:

- 203/203 subset keys map correctly.
- Invalid bbox count: **0**
- Unsorted tracks: **0**
- Tracks with duplicate frame number: **0**

---

## 5) Leakage and Integrity Findings

## 5.1 Issue found

Historical predictions contained future-frame boxes in 14 queries (boxes with `fno >= query_frame`), caused by resumed/stale entries.

## 5.2 Fixes applied

1. **Result sanitization**
   - Removed all boxes with `fno >= query_frame` from:
     - `results/predictions.json`
     - `results/predictions_filtered.json`
   - Also mirrored in package results.

2. **Code hardening in inference**
   - Existing loaded predictions are now clipped to `< query_frame`.
   - Final formatting step also clips to `< query_frame`.

3. **Subset stability hardening**
   - Added fixed subset manifest: `results/eval_subset_keys_203.json`
   - Evaluation and ablation scripts now prefer this manifest, so denominator remains 203 even if per-query media folders are pruned.

4. **Metric table bug fix**
   - SiamRCNN baseline key matching normalized/fixed in fair metrics script.

## 5.3 Post-fix status

- Future-frame violations: **0**
- Non-empty predictions remain: **181**
- Empty predictions remain: **22**

---

## 6) Final Verified Metrics (203-query denominator, post-sanitization)

From `fair_metrics_gap5.py` (after removing all future-frame violations):

- Temporal AP @ IoU=0.25:0.95: **0.1424**
- Temporal AP @ IoU=0.25: **0.2140**
- SpatioTemporal AP @ IoU=0.25:0.95: **0.0866**
- SpatioTemporal AP @ IoU=0.25: **0.1822**
- Tracking recovery @ IoU=0.50: **48.250**
- Success @ IoU=0.05: **47.783**

> **Note:** Earlier pre-sanitization runs reported slightly higher AP values (e.g., stAP@0.25 = 0.1863)
> because 14 queries contained future-frame bounding boxes that inflated scores. After sanitization
> (removing all boxes with `fno >= query_frame`), these are the corrected final numbers.

---

## 7) Files Added/Updated During Audit

- Added:
  - `results/eval_subset_keys_203.json`
  - `END_TO_END_PIPELINE_AUDIT.md`
- Updated:
  - `code/inference_gap5/samuria_inference_ablation.py`
  - `code/evaluation/fair_metrics_gap5.py`
  - `code/ablation/ablation_report_gap5.py`
  - `results/predictions.json`
  - `results/predictions_filtered.json`

Equivalent source files under `ablation_study/` were updated consistently.

---

## 8) Submission Guidance

Use `results/predictions.json` from this audited package as the final GAP=5 result artifact.

---

## 9) Re-Audit Pass (after compression/pruning changes)

A full second end-to-end verification was executed after package pruning.

Re-verified outcomes:

- Subset manifest is present and used: `results/eval_subset_keys_203.json`.
- Subset integrity: **203/203** keys valid, with GT and visual crop metadata available.
- Prediction integrity:
  - Non-empty in subset: **181**
  - Empty in subset: **22**
  - Invalid bbox count: **0**
  - Future-frame bboxes (`fno >= query_frame`): **0**
  - Unsorted tracks: **0**
  - Duplicate-frame tracks: **0**
- Evaluation script and ablation script run successfully on the fixed subset.
- Package media constraint remains valid: **0 MP4 files**.

Conclusion of re-audit: current package state is consistent for submission.
