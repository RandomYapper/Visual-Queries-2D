# SAMURAI VQ2D — GAP=5 Ablation Pipeline
**Group 25 | Ego4D Visual Query 2D Challenge**

---

## What This Package Contains

This package contains the complete **GAP=5 ablation pipeline** for the Ego4D VQ2D task —
from SiamRCNN cache loading through SAM2-based bidirectional tracking to metric evaluation
and ablation comparison.

> **Note:** MP4 video files are excluded from this package due to size constraints.
> Similarity plots (`.png`) for all 203 evaluated queries are included under `results/per_query_similarity/`.

---

## Final Results (203-Query Fair Evaluation)

| Method | stAP @ 0.25 | stAP | tAP @ 0.25 | tAP | Recall % | Success % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SiamRCNN (Baseline)** | 0.1530 | 0.0580 | 0.2250 | 0.1340 | 32.919 | 43.244 |
| **samuriaNew (G=15)** | 0.1527 | 0.0647 | 0.1750 | 0.0943 | **52.652** | **51.724** |
| **samuriaGDino (G=10)** | 0.1420 | 0.0611 | 0.1724 | 0.1032 | 47.244 | 47.291 |
| **GAP=5 (This model)** | **0.1822** | **0.0866** | 0.2140 | **0.1424** | 48.250 | 47.783 |

Produced by: `code/evaluation/fair_metrics_gap5.py`
All 203 queries evaluated — 22 failed tracks included as score=0 (no cherry-picking).

---

## Package Structure

```
GAP5_pipeline_package_no_mp4/
│
├── README.md                          ← This file
├── END_TO_END_PIPELINE_AUDIT.md       ← Full technical pipeline explanation
│
├── code/
│   ├── cache_generation/
│   │   └── extract_vq_detection_scores.py   ← PRE-STEP: SiamRCNN similarity cache
│   │
│   ├── inference_gap5/
│   │   ├── samuria_inference_ablation.py    ← MAIN: Full inference pipeline (GAP=5)
│   │   └── priority_order.json             ← Query processing order (331 queries)
│   │
│   ├── merge/
│   │   └── merge_results_gap5.py           ← Merge per-query tracks → predictions.json
│   │
│   ├── evaluation/
│   │   ├── fair_metrics_gap5.py            ← Fair metric evaluation (203-query subset)
│   │   └── vq2d/                           ← Official Ego4D VQ2D metric library
│   │       ├── structures.py              ← ResponseTrack, BBox data types
│   │       └── metrics/
│   │           ├── metrics.py             ← compute_visual_query_metrics()
│   │           ├── spatio_temporal_metrics.py
│   │           ├── temporal_metrics.py
│   │           ├── tracking_metrics.py
│   │           ├── success_metrics.py
│   │           └── utils.py
│   │
│   └── ablation/
│       ├── ablation_report_gap5.py         ← Ablation table + plot (GAP=5 vs G=10 vs G=15)
│       └── calculate_ablation_metrics.py   ← Conditional ablation metrics script
│
├── results/
│   ├── predictions.json                    ← Final GAP=5 predictions (VQ2D challenge format)
│   ├── predictions_filtered.json           ← Predictions for processed clips only
│   ├── eval_subset_keys_203.json           ← Fixed 203-query evaluation manifest
│   └── per_query_similarity/               ← Similarity plots for each query (181 PNGs)
│       └── {label}/similarity.png
│
└── plots/
    ├── ablation_plot.png                   ← Main ablation bar chart (4 methods)
    ├── ablation_comparison_plot.png        ← G=5 vs G=10 vs G=15 detailed comparison
    └── ablation_output.txt                 ← Raw terminal output from ablation run
```

---

## Pipeline Overview

```
INPUT: Video clip (.mp4) + Query image crop (from vq_val.json)
        ↓
[PRE] SiamRCNN Cache (ret_bboxes, ret_scores per frame)
        ↓
STAGE 1: Temporal Selection
  → Median-smoothed similarity signal
  → Adaptive peak detection (prominence 0.20 → 0.05)
  → Most recent peak above threshold selected
        ↓
STAGE 2: Spatial Validation
  → 2.5x Object-Centric Crop around SiamRCNN bbox
  → SAM2 Image Predictor: confidence + compactness gate
  → Validated mask on up to 5 candidate peaks
        ↓
STAGE 3: Bidirectional Tracking (MAX_GAP = 5)
  → SAM2 Video Predictor: backward + forward propagation
  → Stop if object invisible for > 5 consecutive frames
  → Bidirectional agreement filter (IoU ≥ 0.40)
        ↓
OUTPUT: predictions.json (bounding box track per query)
```

---

## Key Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `MAX_GAP` | **5** | Max consecutive blank frames before tracking stops |
| `SMOOTHING_KERNEL` | 5 | Median filter window for similarity signal |
| `PEAK_DISTANCE` | 25 | Minimum frames between detected peaks |
| `CROP_CONTEXT_MULTIPLIER` | 2.5× | Zoom level for object-centric crop |
| `CONF_THRESHOLD` | 0.85 | SAM2 confidence threshold (quality gate) |
| `COMPACT_THRESHOLD` | 0.10 | Mask compactness threshold (quality gate) |
| `BIDIR_IOU_THRESHOLD` | 0.40 | Bidirectional agreement filter threshold |
| `MIN_TRACK_LEN` | 3 | Minimum frames for a valid track |

---

## How to Run Evaluation

```bash
# Activate the vq2d conda environment
conda activate vq2d

# Run fair metric evaluation on the 203-query subset
cd code/evaluation
python fair_metrics_gap5.py
```

**Expected output:**
```
GAP=5 stAP @ 0.25 = 0.1822
GAP=5 stAP       = 0.0866
GAP=5 tAP @ 0.25 = 0.2140
GAP=5 tAP        = 0.1424
GAP=5 Recall %   = 48.250
GAP=5 Success %  = 47.783
```

**Note:** Requires:
- `vq_val.json` at `/DATA/Group25/data/v2/annoVQ2D/vq_val.json`
- `results/predictions.json` (included)
- `results/eval_subset_keys_203.json` (included)

---

## How to Run Ablation Comparison

```bash
conda activate vq2d
cd code/ablation
python ablation_report_gap5.py
# Outputs: ablation table to terminal + ablation_plot.png
```

---

## Evaluation Methodology Notes

1. **Fair denominator**: All 203 queries that GAP=5 attempted are in the denominator. The 22 where tracking failed score 0 — they are NOT excluded.
2. **Same subset for all methods**: samuriaNew (G=15) and samuriaGDino (G=10) are also evaluated on *only* these same 203 queries. Queries they didn't predict = score 0.
3. **SiamRCNN baseline**: Hardcoded published numbers from the Ego4D paper, evaluated on the full validation set. The comparison is directionally valid but not on the identical 203-query subset.
4. **MAX_GAP selection**: GAP=5 was selected via ablation on the validation split. Final metrics are also reported on the validation split (standard practice; no held-out test set is available publicly).
5. **GT usage**: Ground truth (`response_track`) is loaded during inference **only for visualization** (similarity plots, video overlays, console IoU prints). It is never used for peak selection, SAM2 initialization, or tracking decisions. See the explicit comments at line 784 of `samuria_inference_ablation.py`.
6. **Missing 128 queries**: The validation set contains 331 queries with cached SiamRCNN features. Of these, 203 were processed before inference was stopped due to compute/time constraints (SAM2 Video Predictor requires ~8GB VRAM per worker). The 203-query subset was determined by processing order (priority_order.json, sorted by clip length), NOT by filtering based on performance. All 203 attempted queries — including 22 failures — are in the evaluation denominator.

---

## File Integrity

- **MP4 files**: 0 (excluded by design)
- **Predictions**: 203 queries in manifest, 181 non-empty, 22 empty (score=0)
- **Future-frame violations** (`fno ≥ query_frame`): 0 (sanitized)
- **Invalid bboxes** (`x2 ≤ x1` or `y2 ≤ y1`): 0
- **Duplicate frame indices per track**: 0

See `END_TO_END_PIPELINE_AUDIT.md` for the full technical pipeline walkthrough.
