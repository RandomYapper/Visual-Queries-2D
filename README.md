# Ego4D VQ2D: Robust Spatio-Temporal Localization (Group 25)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ego4D](https://img.shields.io/badge/Dataset-Ego4D-lightgrey.svg)](https://ego4d-data.org/)

This repository contains the official codebase for **Group 25's** submission to the **Ego4D Visual Query 2D (VQ2D) Localization Challenge**. 

Our system solves the problem of localizing the most recent occurrence of an open-vocabulary object in long, untrimmed, and highly noisy egocentric video streams. We implemented a "defense-in-depth" architecture that rejects visual aliases, mathematically validates temporal continuity, and features self-healing track recovery mechanisms.

---

## Pipeline Architecture

Our pipeline processes queries through a strict, multi-stage funnel designed to maximize Precision without sacrificing Recall. The implementation details for the various stages are described below:

### Stage 0: SiamRCNN Cache Generation (The Search Engine)
* **Action:** Scans every frame strictly before the query frame (`[0, query_frame - 1]`) using a Siamese Region-based CNN. Done in [`extract_vq_detection_scores.py`](code/cache_generation/extract_vq_detection_scores.py).
* **Result:** Eliminates ~90% of the video timeline, caching bounding box proposals and similarity scores to drastically reduce the search space for heavy segmentation models.

### Stage 1: Temporal Selection & Signal Processing
* **Action:** Applies median filtering (`kernel=5`) to the raw SiamRCNN similarity time-series in `find_peak_frame` of [`samuria_inference_ablation.py`](code/inference_gap5/samuria_inference_ablation.py).
* **Adaptive Detection:** Uses dynamic prominence thresholds (`0.20` down to `0.05`) to locate Candidate Peaks. If no peaks are found, it triggers a 60-frame argmax fallback recovery to prevent empty initializations.

```python
# Signal smoothing and adaptive peak finding (Stage 1)
scores_smooth = medfilt(raw_scores, kernel_size=SMOOTHING_KERNEL)  # k=5
for prom in [0.20, 0.15, 0.10, 0.05]:
    peaks, _ = find_peaks(scores_smooth, distance=PEAK_DISTANCE, prominence=prom)
    if len(peaks) > 0:
        break
```

### Stage 2: Spatial Validation (The Quality Gate)
* **Action:** Validates the geometric reality of the object at the peak timestamp using **Segment Anything 2 (SAM2)** in `validate_peak_with_crop` of [`samuria_inference_ablation.py`](code/inference_gap5/samuria_inference_ablation.py).
* **2.5x Object-Centric Crop:** Extracts a 2.5x expanded bounding box around the candidate proposal to provide SAM2 with crucial environmental context, preventing "Visual Aliasing."
* **Strict Gating:** Rejects detections lacking high SAM2 Confidence (`> 0.85`) and spatial Compactness (`> 0.10`).

```python
# Object-Centric 2.5x Context Crop (Stage 2)
bbox_w, bbox_h = x2 - x1, y2 - y1
crop_w = min(int(bbox_w * 2.5), int(frame_w / 2.5))
crop_h = min(int(bbox_h * 2.5), int(frame_h / 2.5))
crop_x1 = max(0, center_x - crop_w // 2)
crop_y1 = max(0, center_y - crop_h // 2)
```

### Stage 3: Bidirectional Tracking & Resilience
* **Action:** Propagates the validated mask bi-directionally (forward and backward) using the SAM2 Video Predictor in `track` of [`samuria_inference_ablation.py`](code/inference_gap5/samuria_inference_ablation.py).
* **GAP=5 Rule:** Terminates tracking if the object is occluded for 5 consecutive frames, preventing long-term tracker drift.
* **Truth Check:** Applies a Bidirectional Agreement Filter. If the forward and backward tracks overlap by less than an **IoU of 0.40**, the frame is discarded as drift.

```python
# GAP=5 Occlusion Termination Rule (Stage 3)
if mask.sum() == 0:
    gap += 1
    if gap > MAX_GAP:  # MAX_GAP = 5
        break
else:
    gap = 0
```

---

## Key Innovations & "Self-Healing" Mechanics

Standard trackers crash when they encounter severe occlusion or visual ambiguity. Our pipeline routes around failure:

1.  **Object-Centric Context Cropping:** By zooming out 2.5x before segmentation, we force the model to evaluate the object *within its background*, eliminating texture-based false positives.
2.  **Multi-Hypothesis Tracking (MHT):** When the top two temporal peaks are within a 0.10 confidence margin and have poor compactness, the system refuses to guess. It tracks *both* in parallel and mathematically selects the winner based on track length and average mask confidence.
3.  **Dynamic Track Quality Retry:** If an initialized track hits a wall immediately (survives < 3 frames or has < 0.25 average confidence), the pipeline aborts the track and automatically restarts tracking from the next best peak candidate.
4.  **Semantic Cross-Validation (Optional):** Integrated Grounding DINO logic to double-check semantic alignment between the visual crop and the text label (defaults to off for speed).

---

## Results & Performance

Evaluated against a fixed 203-query subset of the Ego4D validation data. Our optimized `GAP=5` implementation strongly outperforms the baseline in both spatial precision and temporal accuracy.

| Method | stAP@0.25 | stAP | tAP@0.25 | tAP | Recall% | Success% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SiamRCNN (Baseline) | 0.1530 | 0.0580 | 0.2250 | 0.1340 | 32.91% | 43.24% |
| **Ours (GAP=5)** | **0.1822** | **0.0866** | 0.2140 | **0.1424** | **48.25%** | **47.78%** |

---

## Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/your-org/ego4d-vq2d-samurai.git
cd ego4d-vq2d-samurai
```

**2. Activate environment:**
```bash
conda activate vq2d
```

**3. Run Cache Generation (SiamRCNN):**
Build retrieval proposals and similarity scores cache prior to the query frames:
```bash
python code/cache_generation/extract_vq_detection_scores.py
```
*Implementation:* [`extract_vq_detection_scores.py`](code/cache_generation/extract_vq_detection_scores.py)

**4. Run Main SAMURAI Inference (GAP=5):**
Execute tracking and filter results for the 203-query validation subset:
```bash
python code/inference_gap5/samuria_inference_ablation.py
```
*Implementation:* [`samuria_inference_ablation.py`](code/inference_gap5/samuria_inference_ablation.py)

**5. Evaluate Predictions:**
Compute Spatio-Temporal and Temporal Average Precision (AP) metrics:
```bash
python code/evaluation/fair_metrics_gap5.py
```
*Implementation:* [`fair_metrics_gap5.py`](code/evaluation/fair_metrics_gap5.py)

**6. Generate Ablation Report & Plot:**
Calculate performance gains and export the bar charts:
```bash
python code/ablation/ablation_report_gap5.py
```
*Implementation:* [`ablation_report_gap5.py`](code/ablation/ablation_report_gap5.py) (creates `plots/ablation_plot.png`)

---

## Technical Documentation

For a detailed walkthrough of the pipeline design, visual validation parameters, and the no-leakage validation policy, check the internal documentation files:
* [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) — Comprehensive guide on algorithms, coordinate equations, and codebase architecture.
* [`END_TO_END_PIPELINE_AUDIT.md`](END_TO_END_PIPELINE_AUDIT.md) — Leakage verification steps, query lists, and final metrics verification log.

---

## Acknowledgements

Special thanks to our project mentor, **Mrs. Jyoti Nigam**, for her continuous guidance and valuable insights, and to our Deep Learning course instructor, **Aditya Nigam**, for providing the opportunity to explore this research. 

This project was collaboratively developed by the dedicated members of **Group 25**:

* Dishant Jha (B24120)
* Utkarsh Sahu (B24172)
* Divyansh Jindal (B24121)
* Garv Jain (B24124)
* Divyansh Negi (B24122)
* Shivam Soni (B24159)
* Nirupam (B24143)
* Mudit Patial (B24141)
