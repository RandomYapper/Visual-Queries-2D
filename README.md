# Ego4D VQ2D: Robust Spatio-Temporal Localization (Group 25)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ego4D](https://img.shields.io/badge/Dataset-Ego4D-lightgrey.svg)](https://ego4d-data.org/)

This repository contains the official codebase for **Group 25's** submission to the **Ego4D Visual Query 2D (VQ2D) Localization Challenge**. 

Our system solves the problem of localizing the most recent occurrence of an open-vocabulary object in long, untrimmed, and highly noisy egocentric video streams. We implemented a "defense-in-depth" architecture that rejects visual aliases, mathematically validates temporal continuity, and features self-healing track recovery mechanisms.

---

## Pipeline Architecture

Our pipeline processes queries through a strict, multi-stage funnel designed to maximize Precision without sacrificing Recall.

### Stage 0: SiamRCNN Cache Generation (The Search Engine)
* **Action:** Scans every frame strictly before the query frame (`[0, query_frame - 1]`) using a Siamese Region-based CNN.
* **Result:** Eliminates ~90% of the video timeline, caching bounding box proposals and similarity scores to drastically reduce the search space for heavy segmentation models.

### Stage 1: Temporal Selection & Signal Processing
* **Action:** Applies median filtering (`kernel=5`) to the raw SiamRCNN similarity time-series.
* **Adaptive Detection:** Uses dynamic prominence thresholds (0.20 down to 0.05) to locate Candidate Peaks. If no peaks are found, it triggers a 60-frame argmax fallback recovery to prevent empty initializations.

### Stage 2: Spatial Validation (The Quality Gate)
* **Action:** Validates the geometric reality of the object at the peak timestamp using **Segment Anything 2 (SAM2)**.
* **2.5x Object-Centric Crop:** Extracts a 2.5x expanded bounding box around the candidate to provide SAM2 with crucial environmental context, preventing "Visual Aliasing."
* **Strict Gating:** Rejects detections lacking high SAM2 Confidence (> 0.85) and spatial Compactness (> 0.10).

### Stage 3: Bidirectional Tracking & Resilience
* **Action:** Propagates the validated mask bi-directionally (forward and backward) using the SAM2 Video Predictor.
* **GAP=5 Rule:** Terminates tracking if the object is occluded for 5 consecutive frames, preventing long-term tracker drift.
* **Truth Check:** Applies a Bidirectional Agreement Filter. If the forward and backward tracks overlap by less than an **IoU of 0.40**, the frame is discarded as drift.

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
git clone [https://github.com/your-org/ego4d-vq2d-samurai.git](https://github.com/your-org/ego4d-vq2d-samurai.git)
cd ego4d-vq2d-samurai
```

## Acknowledgement

We would like to thank **Mrs. Jyoti Nigam** for her valuable mentorship and guidance throughout this project. We also express our gratitude to **Aditya Nigam**, instructor of the **Deep Learning** course, under which we received the opportunity to work on this project.

This work was carried out by **Group 25**: Dishant Jha, Utkarsh Sahu, Divyansh Jindal, Garv Jain, Divyansh Negi , Shivam Soni, Nirupam, and Mudit Patial.
