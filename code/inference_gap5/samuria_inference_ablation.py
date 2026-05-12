"""
samurai_inference_v4.py — Enhanced VQ2D Inference with Innovations #1 & #2

NEW INNOVATIONS:
  #1: Object-Centric Cropping & Refinement
      - Crop around candidate bbox before SAM2 validation
      - Improves spatial precision for small/cluttered objects
  #2: Semantic Cross-Validation (OPTIONAL, CONSERVATIVE)
      - Uses Grounding DINO to verify semantic match
      - Only rejects peaks when IoU < 0.15 (very permissive)
      - Skips generic labels ("object", "thing")

EXISTING INNOVATIONS (from v3):
  - Adaptive prominence threshold (0.20 → 0.05)
  - Failure recovery fallback (argmax last 60 frames)
  - SAM2 quality gate with recency priority
  - Top-2 multi-hypothesis tracking
  - Bidirectional agreement filtering
  - Track quality validation & retry

Usage (from inside the `samuriaNew/` directory, vq2d conda env):
  CUDA_VISIBLE_DEVICES=0 python samurai_inference_v4.py
"""

import os, sys, json, tempfile, shutil
import numpy as np
import cv2
import torch

# ── path setup ──────────────────────────────────────────────────────────────
ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
SAMURAI_DIR = os.path.abspath(os.path.join(ABLATION_DIR, "../samuriaGDino"))
VQ2D_REPO   = os.path.abspath(os.path.join(SAMURAI_DIR, "../VQ2D"))
sys.path.insert(0, VQ2D_REPO)

from scipy.signal import find_peaks, medfilt
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from vq2d.metrics.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack, BBox
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
VAL_JSON   = "/DATA/Group25/data/v2/annoVQ2D/vq_val.json"
CACHE_DIR  = "/DATA/Group25/data/ImprovedBaselineVectorSpace"
CLIPS_DIR  = "/DATA/Group25/data/clips"

SAM2_CKPT  = os.path.join(SAMURAI_DIR, "checkpoints/sam2_hiera_large.pt")
SAM2_CFG   = "configs/sam2/sam2_hiera_l.yaml"

OUT_DIR    = Path(os.path.join(ABLATION_DIR, "results_gap5"))
DEVICE     = "cuda"

# Signal processing
SMOOTHING_KERNEL = 5
PEAK_DISTANCE    = 25
PEAK_WIDTH       = 3
PEAK_PROMINENCE  = 0.2
PEAK_SIM_THRESH  = 0.0
MAX_GAP          = 5

# ── INNOVATION #1: Object-Centric Cropping ──────────────────────────────────
CROP_CONTEXT_MULTIPLIER = 2.5  # Zoom factor around bbox (2.5x recommended)
MAX_CROP_ZOOM = 2.5            # Max zoom to avoid pixelation
ENABLE_CROPPING = True         # Master switch for Innovation #1

# ── INNOVATION #2: Semantic Cross-Validation ────────────────────────────────
ENABLE_SEMANTIC_GATE = False    # Master switch for Innovation #2
SEMANTIC_IOU_THRESHOLD = 0.15  # Very conservative: only reject if overlap < 15%
SEMANTIC_CONF_THRESHOLD = 0.25 # Grounding DINO confidence threshold
SEMANTIC_SKIP_GENERIC = True   # Skip semantic gate for generic labels
GENERIC_LABELS = {"object", "thing", "item", "stuff", "unknown"}

# ─────────────────────────────────────────────────────────────────────────────
# Grounding DINO Setup (Innovation #2)
# ─────────────────────────────────────────────────────────────────────────────
GDINO_MODEL = None

def load_grounding_dino():
    """Load Grounding DINO model (lazy initialization)."""
    global GDINO_MODEL
    if GDINO_MODEL is not None:
        return GDINO_MODEL
    
    try:
        from groundingdino.util.inference import load_model, predict
        print("Loading Grounding DINO...")
        # Absolute config path from installed groundingdino-py package
        import groundingdino as _gdino_pkg
        _gdino_pkg_dir = os.path.dirname(_gdino_pkg.__file__)
        config_path = os.path.join(_gdino_pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
        ckpt_path = os.path.join(SAMURAI_DIR, "weights", "groundingdino_swint_ogc.pth")
        
        # If weights don't exist, download them
        if not os.path.exists(ckpt_path):
            os.makedirs(os.path.join(SAMURAI_DIR, "weights"), exist_ok=True)
            print("  Downloading Grounding DINO weights...")
            import urllib.request
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, ckpt_path)
        
        GDINO_MODEL = load_model(config_path, ckpt_path)
        print("  Grounding DINO loaded OK.")
        return GDINO_MODEL
    except Exception as e:
        print(f"  [WARN] Could not load Grounding DINO: {e}")
        print("  Semantic validation will be disabled.")
        return None


def semantic_validate_peak(frame_bgr, bbox_xyxy, object_title):
    """
    Innovation #2: Semantic Cross-Validation.
    Returns (is_valid, gdino_bbox, gdino_conf, iou)
    """
    if not ENABLE_SEMANTIC_GATE:
        return True, None, 0.0, 0.0
    
    # Skip generic labels
    if SEMANTIC_SKIP_GENERIC and object_title.lower() in GENERIC_LABELS:
        return True, None, 0.0, 0.0
    
    model = load_grounding_dino()
    if model is None:
        return True, None, 0.0, 0.0  # Fail open if model unavailable
    
    try:
        from groundingdino.util.inference import predict
        
        # Prepare text prompt
        text_prompt = f"a photo of a {object_title}."
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        import groundingdino.datasets.transforms as T
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_source = Image.fromarray(frame_rgb)
        image_transformed, _ = transform(image_source, None)
        
        # Run Grounding DINO
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=SEMANTIC_CONF_THRESHOLD,
            text_threshold=SEMANTIC_CONF_THRESHOLD,
        )
        
        if len(boxes) == 0:
            # No detection → could be false negative, be permissive
            return True, None, 0.0, 0.0
        
        # Get top detection
        best_idx = int(np.argmax(logits))
        gdino_conf = float(logits[best_idx])
        
        # Convert from normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        h, w = frame_bgr.shape[:2]
        cx, cy, bw, bh = boxes[best_idx]
        gx1 = int((cx - bw/2) * w)
        gy1 = int((cy - bh/2) * h)
        gx2 = int((cx + bw/2) * w)
        gy2 = int((cy + bh/2) * h)
        gdino_bbox = (gx1, gy1, gx2, gy2)
        
        # Calculate IoU with SiamRCNN bbox
        sx1, sy1, sx2, sy2 = bbox_xyxy
        ix1 = max(sx1, gx1)
        iy1 = max(sy1, gy1)
        ix2 = min(sx2, gx2)
        iy2 = min(sy2, gy2)
        
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        siam_area = (sx2 - sx1) * (sy2 - sy1)
        gdino_area = (gx2 - gx1) * (gy2 - gy1)
        union_area = siam_area + gdino_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # Very permissive threshold
        is_valid = iou >= SEMANTIC_IOU_THRESHOLD
        
        return is_valid, gdino_bbox, gdino_conf, iou
        
    except Exception as e:
        print(f"    [SEMANTIC] Error: {e}")
        return True, None, 0.0, 0.0  # Fail open


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 / SAMURAI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_sam2_with_compat_ckpt(config_file, ckpt_path, device, video=False):
    """Build SAM2 model tolerating checkpoint mismatches."""
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    hydra_overrides = []
    if video:
        hydra_overrides = ["++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor"]
    hydra_overrides += [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ]
    if video:
        hydra_overrides += [
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [CKPT] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  [CKPT] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    model = model.to(device).eval()
    return model


class SamuraiTracker:
    def __init__(self, device: str):
        print("Loading SAM2 video predictor (SAMURAI)…")
        self.predictor = _build_sam2_with_compat_ckpt(SAM2_CFG, SAM2_CKPT, device, video=True)
        self.image_predictor = SAM2ImagePredictor(
            _build_sam2_with_compat_ckpt(SAM2_CFG, SAM2_CKPT, device, video=False)
        )
        print("  SAM2 loaded OK.")

    def get_init_mask(self, frame_bgr: np.ndarray, bbox_xyxy):
        """Use SAM2 image predictor to get tight mask from a bounding box."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(rgb)
        x1, y1, x2, y2 = bbox_xyxy
        box = np.array([x1, y1, x2, y2], dtype=float)
        masks, scores, _ = self.image_predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(scores))
        return masks[best] > 0.0

    def validate_peak_with_crop(self, frame_bgr: np.ndarray, bbox_xyxy, object_title=""):
        """
        Innovation #1: Object-Centric Cropping + Innovation #3: SAM2 Quality Gate.
        Returns (mask_full_frame, confidence, compactness, crop_info)
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]
        
        # ── Innovation #1: Calculate crop window ─────────────────────────────
        if ENABLE_CROPPING:
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Context window size
            crop_w = int(bbox_w * CROP_CONTEXT_MULTIPLIER)
            crop_h = int(bbox_h * CROP_CONTEXT_MULTIPLIER)
            
            # Limit zoom to avoid pixelation
            max_crop_w = int(w / MAX_CROP_ZOOM)
            max_crop_h = int(h / MAX_CROP_ZOOM)
            crop_w = min(crop_w, max_crop_w)
            crop_h = min(crop_h, max_crop_h)
            
            # Center crop around bbox center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            crop_x1 = max(0, cx - crop_w // 2)
            crop_y1 = max(0, cy - crop_h // 2)
            crop_x2 = min(w, crop_x1 + crop_w)
            crop_y2 = min(h, crop_y1 + crop_h)
            
            # Adjust if hit boundary
            if crop_x2 == w:
                crop_x1 = max(0, crop_x2 - crop_w)
            if crop_y2 == h:
                crop_y1 = max(0, crop_y2 - crop_h)
            
            # Crop frame
            crop_frame = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Translate bbox to crop space
            local_x1 = x1 - crop_x1
            local_y1 = y1 - crop_y1
            local_x2 = x2 - crop_x1
            local_y2 = y2 - crop_y1
            
            # Clamp to crop bounds
            local_x1 = max(0, min(local_x1, crop_frame.shape[1]))
            local_y1 = max(0, min(local_y1, crop_frame.shape[0]))
            local_x2 = max(0, min(local_x2, crop_frame.shape[1]))
            local_y2 = max(0, min(local_y2, crop_frame.shape[0]))
            
            local_bbox = (local_x1, local_y1, local_x2, local_y2)
            crop_info = {
                "crop_used": True,
                "crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2),
                "local_bbox": local_bbox,
            }
        else:
            crop_frame = frame_bgr
            local_bbox = bbox_xyxy
            crop_info = {"crop_used": False}
        
        # ── Run SAM2 on crop (or full frame if cropping disabled) ────────────
        rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        self.image_predictor.set_image(rgb)
        box = np.array(local_bbox, dtype=float)
        masks, scores, _ = self.image_predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(scores))
        mask_local = masks[best] > 0.0
        confidence = float(scores[best])
        
        # Compactness in local space
        mask_area = float(mask_local.sum())
        bbox_area = float((local_bbox[2] - local_bbox[0]) * (local_bbox[3] - local_bbox[1])) + 1e-6
        compactness = mask_area / bbox_area
        
        # ── Project mask back to full frame space ────────────────────────────
        if ENABLE_CROPPING and crop_info["crop_used"]:
            mask_full = np.zeros((h, w), dtype=bool)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_info["crop_coords"]
            mask_full[crop_y1:crop_y2, crop_x1:crop_x2] = mask_local
        else:
            mask_full = mask_local
        
        return mask_full, confidence, compactness, crop_info

    def track(self, frames_dir: str, init_frame_idx: int,
              init_mask: np.ndarray, max_gap: int = MAX_GAP,
              return_separate: bool = False):
        """Bidirectional tracking."""
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        names = [f.name for f in frames]
        target = f"{init_frame_idx:05d}.jpg"
        if target not in names:
            print(f"  [WARN] init frame {target} not found in {frames_dir}")
            return ([], []) if return_separate else []
        init_idx = names.index(target)

        state = self.predictor.init_state(
            video_path=frames_dir,
            async_loading_frames=False,
            offload_video_to_cpu=True,
        )
        self.predictor.add_new_mask(state, frame_idx=init_idx, obj_id=1, mask=init_mask)

        def propagate(reverse):
            res, gap = [], 0
            for fidx, _, logits in self.predictor.propagate_in_video(
                state, start_frame_idx=init_idx, reverse=reverse
            ):
                mask = (logits[0][0] > 0.0).cpu().numpy()
                if mask.sum() == 0:
                    gap += 1
                    if gap > max_gap:
                        break
                else:
                    gap = 0
                    ys, xs = np.where(mask)
                    if len(xs) == 0:
                        continue
                    mask_score = float(logits[0][0][mask].mean()) if mask.sum() > 0 else 0.0
                    res.append({
                        "frame_number": fidx,
                        "x": int(xs.min()), "y": int(ys.min()),
                        "width": int(xs.max() - xs.min()),
                        "height": int(ys.max() - ys.min()),
                        "score": 1.0,
                        "mask_confidence": mask_score,
                    })
            return res

        print("    → BACKWARD propagation…")
        backward = propagate(reverse=True)
        print("    → FORWARD propagation…")
        forward = propagate(reverse=False)
        self.predictor.reset_state(state)

        if return_separate:
            return backward, forward

        seen, merged = set(), []
        for t in backward + forward:
            if t["frame_number"] not in seen:
                seen.add(t["frame_number"])
                merged.append(t)
        merged.sort(key=lambda x: x["frame_number"])
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction from MP4
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames_to_dir(clip_path: str, out_dir: str, max_frame: int = None):
    """Extract frames from an MP4 as 00000.jpg, 00001.jpg …"""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frame is not None and idx >= max_frame:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing — find peak frame from SiamRCNN scores
# ─────────────────────────────────────────────────────────────────────────────
def find_peak_frame(ret_scores, query_frame):
    """Enhanced peak detection with adaptive prominence and fallback."""
    ret_scores = ret_scores[:query_frame]
    score_signal = [float(np.max(s)) if len(s) > 0 else 0.0 for s in ret_scores]
    kernel = SMOOTHING_KERNEL if SMOOTHING_KERNEL % 2 == 1 else SMOOTHING_KERNEL + 1
    score_sm = medfilt(score_signal, kernel_size=kernel)

    recent_peak = None
    peaks = np.array([], dtype=int)
    used_prominence = None
    for prom in [0.20, 0.15, 0.10, 0.05]:
        peaks_candidate, _ = find_peaks(
            score_sm, distance=PEAK_DISTANCE, width=PEAK_WIDTH, prominence=prom,
        )
        if len(peaks_candidate) > 0:
            peaks = peaks_candidate
            used_prominence = prom
            for p in peaks[::-1]:
                if score_sm[p] >= PEAK_SIM_THRESH:
                    recent_peak = p
                    break
            if recent_peak is not None:
                break

    method = f"adaptive_prom={used_prominence}" if recent_peak is not None else None

    if recent_peak is None and len(score_signal) > 0:
        window = 60
        start = max(0, len(score_signal) - window)
        fallback_scores = score_signal[start:]
        if max(fallback_scores) > 0:
            recent_peak = start + int(np.argmax(fallback_scores))
            method = "fallback_argmax_60"
            print(f"    [FALLBACK] No peaks found. Using argmax of last {window} frames → frame {recent_peak}")

    return recent_peak, score_signal, score_sm, peaks, method


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_similarity_plot(score_signal, score_sm, peaks, peak_idx,
                         query_frame, gt_rt, out_path, title="",
                         query_crop=None, peak_crop=None, gt_crop=None):
    has_crops = (query_crop is not None) or (peak_crop is not None) or (gt_crop is not None)
    
    if has_crops:
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        ax_main = fig.add_subplot(gs[0, :])
    else:
        fig, ax_main = plt.subplots(figsize=(14, 4))
    
    ax_main.plot(score_signal, color="#4C9BE8", alpha=0.45, linewidth=1.0, label="Raw score")
    ax_main.plot(score_sm, color="#F4A261", linewidth=2.0, label="Smoothed")

    if gt_rt:
        gt_fns = [t["frame_number"] for t in gt_rt]
        ax_main.axvspan(min(gt_fns), max(gt_fns), alpha=0.2, color="#2A9D8F", label="GT Window")
    
    ax_main.plot(peaks, score_sm[peaks], "x", color="#7209B7", label="Peaks", markersize=8)
    
    if peak_idx is not None:
        ax_main.axvline(peak_idx, color="#E63946", linewidth=2.5, linestyle="--",
                       label=f"Selected Peak ({peak_idx})")
    
    ax_main.axvline(query_frame - 1, color="#2D6A4F", linewidth=1.5, linestyle=":",
                   label=f"Query Frame ({query_frame})")
    
    ax_main.set_xlabel("Frame Number", fontsize=10)
    ax_main.set_ylabel("Score", fontsize=10)
    ax_main.set_title(f"STAGE 1: Temporal Selection — {title}", fontsize=14, fontweight='bold')
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax_main.grid(True, alpha=0.2, linestyle='--')

    if has_crops:
        def add_crop(img, title, idx):
            if img is None or img.size == 0:
                return
            ax = fig.add_subplot(gs[1, idx])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=11)
            ax.axis('off')

        add_crop(query_crop, "Query Image Crop", 0)
        add_crop(gt_crop, "Ground Truth Crop", 1)
        add_crop(peak_crop, f"Predicted Frame {peak_idx if peak_idx is not None else ''}", 2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    Similarity plot → {out_path}")


def save_response_track_video(frames_dir: str, pred_track, gt_rt, out_path: str, max_frame: int = None):
    """Render a full video segment with prediction + GT overlays."""
    frames_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames_paths:
        return
    
    if max_frame is not None:
        frames_paths = frames_paths[:max_frame]
    
    sample = cv2.imread(str(frames_paths[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 15.0, (w, h))

    pred_d = {t["frame_number"]: t for t in pred_track}
    gt_d = {t["frame_number"]: t for t in (gt_rt or [])}

    for fn, frame_path in enumerate(frames_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        if fn in gt_d:
            g = gt_d[fn]
            cv2.rectangle(img, (int(g["x"]), int(g["y"])),
                          (int(g["x"]+g["width"]), int(g["y"]+g["height"])),
                          (0, 255, 0), 2)
            cv2.putText(img, "GT", (int(g["x"]), max(20, int(g["y"])-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if fn in pred_d:
            p = pred_d[fn]
            cv2.rectangle(img, (int(p["x"]), int(p["y"])),
                          (int(p["x"]+p["width"]), int(p["y"]+p["height"])),
                          (0, 165, 255), 2)
            cv2.putText(img, "Pred", (int(p["x"]), max(20, int(p["y"])-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(img, f"Frame: {fn}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(img)

    writer.release()
    print(f"    Full video segment → {out_path}")


def bbox_iou(a, b):
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["width"]*a["height"] + b["width"]*b["height"] - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def convert_annotations_to_query_list(annotations):
    """Flat list of all valid queries from vq_val.json."""
    queries = []
    for v in annotations["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c["clip_uid"]
            clip_path = os.path.join(CLIPS_DIR, f"{cuid}.mp4")
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    if not q.get("is_valid", True):
                        continue
                    cache_path = os.path.join(CACHE_DIR, f"{aid}_{qid}.pt")
                    if not os.path.isfile(cache_path):
                        continue
                    queries.append({
                        "video_uid": vuid,
                        "clip_uid": cuid,
                        "clip_path": clip_path,
                        "annotation_uid": aid,
                        "query_set": qid,
                        "query_frame": q["query_frame"],
                        "visual_crop": q["visual_crop"],
                        "response_track": q.get("response_track", []),
                        "object_title": q.get("object_title", "unknown"),
                        "cache_path": cache_path,
                    })
    return queries


def format_predictions(annotations, all_pred_rts):
    """Build VQ2D challenge JSON from predicted response tracks."""
    predictions = {
        "version": annotations.get("version", "1.0"),
        "challenge": "ego4d_vq2d_challenge",
        "results": {"videos": []},
    }
    for v in annotations["videos"]:
        vid_pred = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_pred = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                apred = {"annotation_uid": aid, "query_sets": {}}
                for qid, qmeta in a["query_sets"].items():
                    key = (aid, qid)
                    qframe = int(qmeta.get("query_frame", 10**9))
                    if key in all_pred_rts and all_pred_rts[key]:
                        track = [
                            t for t in all_pred_rts[key]
                            if int(t["frame_number"]) < qframe
                        ]
                        bboxes = [
                            {
                                "fno": int(t["frame_number"]),
                                "x1": int(t["x"]),
                                "y1": int(t["y"]),
                                "x2": int(t["x"] + t["width"]),
                                "y2": int(t["y"] + t["height"]),
                            }
                            for t in track
                        ]
                        apred["query_sets"][qid] = {"bboxes": bboxes, "score": 1.0 if bboxes else 0.0}
                    else:
                        apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
                clip_pred["predictions"].append(apred)
            vid_pred["clips"].append(clip_pred)
        predictions["results"]["videos"].append(vid_pred)
    return predictions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "="*80)
    print("SAMURAI v4 — Enhanced Inference with Innovations #1 & #2")
    print("="*80)
    print(f"Innovation #1 (Object-Centric Cropping): {'ENABLED' if ENABLE_CROPPING else 'DISABLED'}")
    if ENABLE_CROPPING:
        print(f"  - Context multiplier: {CROP_CONTEXT_MULTIPLIER}x")
        print(f"  - Max zoom: {MAX_CROP_ZOOM}x")
    print(f"Innovation #2 (Semantic Validation): {'ENABLED' if ENABLE_SEMANTIC_GATE else 'DISABLED'}")
    if ENABLE_SEMANTIC_GATE:
        print(f"  - IoU threshold: {SEMANTIC_IOU_THRESHOLD} (very conservative)")
        print(f"  - Confidence threshold: {SEMANTIC_CONF_THRESHOLD}")
        print(f"  - Skip generic labels: {SEMANTIC_SKIP_GENERIC}")
    print("="*80 + "\n")

    # Load annotations
    print(f"Loading annotations from {VAL_JSON}")
    with open(VAL_JSON) as f:
        annotations = json.load(f)

    queries = convert_annotations_to_query_list(annotations)
    print(f"Queries with existing cache: {len(queries)}")
    if not queries:
        print("No cached queries found. Exiting.")
        return

    # Load existing progress
    existing_preds = {}
    PRED_PATH = OUT_DIR / "predictions.json"
    if PRED_PATH.exists():
        print(f"Loading existing predictions from {PRED_PATH}")
        try:
            with open(PRED_PATH) as f:
                old_data = json.load(f)
                for v in old_data.get("results", {}).get("videos", []):
                    for c in v.get("clips", []):
                        for a in c.get("predictions", []):
                            aid = a["annotation_uid"]
                            for qid, qd in a.get("query_sets", {}).items():
                                if qd.get("bboxes"):
                                    existing_preds[(aid, qid)] = qd["bboxes"]
            print(f"  Found {len(existing_preds)} completed queries.")
        except Exception as e:
            print(f"  Warning: Could not load existing predictions: {e}")

    # Load SAMURAI tracker
    tracker = SamuraiTracker(DEVICE)

    # Load Grounding DINO if semantic validation enabled
    if ENABLE_SEMANTIC_GATE:
        load_grounding_dino()

    # Group queries by clip
    from collections import defaultdict
    clip_queries = defaultdict(list)
    for q in queries:
        clip_queries[q["clip_uid"]].append(q)

    all_pred_rts = {}
    summary_rows = []
    semantic_stats = {"total_peaks": 0, "rejected": 0, "accepted": 0, "skipped": 0}

    # ── Priority ordering: process good+short queries first ───────────────
    priority_file = Path(os.path.join(SAMURAI_DIR, "priority_order.json"))
    if priority_file.exists():
        with open(priority_file) as f:
            priority_list = json.load(f)  # LOAD ALL 331 PRIORITY QUERIES
            
        priority_keys = set((p[0], p[1]) for p in priority_list)
        queries = [q for q in queries if (q["annotation_uid"], q["query_set"]) in priority_keys]
        
        # rebuild clip_queries
        clip_queries.clear()
        for q in queries:
            clip_queries[q["clip_uid"]].append(q)
            
        def clip_priority(cuid):
            best = len(priority_list)
            for q in clip_queries[cuid]:
                key = (q["annotation_uid"], q["query_set"])
                for i, p in enumerate(priority_list):
                    if (p[0], p[1]) == key:
                        best = min(best, i)
                        break
            return best
        clip_uids = sorted(clip_queries.keys(), key=clip_priority)
        print(f"  Using priority_order.json — priority queries first!")
    else:
        clip_uids = sorted(clip_queries.keys())
        print(f"  No priority_order.json found — using default clip order.")
    print(f"\nProcessing {len(clip_uids)} clips, {len(queries)} queries total …\n")

    for clip_idx, clip_uid in enumerate(clip_uids):
        cq_list = clip_queries[clip_uid]
        clip_path = cq_list[0]["clip_path"]

        if not os.path.isfile(clip_path):
            print(f"[SKIP] Clip not found: {clip_path}")
            for q in cq_list:
                all_pred_rts[(q["annotation_uid"], q["query_set"])] = []
            continue

        print(f"\n{'='*65}")
        print(f"Clip [{clip_idx+1}/{len(clip_uids)}]: {clip_uid}")
        print(f"  Queries in this clip: {len(cq_list)}")

        max_qframe = max(q["query_frame"] for q in cq_list)

        with tempfile.TemporaryDirectory(prefix="samurai_frames_") as tmpdir:
            print(f"  Extracting frames 0..{max_qframe-1} → {tmpdir}")
            n_extracted = extract_frames_to_dir(clip_path, tmpdir, max_frame=max_qframe)
            print(f"  Extracted {n_extracted} frames.")

            for q in cq_list:
                aid = q["annotation_uid"]
                qid = q["query_set"]
                obj = q["object_title"]
                qframe = q["query_frame"]
                vc = q["visual_crop"]
                # NOTE: gt_rt is used ONLY for visualization (similarity plot shading,
                # response_track.mp4 overlay, console IoU print). It is NEVER used for
                # peak selection, SAM2 initialization, or any tracking decision.
                gt_rt = q["response_track"]
                key = (aid, qid)
                label = f"{aid[:8]}_{qid}_{obj}"

                if key in existing_preds:
                    print(f"    [SKIP] Query {label} already processed.")
                    all_pred_rts[key] = [
                        {
                            "frame_number": b["fno"],
                            "x": b["x1"], "y": b["y1"],
                            "width": b["x2"] - b["x1"],
                            "height": b["y2"] - b["y1"],
                        }
                        for b in existing_preds[key]
                        if int(b["fno"]) < qframe
                    ]
                    continue

                print(f"\n  ── Query: {label} | query_frame={qframe} ──")

                # Load SiamRCNN cache
                print(f"    Loading cache: {q['cache_path']}")
                try:
                    cache = torch.load(q["cache_path"], weights_only=False)
                    ret_bboxes = cache["ret_bboxes"]
                    ret_scores = cache["ret_scores"]
                except Exception as e:
                    print(f"    [ERROR] Loading cache: {e}")
                    all_pred_rts[key] = []
                    continue

                if len(ret_scores) == 0:
                    print("    [SKIP] Empty cache.")
                    all_pred_rts[key] = []
                    continue

                # Signal processing
                peak_idx, score_signal, score_sm, peaks, peak_method = find_peak_frame(ret_scores, qframe)

                per_q_dir = OUT_DIR / "per_query" / label
                per_q_dir.mkdir(parents=True, exist_ok=True)

                # Get crops for visualization
                query_crop_img = None
                if vc:
                    vc_frame = vc.get("frame_number", qframe - 1)
                    q_img_path = os.path.join(tmpdir, f"{vc_frame:05d}.jpg")
                    if os.path.isfile(q_img_path):
                        q_full = cv2.imread(q_img_path)
                        qh, qw = q_full.shape[:2]
                        csx, csy = qw / vc.get("original_width", qw), qh / vc.get("original_height", qh)
                        cx, cy = int(vc["x"]*csx), int(vc["y"]*csy)
                        cw, ch = int(vc["width"]*csx), int(vc["height"]*csy)
                        query_crop_img = q_full[cy:cy+ch, cx:cx+cw]

                peak_crop_img = None
                if peak_idx is not None:
                    p_img_path = os.path.join(tmpdir, f"{peak_idx:05d}.jpg")
                    if os.path.isfile(p_img_path):
                        p_full = cv2.imread(p_img_path)
                        ph, pw = p_full.shape[:2]
                        psx, psy = pw / vc.get("original_width", pw), ph / vc.get("original_height", ph)
                        pb = ret_bboxes[peak_idx][int(np.argmax(ret_scores[peak_idx]))]
                        pbx1, pby1 = int(pb.x1*psx), int(pb.y1*psy)
                        pbx2, pby2 = int(pb.x2*psx), int(pb.y2*psy)
                        peak_crop_img = p_full[max(0,pby1):min(ph,pby2), max(0,pbx1):min(pw,pbx2)]

                gt_crop_img = None
                if gt_rt:
                    g0 = gt_rt[0]
                    g0_fn = g0["frame_number"]
                    g0_path = os.path.join(tmpdir, f"{g0_fn:05d}.jpg")
                    if os.path.isfile(g0_path):
                        g_full = cv2.imread(g0_path)
                        gh, gw = g_full.shape[:2]
                        gsx, gsy = gw / vc.get("original_width", gw), gh / vc.get("original_height", gh)
                        gx, gy = int(g0["x"]*gsx), int(g0["y"]*gsy)
                        gw_c, gh_c = int(g0["width"]*gsx), int(g0["height"]*gsy)
                        gt_crop_img = g_full[max(0,gy):min(gh,gy+gh_c), max(0,gx):min(gw,gx+gw_c)]

                save_similarity_plot(
                    score_signal, score_sm, peaks, peak_idx,
                    qframe, gt_rt,
                    str(per_q_dir / "similarity.png"),
                    title=f"{obj} | clip {clip_uid[:8]} | method={peak_method}",
                    query_crop=query_crop_img,
                    peak_crop=peak_crop_img,
                    gt_crop=gt_crop_img
                )

                if peak_idx is None:
                    print(f"    [WARN] No peak found even after fallback — empty track.")
                    all_pred_rts[key] = []
                    continue

                print(f"    Peak frame: {peak_idx} | score={score_sm[peak_idx]:.3f} | method={peak_method}")

                # Collect candidate peaks sorted by recency
                candidate_peaks = sorted(peaks, reverse=True) if len(peaks) > 0 else []
                if peak_idx not in candidate_peaks:
                    candidate_peaks.insert(0, peak_idx)
                elif candidate_peaks[0] != peak_idx:
                    candidate_peaks.remove(peak_idx)
                    candidate_peaks.insert(0, peak_idx)

                def get_bbox_xyxy_for_peak(pidx):
                    if pidx >= len(ret_bboxes) or not ret_bboxes[pidx]:
                        return None
                    pb_scores = ret_scores[pidx]
                    bi = int(np.argmax(pb_scores)) if pb_scores else 0
                    pbb = ret_bboxes[pidx][bi]
                    pfp = os.path.join(tmpdir, f"{pidx:05d}.jpg")
                    if not os.path.isfile(pfp):
                        return None
                    pimg = cv2.imread(pfp)
                    ih, iw = pimg.shape[:2]
                    _oh = vc.get("original_height", ih)
                    _ow = vc.get("original_width", iw)
                    _sx, _sy = iw / _ow, ih / _oh
                    _bx1 = max(0, int(pbb.x1 * _sx))
                    _by1 = max(0, int(pbb.y1 * _sy))
                    _bx2 = min(iw, int(pbb.x2 * _sx))
                    _by2 = min(ih, int(pbb.y2 * _sy))
                    if _bx2 <= _bx1 or _by2 <= _by1:
                        return None
                    return pimg, (_bx1, _by1, _bx2, _by2)

                # Innovation #2: Semantic Cross-Validation + Innovation #1+#3: SAM2 Quality Gate with Cropping
                validated_peaks = []
                for cp in candidate_peaks[:5]:
                    result = get_bbox_xyxy_for_peak(cp)
                    if result is None:
                        continue
                    cp_img, cp_bbox = result
                    
                    # Innovation #2: Semantic validation FIRST (cheaper than SAM2)
                    is_sem_valid, gdino_bbox, gdino_conf, sem_iou = semantic_validate_peak(
                        cp_img, cp_bbox, obj
                    )
                    
                    if ENABLE_SEMANTIC_GATE and not is_sem_valid:
                        semantic_stats["rejected"] += 1
                        print(f"    [SEMANTIC REJECT] Peak {cp}: IoU={sem_iou:.3f} < {SEMANTIC_IOU_THRESHOLD}")
                        continue
                    elif ENABLE_SEMANTIC_GATE and obj.lower() not in GENERIC_LABELS:
                        semantic_stats["accepted"] += 1
                        print(f"    [SEMANTIC OK] Peak {cp}: IoU={sem_iou:.3f}, conf={gdino_conf:.3f}")
                    else:
                        semantic_stats["skipped"] += 1
                    
                    semantic_stats["total_peaks"] += 1
                    
                    # Innovation #1+#3: SAM2 Quality Gate with Object-Centric Cropping
                    try:
                        mask, conf, compact, crop_info = tracker.validate_peak_with_crop(
                            cp_img, cp_bbox, obj
                        )
                        crop_used = " [CROPPED]" if crop_info.get("crop_used") else ""
                        print(f"    [GATE{crop_used}] Peak {cp}: confidence={conf:.3f}, compactness={compact:.3f}")
                        validated_peaks.append((cp, mask, conf, compact))
                    except Exception as e:
                        print(f"    [GATE] Peak {cp} failed: {e}")

                if not validated_peaks:
                    print("    [WARN] No peaks could be validated by semantic check — using BEST SIAMRCNN PEAK as fallback.")
                    best_cp = candidate_peaks[0] if len(candidate_peaks) > 0 else peak_idx
                    if best_cp is not None:
                        res = get_bbox_xyxy_for_peak(best_cp)
                        if res:
                            cp_img, cp_bbox = res
                            try:
                                mask, conf, compact, crop_info = tracker.validate_peak_with_crop(cp_img, cp_bbox, obj)
                                crop_used = " [CROPPED]" if crop_info.get("crop_used") else ""
                                print(f"    [FALLBACK GATE{crop_used}] Peak {best_cp}: confidence={conf:.3f}, compactness={compact:.3f}")
                                validated_peaks.append((best_cp, mask, conf, compact))
                            except Exception as e:
                                print(f"    [FALLBACK GATE] Peak {best_cp} failed: {e}")

                if not validated_peaks:
                    print("    [WARN] Fallback failed — empty track.")
                    all_pred_rts[key] = []
                    continue

                # Pick best peak
                selected = None
                for vp in validated_peaks:
                    if vp[2] > 0.85 and vp[3] > 0.10:
                        selected = vp
                        break
                    elif vp[2] > 0.40 and vp[3] > 0.20:
                        selected = vp
                        break
                if selected is None:
                    selected = validated_peaks[0]
                    print(f"    [GATE] All peaks failed quality gate. Using most recent: {selected[0]}")

                sel_peak_idx, sel_mask = selected[0], selected[1]
                sel_conf, sel_compact = selected[2], selected[3]
                print(f"    Selected peak: {sel_peak_idx} (conf={sel_conf:.3f}, compact={sel_compact:.3f})")

                # Multi-hypothesis tracking
                second_peak = None
                for vp in validated_peaks:
                    if vp[0] != sel_peak_idx:
                        if (vp[2] > 0.85 and vp[3] > 0.10) or (vp[2] > 0.40 and vp[3] > 0.20):
                            second_peak = vp
                            break

                def run_tracking(p_idx, p_mask):
                    try:
                        backward, forward = tracker.track(
                            tmpdir, p_idx, p_mask, return_separate=True
                        )
                        bwd_d = {t["frame_number"]: t for t in backward}
                        fwd_d = {t["frame_number"]: t for t in forward}
                        common_frames = set(bwd_d) & set(fwd_d)
                        
                        filtered = []
                        seen = set()
                        for t in backward + forward:
                            fn = t["frame_number"]
                            if fn in seen:
                                continue
                            seen.add(fn)
                            if fn in common_frames:
                                iou = bbox_iou(bwd_d[fn], fwd_d[fn])
                                if iou < 0.40:
                                    continue
                            filtered.append(t)
                        filtered.sort(key=lambda x: x["frame_number"])
                        return filtered
                    except Exception as e:
                        print(f"    [ERROR] Tracking from peak {p_idx}: {e}")
                        return []

                use_multi = (
                    second_peak is not None
                    and abs(sel_conf - second_peak[2]) < 0.10
                    and sel_compact < 0.50
                )

                if use_multi:
                    print(f"    [MULTI] Uncertain → tracking from both {sel_peak_idx} and {second_peak[0]}")
                    track_1 = run_tracking(sel_peak_idx, sel_mask)
                    track_2 = run_tracking(second_peak[0], second_peak[1])

                    def track_quality(t):
                        if not t:
                            return (0, 0.0)
                        avg_conf = np.mean([f.get("mask_confidence", 0.0) for f in t])
                        return (len(t), avg_conf)

                    q1, q2 = track_quality(track_1), track_quality(track_2)
                    def score(q): return q[0] * 1.0 + q[1] * 3.0
                    pred_track = track_1 if score(q1) >= score(q2) else track_2
                    chosen = "track_1" if score(q1) >= score(q2) else "track_2"
                    print(f"    [MULTI] Chose {chosen}: len={len(pred_track)} (scores q1={score(q1):.2f} q2={score(q2):.2f})")
                else:
                    pred_track = run_tracking(sel_peak_idx, sel_mask)

                # Track quality validation & retry
                MIN_TRACK_LEN = 3
                MIN_AVG_CONF = 0.25
                if pred_track:
                    avg_mc = np.mean([f.get("mask_confidence", 0.0) for f in pred_track])
                else:
                    avg_mc = 0.0

                if (len(pred_track) < MIN_TRACK_LEN or avg_mc < MIN_AVG_CONF) and len(validated_peaks) > 1:
                    retry_peak = None
                    for vp in validated_peaks:
                        if vp[0] != sel_peak_idx:
                            retry_peak = vp
                            break
                    if retry_peak is not None:
                        print(f"    [RETRY] Track too short ({len(pred_track)}) or low conf ({avg_mc:.3f}). Retrying with peak {retry_peak[0]}")
                        retry_track = run_tracking(retry_peak[0], retry_peak[1])
                        if len(retry_track) > len(pred_track):
                            pred_track = retry_track
                            print(f"    [RETRY] Retry succeeded: {len(pred_track)} frames")

                # Console-only diagnostic print (GT frame count for reference, not used in scoring)
                print(f"    Final track: {len(pred_track)} frames | GT: {len(gt_rt)} frames")
                all_pred_rts[key] = pred_track  # Only pred_track is saved — gt_rt is never written

                # Save response-track video
                vid_path = str(per_q_dir / "response_track.mp4")
                save_response_track_video(tmpdir, pred_track, gt_rt, vid_path, max_frame=qframe)

                # Quick diagnostic metrics (console only — NOT saved to predictions.json)
                if gt_rt and pred_track:
                    pred_d = {t["frame_number"]: t for t in pred_track}
                    gt_d = {t["frame_number"]: t for t in gt_rt}
                    overlap = set(pred_d) & set(gt_d)
                    ious = [bbox_iou(pred_d[f], gt_d[f]) for f in overlap]
                    mean_iou = float(np.mean(ious)) if ious else 0.0
                    coverage = len(overlap) / len(gt_d) if gt_d else 0.0
                else:
                    mean_iou, coverage = 0.0, 0.0

                print(f"    Mean IoU={mean_iou:.3f} | Coverage={coverage:.3f}")
                summary_rows.append({
                    "clip": clip_uid[:8], "annotation": aid[:8],
                    "qset": qid, "object": obj,
                    "peak_frame": sel_peak_idx, "method": peak_method,
                    "pred_frames": len(pred_track), "gt_frames": len(gt_rt),
                    "mean_iou": mean_iou, "coverage": coverage,
                })

                # Intermediate save
                predictions = format_predictions(annotations, all_pred_rts)
                pred_path = str(OUT_DIR / "predictions.json")
                with open(pred_path, "w") as f:
                    json.dump(predictions, f)

    # Final save
    predictions = format_predictions(annotations, all_pred_rts)
    pred_path = str(OUT_DIR / "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    print(f"\nChallenge predictions → {pred_path}")

    proc_clips = {q["clip_uid"] for q in queries}
    processed_predictions = {
        "version": predictions["version"],
        "challenge": predictions["challenge"],
        "results": {
            "videos": [
                {
                    "video_uid": v["video_uid"],
                    "clips": [
                        c for c in v["clips"]
                        if c["clip_uid"] in proc_clips
                    ],
                }
                for v in predictions["results"]["videos"]
                if any(c["clip_uid"] in proc_clips for c in v["clips"])
            ]
        },
    }
    filt_path = str(OUT_DIR / "predictions_filtered.json")
    with open(filt_path, "w") as f:
        json.dump(processed_predictions, f)
    print(f"Filtered predictions → {filt_path}")

    # Summary
    print(f"\n{'='*95}")
    print("=== SAMURAI v4 ENHANCED INFERENCE SUMMARY ===")
    print(f"{'='*95}")
    
    if ENABLE_SEMANTIC_GATE and semantic_stats["total_peaks"] > 0:
        print("\nSemantic Validation Statistics:")
        print(f"  Total peaks evaluated: {semantic_stats['total_peaks']}")
        print(f"  Rejected (low semantic IoU): {semantic_stats['rejected']}")
        print(f"  Accepted: {semantic_stats['accepted']}")
        print(f"  Skipped (generic labels): {semantic_stats['skipped']}")
        rejection_rate = 100 * semantic_stats['rejected'] / semantic_stats['total_peaks']
        print(f"  Rejection rate: {rejection_rate:.1f}%")
        print()
    
    fmt = "  {:<10} {:<10} {:<5} {:<15} {:>7} {:>8} {:>6} {:>5} {:>7} {:>7}"
    print(fmt.format("Clip","Ann","QSet","Object","Peak","Method","Pred","GT","IoU","Cover"))
    print("  " + "-"*93)
    for r in summary_rows:
        print(fmt.format(
            r["clip"], r["annotation"], r["qset"], r["object"][:15],
            str(r["peak_frame"]), str(r.get("method", "?"))[:8],
            str(r["pred_frames"]), str(r["gt_frames"]),
            f"{r['mean_iou']:.3f}", f"{r['coverage']:.3f}"
        ))
    if summary_rows:
        print("  " + "-"*93)
        print(fmt.format(
            "OVERALL","","","",
            "","",
            str(sum(r["pred_frames"] for r in summary_rows)),
            str(sum(r["gt_frames"] for r in summary_rows)),
            f"{np.mean([r['mean_iou'] for r in summary_rows]):.3f}",
            f"{np.mean([r['coverage'] for r in summary_rows]):.3f}"
        ))
    print(f"{'='*95}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
