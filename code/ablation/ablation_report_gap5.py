import sys, os, json
import numpy as np
import matplotlib.pyplot as plt

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(ABLATION_DIR, "../.."))
EPISODIC_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "../.."))
EVAL_CODE_DIR = os.path.join(PACKAGE_ROOT, "code/evaluation")
sys.path.insert(0, EVAL_CODE_DIR)

from vq2d.metrics.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack, BBox

VAL_JSON      = "/DATA/Group25/data/v2/annoVQ2D/vq_val.json"
PRED_G5       = os.path.join(PACKAGE_ROOT, "results/predictions.json")
PRED_NEW      = os.path.join(EPISODIC_ROOT, "samuriaNew/new_results/predictions.json")
PRED_GDINO    = os.path.join(EPISODIC_ROOT, "samuriaGDino/new_results_v4/predictions.json")
SUBSET_KEYS_JSON = os.path.join(PACKAGE_ROOT, "results/eval_subset_keys_203.json")

# Hardcoded SiamRCNN Baseline from User
SIAM_RCNN = {
    "stAP @ 0.25": 0.153,
    "stAP": 0.058,
    "tAP @ 0.25": 0.225,
    "tAP": 0.134,
    "recall": 32.919,
    "success": 43.244,
}

def load_pred_map(path):
    pred_map = {}
    if not os.path.isfile(path): return pred_map
    with open(path) as f:
        data = json.load(f)
    for v in data.get("results", {}).get("videos", []):
        for c in v.get("clips", []):
            for a in c.get("predictions", []):
                aid = a.get("annotation_uid", "")
                for qid, qd in a.get("query_sets", {}).items():
                    bboxes = qd.get("bboxes", [])
                    if bboxes: pred_map[(aid, str(qid))] = bboxes
    return pred_map

def interpolate_bboxes(bboxes):
    if len(bboxes) <= 1: return bboxes
    bboxes = sorted(bboxes, key=lambda b: b['fno'])
    b_map  = {b['fno']: b for b in bboxes}
    out    = []
    for f in range(bboxes[0]['fno'], bboxes[-1]['fno'] + 1):
        if f in b_map: out.append(b_map[f])
        else:
            prev = max(b for b in b_map if b < f)
            nxt  = min(b for b in b_map if b > f)
            r    = (f - prev) / (nxt - prev)
            p, n = b_map[prev], b_map[nxt]
            out.append({'fno': f,
                        'x1': p['x1'] + (n['x1']-p['x1'])*r,
                        'y1': p['y1'] + (n['y1']-p['y1'])*r,
                        'x2': p['x2'] + (n['x2']-p['x2'])*r,
                        'y2': p['y2'] + (n['y2']-p['y2'])*r})
    return out

def make_rt(pred_map, key):
    boxes = pred_map.get(key, [])
    if not boxes: return ResponseTrack([], score=0.0)
    bboxes_out = []
    for b in boxes:
        if "fno" in b: bboxes_out.append(BBox(b["fno"], b["x1"], b["y1"], b["x2"], b["y2"]))
        elif "frame_number" in b:
            bboxes_out.append(BBox(b["frame_number"], b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]))
    if not bboxes_out: return ResponseTrack([], score=0.0)
    dict_bboxes = [{'fno': b.fno, 'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2} for b in bboxes_out]
    filled = interpolate_bboxes(dict_bboxes)
    return ResponseTrack([BBox(b['fno'], b['x1'], b['y1'], b['x2'], b['y2']) for b in filled], score=1.0)

def main():
    print("Running Ablation Study on 203 queries subset...")
    g5_map = load_pred_map(PRED_G5)
    new_map = load_pred_map(PRED_NEW)
    gdino_map = load_pred_map(PRED_GDINO)

    # Use the 203 folders logic
    PER_QUERY_G5 = os.path.join(PACKAGE_ROOT, "results/per_query_similarity")
    with open(VAL_JSON) as f: val_data = json.load(f)
    label_map = {}
    gt_lookup = {}
    for v in val_data["videos"]:
        for c in v["clips"]:
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    obj = q.get("object_title", "unknown")
                    label = f"{aid[:8]}_{qid}_{obj}"
                    label_map[label] = (aid, str(qid))
                    gt_lookup[(aid, str(qid))] = q

    eval_keys = set()
    if os.path.isfile(SUBSET_KEYS_JSON):
        with open(SUBSET_KEYS_JSON) as f:
            for aid, qid in json.load(f):
                eval_keys.add((aid, str(qid)))
    elif os.path.isdir(PER_QUERY_G5):
        for folder in os.listdir(PER_QUERY_G5):
            if folder in label_map:
                eval_keys.add(label_map[folder])
    for k in g5_map: eval_keys.add(k)

    gt_tracks, g5_tracks, new_tracks, gdino_tracks, vc_bboxes = [], [], [], [], []
    for key in sorted(eval_keys):
        q = gt_lookup.get(key)
        if not q or not q.get("visual_crop") or not q.get("response_track"): continue
        gt_rt = ResponseTrack([BBox(t["frame_number"], t["x"], t["y"], t["x"] + t["width"], t["y"] + t["height"]) for t in q["response_track"]])
        vc = q["visual_crop"]
        vc_bbox = BBox(vc["frame_number"], vc["x"], vc["y"], vc["x"] + vc["width"], vc["y"] + vc["height"])
        gt_tracks.append(gt_rt)
        g5_tracks.append([make_rt(g5_map, key)])
        new_tracks.append([make_rt(new_map, key)])
        gdino_tracks.append([make_rt(gdino_map, key)])
        vc_bboxes.append(vc_bbox)

    print(f"Evaluated {len(gt_tracks)} queries.")
    m_g5 = compute_visual_query_metrics(g5_tracks, gt_tracks, vc_bboxes)
    m_new = compute_visual_query_metrics(new_tracks, gt_tracks, vc_bboxes)
    m_gdino = compute_visual_query_metrics(gdino_tracks, gt_tracks, vc_bboxes)

    # Extracted flattened metrics for plot
    # Mapping keys to standard names
    # Use the exact lower-cased collapsed strings from the debug output
    key_map = [
        ("spatiotemporal ap @ iou=0.25:0.95", "stAP"),
        ("spatiotemporal ap @ iou=0.25", "stAP @ 0.25"),
        ("temporal ap @ iou=0.25:0.95", "tAP"),
        ("temporal ap @ iou=0.25", "tAP @ 0.25"),
        ("tracking % recovery (max scr) @ iou=0.50", "recall"),
        ("success (max scr) @ iou=0.05", "success")
    ]

    results = {"SiamRCNN": SIAM_RCNN, "samuriaNew (G15)": {}, "samuriaGDino (G10)": {}, "GAP=5": {}}
    
    # Fill actual results
    def extract(m, target, method_name):
        for pair, d in m.items():
            for k_raw, v in d.items():
                k_clean = " ".join(k_raw.split()).lower()
                for km_key, km_val in key_map:
                    if km_key in k_clean:
                        target[km_val] = v
                        break

    extract(m_g5, results["GAP=5"], "GAP=5")
    extract(m_new, results["samuriaNew (G15)"], "samuriaNew (G15)")
    extract(m_gdino, results["samuriaGDino (G10)"], "samuriaGDino (G10)")

    # Printing Table
    print("\n" + "="*120)
    print(f"{'Method':<20} | {'stAP@0.25':<10} | {'stAP':<10} | {'tAP@0.25':<10} | {'tAP':<10} | {'Recall':<10} | {'Success':<10}")
    print("-" * 120)
    metrics_list = ["stAP @ 0.25", "stAP", "tAP @ 0.25", "tAP", "recall", "success"]
    for method in ["SiamRCNN", "samuriaNew (G15)", "samuriaGDino (G10)", "GAP=5"]:
        row = [f"{method:<20}"]
        for m in metrics_list:
            val = results[method].get(m, 0.0)
            row.append(f"{val:>10.4f}")
        print(" | ".join(row))
    print("="*120)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(results.keys())
    x = np.arange(len(metrics_list))
    width = 0.2

    for i, method in enumerate(methods):
        vals = [results[method].get(m, 0.0) for m in metrics_list]
        # Recall/Success are often 0-100 in table but 0-1 in plot? 
        # Actually compute_visual_query_metrics returns 0-1 for AP and 0-100 for recovery/success?
        # Let's check. 52.6516 sounds like percentage.
        # I'll scale APs to 100 for consistent plotting.
        vals_scaled = [v if i >= 4 else v*100 for i, v in enumerate(vals)]
        ax.bar(x + i*width - 1.5*width, vals_scaled, width, label=method)

    ax.set_ylabel('Score (scaled to 100)')
    ax.set_title('SAMURAI Ablation Study: MAX_GAP Threshold Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ablation_plot.png')
    print("\nPlot saved to ablation_plot.png")

if __name__ == "__main__":
    main()
