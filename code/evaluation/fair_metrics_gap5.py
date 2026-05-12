"""
fair_metrics_gap5.py
====================
Evaluates GAP=5 on the 181 queries it completed inference on.
Baselines (samuriaNew, samuriaGDino) are also scored on that SAME 181-query subset.
Queries missing from a baseline = score 0 (FAIR, no cherry-picking).
"""
import sys, os, json

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(ABLATION_DIR, "../.."))
EPISODIC_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "../.."))
sys.path.insert(0, ABLATION_DIR)

from vq2d.metrics.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack, BBox

VAL_JSON      = "/DATA/Group25/data/v2/annoVQ2D/vq_val.json"
PRED_G5       = os.path.join(PACKAGE_ROOT, "results/predictions.json")
PRED_NEW      = os.path.join(EPISODIC_ROOT, "samuriaNew/new_results/predictions.json")
PRED_GDINO    = os.path.join(EPISODIC_ROOT, "samuriaGDino/new_results_v4/predictions.json")
SUBSET_KEYS_JSON = os.path.join(PACKAGE_ROOT, "results/eval_subset_keys_203.json")

# Known published SiamRCNN baseline numbers (hardcoded)
SIAM_RCNN = {
    "Temporal AP @ IoU=0.25:0.95":      0.1340,
    "Temporal AP @ IoU=0.25":           0.2250,
    "SpatioTemporal AP @ IoU=0.25:0.95":0.0580,
    "SpatioTemporal AP @ IoU=0.25":     0.1530,
    "Tracking recovery @ IoU=0.50":    32.919,
    "Success @ IoU=0.05":              43.244,
}


def load_pred_map(path):
    """Returns {(aid, qid): [{fno,x1,y1,x2,y2}, ...]}"""
    pred_map = {}
    if not os.path.isfile(path):
        print(f"  [MISSING] {path}")
        return pred_map
    with open(path) as f:
        data = json.load(f)
    for v in data.get("results", {}).get("videos", []):
        for c in v.get("clips", []):
            for a in c.get("predictions", []):
                aid = a.get("annotation_uid", "")
                for qid, qd in a.get("query_sets", {}).items():
                    bboxes = qd.get("bboxes", [])
                    if bboxes:
                        pred_map[(aid, str(qid))] = bboxes
    print(f"  Loaded {len(pred_map):4d} predictions  ← {os.path.basename(path)}")
    return pred_map


def interpolate_bboxes(bboxes):
    """Fill gaps so frames are contiguous — required by ResponseTrack."""
    if len(bboxes) <= 1:
        return bboxes
    bboxes = sorted(bboxes, key=lambda b: b['fno'])
    b_map  = {b['fno']: b for b in bboxes}
    out    = []
    for f in range(bboxes[0]['fno'], bboxes[-1]['fno'] + 1):
        if f in b_map:
            out.append(b_map[f])
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
    """Build ResponseTrack from pred_map. Returns empty RT if key missing."""
    boxes = pred_map.get(key, [])
    if not boxes:
        return ResponseTrack([], score=0.0)
    bboxes_out = []
    for b in boxes:
        if "fno" in b:
            bboxes_out.append(BBox(b["fno"], b["x1"], b["y1"], b["x2"], b["y2"]))
        elif "frame_number" in b:
            bboxes_out.append(BBox(
                b["frame_number"], b["x"], b["y"],
                b["x"] + b["width"], b["y"] + b["height"]
            ))
    if not bboxes_out:
        return ResponseTrack([], score=0.0)
    # Normalise to dict format for interpolation
    dict_bboxes = [{'fno': b.fno, 'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2}
                   for b in bboxes_out]
    filled = interpolate_bboxes(dict_bboxes)
    return ResponseTrack(
        [BBox(b['fno'], b['x1'], b['y1'], b['x2'], b['y2']) for b in filled],
        score=1.0
    )


def siam_baseline_for_metric(metric_name: str) -> float:
    name = " ".join(metric_name.split()).lower()
    if name.startswith("spatiotemporal ap") and "0.25:0.95" in name:
        return SIAM_RCNN["SpatioTemporal AP @ IoU=0.25:0.95"]
    if name.startswith("spatiotemporal ap") and "0.25" in name:
        return SIAM_RCNN["SpatioTemporal AP @ IoU=0.25"]
    if name.startswith("temporal ap") and "0.25:0.95" in name:
        return SIAM_RCNN["Temporal AP @ IoU=0.25:0.95"]
    if name.startswith("temporal ap") and "0.25" in name:
        return SIAM_RCNN["Temporal AP @ IoU=0.25"]
    if name.startswith("tracking % recovery"):
        return SIAM_RCNN["Tracking recovery @ IoU=0.50"]
    if name.startswith("success"):
        return SIAM_RCNN["Success @ IoU=0.05"]
    return 0.0


def main():
    print("=" * 70)
    print("  GAP=5 FAIR METRICS  (evaluated on GAP=5 completed subset)")
    print("=" * 70)

    print("\nLoading predictions...")
    g5_map    = load_pred_map(PRED_G5)
    new_map   = load_pred_map(PRED_NEW)
    gdino_map = load_pred_map(PRED_GDINO)

    # The evaluation set = ALL queries GAP=5 attempted (per_query folders),
    # including ones where tracking failed (empty output). Those score 0 for GAP=5.
    # This is the TRULY FAIR denominator.
    PER_QUERY_G5 = os.path.join(PACKAGE_ROOT, "results/per_query_similarity")

    print("\nBuilding label_map from vq_val.json for per_query folder mapping...")
    with open(VAL_JSON) as f:
        val_tmp = json.load(f)
    label_map = {}
    for v in val_tmp["videos"]:
        for c in v["clips"]:
            cuid = c.get("clip_uid", "")
            if not cuid:
                continue
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    obj = q.get("object_title", "unknown")
                    label = f"{aid[:8]}_{qid}_{obj}"
                    label_map[label] = (aid, str(qid))

    eval_keys = set()
    if os.path.isfile(SUBSET_KEYS_JSON):
        with open(SUBSET_KEYS_JSON) as f:
            for aid, qid in json.load(f):
                eval_keys.add((aid, str(qid)))
        print(f"  Loaded fixed GAP=5 subset from {os.path.basename(SUBSET_KEYS_JSON)}")
    elif os.path.isdir(PER_QUERY_G5):
        for folder in os.listdir(PER_QUERY_G5):
            fpath = os.path.join(PER_QUERY_G5, folder)
            if os.path.isdir(fpath) and folder in label_map:
                eval_keys.add(label_map[folder])

    # Also add any non-empty preds not captured by folder scan
    for k in g5_map:
        eval_keys.add(k)

    print(f"\nEvaluation subset: {len(eval_keys)} queries")
    print(f"  (per_query folders mapped: {len(eval_keys)} | non-empty preds: {len(g5_map)})")
    print(f"  Queries with empty/failed tracks (score=0 for GAP=5): {len(eval_keys) - len(g5_map)}")

    print("\nLoading vq_val.json for ground truth...")
    with open(VAL_JSON) as f:
        val = json.load(f)

    # Build GT lookup: {(aid, qid): (response_track, visual_crop)}
    gt_lookup = {}
    for v in val["videos"]:
        for c in v["clips"]:
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    gt_lookup[(aid, str(qid))] = q

    print(f"GT lookup has {len(gt_lookup)} total queries")

    # Build arrays for metrics computation
    gt_tracks    = []
    g5_tracks    = []
    new_tracks   = []
    gdino_tracks = []
    vc_bboxes    = []

    skipped = 0
    g5_empty = 0; new_empty = 0; gdino_empty = 0

    for key in sorted(eval_keys):
        q = gt_lookup.get(key)
        if q is None:
            skipped += 1
            continue

        vc = q.get("visual_crop")
        if vc is None:
            skipped += 1
            continue

        gt_boxes = q.get("response_track", [])
        if not gt_boxes:
            skipped += 1
            continue

        gt_rt = ResponseTrack([
            BBox(t["frame_number"], t["x"], t["y"],
                 t["x"] + t["width"], t["y"] + t["height"])
            for t in gt_boxes
        ])

        vc_bbox = BBox(
            vc["frame_number"], vc["x"], vc["y"],
            vc["x"] + vc["width"], vc["y"] + vc["height"]
        )

        g5_rt    = make_rt(g5_map,    key)
        new_rt   = make_rt(new_map,   key)
        gdino_rt = make_rt(gdino_map, key)

        if not g5_rt.bboxes:    g5_empty    += 1
        if not new_rt.bboxes:   new_empty   += 1
        if not gdino_rt.bboxes: gdino_empty += 1

        gt_tracks.append(gt_rt)
        g5_tracks.append([g5_rt])
        new_tracks.append([new_rt])
        gdino_tracks.append([gdino_rt])
        vc_bboxes.append(vc_bbox)

    N = len(gt_tracks)
    print(f"\nFinal evaluation: {N} queries  (skipped {skipped} — no GT/VC)")
    print(f"  GAP=5    empty tracks: {g5_empty}/{N}")
    print(f"  samuriaNew empty:      {new_empty}/{N}  (queries outside its processed set)")
    print(f"  samuriaGDino empty:    {gdino_empty}/{N}")

    if N == 0:
        print("\nERROR: No queries to evaluate!")
        return

    print(f"\nComputing metrics for {N} queries ...")
    m_g5    = compute_visual_query_metrics(g5_tracks,    gt_tracks, vc_bboxes)
    m_new   = compute_visual_query_metrics(new_tracks,   gt_tracks, vc_bboxes)
    m_gdino = compute_visual_query_metrics(gdino_tracks, gt_tracks, vc_bboxes)

    print("\n" + "=" * 120)
    print(f"  RESULTS: GAP=5 vs Baselines  [{N} queries | same subset for all]")
    print("=" * 120)
    print(f"  {'Metric':<52} | {'SiamRCNN':>10} | {'samuriaNew':>12} | {'samuriaGDino':>12} | {'GAP=5':>10} | Delta(G5-New)")
    print("  " + "-" * 116)

    for pair in m_g5.keys():
        d_g5    = m_g5[pair]
        d_new   = m_new.get(pair, {})
        d_gdino = m_gdino.get(pair, {})

        for k in d_g5.keys():
            v_g5    = d_g5[k]
            v_new   = d_new.get(k, 0.0)
            v_gdino = d_gdino.get(k, 0.0)
            # Match hardcoded SiamRCNN key
            v_siam = siam_baseline_for_metric(k)

            delta = v_g5 - v_new
            flag  = "✅" if delta > 0.001 else ("❌" if delta < -0.001 else "➖")

            print(f"  {k:<52} | {v_siam:>10.4f} | {v_new:>12.4f} | {v_gdino:>12.4f} | {v_g5:>10.4f} | {delta:>+8.4f} {flag}")

    print("=" * 120)
    print(f"\nNote: {g5_empty} queries where GAP=5 tracking failed → score 0 (counted, NOT ignored).")
    print(f"      {new_empty} queries where samuriaNew has no prediction for this subset → score 0.")
    print(f"      All {N} queries contribute to the denominator. FAIR comparison.")


if __name__ == "__main__":
    main()
