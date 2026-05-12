"""
merge_results.py — Fixed version
Merges all per_query/track.json files into results/predictions.json
"""
import json
import os
from pathlib import Path

VAL_JSON = "/DATA/Group25/data/v2/annoVQ2D/vq_val.json"

def update_predictions(data, video_uid, clip_uid, annotation_uid, query_set, bboxes):
    """Update the predictions.json structure in-place. Returns True if found."""
    for v in data["results"]["videos"]:
        if v["video_uid"] == video_uid:
            for c in v["clips"]:
                if c["clip_uid"] == clip_uid:
                    for a in c["predictions"]:
                        if a["annotation_uid"] == annotation_uid:
                            a["query_sets"][str(query_set)] = {
                                "bboxes": bboxes,
                                "score": 1.0 if bboxes else 0.0
                            }
                            return True
    return False


def load_track_json(track_file):
    """
    Load a track.json file and convert to bbox format.
    track.json format: [{frame_number, x, y, width, height, score, mask_confidence}, ...]
    Output format:     [{fno, x1, y1, x2, y2}, ...]
    """
    try:
        with open(track_file) as f:
            track_data = json.load(f)
        if not track_data:
            return None, "empty_list"

        bboxes = []
        for t in track_data:
            # Validate required fields
            if not all(k in t for k in ["frame_number", "x", "y", "width", "height"]):
                return None, f"missing_fields_in_entry: {list(t.keys())}"
            bboxes.append({
                "fno": int(t["frame_number"]),
                "x1":  int(t["x"]),
                "y1":  int(t["y"]),
                "x2":  int(t["x"] + t["width"]),
                "y2":  int(t["y"] + t["height"]),
            })
        return bboxes, "ok"
    except json.JSONDecodeError as e:
        return None, f"json_error: {e}"
    except Exception as e:
        return None, f"read_error: {e}"


def main():
    package_root = Path(__file__).resolve().parents[2]
    out_dir = package_root / "results"
    main_pred = out_dir / "predictions.json"

    if not main_pred.exists():
        print(f"ERROR: {main_pred} not found. Cannot merge.")
        return

    print(f"Loading {main_pred} ...")
    with open(main_pred) as f:
        all_data = json.load(f)

    # Count current state before merging
    pre_count = 0
    for v in all_data["results"]["videos"]:
        for c in v["clips"]:
            for a in c["predictions"]:
                for qid, qdata in a["query_sets"].items():
                    if qdata.get("bboxes"):
                        pre_count += 1
    print(f"  predictions.json before merge: {pre_count} non-empty queries")

    # Build label_map from vq_val.json
    # label → (annotation_uid, query_set_id, video_uid, clip_uid)
    print(f"Loading {VAL_JSON} ...")
    with open(VAL_JSON) as f:
        val_data = json.load(f)

    label_map = {}
    for v in val_data["videos"]:
        vuid = v["video_uid"]
        for c in v["clips"]:
            cuid = c.get("clip_uid", "")
            if not cuid:
                continue
            for a in c["annotations"]:
                aid = a["annotation_uid"]
                for qid, q in a["query_sets"].items():
                    obj = q.get("object_title", "unknown")
                    # Label format must match what inference script saves:
                    # f"{aid[:8]}_{qid}_{obj}"
                    label = f"{aid[:8]}_{qid}_{obj}"
                    label_map[label] = (aid, qid, vuid, cuid)

    print(f"  label_map has {len(label_map)} entries")

    # Merge from per_query folders
    per_query_dir = out_dir / "per_query"
    if not per_query_dir.exists():
        print(f"ERROR: {per_query_dir} does not exist.")
        return

    folders = sorted(per_query_dir.iterdir())
    total_folders = sum(1 for f in folders if f.is_dir())
    print(f"\nFound {total_folders} per_query folders. Merging...")

    stats = {
        "merged_ok": 0,
        "already_had_data": 0,
        "no_track_file": 0,
        "empty_track": 0,
        "track_error": 0,
        "label_not_in_map": 0,
        "update_failed": 0,
    }
    label_mismatches = []

    for pq_folder in sorted(per_query_dir.iterdir()):
        if not pq_folder.is_dir():
            continue

        label = pq_folder.name

        # Check label mapping
        if label not in label_map:
            stats["label_not_in_map"] += 1
            label_mismatches.append(label)
            continue

        aid, qid, vuid, cuid = label_map[label]
        track_file = pq_folder / "track.json"

        if not track_file.exists():
            stats["no_track_file"] += 1
            print(f"  [WARN] No track.json in: {label}")
            continue

        bboxes, status = load_track_json(track_file)

        if status != "ok":
            if status == "empty_list":
                stats["empty_track"] += 1
                print(f"  [EMPTY] {label}: track.json is empty list")
            else:
                stats["track_error"] += 1
                print(f"  [ERROR] {label}: {status}")
            # Still update with empty bboxes so query is "registered"
            update_predictions(all_data, vuid, cuid, aid, qid, [])
            continue

        # Successful merge
        if update_predictions(all_data, vuid, cuid, aid, qid, bboxes):
            stats["merged_ok"] += 1
        else:
            stats["update_failed"] += 1
            print(f"  [FAIL] Could not find slot in predictions.json for: {label}")
            print(f"         aid={aid}, qid={qid}, vuid={vuid}, cuid={cuid}")

    # Save merged results
    print(f"\nSaving merged predictions to {main_pred} ...")
    with open(main_pred, "w") as f:
        json.dump(all_data, f)

    # Count after merge
    post_count = 0
    for v in all_data["results"]["videos"]:
        for c in v["clips"]:
            for a in c["predictions"]:
                for qid, qdata in a["query_sets"].items():
                    if qdata.get("bboxes"):
                        post_count += 1

    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"  Folders processed:          {total_folders}")
    print(f"  Successfully merged:        {stats['merged_ok']}")
    print(f"  Label not in val map:       {stats['label_not_in_map']}")
    print(f"  No track.json file:         {stats['no_track_file']}")
    print(f"  Empty track.json:           {stats['empty_track']}")
    print(f"  Corrupt track.json:         {stats['track_error']}")
    print(f"  Slot not found in JSON:     {stats['update_failed']}")
    print(f"  Before merge (non-empty):   {pre_count}")
    print(f"  After merge  (non-empty):   {post_count}")
    print(f"  Net new additions:          {post_count - pre_count}")
    print("="*60)

    if label_mismatches:
        print(f"\nWARNING: {len(label_mismatches)} folders had no match in vq_val.json:")
        for lm in label_mismatches[:10]:
            print(f"  '{lm}'")
        if len(label_mismatches) > 10:
            print(f"  ... and {len(label_mismatches)-10} more")
        print("\nExpected label format: {annotation_uid[:8]}_{query_set_id}_{object_title}")
        print("Check that inference script uses this exact naming convention.")


if __name__ == "__main__":
    main()
