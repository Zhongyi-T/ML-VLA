"""Export a real Bridge LeRobot sample (image, instruction, actions) and generate
an episode-level train/validation split with zero trajectory overlap.
"""
import argparse
import json
import random
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Inspect Bridge LeRobot dataset and create episode-level split")
    parser.add_argument("--bridge-dir", type=Path, required=True, help="Root directory of bridge_orig_lerobot")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for data_report/ and splits/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode-level split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation episode ratio (default: 0.1)")
    args = parser.parse_args()

    data = args.bridge_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    report_dir = out / "data_report"
    split_dir = out / "splits"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((data / "meta/info.json").read_text(encoding="utf-8"))
    ep_ids = [
        int(json.loads(line)["episode_index"])
        for line in (data / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(args.seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(len(ep_ids) * args.val_ratio)))
    val_ids = sorted(ep_ids[:n_val])
    train_ids = sorted(ep_ids[n_val:])
    assert set(train_ids).isdisjoint(val_ids), "Train and val episode sets overlap!"
    assert len(train_ids) + len(val_ids) == len(ep_ids)

    (split_dir / "train_episodes.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (split_dir / "val_episodes.json").write_text(json.dumps(val_ids), encoding="utf-8")
    split_summary = {
        "seed": args.seed,
        "ratio": f"{int(round((1 - args.val_ratio) * 100))}/{int(round(args.val_ratio * 100))}",
        "split_unit": "complete_episode",
        "source_subset": f"first {len(ep_ids)} complete episodes of IPEC-COMMUNITY/bridge_orig_lerobot",
        "total_episodes": len(ep_ids),
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "overlap": 0,
    }
    (split_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2) + "\n", encoding="utf-8")

    pq_file = sorted((data / "data").rglob("*.parquet"))[0]
    table = pq.read_table(pq_file).to_pydict()
    actions = np.asarray(table["action"], dtype=np.float32)
    states = np.asarray(table["observation.state"], dtype=np.float32)

    tasks_map = {}
    tasks_path = data / "meta/tasks.jsonl"
    if tasks_path.exists():
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                tasks_map[obj["task_index"]] = obj["task"]
    task_idx = int(table["task_index"][0]) if "task_index" in table else 0
    lang = tasks_map.get(task_idx, "<unknown>")

    vid_candidates = sorted((data / "videos").rglob("*image_0*/*.mp4")) or sorted((data / "videos").rglob("*.mp4"))
    img_saved = None
    if vid_candidates:
        with av.open(str(vid_candidates[0])) as container:
            frame = next(container.decode(video=0))
            img = frame.to_image()
            img_path = report_dir / "sample_ep0_frame0.png"
            img.save(img_path)
            img_saved = {"path": str(img_path.relative_to(out)), "size": list(img.size), "mode": img.mode}

    sample_info = {
        "dataset": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "revision": "0e9d76d07e9df3ea3eba257b2520d4913833fad2",
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "sample_parquet": str(pq_file.relative_to(data.parent)),
        "episode_length": int(actions.shape[0]),
        "language_instruction": lang,
        "action_shape": list(actions.shape),
        "action_min_per_dim": [round(float(x), 5) for x in actions.min(axis=0)],
        "action_max_per_dim": [round(float(x), 5) for x in actions.max(axis=0)],
        "action_first_3_steps": [[round(float(v), 5) for v in row] for row in actions[:3]],
        "state_shape": list(states.shape),
        "include_state_in_model": False,
        "normalization": "official Bridge transform (continuous q99, gripper binary), unnorm_key=oxe_bridge",
        "image_sample": img_saved,
    }
    (report_dir / "sample_inspection.json").write_text(
        json.dumps(sample_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_summary, indent=2))
    print(json.dumps(sample_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
