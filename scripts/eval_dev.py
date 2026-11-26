"""Development-only closed-loop sanity check entry for SimplerEnv + StarVLA.

Note: This script is ONLY for development sanity checks (e.g. episode 0).
Formal 40-episode B-class evaluation must use a separate formal evaluator with
frozen initial states, explicit environment/model seeds, and per-episode CSV logging.
"""
import json
from pathlib import Path
import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient


def main():
    args = get_args()
    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        host=args.host,
        port=args.port,
        action_scale=args.action_scale,
    )
    results = maniskill2_evaluator(model, args)
    assert len(results) == args.obj_episode_range[1] - args.obj_episode_range[0]
    folder = Path(args.logging_dir)
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "task": args.env_name,
        "checkpoint": args.ckpt_path,
        "episode_range": args.obj_episode_range,
        "success": [bool(x) for x in results],
        "success_rate": float(np.mean(results)),
        "purpose": "development_only",
    }
    (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record)


if __name__ == "__main__":
    main()
