
"""Evaluate closed-loop visual search in Minecraft."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from minecombat.policies.vpt import DEFAULT_MODEL
from minecombat.train.common import resolve_path
from minecombat.utils import RunLog

from . import evaluate, summarize


def run_search_evaluation(
    env,
    combat_policy,
    predictor,
    *,
    episodes: int = 5,
    seed_start: int = 0,
    max_steps: int = 600,
    yaw_sign: int = 1,
):
    """Evaluate combat plus search and collect per-episode traces."""

    from minecombat.policies.harness.search_controller import (
        ClosedLoopSearchController,
    )
    from minecombat.policies.harness.search_trace import SearchTracePolicy
    from minecombat.policies.harness.target_detector import RGBTargetDetector
    from minecombat.policies.harness.visual_search_policy import (
        CameraActionAdapter,
        VisualSearchPolicy,
    )

    policy = SearchTracePolicy(
        VisualSearchPolicy(
            detector=RGBTargetDetector(predictor),
            controller=ClosedLoopSearchController(),
            combat_policy=combat_policy,
            camera_action=CameraActionAdapter(
                env, yaw_sign=yaw_sign
            ),
        )
    )

    records = evaluate(
        env,
        policy=policy,
        episodes=episodes,
        seed_start=seed_start,
        max_steps=max_steps,
        deterministic=True,
    )

    traces = policy.records()

    if len(traces) != len(records):
        raise RuntimeError("Episode and search trace counts differ")

    search_episodes = []
    for record, trace in zip(records, traces, strict=True):
        if record.steps != trace["steps"]:
            raise RuntimeError("Search trace step count mismatch")

        search_episodes.append({
            "seed": record.seed,
            "outcome": record.outcome,
            "win": record.win,
            **trace,
        })

    total_steps = sum(t["steps"] for t in traces)
    first_steps = [
        t["first_detection_step"]
        for t in traces
        if t["first_detection_step"] is not None
    ]

    search_summary = {
        "episodes": len(traces),
        "model_detection_episode_rate": (
            len(first_steps) / len(traces)
        ),
        "mean_first_detection_step": (
            mean(first_steps) if first_steps else None
        ),
        "search_step_rate": (
            sum(t["search_steps"] for t in traces) / total_steps
        ),
        "align_step_rate": (
            sum(t["align_steps"] for t in traces) / total_steps
        ),
        "combat_step_rate": (
            sum(t["combat_steps"] for t in traces) / total_steps
        ),
        "total_reacquisitions": sum(
            t["reacquisitions"] for t in traces
        ),
    }

    return (
        records,
        search_episodes,
        summarize(records),
        search_summary,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Minecraft closed-loop visual search."
    )

    parser.add_argument("--combat-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--config",
        default="configs/env/zombie_stage1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--yaw-sign", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="experiments",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    if args.seed_start < 0:
        parser.error("seed-start must be non-negative")
    if args.yaw_sign not in (-1, 1):
        parser.error("yaw-sign must be +1 or -1")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("imgsz must be a multiple of 32")

    combat_ckpt = resolve_path(args.combat_checkpoint)
    detector_ckpt = resolve_path(args.detector_checkpoint)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)

    metadata = {
        **vars(args),
        "combat_checkpoint": str(combat_ckpt),
        "detector_checkpoint": str(detector_ckpt),
        "config": str(config_path),
        "output_dir": str(output_dir),
    }

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0

    for path in (combat_ckpt, detector_ckpt, config_path):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")

    from minecombat.envs import make_zombie_env
    from minecombat.envs.zombie_arena import load_config
    from minecombat.policies.harness.ultralytics_predictor import (
        UltralyticsBoxPredictor,
    )
    from minecombat.policies.vpt import load_vpt_policy

    config = load_config(config_path)

    if config.get("action_type", "agent") != "agent":
        parser.error("Requires agent action space")

    if config.get("zombie", {}).get("face_spawn", False):
        parser.error("Evaluation must not use face_spawn")

    if config.get("assist", {}).get("scan_at_reset_steps", 0):
        parser.error("Evaluation must not use scan-at-reset")

    env = make_zombie_env(config_path)

    try:
        combat = load_vpt_policy(
            repo_id=args.model,
            checkpoint=str(combat_ckpt),
            device=args.device,
        )
        predictor = UltralyticsBoxPredictor(
            detector_ckpt,
            device=args.detector_device,
            imgsz=args.imgsz,
        )

        records, search_episodes, metrics, search_metrics = (
            run_search_evaluation(
                env,
                combat,
                predictor,
                episodes=args.episodes,
                seed_start=args.seed_start,
                max_steps=args.max_steps,
                yaw_sign=args.yaw_sign,
            )
        )
    finally:
        env.close()

    run = RunLog(output_dir, "eval_visual_search")
    run.save_config(config_path)
    run.save_metrics(metadata, filename="policy.json")
    run.save_metrics(metrics)
    run.save_metrics(search_metrics, filename="search_summary.json")
    run.save_records(records)
    run.save_git_commit()

    trace_path = run.run_dir / "search_episodes.json"
    trace_path.write_text(
        json.dumps(search_episodes, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "combat": metrics,
        "search": search_metrics,
    }, indent=2))
    print(f"Saved evaluation: {run.run_dir}")
    return 0
