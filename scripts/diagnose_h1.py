#!/usr/bin/env python3
"""Measure tactic-conditioned action-distribution differences.

Loads a trained ConditionedVPTPolicy checkpoint and compares the action logits
for two tactic IDs on the same observations and recurrent state. This is a
conditioning diagnostic, not a behavioral evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minecombat.train.generators import conditioned_policy_generator  # noqa: E402


def resolve(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(ROOT / candidate)


def kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p_a = torch.softmax(logits_a.float(), dim=-1)
    p_b = torch.softmax(logits_b.float(), dim=-1)
    value = (
        p_a * (torch.log(p_a + 1e-12) - torch.log(p_b + 1e-12))
    ).sum(dim=-1).mean()
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare action distributions for two tactic IDs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="CraftJarvis/MineStudio_VPT.foundation_model_1x")
    parser.add_argument("--num-tactics", type=int, default=2)
    parser.add_argument("--tactic-a", type=int, default=0)
    parser.add_argument("--tactic-b", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.tactic_a == args.tactic_b:
        parser.error("tactic IDs must differ")

    policy = conditioned_policy_generator(
        args.model,
        args.num_tactics,
    )
    state = torch.load(
        resolve(args.checkpoint),
        map_location="cpu",
    )
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    result = policy.load_state_dict(state, strict=True)
    policy.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    button_kls: list[float] = []
    camera_kls: list[float] = []
    button_diffs: list[float] = []
    camera_diffs: list[float] = []
    value_diffs: list[float] = []

    for _ in range(args.samples):
        image = torch.rand(
            (1, 1, 128, 128, 3),
            generator=generator,
            dtype=torch.float32,
        ).to(policy.device)

        outputs = {}
        for tactic in (args.tactic_a, args.tactic_b):
            tactic_ids = torch.full(
                (1, 1),
                tactic,
                dtype=torch.long,
                device=policy.device,
            )
            with torch.inference_mode():
                latents, _ = policy(
                    {"image": image, "tactic_ids": tactic_ids},
                    state_in=policy.initial_state(1),
                )
            outputs[tactic] = latents

        for key, kls, diffs in (
            ("buttons", button_kls, button_diffs),
            ("camera", camera_kls, camera_diffs),
        ):
            logits_a = outputs[args.tactic_a]["pi_logits"][key]
            logits_b = outputs[args.tactic_b]["pi_logits"][key]
            kls.append(kl_divergence(logits_a, logits_b))
            diffs.append(float((logits_a - logits_b).abs().max()))

        value_diffs.append(
            float(
                (
                    outputs[args.tactic_a]["vpred"].float()
                    - outputs[args.tactic_b]["vpred"].float()
                ).abs().max()
            )
        )

    summary = {
        "checkpoint": resolve(args.checkpoint),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "tactic_a": args.tactic_a,
        "tactic_b": args.tactic_b,
        "samples": args.samples,
        "button_kl_mean": sum(button_kls) / len(button_kls),
        "camera_kl_mean": sum(camera_kls) / len(camera_kls),
        "button_logit_maxdiff": max(button_diffs),
        "camera_logit_maxdiff": max(camera_diffs),
        "value_maxdiff": max(value_diffs),
    }
    print(json.dumps(summary, indent=2))

    if not all(
        torch.isfinite(torch.tensor(value))
        for value in (
            summary["button_kl_mean"],
            summary["camera_kl_mean"],
            summary["button_logit_maxdiff"],
            summary["camera_logit_maxdiff"],
            summary["value_maxdiff"],
        )
    ):
        print("RESULT non_finite=True", file=sys.stderr)
        return 1

    if summary["button_kl_mean"] == 0.0 and summary["camera_kl_mean"] == 0.0:
        print("RESULT conditioning_ignored=True", file=sys.stderr)
        return 1

    print("RESULT conditioning_effective=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
