
"""Integration test for conditioned policy evaluation CLI."""

import csv
import json
import runpy
import sys
from pathlib import Path

import minecombat.envs as env_module
from minecombat.policies.harness import factory


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_conditioned.py"


def test_conditioned_evaluation_cli(monkeypatch, tmp_path):
    calls = []
    model_loads = []

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def reset(self):
            return {"seed": self.seed}, {
                "health": 20.0,
                "damage_dealt": {"zombie": 0.0},
                "kill_entity": {"zombie": 0.0},
            }

        def step(self, action):
            return {"seed": self.seed}, 1.0, True, False, {
                "health": 20.0,
                "damage_dealt": {"zombie": 10.0},
                "kill_entity": {"zombie": 1.0},
                "combat_outcome": "success",
            }

        def close(self):
            self.closed = True

    class FakeHarness:
        def __init__(self):
            self.tactical_policy = None

        def get_action(
            self,
            obs,
            memory,
            *,
            deterministic,
            input_shape,
        ):
            tactic_id = self.tactical_policy.tactic_id

            calls.append((tactic_id, obs["seed"]))

            return {"tactic_id": tactic_id}, memory

    env = FakeEnv()

    def fake_build_harness(**kwargs):
        model_loads.append(kwargs)
        return FakeHarness()

    monkeypatch.setattr(
        env_module,
        "make_zombie_env",
        lambda config: env,
    )

    monkeypatch.setattr(
        factory,
        "build_conditioned_harness",
        fake_build_harness,
    )

    # CLI requires the checkpoint and config paths to exist.
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fake-checkpoint")

    config = tmp_path / "env.yaml"
    config.write_text("seed: 0\n", encoding="utf-8")

    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_conditioned.py",
            "--checkpoint", str(checkpoint),
            "--config", str(config),
            "--num-tactics", "2",
            "--tactic-ids", "0", "1",
            "--episodes", "2",
            "--seed-start", "7",
            "--output-dir", str(output_dir),
            "--deterministic",
        ],
    )

    # Execute the actual CLI entry point.
    runpy.run_path(
        str(EVAL_SCRIPT),
        run_name="__main__",
    )

    # The model should be constructed only once.
    assert len(model_loads) == 1

    # Both tactics must use exactly the same seeds.
    assert calls == [
        (0, 7),
        (0, 8),
        (1, 7),
        (1, 8),
    ]

    assert env.closed

    # Both evaluations must produce independent result files.
    for tactic_id in (0, 1):
        directories = list(
            output_dir.glob(
                f"*_eval_conditioned_tactic_{tactic_id}"
            )
        )

        assert len(directories) == 1

        run_dir = directories[0]

        policy_info = json.loads(
            (run_dir / "policy.json").read_text()
        )

        assert policy_info["tactic_id"] == tactic_id

        metrics = json.loads(
            (run_dir / "metrics.json").read_text()
        )

        assert metrics["episodes"] == 2
        assert metrics["win_rate"] == 1.0

        with (run_dir / "episodes.csv").open() as f:
            records = list(csv.DictReader(f))

        assert [int(r["seed"]) for r in records] == [7, 8]
