# Offline evaluation summary

`scripts/summarize_evals.py` reads explicitly selected evaluation directories
using the Python standard library. It prints Markdown to stdout by default.
Select `--format json` for JSON output. Each invocation summarizes one task.

```bash
python scripts/summarize_evals.py --task TASK RUN_DIR1 RUN_DIR2 RUN_DIR3 \
  --format markdown
```

For team validation, run these commands from the repository root on the
designated teammate's computer in its existing environment. The summary tool
and focused tests do not start Minecraft, Ray, or GPU training. The tests use
the project's existing pytest dependency.

## Inputs and checks

Each directory supplies `episodes.csv`, with optional `metrics.json`,
`policy.json`, and `config.yaml` metadata. The script recomputes metrics from
the episode rows and checks available saved metrics against them.

Runs in a group must have the same seed set and matching recorded identities
when available: policy, model, checkpoint, tactic, sampling settings, source
commit, and SHA-256 hash of the original `config.yaml` bytes. Conflicts are
rejected, including inconsistent identities recorded within one run. The task
label is supplied by the caller. Missing metadata produces a warning; unknown
JSON values are `null` and Markdown uses `unknown`. An explicit
`"checkpoint": null` in `policy.json` means no checkpoint was supplied and is
distinguished from an absent checkpoint field by `known_metadata_fields`.

A matching checkpoint reference compares recorded paths. It does not verify
the checkpoint file's contents. A matching configuration hash covers the saved
file's original bytes. Inherited `base` files and runtime versions require
their own provenance, which these historical snapshots may lack.
Even comment-only config changes or differently spelled checkpoint paths are
conservatively rejected. Review such groups separately rather than rewriting
their original artifacts to bypass the checks.

## Reading the output

The report includes each run's metrics and an equal-weight macro mean and
minimum–maximum range across runs. The range describes the observed runs.
It is not a confidence interval. Three repetitions of seeds 0–49 represent
150 episodes but only 50 unique seeds. Repeated seeds across runs are retained;
duplicate seeds within a run and duplicate run directories are rejected.
For optional metrics such as time-to-kill, a run with no defined value is
excluded from that metric's mean; `runs_with_value` reports the denominator.

- `zero_damage_death_rate` counts episodes with `death=True` and
  `damage_dealt=0`, divided by all episodes in that run.
- `zero_damage_timeout_rate` counts episodes with `timeout=True` and
  `damage_dealt=0`, divided by all episodes in that run.

Zero-damage failures identify episodes for closer inspection. The CSV has no
per-step target visibility, player-facing direction, or action trace, so it
cannot establish whether the policy failed to see, turn toward, approach, or
attack the zombie. `spawn_angle` records the spawn geometry angle. Treating it
as an angle relative to the player's field of view requires player-facing
information that is absent from these CSVs.

## Teammate validation

First run the focused tests in the existing environment.

```bash
pytest tests/test_summarize_evals.py
```

Summarize the three 2M combined-task runs.

```bash
python scripts/summarize_evals.py --task combined \
  experiments/eval_2m_4090d/20261106_135858_eval_vpt_ckpt \
  experiments/eval_2m_4090d/20261106_143436_eval_vpt_ckpt \
  experiments/eval_2m_4090d/20261106_150837_eval_vpt_ckpt
```

Expected win rates are 0.58, 0.66, and 0.62, with macro mean 0.62 and range
0.58–0.66. The saved metrics are the sources for these expected values.

Summarize the three 2M random-task runs separately.

```bash
python scripts/summarize_evals.py --task random \
  experiments/eval_2m_4090d/20261106_142035_eval_vpt_ckpt \
  experiments/eval_2m_4090d/20261106_145425_eval_vpt_ckpt \
  experiments/eval_2m_4090d/20261106_152754_eval_vpt_ckpt
```

Expected win rates are 0.06, 0.10, and 0.14, with macro mean 0.10 and range
0.06–0.14. Repeat either command with `--format json` and check that the
numbers match its Markdown output. Review the reported warnings and metadata
checks alongside the rates.
