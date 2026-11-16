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

Every episode with `win=True` must supply `time_to_kill`. A missing or blank
value raises `SummaryError` with the CSV path and row location. Empty values
remain valid for non-winning episodes and are preserved as missing values.

When `policy.json` supplies a valid `max_steps` (a positive integer), every
episode must have `steps <= max_steps`. An episode exceeding the limit raises
`SummaryError` identifying the run directory, seed, actual steps, and limit.
An absent or `null` limit remains unknown and skips this check. The tool uses
only the limit recorded in that run's `policy.json` for this check.

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

Markdown lists every identity field in `IDENTITY_FIELDS` order for each run.
The order is `policy`, `model`, `checkpoint`, `checkpoint_sha256`,
`model_revision`, `tactic_id`, `num_tactics`, `deterministic`, `max_steps`,
`config_sha256`, and `git_commit`. Known values use JSON notation, including
`false` for `deterministic=False` and `0` for `tactic_id=0`. Unknown fields
display `unknown`. An explicit `"checkpoint": null` displays as
`checkpoint: null`, while an absent checkpoint field displays as
`checkpoint: unknown`.

The report includes each run's metrics and an equal-weight macro mean and
minimum–maximum range across runs. The range describes the observed runs.
It is not a confidence interval. Three repetitions of seeds 0–49 represent
150 episodes but only 50 unique seeds. Repeated seeds across runs are retained;
duplicate seeds within a run and duplicate run directories are rejected.
For optional metrics, a run with no defined value is excluded from that
metric's mean; `runs_with_value` reports the denominator. Time-to-kill is
undefined for runs with no wins. Within a run with wins, its mean includes
every winning episode, each of which must have a recorded kill time.

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

Check that the regression cases cover partial missing kill times among wins,
steps equal to and above a known limit, absent and `null` limits, and all
Markdown identity fields and their values. Missing kill times and excessive
steps should produce errors with input context. Non-winning empty kill times
and unknown limits should remain valid.

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
