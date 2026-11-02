# Unified Roadmap

Date: 2026-11-02  
Status: active  
See `docs/glossary.md` for the `M`, `RQ`, and `HH` naming rules.

## One line, not two

Flat PPO and the hierarchical harness are one research program with a control
condition and an extension:

```text
VPT 1x
  |
  v
Flat VPT + PPO                 RQ1 / control condition
  |
  v
Stage-1 gate: >=50% win rate
  |
  v
HH1 Conditioned VPT            extension, warm-started from flat PPO
  |
  v
HH2 High-Level PPO             extension, RQ3
  |
  v
Skeleton -> Creeper -> Ender Dragon
```

- **Flat PPO is the control.** It must remain independently executable.
- **HH1/HH2 are extensions.** They are not a second mainline.
- The harness is evaluated only after the flat policy passes the stage-1 gate.
- Both share one environment, reward core, evaluation protocol, and checkpoint
  lineage.

## Shared core

Both lines use:

| Component | Location |
|---|---|
| Environment config with base + overlay | `configs/env/*.yaml`, `minecombat.envs.zombie_arena.load_config` |
| Combat stats and reward deltas | `minecombat.envs.callbacks.combat_reward` |
| Evaluation runner and metrics | `minecombat.eval`, canonical CLI `scripts/eval.py` |
| Policy loading and warm-start | `minecombat.policies.vpt`, `minecombat.train.generators` |
| Ray/W&B launcher | `minecombat.train.ppo`, `minecombat.train.high_level` |
| Common training preflight | `docs/training-common.md` |

A change to the shared core must be validated against both entry points, even if
only one line currently runs experiments. `scripts/eval_baseline.py` and
`scripts/eval_conditioned.py` remain as compatibility wrappers around the
canonical `scripts/eval.py`.

## Checkpoint lineage

```text
VPT 1x
  -> flat PPO best checkpoint
  -> HH1 conditioned checkpoint (warm start)
  -> HH2 high-level checkpoint
```

Do not restart from VPT for every reward/curriculum experiment. Use
`--init-checkpoint` for flat PPO and the HH1 checkpoint loader for harness
training.

## Experiment protocol

- One experiment issue per run.
- One owner per run.
- One GPU queue: no simultaneous long runs on the local GPU.
- Fixed 50-seed evaluation, same seeds across variants.
- Report: metrics JSON, per-episode CSV, checkpoint path, git commit, notes.
- PRs link the experiment issue and include the result, not only the code.

## Current status

| Item | Status |
|---|---|
| M0 environment | complete |
| M1 zombie arena | complete |
| M2 baselines | complete |
| RQ1 flat PPO | in progress; 44% easy, 34% at ±30°, 48% mixed distance, 4% random |
| Stage-1 gate | not passed |
| HH1 conditioned VPT | 10k engineering validation; tactics weak, 0% win |
| HH2 high-level PPO | implemented; GPU validation pending |
| RQ2/RQ3 | not started |

## Immediate next steps

1. Widen the joint curriculum from ±30° to ±60°, then ±90° and ±180°,
   confirming every stage on 50 seeds.
2. Measure random spawn with 50 seeds at every stage; 20 seeds are triage only.
3. Move the 1M consolidation run to the 24 GB / 64 GB host once the angle
   curriculum shows a stable random-spawn trend.
4. Pass the stage-1 gate on the random-spawn task, then resume HH1/HH2.

## Non-goals

- No separate reward, evaluation, or config copies per line.
- No new HH feature merge without a linked experiment issue and evaluation plan.
- No bare `H1`/`H2` labels; use `RQ` or `HH`.
