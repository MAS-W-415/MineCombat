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
- Fixed evaluation task and seeds; **3×50-seed runs for any claim**, 20-seed
  screens are triage only. Minecraft mob AI is not reset between episodes, so
  repeated runs of the same checkpoint differ by up to ~14 points.
- Report: metrics JSON, per-episode CSV, checkpoint path, git commit, notes.
- PRs link the experiment issue and include the result, not only the code.

## Current status

| Item | Status |
|---|---|
| M0 environment | complete |
| M1 zombie arena | complete |
| M2 baselines | complete |
| RQ1 flat PPO | in progress; 1M: 72% easy, 52% ±30°, 40.7% combined, 22.7% random |
| Stage-1 gate | not passed; case B, 1M continuation running |
| HH1 conditioned VPT | 10k engineering validation; tactics weak, 0% win |
| HH2 high-level PPO | L1 and targeted L2 data flow validated; multi-update stability and L3 pending |
| RQ2/RQ3 | not started |

## Immediate next steps

1. Finish the 1M continuation (2.048M total) with `--keep-interval 5` and
   `--wandb-mode offline`; evaluate 1.25M/1.5M/1.75M/2M checkpoints.
2. If the combined mean crosses 50% and random keeps improving, move toward the
   Stage-1 gate. If it plateaus below 50%, analyze camera tracking and consider
   a curriculum/observation change instead of another long run.
3. Keep 360° search as Stage-1b unless random reaches a stable useful level.
4. Keep HH1/HH2 behavioral work paused until the flat gate passes.

## Non-goals

- No separate reward, evaluation, or config copies per line.
- No new HH feature merge without a linked experiment issue and evaluation plan.
- No bare `H1`/`H2` labels; use `RQ` or `HH`.
