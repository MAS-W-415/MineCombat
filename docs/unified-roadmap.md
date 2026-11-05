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
| RQ1 flat PPO | in progress; visible combat is learnable, but full 360° search is unsolved |
| Stage-1 gate | not passed; evaluation variance is being re-baselined |
| HH1 conditioned VPT | 10k engineering validation; tactics weak, 0% win |
| HH2 high-level PPO | L1 and targeted L2 data flow validated; multi-update stability and L3 pending |
| RQ2/RQ3 | not started |

## Immediate next steps

1. Re-baseline the joint checkpoint and the best alternative checkpoints with
   3×50-seed runs on the combined and random tasks.
2. Decide the Stage-1 gate using those intervals: visible combat (±30°, distance
   3–7) versus full 360° search. Off-screen spawns never enter the view without
   an active turn, and camera entropy did not solve this.
3. For visible combat, run 1M on the joint curriculum on the 24 GB / 64 GB host.
4. For full search, add a search mechanism (scan-at-reset, auto-face curriculum,
   or a relative-direction observation for HH).

## Non-goals

- No separate reward, evaluation, or config copies per line.
- No new HH feature merge without a linked experiment issue and evaluation plan.
- No bare `H1`/`H2` labels; use `RQ` or `HH`.
