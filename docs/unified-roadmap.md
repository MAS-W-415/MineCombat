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
| RQ1 flat PPO | in progress; 3×50 combined 16%, random 10%; visible combat is learnable but composition and search are weak |
| Stage-1 gate | not passed; the 1M combined run is the next test |
| HH1 conditioned VPT | 10k engineering validation; tactics weak, 0% win |
| HH2 high-level PPO | L1 and targeted L2 data flow validated; multi-update stability and L3 pending |
| RQ2/RQ3 | not started |

## Immediate next steps

1. Run the 1M combined curriculum on the 4090D (24 GB / 60 GB / 24 h):
   `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`, 20 iterations,
   save every 5, then 3×50 evaluation. See `docs/1m-4090d-handoff.md`.
2. If the 1M combined task stays below ~30%, change the curriculum or task
   definition; do not start another long run on the same setup.
3. Keep 360° search as Stage-1b. Camera entropy did not help; scan-at-reset,
   auto-face, or a relative-direction observation are the options.
4. Keep HH1/HH2 behavioral work paused until the flat gate passes.

## Non-goals

- No separate reward, evaluation, or config copies per line.
- No new HH feature merge without a linked experiment issue and evaluation plan.
- No bare `H1`/`H2` labels; use `RQ` or `HH`.
