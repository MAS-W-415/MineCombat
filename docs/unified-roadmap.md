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
| RQ1 flat PPO | in progress; 2M: 90% easy, 64% ±30°, 62% combined, 10% random |
| Stage-1 gate | Stage-1a visible combat passed; Stage-1b 360° search open |
| Closed-loop search engineering | camera calibration and 9-step VisualSearchPolicy integration accepted; 560-frame search dataset delivered; detector fine-tune v1 complete |
| Mob Radar zero-shot detector | YOLO26 Nano loads and integrates; 56.4% frame recall, 0% false positives on 80-image audit; case B, fine-tune needed |
| Zombie detector annotation | first Train/Val batch complete: 40 images, 27 positives, 13 negatives; Train 32 / Val 8 by episode; Test task created |
| Mob Radar fine-tune v1 | 20 epochs from Mob Radar; pilot audit recall 56.4% -> 82.1%, FPR 0%; independent test pending |
| Independent detector test | `test_40` created: 20 Face + 20 Random images, two jobs, 7 prelabels; waiting for annotation |
| HH1 conditioned VPT | 10k engineering validation; tactics weak, 0% win |
| HH2 high-level PPO | L1 and targeted L2 data flow validated; multi-update stability and L3 pending |
| RQ2/RQ3 | not started |

## Immediate next steps

1. Freeze the 2M checkpoint as the best **visible-combat** flat policy; do not
   continue the same ±30° curriculum expecting random to improve.
2. For **Stage-1b search**, scan-at-reset was tested and rejected (0% win,
   54% timeouts). Choose between an explicit auto-face/split-gate definition
   and a relative-direction observation for HH. Do not run more random PPO
   without one of these. The closed-loop engineering chain is now accepted and
   the 560-frame search dataset is delivered. A third-party Mob Radar zero-shot
   detector was tested (56.4% frame recall at 320, 0% false positives); it is
   usable only as a fine-tuning initialization. The first Train/Val annotation
   batch is complete (40 images: 27 positive, 13 negative; 32 train / 8 val by
   episode). Fine-tune v1 is complete; on the pilot audit set recall improved
   from 56.4% to 82.1% with 0% false positives, but this audit is not an
   independent test. Annotate and freeze the `test_40` task, then report the
   old-vs-fine-tuned Precision/mAP comparison before the 360° evaluation.
3. Ask the team to accept the Stage-1a/1b split. If accepted, initialize HH1
   from the 2M visible-combat checkpoint and keep HH2 behavioral work gated on
   Stage-1b.
4. Keep 3×50 evaluation for all gate claims; 20-seed runs are triage only.

## Non-goals

- No separate reward, evaluation, or config copies per line.
- No new HH feature merge without a linked experiment issue and evaluation plan.
- No bare `H1`/`H2` labels; use `RQ` or `HH`.
