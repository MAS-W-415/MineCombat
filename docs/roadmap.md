# Roadmap

## M0 — Environment bring-up

**Goal:** prove that MineStudio, VPT 1x, and the local renderer work together.

Tasks:

- install JDK 8, MineStudio, and the VPT 1x checkpoint;
- run `python -m minestudio.simulator.entry`;
- run 600 steps of VPT inference and record a video;
- measure steps per second, peak VRAM, and reset time.

Acceptance:

- simulator runs for at least 5 minutes without crashing;
- VPT inference produces a valid 128×128 observation stream;
- performance numbers are recorded in the M0 issue.

## M1 — Zombie arena

**Goal:** a reproducible single-zombie combat environment.

Tasks:

- implement `make_zombie_env`;
- implement arena reset without killing the player;
- randomize zombie ring spawn, distance, and player yaw;
- implement and unit-test the combat reward;
- verify 20 consecutive resets.

Acceptance:

- exactly one zombie after each reset;
- player starts at full health with the iron loadout;
- zombie is not always in front of the camera;
- no natural mobs, daylight cycle, or weather variation;
- 20 resets complete without manual intervention.

## M2 — Baseline and metrics

**Goal:** measure what the pretrained model already does.

Tasks:

- implement the evaluation harness and metrics;
- run 50 fixed-seed episodes with a random policy;
- run 50 fixed-seed episodes with zero-shot VPT 1x;
- record win rate, HP lost, time-to-kill, stuck rate;
- save videos for qualitative review.

Acceptance:

- metrics JSON and per-episode CSV are produced;
- zero-shot baseline is clearly above random;
- failure modes are documented.

## M3 — First PPO run

**Goal:** show that PPO improves over the zero-shot baseline.

Tasks:

- validate the single-GPU PPO config with a 10k-step smoke test;
- run at least 1M environment steps;
- evaluate checkpoints on the fixed seed set;
- log curves and videos to W&B.

Acceptance:

- no OOM during the smoke test;
- training curve is non-degenerate;
- at least one checkpoint beats the zero-shot win rate;
- stage-1 gate decision is recorded.

## Stage-1 gate

Move to skeletons only if:

- win rate is at least 50% on 50 fixed seeds, and
- mean HP lost is lower than the zero-shot baseline, and
- video review shows purposeful approach and attack behavior.

If the gate is not met, document the failure, inspect reward hacking, and run
one targeted ablation before extending the curriculum.

## M4 — Curriculum

- Skeleton arena with projectiles.
- Creeper and mixed-enemy arena.
- High-level harness experiments.
- Ender Dragon scenario.

These are intentionally unspecified until stage 1 is stable.
