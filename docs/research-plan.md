# Research Plan

## 1. Objective

Train a single end-to-end policy that controls Minecraft's native keyboard and
mouse action space and can defeat combat enemies, culminating in the Ender
Dragon. The project uses curriculum learning so that each stage is a small,
measurable RL experiment rather than an attempt to solve the boss fight
immediately.

## 2. Research questions

### H1 — Can PPO from VPT 1x learn minimal-damage zombie combat?

Compare a pretrained VPT 1x policy with the same policy after PPO fine-tuning
on the zombie arena. The primary metric is win rate; secondary metrics are HP
lost per episode and time-to-kill.

### H2 — Does zombie combat transfer to skeletons?

Fine-tune three conditions on the skeleton task:

1. VPT 1x directly on skeletons.
2. A zombie-pretrained policy fine-tuned on skeletons.
3. A zombie-pretrained policy with a high-level harness, fine-tuned on skeletons.

If condition 2 is more sample-efficient than condition 1, the result supports
skill transfer between melee and ranged enemies.

### H3 — Does a high-level harness improve long-horizon boss performance?

Only after the flat policy is strong on stage 1–3. The harness is a separate
decision layer; the flat policy remains the control condition. This ordering
prevents attributing gains to a stronger base policy.

## 3. Curriculum

| Stage | Enemy | New challenge | Gate to next stage |
|---|---|---|---|
| 1 | Zombie | Melee approach, aim, attack, avoid damage | ≥50% win rate, damage falling |
| 2 | Skeleton | Projectiles, distance control, strafing | ≥40% win rate, no degenerate camping |
| 3 | Creeper + mixed | Explosion risk, multiple enemies, terrain | Risk-aware behavior, positive damage ratio |
| 4 | Ender Dragon | Long horizon, crystals, phase changes | Harness comparison complete |

The gates are provisional and will be revised after the first baseline.

## 4. Stage 1 — Zombie arena

### Fixed factors

- Flat plains biome, midnight, clear weather.
- Natural mob spawning disabled.
- One adult zombie.
- Iron sword, iron helmet, chestplate, leggings, boots.
- Full health at reset.

### Randomized factors

- Zombie spawn angle: 0–360°.
- Zombie spawn distance: 4–10 blocks.
- Player yaw at reset: random.
- World location: random plains teleport within a bounded range.

### Episode termination

- Zombie killed.
- Player health reaches zero.
- 600 simulator steps (30 seconds at 20 Hz).

### Observation and action

- Observation: 128×128 RGB, plus the previous action handled by the VPT policy.
- Action: MineStudio `agent` action space (camera and buttons).
- HP is logged for reward and evaluation but is not a policy input in stage 1.
  Adding HP as an input is a later ablation.

## 5. Reward

```text
r_t = 0.1 * Δdamage_dealt
    - 0.2 * Δhp_lost
    - 0.001
    + 10 * kill
    - 5 * death
```

The reward is clipped to `[-10, 20]`. Details and failure modes are in
[rewards.md](rewards.md).

## 6. Baselines

1. **Random policy** — sanity check for the environment and metrics.
2. **Zero-shot VPT 1x** — measures what the pretrained model already knows.
3. **PPO from VPT 1x** — the first learned condition.
4. **Scripted oracle (optional)** — uses privileged mob positions as an upper
   bound and to validate that the environment is winnable.

## 7. Metrics

- Win rate.
- Death rate.
- Timeout rate.
- Stuck rate (timeout with zero damage dealt).
- Mean HP lost per episode.
- Mean HP lost on wins.
- Mean time-to-kill.
- Damage ratio (damage dealt / HP lost).
- Mean episode reward.

Report mean and 95% bootstrap confidence intervals over at least 50 fixed seeds.

## 8. Ablations

Run after the first PPO baseline is stable:

- Sparse kill reward vs dense combat reward.
- HP as an input vs reward-only HP.
- Fixed spawn vs randomized spawn and yaw.
- Full action space vs masked unused actions.
- Context length 32 vs 64 vs 128.
- Frozen image encoder vs full fine-tuning.

## 9. Experiment protocol

Every run gets a directory under `experiments/` containing:

- resolved config,
- git commit,
- random seeds,
- metrics JSON and per-episode CSV,
- W&B run link,
- short notes with the hypothesis and outcome.

A run is only comparable if the environment config, reward config, evaluation
seeds, and action space are identical.

## 10. Risks

| Risk | Mitigation |
|---|---|
| 8 GB VRAM is insufficient for PPO | VPT 1x, batch 1, context 64, freeze encoder if needed; move to 4090/cloud |
| Simulator rendering is slow | 128×128 observations, 640×360 render, fast reset, fewer envs |
| Reward hacking (spin or camp) | Time penalty, stuck metric, video review, damage-dealt term |
| Zombie spawns inside blocks | Spawn 2 blocks above ground, validate with 20-reset test |
| High variance across seeds | Fixed evaluation seeds and confidence intervals |
| Curriculum transfer fails | Report negative result; compare direct fine-tuning control |

## 11. Timeline

| Week | Milestone |
|---|---|
| 1 | M0 environment bring-up |
| 2 | M1 zombie arena and reward |
| 3 | M2 baseline and metrics |
| 4–5 | M3 first PPO run and stage-1 gate |
| 6+ | M4 skeleton curriculum |
