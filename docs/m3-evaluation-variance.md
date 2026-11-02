# M3 — Evaluation Variance and Reproducibility

Date: 2026-11-03  
Checkpoint: `experiments/flat_100k_mixed_angle30_mask_warm/checkpoints/2026-11-03_03-25-05/2/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds (0–49), max 600 steps, deterministic policy

## Observation

Three evaluations of the same checkpoint, config, and seeds produced:

| Run | Win rate | Death rate | HP lost | Run directory |
|---|---:|---:|---:|---|
| 1 | 0.10 | 0.76 | 16.08 | `experiments/20261103_060302_eval_vpt_ckpt/` |
| 2 | 0.24 | 0.52 | 14.28 | `experiments/20261103_061100_eval_vpt_ckpt/` |
| 3 | 0.20 | 0.60 | 15.38 | `experiments/20261103_061800_eval_vpt_ckpt/` |

Runs 2 and 3 recorded spawn geometry. The spawn distance and angle matched for
**50/50 seeds**, but only 38/50 episode outcomes and 26/50 death flags agreed.

Conclusion: spawn placement is deterministic, but the combat rollout is not.
The variance comes from Minecraft's mob AI or physics state, not from the
policy or the spawn RNG.

## Cause

`ArenaResetCallback` uses a fast reset after the first episode: it kills mobs,
teleports the player, and restores health, but does not recreate the world. The
Java-side mob RNG advances across episodes, so the zombie's pathing and attack
timing differ even when the spawn geometry is identical.

A `hard_reset_each_episode` option was prototyped but makes the MineRL client
time out in `_TO_MOVE_quit_current_episode` when reset is called every episode.
It is not usable as a drop-in fix.

## Implications

1. **Single 50-seed runs have a wide interval.** The observed range was 14
   percentage points around a mean of 18%. Do not treat a 5–10 point difference
   between two single runs as a real effect.
2. **Twenty-seed screens are triage only.** The earlier random-spawn screen
   showed 15% (3/20); the 50-seed result was 4% (2/50).
3. **Historical comparisons need re-baselining.** Differences of 20+ points
   (inventory mask, far-spawn transfer, ±30° joint curriculum) are likely real,
   but the exact values are uncertain.
4. **Spawn metadata is now recorded.** Evaluation CSVs include
   `spawn_distance` and `spawn_angle`, so failures can be binned by geometry.

## Revised evaluation protocol

For triage:

```bash
python scripts/eval.py --policy vpt --checkpoint <ckpt> \
  --config <cfg> --episodes 50 --seed-start 0 --max-steps 600 --deterministic
```

For any claim or gate measurement:

- Run **three independent 50-seed evaluations** and report the mean and range
  (or standard deviation).
- Use **100+ seeds** for the final gate; more seeds also average over AI
  randomness.
- Keep the same seeds across checkpoints, but do not assume repeated runs of
  the same checkpoint will match exactly.
- Label 20-seed results as screens, never as results.

## Next

1. Re-baseline the current best checkpoints on the combined task and random
   spawn with 3×50-seed runs.
2. Rework the stage-1 gate decision using those intervals, not single runs.
3. If exact reproducibility is required, the engine needs a proper
   world/instance reset or a way to reseed mob AI. That is separate engine work.

## Artifacts

- `experiments/20261103_060302_eval_vpt_ckpt/` (no spawn columns)
- `experiments/20261103_061100_eval_vpt_ckpt/` (spawn columns)
- `experiments/20261103_061800_eval_vpt_ckpt/` (spawn columns)
