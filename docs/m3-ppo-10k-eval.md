# M3 — 10k PPO Checkpoint Evaluation

Date: 2026-10-30  
Checkpoint: `experiments/ppo_smoke/checkpoints/2026-10-30_18-22-45/20/model.ckpt`  
Seeds: 0–49, max 600 steps, deterministic sampling

## Results

| Metric | Zero-shot VPT | 10k PPO |
|---|---:|---:|
| Win rate | 0.02 | 0.00 |
| Death rate | 0.28 | 0.22 |
| Timeout / stuck rate | 0.70 | 0.78 |
| Mean HP lost | 11.58 | 10.84 |
| Mean steps | 535.7 | 568.2 |
| Mean reward | -2.65 | -2.74 |

## Interpretation

The 10k checkpoint survives slightly better (lower death rate and HP loss) but
does not win. This is expected: 10k steps is a pipeline smoke, not a training
run. It does not show that PPO improves over zero-shot. That question needs the
1M-step run (#10).

This checkpoint was trained before the `custom.damage_dealt` observation fix,
so it did not receive the dense damage term. The evaluation metrics (win,
death, HP, steps) do not depend on the reward and remain comparable. The
dense-damage follow-up is in `docs/m3-dense-damage.md`.

## Artifacts

- `experiments/20261030_190258_eval_vpt_ckpt/`
- `scripts/eval_baseline.py --checkpoint ...` loads a PPO `model.ckpt` on top of
  the VPT 1x weights.
