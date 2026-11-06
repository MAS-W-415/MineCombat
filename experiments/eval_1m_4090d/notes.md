# 1M flat PPO evaluation

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Code: `b54c53d`  
Seeds: 0–49, max 600 steps, deterministic

## Results

| Task | Win | Death | HP lost | Time-to-kill |
|---|---:|---:|---:|---:|
| easy 3–4 | 0.72 | 0.28 | 12.00 | 190.9 |
| ±30° close | 0.52 | 0.46 | 12.50 | 196.5 |
| combined 3–7 + ±30° (3×50 mean) | **0.407** | 0.567 | 14.57 | 227.5 |
| random 360° (3×50 mean) | **0.227** | 0.580 | 15.67 | 282.3 |

Ranges: combined 0.32–0.52; random 0.20–0.26.

## Directories

- `20261106_093929_eval_vpt_ckpt/` — easy, 1×50
- `20261106_095237_eval_vpt_ckpt/` — ±30°, 1×50
- `20261106_100658_eval_vpt_ckpt/` — combined r1
- `20261106_104142_eval_vpt_ckpt/` — combined r2
- `20261106_111247_eval_vpt_ckpt/` — combined r3
- `20261106_102555_eval_vpt_ckpt/` — random r1
- `20261106_105837_eval_vpt_ckpt/` — random r2
- `20261106_113201_eval_vpt_ckpt/` — random r3
- `videos/manifest.json` — 10 recorded episodes (4 wins, 6 deaths)

Videos are stored in the group share, not committed to Git.

See `docs/m3-1m-flat-ppo.md` for the full analysis.
