# 2M flat PPO evaluation

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Code: `b54c53d`  
Seeds: 0–49, max 600 steps, deterministic

## Results

| Task | Win | Death | HP lost | Time-to-kill |
|---|---:|---:|---:|---:|
| easy 3–4 | 0.90 | 0.10 | 6.96 | 166.7 |
| ±30° close | 0.64 | 0.30 | 11.34 | 189.0 |
| combined 3–7 + ±30° (3×50 mean) | **0.620** | 0.340 | 12.15 | 212.1 |
| random 360° (3×50 mean) | **0.100** | 0.560 | 14.93 | 276.1 |

Ranges: combined 0.58–0.62; random 0.06–0.14.

## Directories

- `20261106_133111_eval_vpt_ckpt/` — easy, 1×50
- `20261106_134445_eval_vpt_ckpt/` — ±30°, 1×50
- `20261106_135858_eval_vpt_ckpt/` — combined r1
- `20261106_143436_eval_vpt_ckpt/` — combined r2
- `20261106_150837_eval_vpt_ckpt/` — combined r3
- `20261106_142035_eval_vpt_ckpt/` — random r1
- `20261106_145425_eval_vpt_ckpt/` — random r2
- `20261106_152754_eval_vpt_ckpt/` — random r3
- `videos/manifest.json` — 10 recorded episodes

Videos are in the group share, not committed to Git.

See `docs/m3-2m-continuation.md` for the full analysis and the Stage-1a/1b
decision.
