# 100k face-curriculum probe

Source: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Env: `configs/env/zombie_stage1_random_face.yaml`  
Code: local `main` after PR #65  
Hardware: local RTX 4060, 31 GB RAM

```text
steps: 102,400
iterations: 2
wall time: 1,756.37 s (29.3 min)
checkpoints: 1 and 2 retained
NaN / OOM: none
```

Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/model.ckpt`

Evaluation: `experiments/eval_probe_face_100k/` and
`docs/m3-face-curriculum-probe.md`.
