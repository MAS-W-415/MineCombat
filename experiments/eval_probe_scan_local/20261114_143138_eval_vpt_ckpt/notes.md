# Scan-at-reset probe evaluation

Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/`  
Config: `configs/env/zombie_stage1_random_scan.yaml`  
Protocol: 50 seeds, max 600 steps, deterministic, local RTX 4060

```text
win_rate=0.00 death_rate=0.46 timeout_rate=0.54
stuck_rate=0.54 mean_hp_lost=11.28
mean_steps=479.10 mean_reward=-2.69
```

Negative result: 0 kills, 54% timeouts. See `docs/m3-scan-at-reset.md`.
