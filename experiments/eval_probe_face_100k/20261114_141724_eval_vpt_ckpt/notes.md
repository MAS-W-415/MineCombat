# Face-curriculum probe evaluation — probe combined

Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/`  
Config: `configs/env/zombie_stage1_easy_mixed_angle30_mask.yaml`  
Protocol: 50 seeds, max 600 steps, deterministic, local RTX 4060

```text
win_rate=0.68 death_rate=0.26 timeout_rate=0.06
mean_hp_lost=11.22 mean_steps=249.92
mean_time_to_kill=221.35294117647058 mean_reward=5.84
```

See `docs/m3-face-curriculum-probe.md`.
