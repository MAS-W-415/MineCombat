# Face-curriculum probe evaluation — probe face (100k)

Checkpoint: `experiments/probe_face_100k/checkpoints/2026-11-14_13-27-50/2/`  
Config: `configs/env/zombie_stage1_random_face.yaml`  
Protocol: 50 seeds, max 600 steps, deterministic, local RTX 4060

```text
win_rate=0.52 death_rate=0.34 timeout_rate=0.14
mean_hp_lost=11.48 mean_steps=316.94
mean_time_to_kill=233.46153846153845 mean_reward=3.82
```

See `docs/m3-face-curriculum-probe.md`.
