# Continuation checkpoint evaluation — 1.75M random

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/`  
Task: random 360  
Protocol: 1x50 local, max 600 steps, deterministic

```text
win_rate=0.20 death_rate=0.42 timeout_rate=0.38
mean_hp_lost=14.98 mean_steps=474.80
mean_time_to_kill=411.3 mean_reward=-1.02
```

Local 4060 evaluation, used for the trend only. The final 2M gate evaluation is
on the RTX 4090D. See `docs/m3-2m-continuation.md`.
