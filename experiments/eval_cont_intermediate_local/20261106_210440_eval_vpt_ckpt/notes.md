# Continuation checkpoint evaluation — 1.75M combined

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/`  
Task: combined 3-7 + ±30  
Protocol: 1x50 local, max 600 steps, deterministic

```text
win_rate=0.64 death_rate=0.32 timeout_rate=0.04
mean_hp_lost=12.78 mean_steps=248.76
mean_time_to_kill=222.78125 mean_reward=5.05
```

Local 4060 evaluation, used for the trend only. The final 2M gate evaluation is
on the RTX 4090D. See `docs/m3-2m-continuation.md`.
