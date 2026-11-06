# 1M checkpoint evaluation — random r2

Checkpoint: `experiments/flat_1m_mixed_angle30_4090d/checkpoints/2026-11-06_08-05-32/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Protocol: 50 seeds, rep 2, max 600 steps, deterministic

```text
win_rate=0.26 death_rate=0.66 timeout_rate=0.08
mean_hp_lost=16.46 mean_hp_lost_on_win=13.153846153846153
mean_steps=332.38 mean_time_to_kill=289.2307692307692
mean_reward=-0.31
```

Part of the 1M flat PPO evaluation. See `docs/m3-1m-flat-ppo.md`.
