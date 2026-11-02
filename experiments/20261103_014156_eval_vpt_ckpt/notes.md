# Defense probe — control (50 seeds)

Checkpoint: `experiments/probe_easy_control/checkpoints/2026-11-03_01-10-24/20/`  
Training config: `configs/env/zombie_stage1_easy_mask.yaml`  
Eval config: `configs/env/zombie_stage1_easy.yaml`  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.26 death_rate=0.62 timeout_rate=0.12 stuck_rate=0.08
mean_hp_lost=15.20 mean_hp_lost_on_win=9.69
mean_steps=305.66 mean_time_to_kill=195.85 mean_reward=+0.21
```

A 10k continuation of the easy-AI curriculum. This is the control for the
defense probes; it is statistically unchanged from the 100k parent (30% win,
58% death).
