# Defense probe — far spawn, evaluated on easy (50 seeds)

Checkpoint: `experiments/probe_easy_far/checkpoints/2026-11-03_01-18-07/20/`  
Training config: `configs/env/zombie_stage1_easy_far_mask.yaml` (5–7 blocks)  
Eval config: `configs/env/zombie_stage1_easy.yaml` (3–4 blocks)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.52 death_rate=0.40 timeout_rate=0.08 stuck_rate=0.08
mean_hp_lost=11.64 mean_hp_lost_on_win=6.81
mean_steps=243.90 mean_time_to_kill=184.00 mean_reward=+3.89
```

Training with a farther spawn improved the close-range task: 52% win and 40%
death, versus 26% / 62% for the control. This is the first checkpoint above the
50% stage-1 win threshold on any task, and the best result in the project.

The policy also takes less damage per win (6.8 HP), which suggests the far
curriculum taught approach and spacing rather than only attack spam.
