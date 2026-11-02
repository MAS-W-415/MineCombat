# 100k stationary + attack shaping — 50-seed evaluation

Checkpoint: `experiments/flat_100k_stationary_attack/checkpoints/2026-11-02_19-30-36/2/model.ckpt`  
Config: `configs/env/zombie_stage1_stationary.yaml` (NoAI zombie, 2.5–3.5 blocks in front)  
Seeds: 0–49, max 600 steps, deterministic

```text
win_rate=0.00 death_rate=0.00 timeout_rate=1.00 stuck_rate=0.86
mean_hp_lost=0.56 mean_steps=600.00 mean_reward=-0.62
damage_ratio=1.67
```

The damage ratio is misleading because HP loss is tiny. More useful:

- 7/50 episodes dealt damage (previous stationary baseline: 0/50).
- Maximum damage: 11.8 HP (zombie has 20 HP).
- Mean damage: 0.94 HP/episode.
- Still 0 kills; all episodes timed out.

The attack-attempt bonus increased engagement, but the policy still does not finish
the kill. Attack action frequency remains low in short diagnostics.
