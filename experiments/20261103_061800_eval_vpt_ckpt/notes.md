# Joint ±30 checkpoint — combined task, repeat 3 (50 seeds)

Same checkpoint, config, and seeds as `20261103_061100_eval_vpt_ckpt/`.

```text
win_rate=0.20 death_rate=0.60 mean_hp_lost=15.38
```

Spawn distance and angle matched repeat 2 for 50/50 seeds, but only 38/50
episode outcomes and 26/50 death flags agreed. This proves that the mob AI, not
the spawn RNG, is the source of evaluation variance. See
`docs/m3-evaluation-variance.md`.
