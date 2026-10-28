# Experiments

Each run gets a timestamped directory:

```text
experiments/
└─ 20261105_143000_eval_vpt/
   ├─ config.yaml
   ├─ metrics.json
   ├─ episodes.csv
   ├─ git_commit.txt
   └─ notes.md
```

Large artifacts are ignored by git:

- videos,
- checkpoints,
- replay buffers,
- W&B local files.

## What to commit

- Resolved config.
- Metrics JSON and per-episode CSV.
- Git commit hash.
- A short `notes.md` with:
  - hypothesis,
  - change from the previous run,
  - seed range,
  - result,
  - next action.

## Naming

Use a short descriptive suffix:

- `eval_random`
- `eval_vpt_zero_shot`
- `ppo_zombie_v1`
- `ppo_zombie_sparse_reward`

## Reproducibility checklist

- [ ] Environment config is committed.
- [ ] Reward config is committed.
- [ ] Seed range is recorded.
- [ ] Evaluation uses fixed seeds.
- [ ] Baseline metrics are available for comparison.
- [ ] Video is stored outside git and linked in `notes.md`.
