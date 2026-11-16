# Search dataset — random group (v1)

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1.yaml`  
Seeds: 3000–3009, max 300 steps, frame stride 10  
Branch: `feat/closed-loop-search` (`7b0edc2`)

## Result

```text
episodes: 10
PNG frames: 305
videos: 10
wall time: 276.59 s
outcomes: success=0, death=2, ongoing=8
```

Per-seed frames: 3000=29, 3001=28, all other seeds=31.

All frames are 128×128 RGB, non-black, non-duplicate, and the PNG count matches
`manifest.jsonl` (305/305). Most frames have no visible target, which is useful
as negative data; a few episodes contain close/visible targets.

## Files

- Raw frames + manifest: `policy_frames/`
- Video manifest: `manifest.json`
- Shared delivery:
  `/eaas/default/groups/mine_/home/share/minecombat_results/search_dataset_v1/random/`
- Representative videos:
  `.../search_dataset_v1/videos/random/` (3000, 3001, 3008, 3002, 3005)
- Sample previews: `.../search_dataset_v1/samples/random/`

See `docs/m3-search-dataset-v1.md` for the full acceptance record.
