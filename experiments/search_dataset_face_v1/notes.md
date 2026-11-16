# Search dataset — face group (v1)

Checkpoint: `experiments/flat_1m_cont_mixed_angle30_4090d/checkpoints/2026-11-06_11-45-28/20/`  
Config: `configs/env/zombie_stage1_random_face.yaml`  
Seeds: 2000–2009, max 300 steps, frame stride 10  
Branch: `feat/closed-loop-search` (`7b0edc2`)

## Result

```text
episodes: 10
PNG frames: 255
videos: 10
wall time: 260.81 s
outcomes: success=5, death=2, ongoing=3
```

Per-seed frames: 2000=14, 2001=31, 2002=22, 2003=31, 2004=28,
2005=27, 2006=31, 2007=23, 2008=26, 2009=22.

All frames are 128×128 RGB, non-black, non-duplicate, and the PNG count matches
`manifest.jsonl` (255/255). Early step-0 frames can be before the target is
rendered; later frames contain visible targets.

## Files

- Raw frames + manifest: `policy_frames/`
- Video manifest: `manifest.json`
- Shared delivery:
  `/eaas/default/groups/mine_/home/share/minecombat_results/search_dataset_v1/face/`
- Representative videos:
  `.../search_dataset_v1/videos/face/` (2000, 2004, 2002, 2001, 2006)
- Sample previews: `.../search_dataset_v1/samples/face/`

See `docs/m3-search-dataset-v1.md` for the full acceptance record.
