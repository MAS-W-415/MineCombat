# Mob Radar zero-shot detector validation — v1

Date: 2026-11-16  
Branch: `feat/closed-loop-search` @ `0539f3e`  
Model: `mob_radar` YOLO26 Nano (Ultralytics 8.4.215)  
Task: zero-shot validation only; no training and no weight changes.

## Outcome

**Case B — detects Zombie, but misses too many targets for direct integration
into the formal 360° search evaluation.**

| Metric (imgsz=320) | Value |
|---|---:|
| Audit images | 80 |
| Human-positive frames | 39 |
| Human-negative frames | 41 |
| True positives | 22 |
| False positives | 0 |
| False negatives | 17 |
| Frame Recall | 56.4% |
| False Positive Rate | 0.0% |
| Correct localization among predicted positives | 100% (22/22) |
| Mean end-to-end latency | 10.14 ms |

`imgsz=640` lowered recall to 43.6% (17/39). The 320 result is the better
setting for this small source image.

## Artifacts

Shared root:

```text
/eaas/default/groups/mine_/home/share/minecombat_results/mob_radar_zeroshot_v1/
```

The shared package contains the third-party weights, 80 input PNGs, 320/640
predictions and labels, `audit.csv`, metrics, contact sheets, examples, logs,
and checksums. The model weight stays out of Git.

GitHub stores this small record and `docs/m3-mob-radar-zeroshot.md`.

## Recommendation

Do not use this checkpoint as the final detector for a 360° performance claim.
Use it as an initialization for a small in-domain fine-tune after 100–200 frames
are annotated (with empty labels for negatives and an episode-level split).
A 320-based closed-loop smoke is acceptable only as an engineering check.
