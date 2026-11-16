# Mob Radar YOLO26 Nano — MineCombat 零微调验证报告

Date: 2026-11-16  
Branch: `feat/closed-loop-search`  
Commit: `0539f3e64e21f44048ee7cde4f586520be89ab7c`  
任务性质：只验证第三方现成权重，不训练任何模型。

## 1. 结论

**情况 B：能识别 Zombie，但漏检较多，暂不建议直接跳过人工标注和训练进入正式 Closed-loop Search 性能实验。**

- `imgsz=320` 图像级 Frame Recall：**56.4%**（22/39）
- `imgsz=320` False Positive Rate：**0.0%**（0/41）
- 检测到的 22 个 Zombie 框，人工复核均能框住目标；框定位正确率 **100%**
- 主要问题是漏检，尤其是极近距离、画面边缘、远距离/过小目标和特殊光照
- `imgsz=640` 没有改善召回，反而降到 **43.6%**（17/39）
- 建议保留该权重作为后续少量域内微调的初始化，而不是直接作为最终检测器

这不是标准目标检测 mAP。当前没有逐框真值标注，只报告图像级指标。

## 2. 模型和权重

| 项目 | 结果 |
|---|---|
| 模型 | `mob_radar` YOLO26 Nano（Ultralytics YOLO26n） |
| 来源 | `https://github.com/draco-tlw/mob-radar` |
| 权重 | `runs/detect/mob_radar_v26n_dv5_sz1024_f11_lr1e-2_lrf1e-2_cos/weights/best.pt` |
| 文件大小 | 5,432,709 bytes（与 GitHub 标注一致） |
| SHA256 | `653542fd3af2516f9edbc25725991753a5daecce4a6917ea6a8ae76aca70aa57` |
| 文件格式 | PyTorch ZIP checkpoint，不是 HTML/LFS 指针 |
| 任务 | `detect` |
| 参数量 | 2,505,750 |
| 计算量 | 5.9 GFLOPs |
| 类别 | `creeper`, `skeleton`, `spider`, `zombie`, `enderman` |

**模型加载：PASS**  
**`zombie` 类别存在：PASS**  
**MineCombat `UltralyticsBoxPredictor` + `RGBTargetDetector` 适配器：PASS**

适配器测试使用 `min_confidence=0.5`，在 face 前 5 张中返回了
`center_x` 和 `confidence`，其中目标明显的图片返回置信度约 `0.79–0.90`，
无目标的图片返回 `center_x=None`。

## 3. 测试数据

任务要求的 40 张代表性样本 + 40 张补充样本，共 **80 张真实 128×128 Policy RGB PNG**。

| 项目 | 数量 |
|---|---:|
| 总图片 | 80 |
| Face 组 | 40 |
| Random 组 | 40 |
| 人工确认 Zombie 正样本 | 39 |
| 人工确认背景负样本 | 41 |

补充样本来自不同 Episode/step，目标是让远距离、边缘、近距离和背景场景
都进入审计集。审计集是**有意增强正负样本**的集合，不代表原始数据中的自然
正样本比例。

## 4. 检测结果

### 4.1 `imgsz=320`（主结果）

推理参数：

```text
model: mob_radar_best.pt
imgsz: 320
conf: 0.25
device: cuda:0
batch: 1
```

| 指标 | 结果 |
|---|---:|
| Positive Frames | 39 |
| Negative Frames | 41 |
| True Positive | 22 |
| False Positive | 0 |
| False Negative | 17 |
| Frame Recall | **56.4%**（22/39） |
| False Positive Rate | **0.0%**（0/41） |
| Correct Localization | **100%**（22/22 个预测正框） |

分组：

| 组 | 正样本 | TP | FN | Recall | FP | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Face | 32 | 20 | 12 | 62.5% | 0 | 0% |
| Random | 7 | 2 | 5 | 28.6% | 0 | 0% |

Random 组只有 7 张人工确认的正样本，数量较少；该组召回率只能作为
初步信号，不能作为稳定估计。

### 4.2 `imgsz=640`（对照）

| 指标 | 320 | 640 |
|---|---:|---:|
| TP | 22 | 17 |
| FN | 17 | 22 |
| Frame Recall | **56.4%** | **43.6%** |
| FPR | 0% | 0% |
| Correct Localization | 100% | 100% |

提高输入分辨率没有改善召回。对该权重和当前 128×128 源图，**320 更合适**。

## 5. 推理延迟

测试设备：NVIDIA RTX 4090 D 24GB，CUDA 0，batch=1。  
测量方式：10 张 warmup 后，对 80 张图跑 2 轮，`model.predict` 前后
`torch.cuda.synchronize()`，记录端到端耗时。

| 输入 | 平均 | 中位数 | P95 | 模型内部 inference |
|---|---:|---:|---:|---:|
| 320×320 | 10.14 ms | 9.60 ms | 12.74 ms | 7.15 ms |
| 640×640 | 9.87 ms | 9.73 ms | 10.33 ms | 6.92 ms |

单张图延迟都在约 10 ms 量级；640 在这个小模型上没有明显变慢，但召回更差，
因此没有理由用 640 替代 320。该延迟只代表单环境单张推理，不代表端到端
Minecraft 闭环 FPS。

## 6. 主要失败场景

### 漏检

- 极近距离，Zombie 头部/身体占满画面：`seed_2000_step_00130`、
  `seed_2002_step_00110`、`seed_2006_step_00300`
- 目标贴近画面边缘或被截断：`extra_seed_2008_step_00180`
- 远距离或极小目标：`extra_seed_2001_step_00060`、
  `extra_seed_2005_step_00060`、`extra_seed_3003_step_00030`
- 特殊光照/阴影：`seed_2008_step_00250`（日落暖光下的 Zombie）
- 近距离但模型没有定位：`seed_3008_step_00300`、
  `extra_seed_3000_step_00170`

### 错分类

- `seed_3001_step_00270`：明显的 Zombie 被分类为 `creeper`，
  置信度 `0.64`，因此没有通过 Zombie 类别过滤。
- 部分图片还出现低置信度 `skeleton 0.51` 或 `creeper 0.48` 的额外框，
  但 Zombie 框本身仍存在。

### 误报

- 41 张背景负样本中，没有出现置信度 ≥0.5 的 Zombie 误报。
- 这是本批小样本上的结果；不能据此推断真实 360° 场景不会误报。

## 7. 交付文件

共享目录：

```text
/eaas/default/groups/mine_/home/share/minecombat_results/mob_radar_zeroshot_v1/
```

关键文件：

```text
models/mob_radar_best.pt
models/...                         # 第三方权重
input/face/                        # 40 张 face 测试图
input/random/                      # 40 张 random 测试图
predictions/face/                  # 320 预测 JPG + 标注 PNG + labels
predictions/random/
predictions_640/face/              # 640 对照
predictions_640/random/
audit.csv                          # 320 逐图审计表
audit_640.csv                      # 640 对照审计表
audit_metrics_320.json
audit_metrics_640.json
predictions_320_summary.json
predictions_640_summary.json
latency_320_640.json
adapter_face5.log
model_info.log
model_sha256.txt
audit_sheet_face_320.png
audit_sheet_random_320.png
predicted_positive_sheet_320.png
predicted_positive_sheet_640.png
examples/
report.md
README.md
environment.txt
```

`audit.csv` 的字段：

```text
image,group,human_has_zombie,pred_has_zombie,box_correct,notes
```

`box_correct` 只在人工确认目标且模型有检测时为 1；本批 320 的 22 个
预测正框全部被人工判为正确。

## 8. 典型案例

- 成功：`examples/success/`
- 远距离成功：`examples/far/`
- 漏检：`examples/miss/`
- 类别混淆：`examples/class_confusion/`
- 背景负样本：`examples/background/`
- 误报：本批 41 张负样本中没有 Zombie 误报，因此没有误报样例

每个案例同时保留原始 128×128 PNG 和 320 标注 PNG。

## 9. 最终建议

**不建议把当前权重直接接入正式 360° 搜索性能实验。**

理由：

1. 320 的图像级召回只有 56.4%，低于任务中“情况 A”参考的约 80%。
2. 失败集中在搜索最需要处理的场景：远距离、边缘、近距离遮挡和特殊光照。
3. 640 没有改善召回，说明单纯提高分辨率不是解决方案。
4. 没有真实标注框，无法给出 mAP；当前结果只能作为工程筛选。

建议下一步：

1. 从 560 张数据中选出 100–200 张，按 Episode 划分并标注 Zombie 框；
2. 对确认无目标的图片生成空标签；
3. 以 `mob_radar_best.pt` 为初始化做少量域内微调，而不是从零训练；
4. 微调后重新按本报告的 80 张审计集和 3×50 评估流程复测；
5. 在真实 `best.pt` 指标达到可接受水平前，不做正式 Closed-loop Search
   性能结论。

如果只是做一次工程闭环 smoke，可以暂时使用 320 和当前权重，但必须明确
标注为“工程验证”，不能当作 Stage-1b 搜索能力结论。
