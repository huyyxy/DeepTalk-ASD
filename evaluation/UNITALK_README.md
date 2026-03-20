# DeepTalk-ASD UniTalk 中文数据集验证系统

## 概述

`validate_unitalk.py` 使用 UniTalk 中文数据集对 DeepTalk-ASD 进行全流水线验证。脚本以视频帧和音频帧为输入（精确对齐），输出每个人脸 ID 说了哪段话，并与 CSV 标注对比生成 Markdown 格式的详细报告。

## 功能特性

1. **音视频精确对齐**: 检测 A/V 流 `start_time` 偏移，采样级对齐后喂入 ASD
2. **逐帧说话状态追踪**: 基于 VAD 轮次事件维护每帧每人脸的说话状态
3. **bbox IoU 人脸匹配**: 通过边界框 IoU 将 ASD track_id 映射到标注 entity_id
4. **双层评估指标**:
   - **ASD 指标**: 仅在人脸匹配成功的标注上评估（衡量 ASD 模型本身）
   - **端到端指标**: 含人脸检测缺失情况（衡量全流水线效果）
5. **语音段输出**: 记录每个 turn 的起止时间、说话人 ID、置信度
6. **Markdown 报告**: 混淆矩阵、逐视频指标、语音段明细

## 使用方法

### 基本用法

```bash
# 验证验证集
python3 demo/validate_unitalk.py --split val

# 限制处理前 5 个场景（快速测试）
python3 demo/validate_unitalk.py --split val --max-videos 5

# 自定义报告路径
python3 demo/validate_unitalk.py --split train --output-report train_report.md
```

### 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split` | (必填) | 数据集分割: `train` 或 `val` |
| `--dataset-dir` | `UniTalk_cn/data/videos/scenes` | 场景视频根目录 |
| `--csv-dir` | `UniTalk_cn/csv` | CSV 标注目录 |
| `--output-report` | `validation_report_{split}.md` | 输出报告文件名 |
| `--max-videos` | 无限制 | 最大处理场景数 |
| `--turn-detector` | `silero-vad` | 轮次检测器类型 |
| `--speaker-detector` | `LR-ASD-ONNX` | 说话者检测器类型 |

## 验证流程

```
CSV 标注加载 → 场景视频扫描 → 逐场景处理 → 评估匹配 → 生成报告
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
               音视频对齐    逐帧处理循环    语音段记录
               (ffprobe)   (音频追赶视频)   (turn events)
```

### 逐帧处理详情

1. **音频追赶**: 将 30ms 音频块喂入 ASD 直到追上当前视频帧时间
2. **轮次事件处理**:
   - `TURN_CONFIRMED`: 评估说话者，标记得分最高的 track 为说话状态
   - `SPEAKER_CHANGE`: 重新评估，关闭旧说话者段，开始新段
   - `TURN_END`: 记录完整语音段，清除说话状态
   - `TURN_REJECTED`: 清除说话状态（伪语音段）
3. **视频帧处理**: 检测人脸、归一化 bbox，记录当前说话状态

### 评估逻辑

对每个 CSV 标注 `(frame_timestamp, entity_id, bbox, label)`:

1. **时间匹配**: 找到 ±0.04s 内最近的检测帧
2. **人脸匹配**: 用 bbox IoU（阈值 0.3）将标注 entity 匹配到检测 track
3. **分类判定**:
   - 匹配成功 → 比较预测/实际说话状态 → TP/FP/TN/FN
   - 匹配失败 + 实际说话 → FN（端到端）
   - 匹配失败 + 实际不说话 → TN（端到端）

## 输出报告结构

报告包含 6 个章节:

| 章节 | 内容 |
|------|------|
| 1. 验证配置 | 模型配置、阈值参数 |
| 2. 总体性能指标 | ASD 指标 + 端到端指标 + 混淆矩阵 |
| 3. 语音段检测结果 | **哪个人脸说了哪段话** — 起止时间、Track ID、匹配 Entity、置信度 |
| 4. 各视频详细指标 | 按 video_id 分组的 ASD 指标 |
| 5. 场景处理统计 | 每个场景的时长、帧数、语音段数、处理耗时 |
| 6. 总结 | 汇总统计 + 性能结论 |

## 数据集结构

```
UniTalk_cn/
├── csv/
│   ├── train_orig.csv      # 训练集标注 (~234K 行)
│   └── val_orig.csv        # 验证集标注 (~105K 行)
└── data/videos/scenes/
    ├── train/{video_id}/    # 347 个场景视频
    └── val/{video_id}/      # 103 个场景视频
```

### CSV 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `video_id` | str | YouTube 视频 ID |
| `frame_timestamp` | float | 原始视频中的帧时间戳（秒） |
| `entity_id` | str | 实体 ID，格式 `{video_id}:{编号}` |
| `entity_box_{x1,y1,x2,y2}` | float | 归一化边界框 (0–1) |
| `label` | str | `SPEAKING_AUDIBLE` / `NOT_SPEAKING` |

### 场景视频命名

```
{video_id}_scene_{序号}_{起始时间}-{结束时间}.mp4
```

时间为原始视频中的 **带缓冲** 时间（标注范围 ± 0.5s），因此场景 time=0 对应原始视频的起始时间。

## 关键设计说明

### 时间映射

场景文件名编码了视频在原始视频中的起止时间（含 0.5s 缓冲）:
- `scene_time = 0` → `original_time = filename_start`
- `scene_time = T` → `original_time = filename_start + T`

### Top-1 Speaker 策略

每个 turn 仅标记得分最高的 track 为"说话"。原因:
- 标准 ASD 评估假设每个 turn 只有一个主说话者
- ASD 模型可能对非说话人脸也给出轻微正分（唇部运动噪声）
- Top-1 策略显著提升精确率，F1 提升明显

### VAD 系统性延迟

由于 turn-based 检测的固有特性：
- **~200ms 确认延迟**: 语音段开始后需积累足够帧才确认
- **~500ms 尾部容忍**: 静音需持续 500ms 才判定 turn 结束
- 短于 200ms 的语音段会被 VAD 拒绝

这些导致 per-frame 召回率低于理论值，但语音段级别的检测是准确的。

## 性能基准

| 等级 | ASD F1 | 说明 |
|------|--------|------|
| 优秀 | >= 0.8 | 模型表现出色 |
| 良好 | >= 0.6 | 仍有改进空间 |
| 不足 | < 0.6 | 需要进一步优化 |

> 注：Per-frame F1 受 VAD 延迟影响较大，实际 turn-level 检测质量通常优于 per-frame 数值。

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| ffmpeg 未找到 | `brew install ffmpeg` (macOS) 或 `apt install ffmpeg` (Linux) |
| 所有指标为 0 | 检查 CSV 路径和场景视频是否匹配 |
| 处理速度慢 | 用 `--max-videos` 减少数量 |
| 内存不足 | 减少同时处理的场景数 |
