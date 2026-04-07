# DeepTalk-ASD

**DeepTalk-ASD** is an industrial-grade **Active Speaker Detection (ASD)** library. It integrates multimodal fusion (Audio/Visual), speaker verification, and real-time processing support, optimized for CPU execution via ONNX.

---

## 1. What is DeepTalk-ASD?

DeepTalk-ASD determines in real-time which person in a video is speaking by fusing:
- **Visual Features**: Mouth movements and face tracking.
- **Audio Features**: Speech patterns (VAD) and MFCC.
- **Speaker Embeddings (Voiceprint)**: Advanced matching to effectively handle off-screen voices and speaker overlaps.

It is built upon the SOTA [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) model but provides a complete, production-ready pipeline that runs efficiently on standard CPUs without requiring large GPU environments.

## 2. Key Features

- **High Performance**: Full-pipeline ONNX inference optimized for CPU, supporting real-time operation on laptops and mobile devices.
- **Multimodal Fusion**: Deeply integrates face detection, voice activity detection (VAD), and audio-visual ASD models.
- **Speaker Verification**: Fuses ASD scores with speaker embeddings (Voiceprint) to suppress off-screen interference and "reward" matching tracks.
- **Zero-Configuration**: Models (~46 MB) are automatically downloaded and cached on first use.
- **Modular Design**: Flexible components for Face Detection, Turn Detection (VAD/pVAD), and Speaker Decision.

## 3. Quick Start

### Installation

```bash
pip install deeptalk-asd
```

### Basic Usage

```python
import time
from deeptalk_asd import ASDDetectorFactory, VideoFrame, AudioFrame

# 1. Initialize the detector
# Models are automatically downloaded to ~/.cache/deeptalk_asd/ on first run
asd = ASDDetectorFactory().create()

# 2. In your processing loop:
# Append Video Frame (requires 25 fps alignment)
# video_data: BGR image (H, W, 3)
v_frame = VideoFrame(data=video_data)
asd.append_video(v_frame, create_time=time.perf_counter())

# Append Audio Block (16kHz, 16-bit mono PCM)
# audio_data: np.ndarray
a_frame = AudioFrame(data=audio_data)
asd.append_audio(a_frame, create_time=time.perf_counter())

# 3. Evaluate results
# returns a dict mapping track_id to speaking confidence score
results = asd.evaluate()
for track_id, score in results.items():
    if score > 0:
        print(f"Tracking ID {track_id} is active (Confidence: {score:.2f})")
```

## 4. Configuration

### 4.1 Component Engines
You can customize the underlying face, turn, or speaker engines through the factory:

```python
config = {
    "face_detector": {"type": "inspireface"},
    "turn_detector": {"type": "silero-vad"},  # or "pvad"
    "speaker_detector": {"type": "LR-ASD-ONNX"}
}
asd = ASDDetectorFactory(**config).create()
```

### 4.2 pVAD Configuration (Speaker-Change Detection)

pVAD (Personal VAD) enhances standard VAD by adding **speaker-change detection**. When the system detects that the current speaker has changed, it triggers a `SPEAKER_CHANGE` event, allowing the ASD system to respond faster in multi-speaker scenarios.

```python
config = {
    "turn_detector": {
        "type": "pvad",
        "pvad_model_dir": "weights",               # Required: Directory for pVAD models
        "pvad_model_name": "pvad.onnx",             # Optional: pVAD model filename
        "spk_model_name": "speaker_model.onnx",     # Optional: 192-dim speaker embedding model
        "pvad_threshold": 0.35,                     # Probability threshold for target speaker
        "min_low_frames": 30,                       # Frames to wait before triggering change
        "cooldown_frames": 50,                      # Cooldown period to prevent false triggers
        "vad": {                                    # Internal VAD config (decorator pattern)
            "type": "silero-vad",
            "model_dir": "weights"
        }
    }
}
asd = ASDDetectorFactory(**config).create()
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pvad_model_dir` | str | *Required* | Directory containing pVAD and 192-dim speaker models |
| `pvad_threshold` | float | `0.35` | Probabilities below this are treated as "non-target" |
| `min_low_frames` | int | `30` | Minimum frames below threshold to trigger `SPEAKER_CHANGE` |
| `vad` | dict | `{"type": "silero-vad"}` | Internal VAD engine wrapped by pVAD |

### 4.3 Environment Variables & Offline Mode

You can control the framework globally using these environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEEPTALK_ASD_OFFLINE` | _(unset)_ | Set to `1` to disable all network requests (requires pre-downloaded models) |
| `DEEPTALK_ASD_CACHE_DIR` | `~/.cache/deeptalk_asd/` | Custom directory to store or load model weights |

## 5. Model Management

Models are automatically downloaded on first use. In restricted environments, you can manage them manually:

- **Automatic**: Default behavior (Downloads ~46 MB to cache).
- **Manual Download**: 
  ```bash
  python3 -m deeptalk_asd download-models --cache-dir /path/to/my_weights
  ```
- **Check Info**:
  ```bash
  python3 -m deeptalk_asd info
  ```

---

## [中文说明]

**DeepTalk-ASD** 是一个工业级的**活跃说话者检测 (Active Speaker Detection, ASD)** 库。它集成了多模态融合（音视频）、声纹验证和实时处理支持，并针对 ONNX 在 CPU 上的执行进行了深度优化。

---

### 1. 什么是 DeepTalk-ASD？

DeepTalk-ASD 通过融合以下信息，实时判定视频中哪个人正在说话：
- **视觉特征**：嘴部动作和人脸追踪。
- **音频特征**：语音模式 (VAD) 和 MFCC。
- **声纹嵌入 (Voiceprint)**：高级声纹匹配，可有效处理画外音干扰和说话人重叠。

本项目基于 SOTA 的 [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) 模型构建，但提供了一个完整的、生产级别的流水线，无需庞大的 GPU 环境即可在标准 CPU 上高效运行。

### 2. 核心特性

- **高性能**：全链路 ONNX 推理，针对 CPU 优化，支持在笔记本电脑和移动设备上实时运行。
- **多模态融合**：深度集成人脸检测、语音活动检测 (VAD) 和音视频 ASD 模型。
- **声纹验证**：将 ASD 得分与声纹嵌入结合，压制画外音干扰并“奖励”匹配的追踪目标。
- **零配置**：模型（约 46 MB）在首次使用时会自动下载并缓存。
- **模块化设计**：提供人脸检测、轮次检测 (VAD/pVAD) 和说话人判定等灵活组件。

### 3. 快速开始

#### 安装

```bash
pip install deeptalk-asd
```

#### 基本用法

```python
import time
from deeptalk_asd import ASDDetectorFactory, VideoFrame, AudioFrame

# 1. 初始化检测器
# 模型在首次运行时会自动下载到 ~/.cache/deeptalk_asd/
asd = ASDDetectorFactory().create()

# 2. 在处理循环中：
# 添加视频帧（需要对齐到 25 fps）
# video_data: BGR 图像 (H, W, 3)
v_frame = VideoFrame(data=video_data)
asd.append_video(v_frame, create_time=time.perf_counter())

# 添加音频块（16kHz, 16-bit 单声道 PCM）
# audio_data: np.ndarray
a_frame = AudioFrame(data=audio_data)
asd.append_audio(a_frame, create_time=time.perf_counter())

# 3. 评估结果
# 返回一个字典，映射 track_id 到说话置信度得分
results = asd.evaluate()
for track_id, score in results.items():
    if score > 0:
        print(f"追踪 ID {track_id} 正在说话 (置信度: {score:.2f})")
```

### 4. 配置说明

#### 4.1 组件引擎
你可以通过工厂方法自定义底层的人脸、轮次或说话人引擎：

```python
config = {
    "face_detector": {"type": "inspireface"},
    "turn_detector": {"type": "silero-vad"},  # 或 "pvad"
    "speaker_detector": {"type": "LR-ASD-ONNX"}
}
asd = ASDDetectorFactory(**config).create()
```

#### 4.2 pVAD 配置（说话人切换检测）

pVAD (Personal VAD) 在标准 VAD 基础上增加了**说话人切换检测**。当系统检测到当前说话人发生变化时，会触发 `SPEAKER_CHANGE` 事件，使 ASD 系统在多人通话场景中响应更迅速。

```python
config = {
    "turn_detector": {
        "type": "pvad",
        "pvad_model_dir": "weights",               # 必填：pVAD 模型目录
        "pvad_model_name": "pvad.onnx",             # 可选：pVAD 模型文件名
        "spk_model_name": "speaker_model.onnx",     # 可选：192 维声纹模型
        "pvad_threshold": 0.35,                     # 目标说话人概率阈值
        "min_low_frames": 30,                       # 触发切换前等待的帧数
        "cooldown_frames": 50,                      # 防止误触的冷却期
        "vad": {                                    # 内部 VAD 配置（装饰器模式）
            "type": "silero-vad",
            "model_dir": "weights"
        }
    }
}
asd = ASDDetectorFactory(**config).create()
```

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `pvad_model_dir` | str | *必填* | 包含 pVAD 和 192 维声纹模型的目录 |
| `pvad_threshold` | float | `0.35` | 低于此概率将被视为“非目标说话人” |
| `min_low_frames` | int | `30` | 触发 `SPEAKER_CHANGE` 所需的持续低于阈值的最小帧数 |
| `vad` | dict | `{"type": "silero-vad"}` | 被 pVAD 包装的内部 VAD 引擎 |

#### 4.3 环境变量与离线模式

你可以使用以下环境变量全局控制框架：

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `DEEPTALK_ASD_OFFLINE` | _(未设置)_ | 设为 `1` 以禁用所有网络请求（需要预下载模型） |
| `DEEPTALK_ASD_CACHE_DIR` | `~/.cache/deeptalk_asd/` | 自定义模型权重存储或加载目录 |

### 5. 模型管理

模型在首次使用时自动下载。在受限环境中，你可以手动管理它们：

- **自动下载**：默认行为（下载约 46 MB 到缓存）。
- **手动下载**：
  ```bash
  python3 -m deeptalk_asd download-models --cache-dir /path/to/my_weights
  ```
- **查看信息**：
  ```bash
  python3 -m deeptalk_asd info
  ```

---

## License

Code is licensed under **MIT License**. Integrated models are subject to their respective licenses (InspireFace, Silero VAD, LR-ASD, Sherpa-ONNX). Please verify compliance for commercial use.
