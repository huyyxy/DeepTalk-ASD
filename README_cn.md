# DeepTalk-ASD

[LR-ASD](https://github.com/Junhua-Liao/LR-ASD) 是 SOTA 的活跃说话者检测模型。本项目在 LR-ASD 的基础上，提供了一个**全流程 ONNX 化、集成声纹识别、支持实时处理**的工业级 ASD 系统，无需依赖庞大的 PyTorch/GPU 环境即可在 CPU 上高效运行。

---

DeepTalk-ASD 是一个高效的活跃说话者检测 (Active Speaker Detection, ASD) 系统。它通过融合音频、视频特征以及声纹信息，实时判定视频画面中哪个人脸正在说话。

## 核心特性

- **多模态融合**: 深度结合人脸检测、语音轮次检测 (VAD)、音视频 ASD 模型以及**声纹特征 (Speaker Embedding)**。
- **声纹增强 (New)**: 
    - **特征融合**: 将 ASD 概率与音频声纹相似度加权融合，有效压制画外音干扰。
    - **快速匹配**: 支持对已知声纹的 Track 进行加速判定，跳过复杂计算。
    - **自动档案更新**: 针对检测到的说话人，利用 EMA (滑动平均) 自动更新其声纹特征档案。
- **模块化设计**: 
    - **FaceDetector**: 提供人脸检测与追踪（当前主流支持 InspireFace）。
    - **TurnDetector**: 音频 VAD 与轮次管理（集成 Silero VAD）。
    - **SpeakerDetector**: 核心决策层，执行音视频特征提取、融合决策及声纹比对（基于 LR-ASD ONNX）。
- **高性能**: 全链路 ONNX 推理，针对 CPU 优化，支持在移动端或普通笔记本上实时运行。
- **模型自动管理**: 首次使用时自动下载模型；支持离线模式和自定义缓存目录。
- **多场景 Demo**: 提供实时摄像头、视频文件、声纹提取及 pVAD 等多个演示方案。

## 系统架构

系统由三个主要子组件协同工作：
1. **人脸检测 (FaceDetector)**: 在每一帧视频中定位并追踪人脸（Track）。
2. **轮次检测 (TurnDetector)**: 判断当前音频流是否包含语音段及其生命周期（START, CONTINUE, END）。
3. **说话者检测 (SpeakerDetector)**: 
    - 提取音频 MFCC 特征与 112x112 嘴部灰度图。
    - 结合 **Sherpa-ONNX** 提取声纹特征（Voiceprint）。
    - 通过三阶段 ONNX 模型 (Audio/Visual Frontend + AV Backend) 计算 ASD 原始得分。
    - 最终得分由 ASD 得分与声纹匹配得分动态融合而得。

## 快速开始

### 1. 安装

```bash
pip install deeptalk_asd
```

### 2. 零配置使用

首次使用时**自动下载**所需模型（约 46 MB），无需手动配置：

```python
from deeptalk_asd import ASDDetectorFactory

asd = ASDDetectorFactory().create()
```

### 3. 运行 Demos

#### 核心 ASD 演示

*   **实时摄像头演示** (推荐首选):
    ```bash
    python3 demo/realtime_asd_demo.py
    ```
*   **视频文件离线处理**:
    ```bash
    python3 demo/video_asd_demo.py --input demo/demo.mp4 --display
    ```

## 模型管理

### 自动下载（默认）

模型文件在首次使用时自动下载到 `~/.cache/deeptalk_asd/`，后续运行直接从缓存加载。

| 模型 | 大小 | 说明 |
| :--- | ---: | :--- |
| `audio_frontend.onnx` | 0.9 MB | LR-ASD 音频前端 |
| `visual_frontend.onnx` | 1.5 MB | LR-ASD 视觉前端 |
| `av_backend.onnx` | 0.8 MB | LR-ASD 音视频融合后端 |
| `silero_vad.onnx` | 2.2 MB | Silero VAD 语音轮次检测 |
| `Pikachu` | 16 MB | InspireFace 人脸检测资源包 |
| `wespeaker_zh_cnceleb_resnet34.onnx` | 25 MB | WeSpeaker 声纹特征提取 |

### 预下载模型（离线环境）

```bash
# 下载所有模型
python3 -m deeptalk_asd download-models

# 下载到指定目录（方便拷贝到离线机器）
python3 -m deeptalk_asd download-models --cache-dir /path/to/models

# 查看缓存状态
python3 -m deeptalk_asd info
```

### 离线模式

设置 `DEEPTALK_ASD_OFFLINE=1` 可禁止所有网络请求，模型需提前下载好：

```bash
export DEEPTALK_ASD_OFFLINE=1
export DEEPTALK_ASD_CACHE_DIR=/path/to/models
```

### 环境变量

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `DEEPTALK_ASD_OFFLINE` | _未设置_ | 设为 `1` 启用离线模式，禁止网络下载 |
| `DEEPTALK_ASD_CACHE_DIR` | `~/.cache/deeptalk_asd/` | 自定义模型缓存目录 |

## 配置说明

通过工厂方法可以精细化控制各个组件及其参数：

```python
from deeptalk_asd import ASDDetectorFactory

# 零配置（推荐）
asd = ASDDetectorFactory().create()

# 自定义配置
config = {
    "face_detector": {
        "type": "inspireface",
        "device": "cpu"
    },
    "turn_detector": {
        "type": "silero-vad", 
        "model_path": "weights/silero_vad.onnx"
    },
    "speaker_detector": {
        "type": "LR-ASD-ONNX", 
        "onnx_dir": "weights",
        "voiceprint_model_path": "weights/wespeaker_zh_cnceleb_resnet34.onnx"
    }
}

asd = ASDDetectorFactory(**config).create()
```

## 许可证说明

本项目代码遵循 **MIT 许可证**。但其集成的预训练模型受各自许可证约束：

1.  **InspireFace**: 代码开源，权重文件通常限于**非商业研究**。商业使用请关注其官方说明。
2.  **Silero VAD**: 遵循 **MIT 许可证**。
3.  **LR-ASD**: 遵循 **MIT 许可证**。
4.  **Sherpa-ONNX**: 遵循 **Apache-2.0 许可证**。

**注意**：在发布或商用基于本项目的应用时，必须自检各模型权重的许可证合规性。
