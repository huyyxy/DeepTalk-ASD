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

### 1. 环境准备

建议使用 Python 3.8 或更高版本。

```bash
# 克隆仓库
git clone <repository_url>
cd DeepTalk-ASD

# 安装核心依赖
python3 -m pip install -r requirements.txt

# 以可编辑模式安装项目
pip3 install -e .
```

### 2. 模型权重

请确保 `weights` 目录下包含以下 ONNX 模型文件：

| 模型分类 | 文件名 | 说明 |
| :--- | :--- | :--- |
| **LR-ASD** | `audio_frontend.onnx`, `visual_frontend.onnx`, `av_backend.onnx` | 音视频 ASD 核心模型 |
| **VAD** | `silero_vad.onnx` | 语音轮次检测模型 |
| **Face** | `Pikachu` | InspireFace 人脸检测所需资源 |
| **Voiceprint** | `wespeaker_zh_cnceleb_resnet34.onnx` | [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) 声纹提取模型 (推荐) |

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

## 配置说明

通过工厂方法可以精细化控制各个组件及其参数：

```python
from deeptalk_asd import ASDDetectorFactory

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
