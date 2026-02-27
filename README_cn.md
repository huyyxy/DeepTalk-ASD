# DeepTalk-ASD

[LR-ASD](https://github.com/Junhua-Liao/LR-ASD)是SOTA的活跃说话者检测模型。它有卓越的性能，但官方开源项目依赖于GPU和相对沉旧的人脸检测等模型。本项目致力于提供一个业界可开箱即用的ASD系统。
---

DeepTalk-ASD 是一个高效的活跃说话者检测 (Active Speaker Detection, ASD) 系统。它通过融合音频和视频特征，实时判定视频画面中哪个人脸正在说话。

## 核心特性

- **多模态融合**: 结合人脸检测、语音轮次检测 (VAD) 和说话者辨别模型。
- **模块化设计**: 
    - **FaceDetector**: 检测并追踪人脸（支持 InspireFace）。
    - **TurnDetector**: 音频 VAD 检测（支持 Silero VAD）。
    - **SpeakerDetector**: 音视频特征融合判定（基于 LR-ASD）。
- **高性能**: 支持 ONNX 推理，适合实时应用场景。
- **易于使用**: 提供命令行演示程序，支持实时摄像头输入和视频文件处理。

## 系统架构

系统由三个主要子组件串行/并行编排而成：
1. **人脸检测 (FaceDetector)**: 负责在每一帧视频中定位人脸。
2. **轮次检测 (TurnDetector)**: 负责判断当前音频流是否包含语音。
3. **说话者检测 (SpeakerDetector)**: 核心决策层，根据 VAD 结果和人脸图像序列计算说话概率。

## 快速开始

### 1. 环境准备

建议使用 Python 3.8 或更高版本。

```bash
# 克隆仓库
git clone <repository_url>
cd DeepTalk-ASD

# 安装依赖
python3 -m pip install -r requirements.txt

# 以可编辑模式安装项目
pip3 install -e .
```

### 2. 验证安装

可以通过 Python 交互环境验证安装是否成功：

```python
python3
>>> import deeptalk_asd
>>> print(deeptalk_asd.__version__)
```

### 3. 模型权重

请确保 `weights` 目录下包含必要的模型文件。项目提供了将模型转换为 ONNX 的支持。
- `Pikachu`: InspireFace 人脸检测相关模型
- `silero_vad.onnx`: Silero VAD 模型
- LR-ASD 相关模型: `audio_frontend.onnx`, `visual_frontend.onnx`, `av_backend.onnx`

### 4. 运行 Demos

#### 视频文件处理演示
```bash
python3 demo/video_asd_demo.py --input demo/demo.mp4 --display
```

#### 实时摄像头演示
```bash
python3 demo/realtime_asd_demo.py
```

## 配置说明

可以通过工厂方法灵活配置各个组件：

```python
from deeptalk_asd import ASDDetectorFactory

config = {
    "face_detector": {"type": "inspireface"},
    "turn_detector": {"type": "silero-vad", "model_path": "weights/silero_vad.onnx"},
    "speaker_detector": {"type": "LR-ASD-ONNX", "onnx_dir": "weights"}
}

factory = ASDDetectorFactory(**config)
asd = factory.create()
```

## 许可证说明

本项目代码部分遵循 **MIT 许可证**。但请注意，本项目集成的各个模型及其相关代码需遵循其各自的许可证：

1. **InspireFace**: 核心代码遵循 MIT，但其提供的训练模型通常仅限于 **非商业研究用途**。商业使用请参考 [InsightFace](https://github.com/deepinsight/insightface) 相关说明。
2. **Silero VAD**: 遵循 **MIT 许可证**。
3. **LR-ASD**: 遵循 **MIT 许可证**。

您可以基于 MIT 许可证发布本项目的代码，但必须在文档中明确告知用户：**在使用特定的预训练模型（尤其是人脸检测相关模型）时，必须遵守原作者的非商业性限制。** 如果用户需要商业化，则需要更换为商业友好的模型或联系原作者获取授权。
