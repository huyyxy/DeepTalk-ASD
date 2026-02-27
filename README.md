# DeepTalk-ASD

[LR-ASD](https://github.com/Junhua-Liao/LR-ASD) is a SOTA Active Speaker Detection (ASD) model. While it offers exceptional performance, the official open-source project relies on GPUs and relatively older face detection models. This project aims to provide a production-ready, out-of-the-box ASD system.

---

DeepTalk-ASD is an efficient Active Speaker Detection (ASD) system. By fusing audio and video features, it determines in real-time which face in a video frame is speaking.

## Key Features

- **Multimodal Fusion**: Combines face detection, voice activity detection (VAD), and speaker identification models.
- **Modular Design**: 
    - **FaceDetector**: Detects and tracks faces (supports InspireFace).
    - **TurnDetector**: Audio VAD detection (supports Silero VAD).
    - **SpeakerDetector**: Audio-visual feature fusion and decision making (based on LR-ASD).
- **High Performance**: Supports ONNX inference, suitable for real-time applications.
- **Easy to Use**: Provides command-line demos supporting real-time camera input and video file processing.

## System Architecture

The system is orchestrated through three main sub-components:
1. **FaceDetector**: Responsible for locating faces in each video frame.
2. **TurnDetector**: Responsible for determining if the current audio stream contains speech.
3. **SpeakerDetector**: The core decision layer that calculates speaking probability based on VAD results and face image sequences.

## Quick Start

### 1. Environment Setup

Python 3.8 or higher is recommended.

```bash
# Clone the repository
git clone <repository_url>
cd DeepTalk-ASD

# Install dependencies
python3 -m pip install -r requirements.txt

# Install the project in editable mode
pip3 install -e .
```

### 2. Verify Installation

Check the installation via the Python interactive environment:

```python
python3
>>> import deeptalk_asd
>>> print(deeptalk_asd.__version__)
```

### 3. Model Weights

Ensure the `weights` directory contains the necessary model files. The project includes support for converting models to ONNX.
- `Pikachu`: InspireFace related models.
- `silero_vad.onnx`: Silero VAD model.
- LR-ASD related models: `audio_frontend.onnx`, `visual_frontend.onnx`, `av_backend.onnx`.

### 4. Running Demos

#### Video File Processing Demo
```bash
python3 demo/video_asd_demo.py --input demo/demo.mp4 --display
```

#### Real-time Camera Demo
```bash
python3 demo/realtime_asd_demo.py
```

## Configuration

The components can be flexibly configured using the factory method:

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

## License

The code in this project is licensed under the **MIT License**. However, please note that the integrated models and their related code are subject to their respective licenses:

1. **InspireFace**: Core code is MIT, but the provided pre-trained models are typically restricted to **non-commercial research use**. For commercial use, please refer to [InsightFace](https://github.com/deepinsight/insightface) documentation.
2. **Silero VAD**: Licensed under the **MIT License**.
3. **LR-ASD**: Licensed under the **MIT License**.

While you may redistribute the code of this project under the MIT license, you must explicitly inform users in the documentation that: **When using specific pre-trained models (especially face detection models), users must comply with the non-commercial restrictions of the original authors.** If commercialization is required, users should replace them with commercially-friendly models or contact the original authors for authorization.
