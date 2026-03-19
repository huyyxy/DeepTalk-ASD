# DeepTalk-ASD

[LR-ASD](https://github.com/Junhua-Liao/LR-ASD) is a SOTA Active Speaker Detection (ASD) model. Building upon LR-ASD, this project provides an industrial-grade ASD system featuring **full-pipeline ONNX conversion, integrated speaker verification, and real-time processing support**. It runs efficiently on CPUs without relying on large PyTorch/GPU environments.

---

DeepTalk-ASD is an efficient Active Speaker Detection (ASD) system. By fusing audio, video features, and speaker embeddings, it determines in real-time which face in a video frame is speaking.

## Key Features

- **Multimodal Fusion**: Deeply integrates face detection, voice activity detection (VAD), audio-visual ASD models, and **Speaker Embeddings**.
- **Speaker Verification Enhancement (New)**: 
    - **Feature Fusion**: Weighted fusion of ASD probability and audio speaker similarity to effectively suppress off-screen voice interference.
    - **Fast Matching**: Supports accelerated determination for tracks with known speaker embeddings, skipping complex calculations.
    - **Automatic Profile Update**: Automatically updates speaker embedding profiles for detected speakers using EMA (Exponential Moving Average).
- **Modular Design**: 
    - **FaceDetector**: Provides face detection and tracking (currently supports InspireFace).
    - **TurnDetector**: Audio VAD and turn management (integrates Silero VAD).
    - **SpeakerDetector**: Core decision layer performing audio-visual feature extraction, fusion decision, and speaker comparison (based on LR-ASD ONNX).
- **High Performance**: Full-pipeline ONNX inference optimized for CPU, supporting real-time operation on mobile devices or standard laptops.
- **Multiple Scenarios**: Provides demos for real-time camera, video files, speaker embedding extraction, and pVAD.

## System Architecture

The system works through three main sub-components:
1. **Face Detection (FaceDetector)**: Locates and tracks faces (Tracks) in each video frame.
2. **Turn Detection (TurnDetector)**: Determines if the current audio stream contains speech segments and manages their lifecycle (START, CONTINUE, END).
3. **Speaker Detection (SpeakerDetector)**: 
    - Extracts audio MFCC features and 112x112 grayscale mouth images.
    - Uses **Sherpa-ONNX** to extract speaker embeddings (Voiceprint).
    - Calculates raw ASD scores through a three-stage ONNX model (Audio/Visual Frontend + AV Backend).
    - Final scores are dynamically fused from ASD scores and speaker matching scores.

## Quick Start

### 1. Environment Setup

Python 3.8 or higher is recommended.

```bash
# Clone the repository
git clone <repository_url>
cd DeepTalk-ASD

# Install core dependencies
python3 -m pip install -r requirements.txt

# Install the project in editable mode
pip3 install -e .
```

### 2. Model Weights

Ensure the `weights` directory contains the following ONNX model files:

| Category | Filename | Description |
| :--- | :--- | :--- |
| **LR-ASD** | `audio_frontend.onnx`, `visual_frontend.onnx`, `av_backend.onnx` | Core Audio-Visual ASD models |
| **VAD** | `silero_vad.onnx` | Voice Activity Detection model |
| **Face** | `Pikachu` (directory) | Resources for InspireFace detection |
| **Voiceprint** | `wespeaker_zh_cnceleb_resnet34.onnx` | [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) Speaker Embedding model (Recommended) |

### 3. Running Demos

#### Core ASD Demos

*   **Real-time Camera Demo** (Recommended first choice):
    ```bash
    python3 demo/realtime_asd_demo.py
    ```
*   **Offline Video File Processing**:
    ```bash
    python3 demo/video_asd_demo.py --input demo/demo.mp4 --display
    ```

## Configuration

Control components and their parameters precisely through the factory method:

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

## License

The code in this project is licensed under the **MIT License**. However, the integrated pre-trained models are subject to their respective licenses:

1.  **InspireFace**: Code is open-source, but weight files are typically restricted to **non-commercial research**. Please check official terms for commercial use.
2.  **Silero VAD**: Licensed under the **MIT License**.
3.  **LR-ASD**: Licensed under the **MIT License**.
4.  **Sherpa-ONNX**: Licensed under the **Apache-2.0 License**.

**Note**: When releasing or commercializing applications based on this project, you must verify the license compliance for each model's weights.
