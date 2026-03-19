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
- **Auto Model Management**: Models are automatically downloaded on first use; supports offline mode and custom cache directories.
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

### 1. Install

```bash
pip install deeptalk_asd
```

### 2. Zero-Configuration Usage

Models are **automatically downloaded** on first use (~46 MB total). No manual setup required:

```python
from deeptalk_asd import ASDDetectorFactory

asd = ASDDetectorFactory().create()
```

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

## Model Management

### Auto-Download (Default)

Models are downloaded to `~/.cache/deeptalk_asd/` on first use. Subsequent runs load from cache instantly.

| Model | Size | Description |
| :--- | ---: | :--- |
| `audio_frontend.onnx` | 0.9 MB | LR-ASD Audio Frontend |
| `visual_frontend.onnx` | 1.5 MB | LR-ASD Visual Frontend |
| `av_backend.onnx` | 0.8 MB | LR-ASD AV Backend |
| `silero_vad.onnx` | 2.2 MB | Silero VAD |
| `Pikachu` | 16 MB | InspireFace Detection Resources |
| `wespeaker_zh_cnceleb_resnet34.onnx` | 25 MB | Speaker Embedding (WeSpeaker) |

### Pre-Download Models (for offline environments)

```bash
# Download all models
python3 -m deeptalk_asd download-models

# Download to a specific directory
python3 -m deeptalk_asd download-models --cache-dir /path/to/models

# Check cache status
python3 -m deeptalk_asd info
```

### Offline Mode

Set `DEEPTALK_ASD_OFFLINE=1` to disable all network requests. Models must be pre-downloaded:

```bash
export DEEPTALK_ASD_OFFLINE=1
export DEEPTALK_ASD_CACHE_DIR=/path/to/models
```

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEEPTALK_ASD_OFFLINE` | _(unset)_ | Set to `1` to enable offline mode |
| `DEEPTALK_ASD_CACHE_DIR` | `~/.cache/deeptalk_asd/` | Custom model cache directory |

## Configuration

Control components and their parameters precisely through the factory method:

```python
from deeptalk_asd import ASDDetectorFactory

# Zero-configuration (recommended)
asd = ASDDetectorFactory().create()

# Custom configuration
config = {
    "face_detector": {
        "type": "inspireface",
        "model_dir": "weights"
    },
    "turn_detector": {
        "type": "silero-vad", 
        "model_dir": "weights"
    },
    "speaker_detector": {
        "type": "LR-ASD-ONNX", 
        "model_dir": "weights",
        "voiceprint_model_name": "wespeaker_zh_cnceleb_resnet34.onnx"
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
