import os
import numpy as np
from collections import deque, defaultdict
import time
import cv2
import math
import python_speech_features
import onnxruntime as ort
from .interface import SpeakerDetectorInterface

from ..deeptalk_logger import DeepTalkLogger

logger = DeepTalkLogger(__name__)


class LRASDOnnxSpeakerDetector(SpeakerDetectorInterface):
    """Active Speaker Detection using LR-ASD ONNX models (pure onnxruntime, no PyTorch).
    
    使用 3 个独立的 ONNX 模型进行推理:
      - audio_frontend.onnx: 提取音频嵌入
      - visual_frontend.onnx: 提取视觉嵌入
      - av_backend.onnx: 融合音视频特征并输出分数
    """

    def __init__(self, **kwargs):
        """
        Initialize the LR-ASD ONNX speaker detector.

        Args:
            onnx_dir (str): ONNX 模型目录路径，默认 'LR_ASD_ONNX/weights'
            device (str): 'cuda' 或 'cpu'
            video_frame_rate (int): 视频帧率，默认 25
            audio_sample_rate (int): 音频采样率，默认 16000
        """
        self.video_frame_rate = kwargs.get('video_frame_rate', 25)
        self.audio_sample_rate = kwargs.get('audio_sample_rate', 16000)
        device = kwargs.get('device', 'cpu')
        onnx_dir = kwargs.get('onnx_dir', 'weights')

        # 选择 ONNX Runtime provider
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # 加载 3 个独立的 ONNX 模型
        self.sess_audio = ort.InferenceSession(
            os.path.join(onnx_dir, 'audio_frontend.onnx'), providers=providers
        )
        self.sess_visual = ort.InferenceSession(
            os.path.join(onnx_dir, 'visual_frontend.onnx'), providers=providers
        )
        self.sess_backend = ort.InferenceSession(
            os.path.join(onnx_dir, 'av_backend.onnx'), providers=providers
        )
        logger.info(
            f"LR-ASD ONNX models loaded from '{onnx_dir}' "
            f"(provider: {self.sess_audio.get_providers()[0]})"
        )

        # Buffers for audio and video data
        self.audio_buffer = deque(maxlen=16000 * 10)  # 10 seconds buffer
        self.video_buffer = defaultdict(list)  # track_id -> list of (frame, timestamp)
        self.last_face_timestamps = {}  # track_id -> last create time
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0

        # Configuration
        self.audio_window = 0.025  # seconds
        self.audio_stride = 0.01  # seconds
        self.max_track_age = 5.0  # seconds before considering a track stale

    def append_video(self, frame_faces, create_time=None):
        """
        添加视频帧中已检测的人脸信息。

        Args:
            frame_faces: 当前帧的人脸检测结果列表，每个元素为 {'id': tracker_id, 'image': face_gray}
            create_time: 帧的创建时间
        """
        if create_time is None:
            create_time = time.perf_counter()
        self.last_video_timestamp = create_time

        for face in frame_faces:
            track_id = face['id']
            face_img = face['image']

            # Resize to 112x112 grayscale
            resized_mouth_img = cv2.resize(face_img, (112, 112))
            self.video_buffer[track_id].append((resized_mouth_img, create_time))
            self.last_face_timestamps[track_id] = create_time

        # Clean up old tracks
        self._cleanup_old_tracks()

    def append_audio(self, audio_chunk, create_time=None):
        """
        添加音频块到处理队列。

        Args:
            audio_chunk: PCM audio data (16-bit, 16kHz, mono)
            create_time: 音频块的创建时间
        """
        if create_time is None:
            create_time = time.perf_counter()

        if isinstance(audio_chunk, (bytes, bytearray)):
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)

        self.audio_buffer.extend(audio_chunk)

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently."""
        current_time = time.perf_counter()
        stale_tracks = [
            track_id for track_id, last_time in self.last_face_timestamps.items()
            if current_time - last_time > self.max_track_age
        ]
        for track_id in stale_tracks:
            if track_id in self.video_buffer:
                del self.video_buffer[track_id]
            if track_id in self.last_face_timestamps:
                del self.last_face_timestamps[track_id]

    def _preprocess_audio(self):
        """Convert raw audio buffer to MFCC features."""
        audio_data = np.array(list(self.audio_buffer), dtype=np.int16)

        mfcc = python_speech_features.mfcc(
            audio_data,
            samplerate=self.audio_sample_rate,
            winlen=self.audio_window,
            winstep=self.audio_stride,
            numcep=13,
        )
        return mfcc.astype(np.float32)

    def _preprocess_video(self, face_frames):
        """Convert list of (mouth_img, timestamp) to numpy array."""
        mouth_imgs = [frame[0] for frame in face_frames]
        return np.array(mouth_imgs, dtype=np.float32)

    def _run_inference(self, audio_feature, video_feature):
        """
        单次 ONNX 推理（参考 demo_onnx.py 的 run_inference）。

        Args:
            audio_feature: (T*4, 13) MFCC 特征
            video_feature: (T, 112, 112) 灰度人脸帧

        Returns:
            scores: (T,) 每帧的说话概率分数
        """
        audio_input = audio_feature[np.newaxis, ...]   # (1, T*4, 13)
        visual_input = video_feature[np.newaxis, ...]  # (1, T, 112, 112)

        audio_embed = self.sess_audio.run(None, {"audio_feature": audio_input})[0]
        visual_embed = self.sess_visual.run(None, {"visual_feature": visual_input})[0]
        scores = self.sess_backend.run(
            None, {"audio_embed": audio_embed, "visual_embed": visual_embed}
        )[0]

        return scores

    def evaluate(self):
        """
        评估当前活动说话者（多时长滑窗推理，参考 demo_onnx.py 的 run_multi_duration）。

        Returns:
            dict: track_id -> average score (正数=说话, 负数=未说话)
        """
        if not self.audio_buffer or not self.video_buffer:
            return {}

        audio_features = self._preprocess_audio()
        audio_feature_rate = int(1.0 / self.audio_stride)  # 100

        duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
        all_scores = {}

        for track_id, face_frames in list(self.video_buffer.items()):
            if not face_frames:
                continue

            video_features = self._preprocess_video(face_frames)

            # 计算对齐后的有效长度（秒）
            length = min(
                (audio_features.shape[0] - audio_features.shape[0] % 4) / audio_feature_rate,
                video_features.shape[0] / self.video_frame_rate,
            )
            num_frames = int(round(length * self.video_frame_rate))
            audio_feat = audio_features[:int(round(length * audio_feature_rate)), :]
            video_feat = video_features[:num_frames, :, :]

            logger.info(f"[LR-ASD-ONNX] track={track_id}, length={length:.2f}s, "
                        f"audio={audio_feat.shape}, video={video_feat.shape}")

            track_all_scores = []

            for duration in duration_set:
                batch_size = int(math.ceil(length / duration))
                scores = []

                for i in range(batch_size):
                    a_start = i * duration * audio_feature_rate
                    a_end = (i + 1) * duration * audio_feature_rate
                    v_start = i * duration * self.video_frame_rate
                    v_end = (i + 1) * duration * self.video_frame_rate

                    audio_chunk = audio_feat[a_start:a_end, :]
                    video_chunk = video_feat[v_start:v_end, :, :]

                    if audio_chunk.shape[0] == 0 or video_chunk.shape[0] == 0:
                        continue

                    # 对齐音频帧到视频帧
                    actual_v_frames = video_chunk.shape[0]
                    target_a_len = actual_v_frames * 4
                    if audio_chunk.shape[0] < target_a_len:
                        audio_chunk = np.pad(
                            audio_chunk,
                            ((0, target_a_len - audio_chunk.shape[0]), (0, 0)),
                            'wrap',
                        )
                    audio_chunk = audio_chunk[:target_a_len, :]

                    start_time = time.perf_counter()
                    chunk_scores = self._run_inference(audio_chunk, video_chunk)
                    elapsed = time.perf_counter() - start_time
                    logger.debug(f"[LR-ASD-ONNX] inference chunk cost {elapsed:.4f}s")

                    scores.extend(chunk_scores.flatten().tolist())

                if len(scores) >= num_frames:
                    track_all_scores.append(scores[:num_frames])

            if track_all_scores:
                avg_scores = np.round(
                    np.mean(np.array(track_all_scores), axis=0), 1
                ).astype(float)
                all_scores[track_id] = float(np.mean(avg_scores))
            else:
                all_scores[track_id] = 0.0

        logger.info(f"[LR-ASD-ONNX] allScores => {all_scores}")
        return all_scores

    def reset(self):
        """Reset the detector state."""
        self.audio_buffer.clear()
        self.video_buffer.clear()
        self.last_face_timestamps.clear()
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0
