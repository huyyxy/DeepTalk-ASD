import os
import numpy as np
from collections import deque, defaultdict
import time
import cv2
import math
import python_speech_features
import onnxruntime as ort
from .interface import SpeakerDetectorInterface

from .voiceprint import SpeakerEmbeddingExtractor, cosine_similarity

from ..deeptalk_logger import DeepTalkLogger

# 阈值与配置（可从环境变量覆盖）
# 声纹快速匹配：当前音频与某 track 声纹相似度超过此阈值即视为匹配，用于唯一说话人直接判定
VOICEPRINT_FAST_MATCH_THRESHOLD = float(os.getenv("LRASD_VOICEPRINT_FAST_MATCH_THRESHOLD", "0.65"))
# 声纹分数融合：相似度高于此值认为“声音极度吻合”，会拉高最终分数
VOICEPRINT_SIMILARITY_HIGH = float(os.getenv("LRASD_VOICEPRINT_SIMILARITY_HIGH", "0.65"))
# 声纹分数融合：相似度低于此值认为“声音截然不同”（如画外音），会压制最终分数
VOICEPRINT_SIMILARITY_LOW = float(os.getenv("LRASD_VOICEPRINT_SIMILARITY_LOW", "0.3"))
# 声纹档案更新时的 EMA 平滑系数（0~1），越大越保留历史声纹
VOICEPRINT_EMA_ALPHA = float(os.getenv("LRASD_VOICEPRINT_EMA_ALPHA", "0.8"))
# 提取声纹所需的最短音频时长（秒），短于此长度不提取声纹
VOICEPRINT_MIN_AUDIO_SEC = float(os.getenv("LRASD_VOICEPRINT_MIN_AUDIO_SEC", "0.5"))
# 最终分数超过此阈值且有声纹时，会创建或更新该 track 的声纹档案
ASD_CONFIRM_SPEAKING_THRESHOLD = float(os.getenv("LRASD_ASD_CONFIRM_SPEAKING_THRESHOLD", "0.85"))
# 声纹极度吻合时，最终分数 = max(asd_score, similarity * 此权重)
VOICEPRINT_BOOST_WEIGHT = float(os.getenv("LRASD_VOICEPRINT_BOOST_WEIGHT", "0.9"))
# 声纹截然不同时，最终分数 = asd_score * 此系数（压制画外音）
VOICEPRINT_REJECT_SCALE = float(os.getenv("LRASD_VOICEPRINT_REJECT_SCALE", "0.3"))
# 声纹模棱两可时的混合权重：final_score = asd_score * BLEND_ASD + similarity * BLEND_SIM
VOICEPRINT_BLEND_ASD = float(os.getenv("LRASD_VOICEPRINT_BLEND_ASD", "0.7"))
VOICEPRINT_BLEND_SIM = float(os.getenv("LRASD_VOICEPRINT_BLEND_SIM", "0.3"))
# 声纹快速匹配中，未匹配到的 track 赋予的分数（通常为负表示未说话）
NON_MATCHED_TRACK_SCORE = float(os.getenv("LRASD_NON_MATCHED_TRACK_SCORE", "-1.0"))
# MFCC 提取的窗口长度（秒）
AUDIO_WINDOW = float(os.getenv("LRASD_AUDIO_WINDOW", "0.025"))
# MFCC 提取的帧移/步长（秒）
AUDIO_STRIDE = float(os.getenv("LRASD_AUDIO_STRIDE", "0.01"))
# 人脸轨迹过期时间（秒），超过此时间未更新的人脸 track 会被清理
MAX_TRACK_AGE = float(os.getenv("LRASD_MAX_TRACK_AGE", "5.0"))

logger = DeepTalkLogger(__name__)


class LRASDOnnxSpeakerDetector(SpeakerDetectorInterface):
    """基于 LR-ASD ONNX 模型的主动说话者检测 (纯 onnxruntime 实现，无 PyTorch 依赖)。
    
    使用 3 个独立的 ONNX 模型进行推理:
      - audio_frontend.onnx: 提取音频嵌入
      - visual_frontend.onnx: 提取视觉嵌入
      - av_backend.onnx: 融合音视频特征并输出分数
    """

    def __init__(self, **kwargs):
        """
        初始化 LR-ASD ONNX 说话者检测器。

        Args:
            onnx_dir (str): ONNX 模型目录路径，默认 'weights'
            device (str): 推理设备，'cuda' 或 'cpu'，默认 'cpu'
            video_frame_rate (int): 视频帧率，默认 25
            audio_sample_rate (int): 音频采样率，默认 16000
            voiceprint_model_path (str): 声纹特征提取 ONNX 模型路径，默认查找 onnx_dir 下的 'wespeaker_zh_cnceleb_resnet34.onnx'
        """
        self.video_frame_rate = kwargs.get('video_frame_rate', 25)
        self.audio_sample_rate = kwargs.get('audio_sample_rate', 16000)
        device = kwargs.get('device', 'cpu')
        onnx_dir = kwargs.get('onnx_dir', 'weights')
        default_voiceprint_model_path = os.path.join(onnx_dir, 'wespeaker_zh_cnceleb_resnet34.onnx')
        voiceprint_model_path = kwargs.get('voiceprint_model_path', default_voiceprint_model_path)

        # Voiceprint Integration
        self.voice_extractor = None
        if voiceprint_model_path and os.path.exists(voiceprint_model_path):
            try:
                self.voice_extractor = SpeakerEmbeddingExtractor(
                    model_path=voiceprint_model_path,
                    provider=device
                )
            except Exception as e:
                logger.error(f"无法初始化声纹提取器: {e}")
        else:
            logger.critical(f"[VOICEPRINT] without voiceprint_model_path")
        
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

        # 音视频数据缓冲区
        self.audio_buffer = deque(maxlen=16000 * 10)  # 10秒的音频缓冲区
        self.video_buffer = defaultdict(list)  # 追踪ID -> (视频帧, 时间戳)列表
        self.last_face_timestamps = {}  # 追踪ID -> 最后一次检测到人脸的时间
        
        # 声纹档案
        self.voice_profiles = {}  # 追踪ID -> 对应的声纹嵌入特征 (np.ndarray)
        
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0

        # 配置参数（默认值可由环境变量覆盖）
        self.audio_window = AUDIO_WINDOW  # 音频窗口大小（秒）
        self.audio_stride = AUDIO_STRIDE  # 音频步长（秒）
        self.max_track_age = MAX_TRACK_AGE  # 人脸轨迹过期时间（秒），超过此时间未更新的轨迹将被清理
        # 帧率降采样：只保留间隔 >= 1/video_frame_rate 的帧，使任意摄像头帧率对齐到模型期望的 25fps
        self._min_frame_interval = 1.0 / self.video_frame_rate
        self._last_appended_video_time = {}  # track_id -> 上次写入 video_buffer 的时间戳

    def append_video(self, frame_faces, create_time=None):
        """
        添加视频帧中已检测的人脸信息。

        按时间戳做帧率降采样：每个 track 只保留间隔 >= 1/video_frame_rate 秒的帧，
        使高帧率摄像头（如 30fps）输入在 buffer 中等效为 25fps，与模型训练时的音视频对齐一致。

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

            # 帧率降采样：该 track 距上次写入已超过 min_frame_interval 才写入
            last_ts = self._last_appended_video_time.get(track_id, -float('inf'))
            if create_time - last_ts < self._min_frame_interval:
                continue
            self._last_appended_video_time[track_id] = create_time

            # 缩放至 112x112 灰度人脸图像
            resized_mouth_img = cv2.resize(face_img, (112, 112))
            self.video_buffer[track_id].append((resized_mouth_img, create_time))
            self.last_face_timestamps[track_id] = create_time

        # 清理旧的轨迹数据
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
            if len(audio_chunk) % 2 != 0:
                logger.warning("音频数据长度不是偶数，可能不是 16-bit PCM")
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)

        self.audio_buffer.extend(audio_chunk)
        self.last_audio_timestamp = create_time

    def _cleanup_old_tracks(self):
        """移除近期未更新且已过期的人脸轨迹。"""
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
            if track_id in self._last_appended_video_time:
                del self._last_appended_video_time[track_id]
            if track_id in self.voice_profiles:
                del self.voice_profiles[track_id]

    def _preprocess_audio(self, audio_data=None):
        """将原始音频缓冲区数据转换为 MFCC 特征。"""
        if audio_data is None:
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
        """将包含 (mouth_img, timestamp) 的列表转换为 numpy 数组。"""
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

    def evaluate(self, start_time: float = None, end_time: float = None):
        """
        评估当前活动说话者（多时长滑窗推理，参考 demo_onnx.py 的 run_multi_duration）。

        Args:
            start_time: 评估起始时间（time.perf_counter 时间戳），可选
            end_time: 评估结束时间（time.perf_counter 时间戳），可选
            若均提供，则只对该时间范围内的音视频做推理；
            若时间范围超过缓冲区长度，则回退为使用整个缓冲区。
            若不传，则使用整个缓冲区。

        Returns:
            dict: track_id -> average score (正数=说话, 负数=未说话)
        """
        if not self.audio_buffer or not self.video_buffer:
            return {}

        effective_start = None
        effective_end = None

        if start_time is not None and end_time is not None:
            buf_len = len(self.audio_buffer)
            buffer_duration = buf_len / self.audio_sample_rate
            buffer_end_time = self.last_audio_timestamp
            buffer_start_time = buffer_end_time - buffer_duration

            if end_time - start_time <= buffer_duration:
                effective_start = max(start_time, buffer_start_time)
                effective_end = min(end_time, buffer_end_time)

                start_idx = max(0, int((effective_start - buffer_start_time) * self.audio_sample_rate))
                end_idx = min(buf_len, int((effective_end - buffer_start_time) * self.audio_sample_rate))

                audio_data = np.array(list(self.audio_buffer)[start_idx:end_idx], dtype=np.int16)
                audio_features = self._preprocess_audio(audio_data)

        if effective_start is None:
            audio_features = self._preprocess_audio()

        # 为当前的音频块提取声纹特征
        current_voice_emb = None
        if self.voice_extractor is not None and effective_start is not None:
            try:
                # 至少需要 VOICEPRINT_MIN_AUDIO_SEC 秒的音频才能提取到较为合理的声纹嵌入特征
                if (effective_end - effective_start) > VOICEPRINT_MIN_AUDIO_SEC:
                    current_voice_emb = self.voice_extractor.extract_from_samples(
                        audio_data.astype(np.float32) / 32768.0, 
                        sample_rate=self.audio_sample_rate
                    )
                    logger.critical(f"[VOICEPRINT] extract_from_samples success")
                else:
                    logger.critical(f"[VOICEPRINT] without extract_from_samples")
            except Exception as e:
                logger.warning(f"声纹提取失败: {e}")

        # --- 声纹快速匹配：若仅有唯一 track 匹配，直接判定说话人 ---
        if current_voice_emb is not None and self.voice_profiles:
            matched_tracks = []
            for tid, profile_emb in self.voice_profiles.items():
                sim = cosine_similarity(current_voice_emb, profile_emb)
                logger.info(f"[VOICEPRINT-FAST] track={tid}, similarity={sim:.3f}")
                if sim > VOICEPRINT_FAST_MATCH_THRESHOLD:
                    matched_tracks.append((tid, sim))

            if len(matched_tracks) == 1:
                matched_tid, matched_sim = matched_tracks[0]
                logger.info(f"[VOICEPRINT-FAST] 唯一匹配 track={matched_tid}, "
                            f"similarity={matched_sim:.3f}, 直接判定为说话人")
                fast_scores = {}
                for track_id in self.video_buffer:
                    if track_id == matched_tid:
                        fast_scores[track_id] = float(matched_sim)
                    else:
                        fast_scores[track_id] = NON_MATCHED_TRACK_SCORE
                # EMA 平滑更新匹配到的声纹档案
                updated_emb = VOICEPRINT_EMA_ALPHA * self.voice_profiles[matched_tid] + (1 - VOICEPRINT_EMA_ALPHA) * current_voice_emb
                updated_emb /= np.linalg.norm(updated_emb)
                self.voice_profiles[matched_tid] = updated_emb
                return fast_scores

        audio_feature_rate = int(1.0 / self.audio_stride)  # 音频特征帧率，通常为 100

        duration_set = [1, 2]
        # duration_set = [1, 2, 4, 6]
        # duration_set = [1, 2, 3, 4, 5, 6]
        # duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]  # 使结果更可靠
        all_scores = {}

        for track_id, face_frames in list(self.video_buffer.items()):
            if not face_frames:
                continue

            if effective_start is not None:
                filtered_frames = [(f, t) for f, t in face_frames
                                   if effective_start <= t <= effective_end]
                if not filtered_frames:
                    continue
                video_features = self._preprocess_video(filtered_frames)
            else:
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

                    t0 = time.perf_counter()
                    chunk_scores = self._run_inference(audio_chunk, video_chunk)
                    elapsed = time.perf_counter() - t0
                    logger.debug(f"[LR-ASD-ONNX] inference chunk cost {elapsed:.3f}s")

                    scores.extend(chunk_scores.flatten().tolist())

                if len(scores) >= num_frames:
                    track_all_scores.append(scores[:num_frames])

            if track_all_scores:
                avg_scores = np.round(
                    np.mean(np.array(track_all_scores), axis=0), 1
                ).astype(float)
                asd_score = float(np.mean(avg_scores))
                
                # --- 声纹分数融合策略 (Score Fusion) ---
                final_score = asd_score
                if current_voice_emb is not None:
                    # 获取该 track_id 历史声纹档案
                    profile_emb = self.voice_profiles.get(track_id)
                    
                    if profile_emb is not None:
                        similarity = cosine_similarity(current_voice_emb, profile_emb)
                        logger.critical(f"[VOICEPRINT] track={track_id}, similarity={similarity:.3f}, asd_score={asd_score:.3f}")
                        
                        if similarity > VOICEPRINT_SIMILARITY_HIGH:
                            # 声音极度吻合，拉高分数弥补侧脸/嘴遮挡导致的 ASD 偏低
                            final_score = max(asd_score, similarity * VOICEPRINT_BOOST_WEIGHT)
                        elif similarity < VOICEPRINT_SIMILARITY_LOW:
                            # 声音截然不同（可能是画外音），即使有唇动也判定为非本说话人
                            final_score = asd_score * VOICEPRINT_REJECT_SCALE
                        else:
                            # 模棱两可：保留混合趋势
                            final_score = asd_score * VOICEPRINT_BLEND_ASD + similarity * VOICEPRINT_BLEND_SIM

                # 更新档案：若高精度 ASD 确认这人在说话
                if final_score > ASD_CONFIRM_SPEAKING_THRESHOLD and current_voice_emb is not None:
                    if track_id not in self.voice_profiles:
                        self.voice_profiles[track_id] = current_voice_emb
                        logger.info(f"[VOICEPRINT] track={track_id} 初始化声纹档案")
                    else:
                        # EMA 平滑更新
                        updated_emb = VOICEPRINT_EMA_ALPHA * self.voice_profiles[track_id] + (1 - VOICEPRINT_EMA_ALPHA) * current_voice_emb
                        # 重新归一化
                        updated_emb /= np.linalg.norm(updated_emb)
                        self.voice_profiles[track_id] = updated_emb

                all_scores[track_id] = float(final_score)
            else:
                all_scores[track_id] = 0.0

        logger.info(f"[LR-ASD-ONNX] allScores => {all_scores}")
        return all_scores

    def reset(self):
        """重置检测器状态，清空所有缓冲区和档案。"""
        self.audio_buffer.clear()
        self.video_buffer.clear()
        self.last_face_timestamps.clear()
        self._last_appended_video_time.clear()
        self.voice_profiles.clear()
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0
