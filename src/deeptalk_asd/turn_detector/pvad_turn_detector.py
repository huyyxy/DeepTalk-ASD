"""
PVADTurnDetector — 基于 Personal VAD 的说话人切换检测装饰器

使用装饰器模式包装一个底层 TurnDetector（如 SileroVadTurnDetector），
在 TURN_CONFIRMED 到 TURN_END 之间叠加 pVAD 逐帧监测：
当目标说话人的 pVAD 概率持续低于阈值，判定发生说话人切换并输出 SPEAKER_CHANGE 状态。

依赖模型:
    - pvad.onnx: pVAD 推理模型 (来自 FireRedTeam/fireredchat-pvad)
    - 192 维声纹模型: 用于提取目标说话人 embedding (如 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx)
"""

import os
from typing import Optional

import numpy as np
import onnxruntime as ort

from ..audio_frame import AudioFrame
from ..deeptalk_logger import DeepTalkLogger
from .interface import TurnDetectorInterface
from .utterance import Utterance, TurnState

logger = DeepTalkLogger(__name__)

PVAD_SAMPLE_RATE = 16000
PVAD_FRAME_SAMPLES = 160  # 10ms @ 16kHz
PVAD_EMBED_DIM = 192


class PVADTurnDetector(TurnDetectorInterface):
    """装饰器模式的 TurnDetector，在底层 VAD 基础上叠加 pVAD 说话人切换检测。

    工作原理:
        1. detect() 首先委托给 inner_detector 获取标准 VAD 状态
        2. 在 TURN_CONTINUE 阶段，如果 pVAD 已激活，逐帧跑 pVAD 推理
        3. 当 pVAD 概率持续低于阈值达 min_low_frames 帧，输出 SPEAKER_CHANGE
        4. on_speaker_identified() 被 ASD 编排层在 evaluate() 之后调用，
           用于提取目标声纹并激活/更新 pVAD 监测
    """

    def __init__(
        self,
        inner_detector: TurnDetectorInterface,
        *,
        model_dir: str,
        pvad_model_name: str = "pvad.onnx",
        spk_model_name: str = "speaker_model.onnx",
        pvad_threshold: float = 0.35,
        min_low_frames: int = 30,
        cooldown_frames: int = 50,
    ):
        """
        参数:
            inner_detector: 底层轮次检测器实例（如 SileroVadTurnDetector）
            model_dir: 模型资源目录，应包含 pvad.onnx 和声纹模型
            pvad_model_name: pVAD ONNX 模型文件名
            spk_model_name: 192 维声纹提取模型文件名
            pvad_threshold: pVAD 概率低于此阈值视为"目标说话人未在说话"
            min_low_frames: 连续低于阈值的帧数达到此值才触发 SPEAKER_CHANGE
            cooldown_frames: SPEAKER_CHANGE 触发后的冷却帧数，防止重新激活后立即误触
        """
        self._inner = inner_detector

        # 参数
        self._threshold = pvad_threshold
        self._min_low_frames = min_low_frames
        self._cooldown_frames = cooldown_frames

        # 加载 pVAD ONNX 模型
        pvad_model_path = os.path.join(model_dir, pvad_model_name)
        if not os.path.exists(pvad_model_path):
            raise FileNotFoundError(f"pVAD 模型不存在: {pvad_model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self._pvad_session = ort.InferenceSession(
            pvad_model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        logger.info(f"pVAD 模型加载成功: {pvad_model_path}")

        # 加载 192 维声纹提取模型
        spk_model_path = os.path.join(model_dir, spk_model_name)
        if not os.path.exists(spk_model_path):
            raise FileNotFoundError(f"pVAD 声纹模型不存在: {spk_model_path}")

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("pVAD 需要 sherpa-onnx，请安装: pip install sherpa-onnx")

        spk_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=spk_model_path,
            num_threads=2,
            debug=False,
            provider="cpu",
        )
        if not spk_config.validate():
            raise ValueError(f"无效的声纹模型配置: {spk_model_path}")

        self._spk_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(spk_config)
        spk_dim = self._spk_extractor.dim
        if spk_dim != PVAD_EMBED_DIM:
            raise ValueError(
                f"声纹模型输出维度 {spk_dim} 与 pVAD 要求的 {PVAD_EMBED_DIM} 不匹配"
            )
        logger.info(f"pVAD 声纹模型加载成功: {spk_model_path}, dim={spk_dim}")

        # pVAD 运行时状态
        self._target_embedding: Optional[np.ndarray] = None  # (1, 192)
        self._mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self._gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        self._active = False
        self._low_count = 0
        self._cooldown_remaining = 0

    def detect(self, audio_frame: AudioFrame) -> Utterance:
        """
        检测轮次信息，在底层 VAD 结果基础上叠加 pVAD 说话人切换检测。

        参数:
            audio_frame: 当前音频帧

        返回:
            轮次检测结果。当 pVAD 检测到说话人切换时，turn_state 为 SPEAKER_CHANGE。
        """
        utterance = self._inner.detect(audio_frame)

        if utterance.turn_state in (TurnState.TURN_END, TurnState.TURN_REJECTED):
            self._deactivate()
            return utterance

        if not self._active or utterance.turn_state != TurnState.TURN_CONTINUE:
            return utterance

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return utterance

        pvad_prob = self._run_pvad(audio_frame)

        if pvad_prob < self._threshold:
            self._low_count += 1
        else:
            self._low_count = 0

        if self._low_count >= self._min_low_frames:
            utterance.turn_state = TurnState.SPEAKER_CHANGE
            self._deactivate()
            self._cooldown_remaining = self._cooldown_frames
            logger.info(
                f"[pVAD] 检测到说话人切换, prob={pvad_prob:.3f}, "
                f"连续低帧数={self._low_count}"
            )

        return utterance

    def on_speaker_identified(self, audio_samples: np.ndarray, sample_rate: int = 16000):
        """
        evaluate() 识别出说话人后由 ASD 编排层调用。
        从传入的音频中提取 192 维目标声纹并激活 pVAD 监测。

        参数:
            audio_samples: 说话人所在时间段的原始音频 (float32, 单声道)
            sample_rate: 采样率
        """
        if audio_samples is None or len(audio_samples) == 0:
            return

        embedding = self._extract_embedding(audio_samples, sample_rate)
        if embedding is None:
            logger.warning("[pVAD] 目标声纹提取失败，pVAD 未激活")
            return

        self._target_embedding = embedding
        self._reset_pvad_buffers()
        self._active = True
        self._low_count = 0
        logger.info("[pVAD] 目标声纹已设置，pVAD 已激活")

    def _extract_embedding(self, audio_samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """用内置 192 维声纹模型从音频中提取 embedding。"""
        if audio_samples.dtype != np.float32:
            audio_samples = audio_samples.astype(np.float32)

        stream = self._spk_extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio_samples)
        stream.input_finished()

        if not self._spk_extractor.is_ready(stream):
            return None

        embedding = np.array(self._spk_extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.reshape(1, PVAD_EMBED_DIM)

    def _run_pvad(self, audio_frame: AudioFrame) -> float:
        """将 AudioFrame 拆成 10ms 帧逐帧跑 pVAD 推理，返回最后一帧概率。"""
        audio_i16 = np.ctypeslib.as_array(audio_frame.data)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        prob = 0.0
        for offset in range(0, len(audio_f32), PVAD_FRAME_SAMPLES):
            frame = audio_f32[offset:offset + PVAD_FRAME_SAMPLES]
            if len(frame) < PVAD_FRAME_SAMPLES:
                break

            outputs = self._pvad_session.run(None, {
                "input_audio": frame.reshape(1, PVAD_FRAME_SAMPLES),
                "spkemb": self._target_embedding,
                "mel_buffer": self._mel_buffer,
                "gru_buffer": self._gru_buffer,
            })
            prob = float(outputs[1][0][0])
            self._mel_buffer = outputs[2]
            self._gru_buffer = outputs[3]

        return prob

    def _reset_pvad_buffers(self):
        """重置 pVAD 的内部状态缓冲区（mel + GRU）。"""
        self._mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self._gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)

    def _deactivate(self):
        """关闭 pVAD 监测并重置状态。"""
        self._active = False
        self._low_count = 0
        self._target_embedding = None
        self._reset_pvad_buffers()
