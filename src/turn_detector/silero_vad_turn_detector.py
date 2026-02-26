import numpy as np
from typing import List

from ..audio_frame import AudioFrame
from .interface import TurnDetectorInterface
from .utterance import Utterance, TurnState
from .vad.silero_vad import SileroVAD


class SileroVadTurnDetector(TurnDetectorInterface):
    """基于 Silero VAD 的轮次检测器

    状态机说明:
        IDLE ──(voice)──> TURN_START
        TURN_START ──(voice, 累计人声 >= min_voice_duration_ms)──> TURN_CONFIRMED
        TURN_START ──(voice, 累计人声 < min_voice_duration_ms)──> TURN_CONTINUE
        TURN_CONTINUE ──(voice, 累计人声 >= min_voice_duration_ms)──> TURN_CONFIRMED
        TURN_CONFIRMED ──(voice)──> TURN_CONFIRMED
        TURN_CONFIRMED ──(silence)──> TURN_SILENCE
        TURN_SILENCE ──(voice)──> TURN_CONFIRMED
        TURN_SILENCE ──(silence, 累计静音 >= silence_duration_ms)──> TURN_END / TURN_REJECTED
        TURN_END / TURN_REJECTED ──> IDLE (自动重置)

    判定结果:
        - TURN_END: 人声段通过 loudness_percentile_95 验证
        - TURN_REJECTED: 人声段未通过 loudness_percentile_95 验证
    """

    def __init__(
        self,
        model_path: str,
        *,
        vad_threshold: float = 0.5,
        prefix_padding_ms: int = 300,
        min_voice_duration_ms: int = 200,
        silence_duration_ms: int = 500,
        abs_amplitude_threshold: float = 0.02,
    ):
        """初始化 SileroVadTurnDetector

        Args:
            model_path: Silero VAD ONNX 模型路径
            vad_threshold: VAD 语音概率阈值 (0.0~1.0)，超过此值视为人声帧
            prefix_padding_ms: 语音段前置非人声时间（毫秒），
                确定语音开始后，会回溯包含这段前置静音帧
            min_voice_duration_ms: 语音段中最短人声时间（毫秒），
                累计人声时长达到此值后才确认为真正的语音段
            silence_duration_ms: 语音段后置非人声时间（毫秒），
                确认语音后，静音持续超过此时长则结束当前语音段
            abs_amplitude_threshold: loudness_percentile_95 的阈值，
                超过此值的语音段才被接受（TURN_END），否则被拒绝（TURN_REJECTED）
        """
        self._vad = SileroVAD(model_path)
        self._vad_threshold = vad_threshold
        self._prefix_padding_ms = prefix_padding_ms
        self._min_voice_duration_ms = min_voice_duration_ms
        self._silence_duration_ms = silence_duration_ms
        self._abs_amplitude_threshold = abs_amplitude_threshold

        # 内部状态
        self._state = TurnState.IDLE
        self._voice_frames: List[AudioFrame] = []       # 当前语音段的帧缓冲
        self._prefix_buffer: List[AudioFrame] = []      # 前置静音帧环形缓冲
        self._voice_duration_ms: int = 0                 # 当前语音段累积人声时长
        self._silence_duration_ms_acc: int = 0           # 当前静音段累积静音时长
        self._is_confirmed: bool = False                 # 是否已通过 min_voice_duration 验证

    def _max_prefix_frames(self, frame_duration_ms: int) -> int:
        """根据帧时长计算前置缓冲区最大帧数"""
        if frame_duration_ms <= 0:
            return 0
        return max(1, self._prefix_padding_ms // frame_duration_ms)

    def _reset(self):
        """重置内部状态到 IDLE"""
        self._state = TurnState.IDLE
        self._voice_frames.clear()
        self._voice_duration_ms = 0
        self._silence_duration_ms_acc = 0
        self._is_confirmed = False
        # 注意: _prefix_buffer 不在此处清空，保持滚动缓冲

    def detect(self, audio_frame: AudioFrame) -> Utterance:
        """根据输入的音频帧，检测轮次信息

        Args:
            audio_frame: 当前音频帧

        Returns:
            Utterance: 轮次检测结果
        """
        # 计算 VAD 概率
        audio_f32 = np.ctypeslib.as_array(audio_frame.data).astype(np.float32) / 32768.0
        prob = self._vad.prob(audio_f32, sr=audio_frame.sample_rate)
        is_voice = prob >= self._vad_threshold

        frame_duration_ms = audio_frame.duration_ms

        # ── 上一轮结束/拒绝后，自动重置 ──
        if self._state in (TurnState.TURN_END, TurnState.TURN_REJECTED):
            self._reset()

        # ── IDLE 状态 ──
        if self._state == TurnState.IDLE:
            if is_voice:
                # 语音开始：取出前置缓冲，进入 TURN_START
                self._state = TurnState.TURN_START
                self._voice_frames = list(self._prefix_buffer)
                self._voice_frames.append(audio_frame)
                self._voice_duration_ms = frame_duration_ms
                self._silence_duration_ms_acc = 0
                self._is_confirmed = False

                return Utterance(
                    face_id=-1,
                    turn_state=TurnState.TURN_START,
                    frames=list(self._voice_frames),
                )
            else:
                # 维护前置静音缓冲区（环形）
                max_frames = self._max_prefix_frames(frame_duration_ms)
                self._prefix_buffer.append(audio_frame)
                while len(self._prefix_buffer) > max_frames:
                    self._prefix_buffer.pop(0)

                return Utterance(
                    face_id=-1,
                    turn_state=TurnState.IDLE,
                    frames=[audio_frame],
                )

        # ── TURN_START / TURN_CONTINUE: 等待累计人声达到 min_voice_duration_ms ──
        if self._state in (TurnState.TURN_START, TurnState.TURN_CONTINUE):
            self._voice_frames.append(audio_frame)

            if is_voice:
                self._voice_duration_ms += frame_duration_ms
                self._silence_duration_ms_acc = 0

                if self._voice_duration_ms >= self._min_voice_duration_ms:
                    self._state = TurnState.TURN_CONFIRMED
                    self._is_confirmed = True
                    return Utterance(
                        face_id=-1,
                        turn_state=TurnState.TURN_CONFIRMED,
                        frames=list(self._voice_frames),
                    )
                else:
                    self._state = TurnState.TURN_CONTINUE
                    return Utterance(
                        face_id=-1,
                        turn_state=TurnState.TURN_CONTINUE,
                        frames=list(self._voice_frames),
                    )
            else:
                # 在未确认阶段收到静音
                self._silence_duration_ms_acc += frame_duration_ms
                if self._silence_duration_ms_acc >= self._silence_duration_ms:
                    # 未确认就超时 -> 伪语音段，拒绝
                    result = Utterance(
                        face_id=-1,
                        turn_state=TurnState.TURN_REJECTED,
                        frames=list(self._voice_frames),
                    )
                    self._state = TurnState.TURN_REJECTED
                    return result
                else:
                    return Utterance(
                        face_id=-1,
                        turn_state=self._state,
                        frames=list(self._voice_frames),
                    )

        # ── TURN_CONFIRMED: 已确认的语音段 ──
        if self._state == TurnState.TURN_CONFIRMED:
            self._voice_frames.append(audio_frame)

            if is_voice:
                self._voice_duration_ms += frame_duration_ms
                self._silence_duration_ms_acc = 0
                return Utterance(
                    face_id=-1,
                    turn_state=TurnState.TURN_CONFIRMED,
                    frames=list(self._voice_frames),
                )
            else:
                # 进入静音检测阶段
                self._silence_duration_ms_acc = frame_duration_ms
                self._state = TurnState.TURN_SILENCE
                return Utterance(
                    face_id=-1,
                    turn_state=TurnState.TURN_SILENCE,
                    frames=list(self._voice_frames),
                )

        # ── TURN_SILENCE: 确认后的静音段 ──
        if self._state == TurnState.TURN_SILENCE:
            self._voice_frames.append(audio_frame)

            if is_voice:
                # 静音被打断，回到 CONFIRMED
                self._voice_duration_ms += frame_duration_ms
                self._silence_duration_ms_acc = 0
                self._state = TurnState.TURN_CONFIRMED
                return Utterance(
                    face_id=-1,
                    turn_state=TurnState.TURN_CONFIRMED,
                    frames=list(self._voice_frames),
                )
            else:
                self._silence_duration_ms_acc += frame_duration_ms
                if self._silence_duration_ms_acc >= self._silence_duration_ms:
                    # 静音超时 -> 结束语音段
                    utterance = Utterance(
                        face_id=-1,
                        turn_state=TurnState.TURN_END,
                        frames=list(self._voice_frames),
                    )
                    # 响度验证
                    loudness = utterance.get_loudness_percentile_95()
                    if loudness < self._abs_amplitude_threshold:
                        utterance.turn_state = TurnState.TURN_REJECTED
                        self._state = TurnState.TURN_REJECTED
                    else:
                        self._state = TurnState.TURN_END

                    return utterance
                else:
                    return Utterance(
                        face_id=-1,
                        turn_state=TurnState.TURN_SILENCE,
                        frames=list(self._voice_frames),
                    )

        # 兜底（不应到达）
        return Utterance(
            face_id=-1,
            turn_state=self._state,
            frames=[audio_frame],
        )
