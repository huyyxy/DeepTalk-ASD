import numpy as np

from ..audio_frame import AudioFrame
from .utterance import Utterance


class TurnDetectorInterface:
    """轮次检测核心类"""
    
    def detect(self, audio_frame: AudioFrame) -> Utterance:
        """
        根据输入的音频帧，检测轮次信息

        参数:
            audio_frame: 当前音频帧

        返回:
            轮次检测结果
        """
        pass

    def on_speaker_identified(self, audio_samples: np.ndarray, sample_rate: int = 16000):
        """
        当 evaluate() 识别出说话人后由 ASD 编排层调用。

        默认空实现（no-op）。PVADTurnDetector 覆写此方法以提取目标声纹并激活 pVAD 监测。

        参数:
            audio_samples: 说话人所在时间段的原始音频 (float32, 单声道)
            sample_rate: 采样率，默认 16000
        """
        pass
