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
