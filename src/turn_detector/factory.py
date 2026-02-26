from ..deeptalk_logger import DeepTalkLogger
from .interface import TurnDetectorInterface
import traceback
import sys
import os


logger = DeepTalkLogger(__name__)

class TurnDetectorFactory:
    def __init__(self, type: str, **kwargs):
        """
        使用指定的 类型创建 图像质量 实例。
        
        :param type: 'LR-ASD-ONNX' 等可选的 活跃说话人模型 类型。
        """
        self.type = type
        self.kwargs = kwargs

    def turn_detector(self)-> TurnDetectorInterface:
        """
        根据 type 获得图像检测器，。
        
        :return: SpeakerDetectorInterface 实例
        """
        try:
            if self.type == 'silero-vad':
                from .silero_vad_turn_detector import SileroVadTurnDetector
                detector = SileroVadTurnDetector(**self.kwargs)
                logger.info("use Silero VAD turn detector")
                return detector
            elif self.type == 'LR-ASD-ONNX':
                from speaker_detector.lrasd_onnx import LRASDOnnxSpeakerDetector
                detector = LRASDOnnxSpeakerDetector(**self.kwargs)
                logger.info("use LR-ASD-ONNX model")
                return detector
        except Exception as e:
            traceback.print_exc()
        return None

