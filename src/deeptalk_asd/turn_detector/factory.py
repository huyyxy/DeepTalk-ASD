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
                logger.info(f"创建 SileroVadTurnDetector, kwargs={self.kwargs}")
                from .silero_vad_turn_detector import SileroVadTurnDetector
                detector = SileroVadTurnDetector(**self.kwargs)
                logger.info("use Silero VAD turn detector")
                return detector
            elif self.type == 'pvad':
                kwargs = dict(self.kwargs)
                vad_config = kwargs.pop('vad', None)
                if vad_config is not None:
                    vad_config = dict(vad_config)
                    vad_type = vad_config.pop('type', 'silero-vad')
                else:
                    vad_type = 'silero-vad'
                    pvad_model_name = kwargs.pop('model_name', None)
                    _PVAD_KEYS = ('pvad_model_name', 'spk_model_name', 'pvad_threshold',
                                  'min_low_frames', 'cooldown_frames')
                    pvad_kwargs = {}
                    for key in _PVAD_KEYS:
                        if key in kwargs:
                            pvad_kwargs[key] = kwargs.pop(key)
                    if pvad_model_name and 'pvad_model_name' not in pvad_kwargs:
                        pvad_kwargs['pvad_model_name'] = pvad_model_name
                    vad_config = kwargs
                    kwargs = pvad_kwargs
                    if 'model_dir' in vad_config:
                        kwargs['model_dir'] = vad_config['model_dir']

                logger.info(f"创建 pVAD: inner_vad_type={vad_type}, vad_config={vad_config}, pvad_kwargs={kwargs}")
                inner_detector = TurnDetectorFactory(vad_type, **vad_config).turn_detector()
                if inner_detector is None:
                    logger.error(f"创建内部 TurnDetector(type={vad_type}) 失败")
                    return None

                logger.info(f"内部 TurnDetector 创建成功, 开始创建 PVADTurnDetector, kwargs={kwargs}")
                from .pvad_turn_detector import PVADTurnDetector
                detector = PVADTurnDetector(inner_detector=inner_detector, **kwargs)
                logger.info(f"use pVAD turn detector (inner: {vad_type})")
                return detector
            elif self.type == 'LR-ASD-ONNX':
                from speaker_detector.lrasd_onnx import LRASDOnnxSpeakerDetector
                detector = LRASDOnnxSpeakerDetector(**self.kwargs)
                logger.info("use LR-ASD-ONNX model")
                return detector
            else:
                logger.error(f"未知的 TurnDetector 类型: {self.type}")
        except Exception as e:
            logger.error(f"创建 TurnDetector(type={self.type}) 异常: {e}", exc_info=True)
        return None

