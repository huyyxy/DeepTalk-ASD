from deeptalk_logger import DeepTalkLogger
from asd_interface import ASDInterface
import torch
import traceback
import sys
import os


logger = DeepTalkLogger(__name__)

class ASDDetectorFactory:
    def __init__(self, type: str, **kwargs):
        """
        使用指定的 类型创建 图像质量 实例。
        
        :param type: 'TalkNet' 等可选的 活跃说话人模型 类型。
        """
        self.type = type
        self.kwargs = kwargs

    def asd_detector(self)-> ASDInterface:
        """
        根据 type 获得图像检测器，。
        
        :return: ASDInterface 实例
        """
        try:
            if self.type == 'TalkNet':
                # 获取当前文件的路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                submodule_path = os.path.join(current_dir, 'TalkNet_ASD')

                # 将子模块路径添加到 sys.path
                sys.path.insert(0, submodule_path)

                from speaker_detector.talknet import TalkNetSpeakerDetector
                detector = TalkNetSpeakerDetector(**self.kwargs)
                logger.info("use TalkNet-ASD model")
                return detector
            elif self.type == 'LoCoNet':
                pass
            elif self.type == 'EASEE-50':
                pass
            elif self.type == 'Light-ASD':
                # 获取当前文件的路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                submodule_path = os.path.join(current_dir, 'Light_ASD')

                # 将子模块路径添加到 sys.path
                sys.path.insert(0, submodule_path)

                from speaker_detector.lightasd import LightASDSpeakerDetector
                detector = LightASDSpeakerDetector(**self.kwargs)
                logger.info("use Light-ASD model")
                return detector
            elif self.type == 'LR-ASD':
                # 获取当前文件的路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                submodule_path = os.path.join(current_dir, 'LR_ASD')

                # 将子模块路径添加到 sys.path
                sys.path.insert(0, submodule_path)

                from speaker_detector.lrasd import LRASDSpeakerDetector
                detector = LRASDSpeakerDetector(**self.kwargs)
                logger.info("use LR-ASD model")
                return detector
        except Exception as e:
            traceback.print_exc()
        return None

