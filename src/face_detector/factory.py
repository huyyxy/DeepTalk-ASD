from ..deeptalk_logger import DeepTalkLogger
from .interface import FaceDetectorInterface
import traceback


logger = DeepTalkLogger(__name__)

class FaceDetectorFactory:
    def __init__(self, type: str, **kwargs):
        """
        使用指定的 类型创建 图像质量 实例。
        
        :param type: 'inspireface' 等可选的人脸检测模型 类型。
        """
        self.type = type
        self.kwargs = kwargs

    def face_detector(self)-> FaceDetectorInterface:
        """
        根据 type 获得图像检测器，。
        
        :return: FaceDetectorInterface 实例
        """
        try:
            if self.type == 'inspireface':
                from .inspireface_detector import InspireFaceDetector
                detector = InspireFaceDetector(**self.kwargs)
                logger.info("use inspireface model")
                return detector
        except Exception as e:
            traceback.print_exc()
        return None

