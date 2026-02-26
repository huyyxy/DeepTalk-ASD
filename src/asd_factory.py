"""
文件名: asd_factory.py
描述:
    ASD 检测器工厂，通过配置信息分别创建 FaceDetector、TurnDetector、SpeakerDetector，
    然后组装为 ASD 实例。
"""
from .deeptalk_logger import DeepTalkLogger
from .asd_interface import ASDInterface
import traceback

logger = DeepTalkLogger(__name__)


class ASDDetectorFactory:
    def __init__(self, **kwargs):
        """
        使用配置信息创建 ASD 实例。

        参数:
            kwargs: 包含三个子组件的配置信息:
                face_detector: dict, 人脸检测器配置, 如 {"type": "inspireface", ...}
                turn_detector: dict, 轮次检测器配置, 如 {"type": "silero-vad", "model_path": "...", ...}
                speaker_detector: dict, 说话者检测器配置, 如 {"type": "LR-ASD-ONNX", "onnx_dir": "...", ...}
        """
        self.kwargs = kwargs

    def create(self) -> ASDInterface:
        """
        根据配置创建 ASD 实例。

        :return: ASDInterface 实例，创建失败时返回 None
        """
        try:
            # 1. 创建 FaceDetector
            face_config = dict(self.kwargs.get('face_detector', {}))
            face_type = face_config.pop('type', None)
            if face_type is None:
                logger.error("face_detector 配置缺少 'type' 字段")
                return None

            from .face_detector.factory import FaceDetectorFactory
            face_detector = FaceDetectorFactory(face_type, **face_config).face_detector()
            if face_detector is None:
                logger.error(f"创建 FaceDetector(type={face_type}) 失败")
                return None
            logger.info(f"FaceDetector 创建成功: type={face_type}")

            # 2. 创建 TurnDetector
            turn_config = dict(self.kwargs.get('turn_detector', {}))
            turn_type = turn_config.pop('type', None)
            if turn_type is None:
                logger.error("turn_detector 配置缺少 'type' 字段")
                return None

            from .turn_detector.factory import TurnDetectorFactory
            turn_detector = TurnDetectorFactory(turn_type, **turn_config).turn_detector()
            if turn_detector is None:
                logger.error(f"创建 TurnDetector(type={turn_type}) 失败")
                return None
            logger.info(f"TurnDetector 创建成功: type={turn_type}")

            # 3. 创建 SpeakerDetector
            speaker_config = dict(self.kwargs.get('speaker_detector', {}))
            speaker_type = speaker_config.pop('type', None)
            if speaker_type is None:
                logger.error("speaker_detector 配置缺少 'type' 字段")
                return None

            from .speaker_detector.factory import SpeakerDetectorFactory
            speaker_detector = SpeakerDetectorFactory(speaker_type, **speaker_config).speaker_detector()
            if speaker_detector is None:
                logger.error(f"创建 SpeakerDetector(type={speaker_type}) 失败")
                return None
            logger.info(f"SpeakerDetector 创建成功: type={speaker_type}")

            # 4. 组装 ASD 实例
            from .asd import ASD
            asd = ASD(
                face_detector=face_detector,
                turn_detector=turn_detector,
                speaker_detector=speaker_detector,
            )
            logger.info("ASD 实例创建成功")
            return asd

        except Exception as e:
            logger.error(f"创建 ASD 实例失败: {e}")
            traceback.print_exc()
            return None
