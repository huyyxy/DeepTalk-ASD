"""
文件名: asd_factory.py
描述:
    ASD 检测器工厂，通过配置信息分别创建 FaceDetector、TurnDetector、SpeakerDetector，
    然后组装为 ASD 实例。

    支持零配置使用：不传参数时自动下载并加载默认模型。
"""
from .deeptalk_logger import DeepTalkLogger
from .asd_interface import ASDInterface
from .model_manager import ensure_model, get_model_cache_dir
import traceback

logger = DeepTalkLogger(__name__)

# 默认组件配置（零配置时使用）
_DEFAULT_FACE_DETECTOR = {"type": "inspireface"}
_DEFAULT_TURN_DETECTOR = {"type": "silero-vad"}
_DEFAULT_SPEAKER_DETECTOR = {"type": "LR-ASD-ONNX"}


class ASDDetectorFactory:
    def __init__(self, **kwargs):
        """
        使用配置信息创建 ASD 实例。

        支持零配置调用:
            ASDDetectorFactory().create()
        
        也支持完整配置:
            ASDDetectorFactory(
                face_detector={"type": "inspireface", "model_dir": "..."},
                turn_detector={"type": "silero-vad", "model_dir": "..."},
                speaker_detector={"type": "LR-ASD-ONNX", "model_dir": "..."},
            ).create()

        参数:
            kwargs: 包含三个子组件的配置信息，所有组件统一使用 model_dir 指定模型目录:
                face_detector: dict, 人脸检测器配置, 如 {"type": "inspireface", "model_dir": "..."}
                turn_detector: dict, 轮次检测器配置, 如 {"type": "silero-vad", "model_dir": "..."}
                speaker_detector: dict, 说话者检测器配置, 如 {"type": "LR-ASD-ONNX", "model_dir": "..."}
        """
        self.kwargs = kwargs

    def _resolve_face_detector_config(self, face_config: dict) -> dict:
        """解析人脸检测器配置，自动补充默认模型目录"""
        config = dict(face_config)
        # model_dir 可选；未设置时 InspireFaceDetector 内部自动解析
        return config

    def _resolve_turn_detector_config(self, turn_config: dict) -> dict:
        """解析轮次检测器配置，自动补充默认模型目录"""
        config = dict(turn_config)
        if config.get("type") == "silero-vad" and "model_dir" not in config:
            model_path = ensure_model("silero_vad.onnx")
            config["model_dir"] = str(model_path.parent)
        elif config.get("type") == "pvad":
            vad_config = config.get("vad", {"type": "silero-vad"})
            if vad_config.get("type") == "silero-vad" and "model_dir" not in vad_config:
                model_path = ensure_model("silero_vad.onnx")
                vad_config["model_dir"] = str(model_path.parent)
            config["vad"] = vad_config
        return config

    def _resolve_speaker_detector_config(self, speaker_config: dict) -> dict:
        """解析说话者检测器配置，自动补充默认模型目录"""
        config = dict(speaker_config)
        if config.get("type") == "LR-ASD-ONNX":
            if "model_dir" not in config:
                cache_dir = get_model_cache_dir()
                ensure_model("audio_frontend.onnx", cache_dir)
                ensure_model("visual_frontend.onnx", cache_dir)
                ensure_model("av_backend.onnx", cache_dir)
                ensure_model("wespeaker_zh_cnceleb_resnet34.onnx", cache_dir)
                config["model_dir"] = str(cache_dir)
        return config

    def create(self) -> ASDInterface:
        """
        根据配置创建 ASD 实例。

        :return: ASDInterface 实例，创建失败时返回 None
        """
        try:
            # 1. 创建 FaceDetector
            face_config = self._resolve_face_detector_config(
                self.kwargs.get('face_detector', _DEFAULT_FACE_DETECTOR)
            )
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
            turn_config = self._resolve_turn_detector_config(
                self.kwargs.get('turn_detector', _DEFAULT_TURN_DETECTOR)
            )
            turn_type = turn_config.pop('type', None)
            if turn_type is None:
                logger.error("turn_detector 配置缺少 'type' 字段")
                return None

            from .turn_detector.factory import TurnDetectorFactory
            logger.info(f"开始创建 TurnDetector: type={turn_type}, config={turn_config}")
            turn_detector = TurnDetectorFactory(turn_type, **turn_config).turn_detector()
            if turn_detector is None:
                logger.error(f"创建 TurnDetector(type={turn_type}) 失败，工厂返回 None（详细原因见上方日志）")
                return None
            logger.info(f"TurnDetector 创建成功: type={turn_type}")

            # 3. 创建 SpeakerDetector
            speaker_config = self._resolve_speaker_detector_config(
                self.kwargs.get('speaker_detector', _DEFAULT_SPEAKER_DETECTOR)
            )
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

