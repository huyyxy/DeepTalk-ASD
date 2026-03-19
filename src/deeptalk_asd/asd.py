"""
文件名: asd.py
描述:
    ASDInterface 的实现，编排 FaceDetector、TurnDetector、SpeakerDetector 三个子组件
    完成活动说话者检测的完整流程。
"""
from typing import List

from .asd_interface import ASDInterface
from .video_frame import VideoFrame
from .audio_frame import AudioFrame
from .face_detector.face_info import FaceProfile
from .turn_detector.interface import TurnDetectorInterface
from .turn_detector.utterance import Utterance, TurnState
from .face_detector.interface import FaceDetectorInterface
from .speaker_detector.interface import SpeakerDetectorInterface, FaceData

from .deeptalk_logger import DeepTalkLogger

logger = DeepTalkLogger(__name__)


class ASD(ASDInterface):
    """活动说话者检测实现类

    编排三个子组件：
    - FaceDetector: 从视频帧中检测人脸
    - TurnDetector: 从音频帧中检测语音轮次 (VAD)
    - SpeakerDetector: 融合音视频特征，判定活跃说话者
    """

    def __init__(
        self,
        face_detector: FaceDetectorInterface,
        turn_detector: TurnDetectorInterface,
        speaker_detector: SpeakerDetectorInterface,
        **kwargs,
    ):
        """
        初始化活动说话者检测系统

        参数:
            face_detector: 人脸检测器实例
            turn_detector: 语音轮次检测器实例
            speaker_detector: 说话者检测器实例
            kwargs: 可选参数，保留用于未来扩展
        """
        super().__init__(**kwargs)
        self._face_detector = face_detector
        self._turn_detector = turn_detector
        self._speaker_detector = speaker_detector

    def append_video(self, video_frame: VideoFrame, create_time: float = None) -> List[FaceProfile]:
        """
        添加视频帧，视频帧率缺省为25

        流程:
            1. 使用 FaceDetector 检测视频帧中的人脸
            2. 将人脸检测结果传给 SpeakerDetector

        参数:
            video_frame: 当前视频帧
            create_time: 此块视频帧的创建时间
        """
        face_profiles = self._face_detector.detect(video_frame)
        if face_profiles:
            face_data_list = [
                FaceData(
                    id=p.id,
                    face_image=p.face_image,
                    face_rect_x=p.face_rectangle.x,
                    face_rect_y=p.face_rectangle.y,
                    face_rect_width=p.face_rectangle.width,
                    face_rect_height=p.face_rectangle.height,
                    five_key_points=p.five_key_points,
                )
                for p in face_profiles
                if p.face_image is not None
            ]
            if face_data_list:
                self._speaker_detector.append_video(face_data_list, create_time)
        return face_profiles

    def append_audio(self, audio_frame: AudioFrame, create_time: float = None) -> Utterance:
        """
        添加音频块到处理队列

        流程:
            1. 将音频 PCM 数据传给 SpeakerDetector
            2. 使用 TurnDetector 进行 VAD 检测

        参数:
            audio_frame: 当前音频帧
            create_time: 此块音频帧的创建时间

        返回:
            轮次检测结果 (Utterance)
        """
        # 1. 将音频数据传入 SpeakerDetector
        audio_data = bytes(audio_frame.data)
        self._speaker_detector.append_audio(audio_data, create_time)

        # 2. TurnDetector 进行 VAD 轮次检测
        utterance = self._turn_detector.detect(audio_frame)
        if utterance.turn_state in [TurnState.TURN_START, TurnState.TURN_CONFIRMED, TurnState.TURN_END, TurnState.TURN_REJECTED, TurnState.SPEAKER_CHANGE]:
            logger.critical(f"Utterance: {utterance.turn_state.name}")
        return utterance

    def evaluate(self, start_time: float = None, end_time: float = None):
        """
        评估当前活动说话者

        参数:
            start_time: 评估起始时间（time.perf_counter 时间戳），可选
            end_time: 评估结束时间（time.perf_counter 时间戳），可选

        返回:
            最新时间点的活动说话者的 track_id 和置信度得分
            格式: {track_id: score, ...}
        """
        scores = self._speaker_detector.evaluate(start_time, end_time)

        if scores:
            best_track = max(scores, key=scores.get)
            if scores[best_track] > 0:
                audio_samples = self._speaker_detector.get_audio_samples(start_time, end_time)
                if audio_samples is not None and len(audio_samples) > 0:
                    self._turn_detector.on_speaker_identified(
                        audio_samples.astype('float32') / 32768.0,
                        getattr(self._speaker_detector, 'audio_sample_rate', 16000),
                    )

        return scores

    def reset(self):
        """重置系统状态"""
        self._speaker_detector.reset()
