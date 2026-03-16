"""
文件名: asd.py
描述:
    ASDInterface 的实现，编排 FaceDetector、TurnDetector、SpeakerDetector 三个子组件
    完成活动说话者检测的完整流程。
"""
import cv2
import numpy as np
from typing import List

from .asd_interface import ASDInterface
from .video_frame import VideoFrame
from .audio_frame import AudioFrame
from .face_detector.face_info import FaceProfile
from .turn_detector.interface import TurnDetectorInterface
from .turn_detector.utterance import Utterance, TurnState
from .face_detector.interface import FaceDetectorInterface
from .speaker_detector.interface import SpeakerDetectorInterface

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
            2. 将每个人脸的灰度图与 track_id 传给 SpeakerDetector

        参数:
            video_frame: 当前视频帧
            create_time: 此块视频帧的创建时间
        """
        # 1. 人脸检测
        face_profiles = self._face_detector.detect(video_frame)

        # 2. 转换为 SpeakerDetector 需要的格式: [{'id': track_id, 'image': face_gray}, ...]
        frame_faces = []
        for profile in face_profiles:
            face_gray = self._extract_mouth_image(profile)
            if face_gray is not None:
                frame_faces.append({
                    'id': profile.id,
                    'image': face_gray,
                })

        # 3. 传入 SpeakerDetector
        if frame_faces:
            self._speaker_detector.append_video(frame_faces, create_time)
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
        if utterance.turn_state in [TurnState.TURN_START, TurnState.TURN_CONFIRMED, TurnState.TURN_END, TurnState.TURN_REJECTED]:
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
        return self._speaker_detector.evaluate(start_time, end_time)

    def reset(self):
        """重置系统状态"""
        self._speaker_detector.reset()

    def _extract_mouth_image(self, profile: FaceProfile) -> np.ndarray:
        """根据人脸和 5 个关键点，截取以嘴部为中心的 112x112 灰度人脸图像"""
        face_image = profile.face_image
        if face_image is None:
            return None
            
        # 转换为灰度图
        if len(face_image.shape) == 3:
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_image.copy()
            
        h, w = face_gray.shape[:2]
        if h <= 0 or w <= 0:
            return face_gray

        # 将灰度人脸统一 resize 到 224x224
        # 这是为了确保人脸比例及大小相对恒定，无论原本截取的分辨率是多少
        TARGET_SIZE = 224
        CROP_SIZE = 112
        face_gray_resized = cv2.resize(face_gray, (TARGET_SIZE, TARGET_SIZE))

        # 如果有五点标记，基于关键点计算嘴部中心
        pts = profile.five_key_points
        if pts is not None and len(pts) >= 5:
            start_x = max(int(profile.face_rectangle.x), 0)
            start_y = max(int(profile.face_rectangle.y), 0)
            
            mouth_left = pts[3]
            mouth_right = pts[4]
            mouth_cx_abs = (mouth_left[0] + mouth_right[0]) / 2.0
            mouth_cy_abs = (mouth_left[1] + mouth_right[1]) / 2.0
            
            mouth_cx_local = mouth_cx_abs - start_x
            mouth_cy_local = mouth_cy_abs - start_y
            
            # 缩放到 224 尺寸下的坐标
            mouth_cx_scaled = int(mouth_cx_local * (TARGET_SIZE / w))
            mouth_cy_scaled = int(mouth_cy_local * (TARGET_SIZE / h))
        else:
            # 没有关键点则默认在正下半部分中心截取
            mouth_cx_scaled = TARGET_SIZE // 2
            mouth_cy_scaled = TARGET_SIZE // 2 + TARGET_SIZE // 4

        half_c = CROP_SIZE // 2
        crop_x1 = mouth_cx_scaled - half_c
        crop_y1 = mouth_cy_scaled - half_c
        crop_x2 = mouth_cx_scaled + half_c
        crop_y2 = mouth_cy_scaled + half_c
        
        # 边界挪移：如果某一边超出边界，向反方向移回差值，保证一定能截取到 CROP_SIZE 宽高
        if crop_x1 < 0:
            crop_x2 += (0 - crop_x1)
            crop_x1 = 0
        if crop_x2 > TARGET_SIZE:
            crop_x1 -= (crop_x2 - TARGET_SIZE)
            crop_x2 = TARGET_SIZE
            
        if crop_y1 < 0:
            crop_y2 += (0 - crop_y1)
            crop_y1 = 0
        if crop_y2 > TARGET_SIZE:
            crop_y1 -= (crop_y2 - TARGET_SIZE)
            crop_y2 = TARGET_SIZE
            
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(TARGET_SIZE, crop_x2)
        crop_y2 = min(TARGET_SIZE, crop_y2)
        
        mouth_crop = face_gray_resized[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 防呆逻辑：若不是 112x112 则 fallback 为强行 resize（一般在边界挪移后不会触发）
        if mouth_crop.shape[0] != CROP_SIZE or mouth_crop.shape[1] != CROP_SIZE:
            try:
                mouth_crop = cv2.resize(mouth_crop, (CROP_SIZE, CROP_SIZE))
            except Exception:
                pass
                
        return mouth_crop
