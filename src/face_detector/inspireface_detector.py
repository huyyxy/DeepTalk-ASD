"""
文件名: inspireface_detector.py
作者: William Hu
邮箱: huyiyang@tiwater.com
创建日期: 2025-05-25
版本: 0.1

描述:
    基于 InspireFace 的人脸检测器实现，作为 FaceDetectorInterface 的派生类。
    直接使用 inspireface SDK 进行人脸检测，将 VideoFrame 转换为 numpy 图像后进行检测。
"""
import os
import time
import traceback
from collections import deque, OrderedDict

import cv2
import numpy as np
import inspireface as isf

from ..video_frame import VideoFrame, VideoBufferType
from .interface import FaceDetectorInterface
from .face_info import FaceProfile, FaceRectangle, HeadPose


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INSPIREFACE_RESOURCE_PATH = os.environ.get('INSPIREFACE_RESOURCE_PATH', os.path.join(_PROJECT_ROOT, 'weights', 'Pikachu'))

WINDOWS_FACE_PROFILES_FRAMES = int(os.environ.get('WINDOWS_FACE_PROFILES_FRAMES', '2'))
MIN_FRAMES_FOR_FACE_PRESENTATION = int(os.environ.get('MIN_FRAMES_FOR_FACE_PRESENTATION', '1'))
REMOVE_STALE_FACE_INTERVAL_SECONDS = int(os.environ.get('REMOVE_STALE_FACE_INTERVAL_SECONDS', '60'))
FACE_DISAPPEARANCE_INTERVAL_SECONDS = int(os.environ.get('FACE_DISAPPEARANCE_INTERVAL_SECONDS', '0'))
TRACK_MODE_SMOOTH_RATIO = float(os.environ.get('TRACK_MODE_SMOOTH_RATIO', '0.06'))
TRACK_MODE_NUM_SMOOTH_CACHE_FRAME = int(os.environ.get('TRACK_MODE_NUM_SMOOTH_CACHE_FRAME', '15'))
FILTER_MINIMUM_FACE_PIXEL_SIZE = int(os.environ.get('FILTER_MINIMUM_FACE_PIXEL_SIZE', '0'))
TRACK_MODEL_DETECT_INTERVAL = int(os.environ.get('TRACK_MODEL_DETECT_INTERVAL', '0'))

race_tags = ["Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"]
gender_tags = ["Female", "Male"]
age_bracket_tags = [
    "0-2 years old", "3-9 years old", "10-19 years old", "20-29 years old", "30-39 years old",
    "40-49 years old", "50-59 years old", "60-69 years old", "more than 70 years old"
]
emotion_tags = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]


class InspireFaceDetector(FaceDetectorInterface):
    """基于 InspireFace SDK 的人脸检测器"""

    def __init__(self, **kwargs):
        """
        初始化 InspireFace 人脸检测器。

        参数:
            kwargs: 可选参数，当前未使用，保留用于未来扩展。
        """
        super().__init__(**kwargs)

        isf.launch(resource_path=INSPIREFACE_RESOURCE_PATH)

        opt = (isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_QUALITY |
               isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION)
        self.session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_LIGHT_TRACK)
        self.session.set_detection_confidence_threshold(0.5)
        self.session.set_track_mode_smooth_ratio(TRACK_MODE_SMOOTH_RATIO)
        self.session.set_track_mode_num_smooth_cache_frame(TRACK_MODE_NUM_SMOOTH_CACHE_FRAME)
        self.session.set_filter_minimum_face_pixel_size(FILTER_MINIMUM_FACE_PIXEL_SIZE)
        self.session.set_track_model_detect_interval(TRACK_MODEL_DETECT_INTERVAL)

        self.window_face_profiles = deque(maxlen=WINDOWS_FACE_PROFILES_FRAMES)
        self.history_faces = OrderedDict()
        self.face_profiles = {"face_count": 0, "profiles": {}}
        self.last_update_face_profiles_time = time.perf_counter()
        self.id_mapping = {}

    def detect(self, video_frame: VideoFrame) -> list[FaceProfile]:
        """
        检测视频帧中的人脸。

        参数:
            video_frame: 输入的视频帧

        返回:
            检测到的人脸列表 (FaceProfile)
        """
        self._remove_stale_faces()

        try:
            image_bgr = self._video_frame_to_bgr(video_frame)
            current_face_profiles = self._detect(image_bgr)
            self._update_window_face_profiles(current_face_profiles)
            profiles: dict = self.face_profiles.get("profiles", {})
            return list(profiles.values())
        except Exception as e:
            traceback.print_exc()
            return []

    def _detect(self, image: np.ndarray) -> dict:
        """检测图像中的人脸并更新人脸信息"""
        faces = self.session.face_detection(image)
        if faces and len(faces) > 0:
            exts = self.session.face_pipeline(
                image, faces,
                isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION
            )

        all_face_profiles = self._get_all_profiles()
        all_profiles = all_face_profiles.get("profiles")

        current_face_profiles = {"face_count": 0, "profiles": {}}
        for idx, face in enumerate(faces):
            tracker_id = face.track_id
            x1, y1, x2, y2 = face.location
            width = x2 - x1
            height = y2 - y1

            face_image_bgr = self._expand_face_area_by_ratio(image, x1, y1, width, height, expand_ratio=1)

            ext = exts[idx]
            face_image_score = ext.quality_confidence
            if abs(face.yaw) > 20 or abs(face.pitch) > 10 or abs(face.roll) > 10:
                face_image_score = 0
            else:
                face_image_score = ext.quality_confidence - abs(face.yaw) - abs(face.pitch) - abs(face.roll) * 0.8

            emotion = emotion_tags[ext.emotion]

            face_id = self._get_face_id_by_track_id(face.track_id)
            history_profile = all_profiles.get(face_id)
            history_face = self.history_faces.get(face_id)
            last_appearance_time = history_face.get("appearance_time") if history_face else None

            if history_profile:
                face_continue_frame = history_profile.face_continue_frame + 1
                first_appearance_time = history_profile.first_appearance_time
                history_best_face_image_score = (
                    history_profile.best_face_image_score
                    if history_profile.best_face_image_score is not None else -1
                )
                if face_image_score >= history_best_face_image_score:
                    best_face_image = face_image_bgr
                    best_face_image_score = face_image_score
                    feature = self.session.face_feature_extract(image, face)
                    face_id = self._match_existing_face(face.track_id, feature)
                else:
                    best_face_image = history_profile.best_face_image
                    best_face_image_score = history_best_face_image_score
                    feature = history_profile.best_face_embedding
            else:
                face_continue_frame = 1
                first_appearance_time = time.time()
                best_face_image = face_image_bgr
                best_face_image_score = face_image_score
                feature = self.session.face_feature_extract(image, face)
                face_id = self._match_existing_face(face.track_id, feature)

            profile = FaceProfile(
                id=face_id,
                track_id=face.track_id,
                face_continue_frame=face_continue_frame,
                face_rectangle=FaceRectangle(x=x1, y=y1, width=width, height=height),
                head_pose=HeadPose(yaw=face.yaw, pitch=face.pitch, roll=face.roll),
                gender=gender_tags[ext.gender],
                age=age_bracket_tags[ext.age_bracket],
                emotion=emotion,
                face_image=face_image_bgr,
                face_image_score=ext.quality_confidence,
                best_face_image=best_face_image,
                best_face_image_score=best_face_image_score,
                best_face_embedding=feature,
                first_appearance_time=first_appearance_time,
                appearance_time=time.time(),
                last_appearance_time=last_appearance_time,
            )

            current_face_profiles["profiles"][face_id] = profile
            current_face_profiles["face_count"] += 1
            self._update_history_face(profile)

        return current_face_profiles

    # ---- ID 映射 ----

    def _get_face_id_by_track_id(self, track_id: int) -> int:
        return self.id_mapping.get(track_id, track_id)

    def _bind_face_id_and_track_id(self, track_id: int, face_id: int):
        self.id_mapping[track_id] = face_id

    # ---- 人脸匹配 ----

    def _match_existing_face(self, current_id: int, embedding) -> int:
        """将嵌入与历史人脸进行匹配来统一 ID"""
        from .face_compare_helper import is_same_face_by_embedding

        for history_id, history_data in list(self.history_faces.items()):
            if history_id >= current_id:
                continue
            if is_same_face_by_embedding(embedding, history_data.get("best_face_embedding")):
                self._bind_face_id_and_track_id(current_id, history_id)
                return history_id
        return current_id

    # ---- 历史人脸管理 ----

    def _update_history_face(self, profile: FaceProfile):
        """更新或添加新的面部记录到历史中"""
        face_id = profile.id
        if profile.face_continue_frame <= 1:
            self.history_faces[face_id] = {
                "id": face_id,
                "best_face_embedding": profile.best_face_embedding,
                "first_appearance_time": profile.first_appearance_time,
                "appearance_time": profile.appearance_time,
            }
        elif face_id in self.history_faces:
            self.history_faces[face_id]["appearance_time"] = profile.appearance_time
            self.history_faces[face_id]["best_face_embedding"] = profile.best_face_embedding

    def _remove_stale_faces(self):
        """移除历史数据中超过一定时间未出现的人脸"""
        now = time.time()
        for history_id in list(self.history_faces):
            if now - self.history_faces[history_id]["appearance_time"] >= REMOVE_STALE_FACE_INTERVAL_SECONDS:
                del self.history_faces[history_id]

    # ---- 滑动窗口管理 ----

    def _update_window_face_profiles(self, current_face_profiles: dict):
        """更新滑动窗口中的人脸配置"""
        self.window_face_profiles.append(current_face_profiles)
        if len(self.window_face_profiles) < WINDOWS_FACE_PROFILES_FRAMES:
            this_face_profiles = {"face_count": 0, "profiles": {}}
        else:
            this_face_profiles = self._aggregate_profiles_from_window()
        if this_face_profiles.get('face_count') > 0:
            self.last_update_face_profiles_time = time.perf_counter()
            self.face_profiles = this_face_profiles
        elif time.perf_counter() - self.last_update_face_profiles_time > FACE_DISAPPEARANCE_INTERVAL_SECONDS:
            self.last_update_face_profiles_time = time.perf_counter()
            self.face_profiles = this_face_profiles

    def _aggregate_profiles_from_window(self) -> dict:
        """跨滑动窗口聚合配置数据"""
        id_counts = self._get_id_counts(self.window_face_profiles)
        profiles = {}
        l_range = WINDOWS_FACE_PROFILES_FRAMES - MIN_FRAMES_FOR_FACE_PRESENTATION + 1

        for face_id, count in id_counts.items():
            if count >= MIN_FRAMES_FOR_FACE_PRESENTATION:
                profile = self._find_recent_profile(self.window_face_profiles, face_id, l_range)
                if profile:
                    profiles[face_id] = profile

        return {"face_count": len(profiles), "profiles": profiles}

    def _find_recent_profile(self, window_face_profiles: deque, face_id: int, l_range: int):
        """找到窗口中给定 ID 的最新配置"""
        for i in range(1, l_range):
            recent_profiles = window_face_profiles[-i]
            profile = recent_profiles["profiles"].get(face_id)
            if profile:
                return profile
        return None

    def _get_all_profiles(self) -> dict:
        """检索窗口内的所有脸部配置"""
        id_counts = self._get_id_counts(self.window_face_profiles)
        profiles = {}
        for face_id in id_counts:
            profiles[face_id] = self._select_most_recent_profile(face_id)
        return {"face_count": len(profiles), "profiles": profiles}

    def _select_most_recent_profile(self, face_id: int):
        """为给定 ID 选择最新的配置"""
        for face_profiles in reversed(self.window_face_profiles):
            profile = face_profiles["profiles"].get(face_id)
            if profile:
                return profile
        return None

    @staticmethod
    def _get_id_counts(window_face_profiles: deque) -> dict:
        """计算窗口中每个人脸 ID 的出现次数"""
        id_counts = {}
        for face_profiles in window_face_profiles:
            for face_id in face_profiles["profiles"]:
                id_counts[face_id] = id_counts.get(face_id, 0) + 1
        return id_counts

    # ---- 图像处理工具 ----

    @staticmethod
    def _expand_face_area_by_ratio(
        image: np.ndarray, x: int, y: int, width: int, height: int, expand_ratio: float = 0.1
    ) -> np.ndarray:
        """按比例扩大并提取人脸区域图像"""
        face_area = width * height
        expand_area = face_area * expand_ratio
        expand_pixels = int(np.sqrt(expand_area) / 2)

        start_y = max(y - expand_pixels, 0)
        start_x = max(x - expand_pixels, 0)
        end_y = min(y + height + expand_pixels, image.shape[0])
        end_x = min(x + width + expand_pixels, image.shape[1])

        return image[start_y:end_y, start_x:end_x]

    @staticmethod
    def _video_frame_to_bgr(video_frame: VideoFrame) -> np.ndarray:
        """
        将 VideoFrame 转换为 BGR 格式的 numpy 数组。

        参数:
            video_frame: 输入的视频帧

        返回:
            BGR 格式的 numpy 图像数组
        """
        width = video_frame.width
        height = video_frame.height
        buf_type = video_frame.type
        raw = bytes(video_frame.data)

        if buf_type == VideoBufferType.BGRA:
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        elif buf_type == VideoBufferType.RGBA:
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        elif buf_type == VideoBufferType.ARGB:
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            rgb = img[:, :, 1:4]
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif buf_type == VideoBufferType.ABGR:
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return img[:, :, 1:4].copy()

        elif buf_type == VideoBufferType.RGB24:
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        elif buf_type == VideoBufferType.I420:
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        elif buf_type == VideoBufferType.NV12:
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

        else:
            raise ValueError(f"不支持的 VideoBufferType: {buf_type}")

    def destroy(self):
        """释放检测器资源"""
        self.session = None
        self.history_faces.clear()
        self.window_face_profiles.clear()
        self.id_mapping.clear()
