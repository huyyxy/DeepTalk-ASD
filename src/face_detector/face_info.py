"""
文件名: face_info.py
作者: William Hu
邮箱: huyiyang@tiwater.com
创建日期: 2025-05-25
版本: 0.1

描述:
    定义人脸检测结果的数据类，用于结构化表示检测到的人脸属性信息。
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FaceRectangle:
    """人脸边界框"""
    x: float
    y: float
    width: float
    height: float


@dataclass
class HeadPose:
    """头部姿态角度"""
    yaw: float
    pitch: float
    roll: float


@dataclass
class FaceProfile:
    """
    人脸检测结果的结构化表示。

    包含人脸的位置、姿态、属性（性别、年龄、表情）、
    图像质量评分、最佳人脸图像及其嵌入向量，以及出现时间等信息。
    """
    id: int  # 会根据face embedding回溯到一开始相似的track_id，能在异常情况下确保track_id的不变
    track_id: int  # 人脸检测算法直接返回的track_id
    face_continue_frame: int
    face_rectangle: FaceRectangle
    head_pose: HeadPose
    gender: str
    age: str
    emotion: str
    face_image: Optional[np.ndarray] = None
    five_key_points: Optional[list] = None
    expand_face_image: Optional[np.ndarray] = None
    face_image_score: float = 0.0
    best_face_image: Optional[np.ndarray] = None
    expand_best_face_image: Optional[np.ndarray] = None # 做人脸识别比对需要扩大的最佳人脸
    best_face_image_score: float = 0.0
    best_face_embedding: Optional[np.ndarray] = None
    first_appearance_time: Optional[float] = None
    appearance_time: Optional[float] = None
    last_appearance_time: Optional[float] = None

    def to_dict(self) -> dict:
        """转换为字典格式，保持向后兼容"""
        return {
            "id": self.id,
            "track_id": self.track_id,
            "face_continue_frame": self.face_continue_frame,
            "face_rectangle": {
                "x": self.face_rectangle.x,
                "y": self.face_rectangle.y,
                "width": self.face_rectangle.width,
                "height": self.face_rectangle.height,
            },
            "head_pose": {
                "yaw": self.head_pose.yaw,
                "pitch": self.head_pose.pitch,
                "roll": self.head_pose.roll,
            },
            "gender": self.gender,
            "age": self.age,
            "emotion": self.emotion,
            "face_image": self.face_image,
            "five_key_points": self.five_key_points,
            "expand_face_image": self.expand_face_image,
            "face_image_score": self.face_image_score,
            "best_face_image": self.best_face_image,
            "expand_best_face_image": self.expand_best_face_image,
            "best_face_image_score": self.best_face_image_score,
            "best_face_embedding": self.best_face_embedding,
            "first_appearance_time": self.first_appearance_time,
            "appearance_time": self.appearance_time,
            "last_appearance_time": self.last_appearance_time,
        }
