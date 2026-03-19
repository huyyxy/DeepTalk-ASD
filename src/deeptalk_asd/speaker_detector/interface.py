from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class FaceData:
    """SpeakerDetector 所需的人脸输入数据，与 FaceDetector 的具体实现解耦。"""
    id: int
    face_image: np.ndarray
    face_rect_x: float
    face_rect_y: float
    face_rect_width: float
    face_rect_height: float
    five_key_points: Optional[list] = None


class SpeakerDetectorInterface:
    """活动说话者检测核心类"""

    def append_video(self, face_data_list: List[FaceData], create_time: Optional[float] = None):
        """
        添加视频帧中已检测的人脸信息，视频帧率缺省为25

        参数:
            face_data_list: 当前帧的人脸数据列表 (FaceData)
            create_time: 帧的创建时间，调用方一般使用time.perf_counter()获得，用于和音频块时间对齐
        """
        pass

    def append_audio(
        self,
        audio_chunk: Union[np.ndarray, bytes, bytearray],
        create_time: Optional[float] = None,
    ):
        """
        添加音频块到处理队列

        参数:
            audio_chunk: pcm_data，一般为16bit，单声道，16000Hz，30ms时长音频块；可为 ndarray 或 bytes/bytearray
            create_time: 此块audio_chunk的创建时间，调用方一般使用time.perf_counter()获得，用于和视频帧时间对其
        """
        pass

    def evaluate(self, start_time: float = None, end_time: float = None):
        """
        评估当前活动说话者，一般是前置VAD检测发现结束说话时调用

        参数:
            start_time: 评估起始时间（time.perf_counter 时间戳），可选
            end_time: 评估结束时间（time.perf_counter 时间戳），可选
            若均提供，则只对该时间范围内的音视频做推理；
            若时间范围超过缓冲区长度，则回退为使用整个缓冲区。
            若不传，则使用整个缓冲区。

        返回:
            最新时间点的活动说话者的tracker_id和置信度得分
        """
        pass

    def get_audio_samples(self, start_time: float = None, end_time: float = None) -> Optional[np.ndarray]:
        """
        获取指定时间范围内的原始音频样本。

        参数:
            start_time: 起始时间 (time.perf_counter 时间戳)，可选
            end_time: 结束时间 (time.perf_counter 时间戳)，可选
            若均为 None，返回整个音频缓冲区。

        返回:
            int16 格式的音频数据 (np.ndarray)，或 None
        """
        pass

    def reset(self):
        """重置系统状态"""
        pass
