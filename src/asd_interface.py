from .video_frame import VideoFrame
from .audio_frame import AudioFrame
from .turn_detector.utterance import Utterance
from .face_detector.interface import FaceProfile
from typing import List


class ASDInterface:
    """活动说话者检测核心类"""
    def __init__(self, **kwargs):
        """
        初始化活动说话者检测系统
        
        参数:
            kwargs: 可选参数，当前未使用，保留用于未来扩展。
        """
        pass
    
    def append_video(self, video_frame: VideoFrame, create_time: float = None) -> List[FaceProfile]:
        """
        添加视频帧，视频帧率缺省为25

        参数:
            video_frame: 当前视频帧
            create_time: 此块视频帧的创建时间，调用方一般使用time.perf_counter()获得，用于和音频块时间对齐
        """
        pass
    
    def append_audio(self, audio_frame: AudioFrame, create_time: float = None) -> Utterance:
        """
        添加音频块到处理队列
        
        参数:
            audio_frame: 当前音频帧
            create_time: 此块音频帧的创建时间，调用方一般使用time.perf_counter()获得，用于和视频帧时间对齐
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
    
    def reset(self):
        """重置系统状态"""
        pass
