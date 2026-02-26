from ..video_frame import VideoFrame
from .face_info import FaceProfile


class FaceDetectorInterface:
    """人脸检测核心类"""
    
    def detect(self, video_frame: VideoFrame) -> list[FaceProfile]:
        """
        根据输入的视频帧，检测人脸信息

        参数:
            video_frame: 视频帧

        返回:
            人脸检测结果列表
        """
        pass
