


class ASDInterface:
    """活动说话者检测核心类"""
    def __init__(self, **kwargs):
        """
        初始化活动说话者检测系统
        
        参数:
            video_fps: 视频帧率
            audio_sample_rate: 音频采样率
            device: 计算设备(cpu或cuda)
        """
        pass
    
    def append_video(self, frame_faces, create_time = None):
        """
        添加视频帧中已检测的人脸信息
        
        参数:
            frame_faces: 当前帧的人脸检测结果列表，每个元素为字典:
                [
                    {
                        'id': tracker_id,   # 人脸的tracker id
                        'bbox': [x1, y1, x2, y2],  # 边界框坐标
                        'image': face_image         # 裁剪后的人脸图像
                    },
                    ... # 可能多个人脸
                ]
        """
        pass
    
    def append_audio(self, audio_chunk, create_time = None):
        """添加音频块到处理队列"""
        pass
    
    def evaluate(self, window_seconds=1.0):
        """
        评估当前活动说话者
        
        参数:
            window_seconds: 评估窗口大小(秒)
        
        返回:
            活动说话者的tracker_id和置信度得分
        """
        pass
    
    def reset(self):
        """重置系统状态"""
        pass
