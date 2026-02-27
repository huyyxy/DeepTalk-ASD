


class SpeakerDetectorInterface:
    """活动说话者检测核心类"""
    
    def append_video(self, frame_faces, create_time = None):
        """
        添加视频帧中已检测的人脸信息，视频帧率缺省为25

        参数:
            frame_faces: 当前帧的人脸检测结果列表，每个元素为字典:
                [
                    {
                        'id': tracker_id1,   # 人脸的tracker id
                        'image': face_gray1         # 裁剪后的人脸图像，灰度图
                    },
                    {
                        'id': tracker_id2,   # 人脸的tracker id
                        'image': face_gray2         # 裁剪后的人脸图像，灰度图
                    },
                    ... # 可能多个人脸
                ]
            create_time: 此块audio_chunk的创建时间，调用方一般使用time.perf_counter()获得，用于和音频块时间对齐
        """
        pass
    
    def append_audio(self, audio_chunk, create_time = None):
        """
        添加音频块到处理队列
        
        参数:
            audio_chunk: pcm_data，一般为16bit，单声道，16000Hz，30ms时长音频块
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
    
    def reset(self):
        """重置系统状态"""
        pass
