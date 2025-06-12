import numpy as np
import cv2
import torch
import torch.nn as nn
import python_speech_features
from collections import defaultdict, deque
import time
from LR_ASD.ASD import ASD


class LRASDSpeakerDetector:
    """活动说话者检测核心类"""
    def __init__(self, **kwargs):
        """
        初始化活动说话者检测系统
        
        参数:
            video_fps: 视频帧率 (默认25)
            audio_sample_rate: 音频采样率 (默认16000)
            device: 计算设备(cpu或cuda) (默认'cuda')
            model_path: 预训练模型路径
            min_track_length: 有效跟踪的最小帧数 (默认10)
            smooth_window: 分数平滑窗口大小 (默认5)
        """
        self.video_fps = kwargs.get('video_fps', 30)
        self.audio_sample_rate = kwargs.get('audio_sample_rate', 16000)
        self.device = kwargs.get('device', 'cuda')
        self.min_track_length = kwargs.get('min_track_length', 10)
        self.smooth_window = kwargs.get('smooth_window', 5)
        
        # 初始化模型
        self.model = ASD()
        model_path = kwargs.get('model_path')
        print(f"\r\n\r\nlr asd model path ======> {model_path}\r\n\r\n")
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # 数据存储
        self.reset()
        
        # 时间追踪
        self.start_time = time.time()
        self.last_video_time = 0
        self.last_audio_time = 0
    
    def append_video(self, frame_faces, create_time):
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
        current_time = time.time() - self.start_time
        
        # 处理每个人脸
        for face in frame_faces:
            tracker_id = face['id']
            face_image = face['image']
            
            # 预处理人脸图像
            if face_image.ndim == 3:
                face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_image
            face_resized = cv2.resize(face_gray, (112, 112))
            
            # 存储到对应的tracker
            if tracker_id not in self.tracks:
                self.tracks[tracker_id] = {
                    'faces': deque(maxlen=100),
                    'timestamps': deque(maxlen=100),
                    'scores': deque(maxlen=100),
                    'last_score': 0
                }
            
            self.tracks[tracker_id]['faces'].append(face_resized)
            self.tracks[tracker_id]['timestamps'].append(create_time)
        
        self.last_video_time = create_time
    
    def append_audio(self, audio_chunk, create_time):
        """添加音频块到处理队列"""
        current_time = time.time() - self.start_time
        
        # 将音频块添加到缓冲区
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk)) 
        self.audio_timestamps.append(create_time)
        
        # 更新音频持续时间
        self.audio_duration = len(self.audio_buffer) / self.audio_sample_rate
        # print(f"self.audio_duration ss = {self.audio_duration}")
        self.last_audio_time = create_time
    
    def _extract_features(self, audio_segment, face_segment):
        """提取音频和视频特征"""
        # 提取MFCC特征
        mfcc = python_speech_features.mfcc(
            audio_segment, 
            self.audio_sample_rate,            
            winlen=0.025,
            # winstep=0.01,
            winstep=0.00825,
            numcep=13
        )
        
        # 标准化MFCC特征
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # 处理视频帧
        video_frames = np.array(face_segment)
        video_frames = (video_frames - 127.5) / 127.5  # 归一化到[-1, 1]
        
        return mfcc, video_frames
    
    def _evaluate_segment(self, tracker_id, audio_segment, face_segment):
        """评估单个音频-视频片段"""
        # 提取特征
        mfcc, video_frames = self._extract_features(audio_segment, face_segment)
        
        # 转换为张量
        audio_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        video_tensor = torch.FloatTensor(video_frames).unsqueeze(0).to(self.device)
        
        # 调试打印
        # print(f"Audio input shape: {audio_tensor.shape}")  # 应为 [1, T, 13]
        # print(f"Video input shape: {video_tensor.shape}")  # 应为 [1, F, 112, 112]
        
        with torch.no_grad():
            # 通过内部模型调用
            embedA = self.model.model.forward_audio_frontend(audio_tensor)
            embedV = self.model.model.forward_visual_frontend(video_tensor)
            # print(f"embedA shape: {embedA.shape}")
            # print(f"embedV shape: {embedV.shape}")
            out = self.model.model.forward_audio_visual_backend(embedA, embedV)
            print(f"out ======> {out}")
            # score = self.model.lossAV.forward(out, labels = None)
            # print(f"score ======> {score}")
            # score = np.round((np.mean(np.array(score), axis = 0)), 1).astype(float)
            # print(f"====== after self.model.model.forward_audio_visual_backend({out.shape}) ======")
            # score = torch.sigmoid(out).item()

            # 应用 sigmoid 激活函数
            probabilities = torch.sigmoid(out)
        
            # 计算平均得分作为最终结果
            score = probabilities.mean().item()
        
        return score
    
    def evaluate(self, window_seconds=1.0):
        """
        评估当前活动说话者
        
        参数:
            window_seconds: 评估窗口大小(秒)
        
        返回:
            活动说话者的tracker_id和置信度得分
        """
        # print(f"\r\n\r\n\r\n====== LR ASD evaluate ======\r\n\r\n\r\n")
        current_time = time.time() - self.start_time
        results = {}
        
        # 检查是否有足够的音频数据
        if self.audio_duration < window_seconds:
            # print(f"LR ASD = {self.audio_duration}")
            return results
        
        # 计算音频段的起止时间
        audio_start_time = max(0, current_time - window_seconds)
        audio_start_index = int(audio_start_time * self.audio_sample_rate)
        audio_end_index = int(current_time * self.audio_sample_rate)
        audio_segment = self.audio_buffer[audio_start_index:audio_end_index]
        
        # 如果音频段不足，填充静音
        required_length = int(window_seconds * self.audio_sample_rate)
        if len(audio_segment) < required_length:
            padding = np.zeros(required_length - len(audio_segment), dtype=np.int16)
            audio_segment = np.concatenate((audio_segment, padding))
        
        # 评估每个tracker
        for tracker_id, track_data in self.tracks.items():
            # 跳过过短的track
            if len(track_data['faces']) < self.min_track_length:
                continue
            
            # 获取时间窗口内的视频帧
            face_segment = []
            for i in range(len(track_data['timestamps'])):
                if audio_start_time <= track_data['timestamps'][i] <= current_time:
                    face_segment.append(track_data['faces'][i])
            
            # 如果视频帧不足，使用最近的帧填充
            required_frames = int(window_seconds * self.video_fps)
            if len(face_segment) < required_frames:
                last_frame = face_segment[-1] if face_segment else np.zeros((112, 112))
                face_segment += [last_frame] * (required_frames - len(face_segment))
            
            # 如果视频帧过多，均匀采样
            if len(face_segment) > required_frames:
                step = len(face_segment) / required_frames
                indices = [int(i * step) for i in range(required_frames)]
                face_segment = [face_segment[i] for i in indices]
            
            # 评估该tracker
            score = self._evaluate_segment(tracker_id, audio_segment, face_segment)
            
            # 平滑分数
            # track_data['scores'].append(score)
            # if len(track_data['scores']) > 1:
            #     smoothed_score = sum(track_data['scores']) / len(track_data['scores'])
            #     # 应用指数平滑
            #     track_data['last_score'] = 0.7 * track_data['last_score'] + 0.3 * smoothed_score
            # else:
            #     track_data['last_score'] = score
            
            # results[tracker_id] = track_data['last_score']
            results[tracker_id] = score
        print(f"LR ASD evaluat = {results}")
        self.reset()
        return results
    
    def reset(self):
        """重置系统状态"""
        self.tracks = {}  # 存储每个tracker的数据
        self.audio_buffer = np.array([], dtype=np.int16)  # 音频缓冲区
        self.audio_timestamps = deque(maxlen=1000)  # 音频时间戳
        self.audio_duration = 0  # 当前音频长度(秒)
        self.start_time = time.time()
        self.last_video_time = 0
        self.last_audio_time = 0
