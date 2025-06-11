from asd_interface import ASDInterface
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import python_speech_features
from TalkNet_ASD.talkNet import talkNet

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

class TalkNetSpeakerDetector(ASDInterface):
    """活动说话者检测核心类"""
    def __init__(self, video_fps=25, audio_sample_rate=16000, device='cuda', model_path=''):
        """
        初始化活动说话者检测系统
        
        参数:
            video_fps: 视频帧率
            audio_sample_rate: 音频采样率
            device: 计算设备(cpu或cuda)
        """
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.device = device
        
        # 初始化活动说话者检测模型
        self.model = talkNet().to(device)
        self._load_model(model_path)
        
        # 初始化队列
        self.face_detections = []  # 存储每帧的人脸检测结果
        self.audio_chunks = []     # 存储音频块
        self.face_sequences = {}   # 存储人脸序列 {tracker_id: {'frames': [], 'bboxes': [], 'images': []}}
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}, using untrained model")
    
    def append_video(self, frame_faces):
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
        self.face_detections.append(frame_faces)
        
        # 更新人脸序列
        for face in frame_faces:
            tracker_id = face['id']
            bbox = face['bbox']
            face_img = face['image']
            
            # 创建或更新该tracker_id的序列
            if tracker_id not in self.face_sequences:
                self.face_sequences[tracker_id] = {
                    'frames': [], 
                    'bboxes': [], 
                    'images': [],
                    'last_seen': len(self.face_detections) - 1
                }
            
            # 添加到序列
            self.face_sequences[tracker_id]['frames'].append(len(self.face_detections) - 1)
            self.face_sequences[tracker_id]['bboxes'].append(bbox)
            self.face_sequences[tracker_id]['images'].append(face_img)
            self.face_sequences[tracker_id]['last_seen'] = len(self.face_detections) - 1
    
    def append_audio(self, audio_chunk):
        """添加音频块到处理队列"""
        self.audio_chunks.append(audio_chunk)
    
    def _cleanup_old_sequences(self, max_missing_frames=15):
        """清理长时间未出现的人脸序列"""
        current_frame = len(self.face_detections) - 1
        ids_to_remove = []
        
        for tracker_id, seq in self.face_sequences.items():
            if current_frame - seq['last_seen'] > max_missing_frames:
                ids_to_remove.append(tracker_id)
        
        for tracker_id in ids_to_remove:
            del self.face_sequences[tracker_id]
    
    def _process_audio(self):
        """处理音频数据"""
        if not self.audio_chunks:
            return None
        
        # 合并所有音频块
        audio_data = np.concatenate(self.audio_chunks)
        
        # 计算MFCC特征
        mfcc = python_speech_features.mfcc(audio_data, self.audio_sample_rate)
        return mfcc
    
    def evaluate(self, window_seconds=1.0):
        """
        评估当前活动说话者
        
        参数:
            window_seconds: 评估窗口大小(秒)
        
        返回:
            活动说话者的tracker_id和置信度得分
        """
        # 清理旧序列
        self._cleanup_old_sequences()
        
        # 处理音频
        audio_features = self._process_audio()
        if audio_features is None:
            return None, 0.0
        
        # 计算评估窗口大小
        video_window_frames = int(window_seconds * self.video_fps)
        
        # 获取当前可用的序列
        active_sequences = []
        for tracker_id, seq in self.face_sequences.items():
            if len(seq['images']) >= video_window_frames:
                # 获取最近的视频窗口
                video_seq = seq['images'][-video_window_frames:]
                # 转换为模型输入
                video_input = np.stack(video_seq)
                video_input = torch.FloatTensor(video_input).unsqueeze(0).unsqueeze(0).to(self.device)
                active_sequences.append((tracker_id, video_input))
        
        if not active_sequences:
            return None, 0.0
        
        # 准备音频输入
        audio_input = audio_features[-int(window_seconds * 100):]  # 100帧/秒
        audio_input = torch.FloatTensor(audio_input).unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        # 评估每个序列
        best_score = -1
        best_tracker = None
        
        for tracker_id, video_input in active_sequences:
            # 模型前向传播
            audio_feat = self.model.forward_audio_frontend(audio_input)
            video_feat = self.model.forward_visual_frontend(video_input)
            audio_feat, video_feat = self.model.forward_cross_attention(audio_feat, video_feat)
            out = self.model.forward_audio_visual_backend(audio_feat, video_feat)
            
            # 计算说话概率
            score = F.softmax(out, dim=1)[0, 1].item()
            
            if score > best_score:
                best_score = score
                best_tracker = tracker_id
        
        return best_tracker, best_score
    
    def visualize(self, frame, active_tracker_id=None):
        """可视化当前帧和检测结果"""
        display_frame = frame.copy()
        
        # 绘制当前帧的所有人脸边界框
        if self.face_detections and len(self.face_detections) > 0:
            current_frame_faces = self.face_detections[-1]
            for face in current_frame_faces:
                tracker_id = face['id']
                x1, y1, x2, y2 = map(int, face['bbox'])
                
                # 活动说话者用绿色框，其他用红色框
                color = (0, 255, 0) if tracker_id == active_tracker_id else (0, 0, 255)
                thickness = 3 if tracker_id == active_tracker_id else 2
                
                # 绘制边界框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 显示tracker_id
                cv2.putText(display_frame, f"ID: {tracker_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 如果是活动说话者，添加额外标签
                if tracker_id == active_tracker_id:
                    cv2.putText(display_frame, "SPEAKING", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def reset(self):
        """重置系统状态"""
        self.face_detections = []
        self.audio_chunks = []
        self.face_sequences = {}
