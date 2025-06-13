import os
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
import time
from LR_ASD.ASD import ASD
import cv2
import math
import python_speech_features
from utils.image_helper import save_image_to_tmp


class LRASDSpeakerDetector:
    """Active Speaker Detection using TalkNet model"""
    
    def __init__(self, **kwargs):
        """
        Initialize the TalkNet speaker detector.
        
        Args:
            model_path (str, optional): Path to pretrained TalkNet model weights
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.video_frame_rate = kwargs.get('video_frame_rate', 25)
        self.audio_sample_rate = kwargs.get('audio_sample_rate', 16000)
        # Get device from kwargs or use default
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set model path
        model_path = kwargs.get('model_path', 'LR_ASD/weight/finetuning_TalkSet.model')
        self.model = ASD()
        self.model.loadParameters(model_path)
        self.model.eval()

        # self.model = self._load_model(model_path)
        # self.model.eval()
        # self.lossAV = lossAV().to(self.device)
        
        # Buffers for audio and video data
        self.audio_buffer = deque(maxlen=16000 * 10)  # 10 seconds buffer
        self.video_buffer = defaultdict(list)  # track_id -> list of (frame, timestamp)
        self.last_face_timestamps = {}  # track_id -> last create time
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0 # 这里有个前置假设，最后一个音频块到来时，前面几秒的buffer都存在
        
        # Configuration
        self.audio_window = 0.025  # seconds of audio for each prediction
        self.audio_stride = 0.01  # seconds between predictions
        self.min_face_size = 64  # minimum face size in pixels
        self.max_track_age = 5.0  # seconds before considering a track stale
        
    # def _load_model(self, model_path):
    #     """Load the TalkNet model with pretrained weights"""
    #     model = talkNetModel().to(self.device)
    #     if model_path and os.path.exists(model_path):
    #         state_dict = torch.load(model_path, map_location=self.device)
    #         model.load_state_dict(state_dict, strict=False)
    #     return model
    
    def append_video(self, frame_faces, create_time=None):
        """
        Add video frame with detected faces to the buffer.
        
        Args:
            frame_faces: List of dictionaries containing face information
            create_time: Timestamp when the frame was created
        """
        if create_time is None:
            create_time = time.perf_counter()
        self.last_video_timestamp = create_time
            
        for face in frame_faces:
            track_id = face['id']
            face_img = face['image']
                
            # Resize face to expected input size (assuming 224x224 for TalkNet)
            # resized_face_img = cv2.resize(face_img, (224, 224))

            # resized_mouth_img = resized_face_img[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]

            resized_mouth_img = cv2.resize(face_img, (112, 112))
            
            # Normalize and convert to tensor
            # face_tensor = torch.FloatTensor(resized_face_img).permute(2, 0, 1) / 255.0
            # save_image_to_tmp(resized_mouth_img)
            
            self.video_buffer[track_id].append((resized_mouth_img, create_time))
            self.last_face_timestamps[track_id] = create_time
            
        # Clean up old tracks
        self._cleanup_old_tracks()
    
    def append_audio(self, audio_chunk, create_time=None):
        """
        Add audio chunk to the buffer.
        
        Args:
            audio_chunk: PCM audio data (16-bit, 16kHz, mono)
            create_time: Timestamp when the audio was created
        """
        if create_time is None:
            create_time = time.perf_counter()
            
        # Convert to numpy array if needed
        if isinstance(audio_chunk, (bytes, bytearray)):
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
        
        self.audio_buffer.extend(audio_chunk)
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        current_time = time.perf_counter()
        stale_tracks = [
            track_id for track_id, last_time in self.last_face_timestamps.items()
            if current_time - last_time > self.max_track_age
        ]
        for track_id in stale_tracks:
            if track_id in self.video_buffer:
                del self.video_buffer[track_id]
            if track_id in self.last_face_timestamps:
                del self.last_face_timestamps[track_id]
    
    def _preprocess_audio(self):
        """Convert raw audio to MFCC features"""
        # 这里输入的是int16的nparray
        # Convert to numpy array if needed
        audio_data = list(self.audio_buffer)
        audio_data = np.array(audio_data, dtype=np.int16)
        
        # Normalize demoTalkNet没有做这一步，但一般算法建议做，后续可以比较一下效果差异
        # audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC features
        mfcc = python_speech_features.mfcc(
            audio_data,
            samplerate=self.audio_sample_rate,
            winlen=self.audio_window,
            winstep=self.audio_stride,
            numcep=13,
            # nfilt=26,
            # nfft=512,
            # preemph=0.97,
            # appendEnergy=True
        )
        
        # # Add delta and delta-delta features
        # delta = python_speech_features.delta(mfcc, 2)
        # delta_delta = python_speech_features.delta(delta, 2)
        # features = np.concatenate((mfcc, delta, delta_delta), axis=1)
        
        # # Convert to tensor and add batch dimension
        # features = torch.FloatTensor(features.T).unsqueeze(0).to(self.device)
        return mfcc
    
    def _preprocess_video(self, face_frames):
        """Convert list of face frames to model input"""
        # # input is [(mouth_img, create_time), (mouth_img, create_time)]
        # # if not face_frames:
        # #     return None
        
        # data_array = np.array(face_frames)
        # videoFeature = data_array[:, 0]
            
        # # # Get the most recent frame
        # # face_tensor, _ = face_frames[-1]
        
        # # # Add batch and sequence dimensions
        # # face_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        # return videoFeature
        # 提取口部图像部分
        mouth_imgs = [frame[0] for frame in face_frames]

        # 根据提取的口部图像创建 NumPy 数组
        videoFeature = np.array(mouth_imgs)
        
        return videoFeature
    
    def evaluate(self):
        """
        Evaluate the current state and return active speaker predictions.
        具体流程：
        1. 先预先处理self.audio_buffer, self.video_buffer，从尾部看哪个时间更早，使用这个时间
        往前回溯6秒，即拿到100*6个audio的帧和25*6个video的帧（假设python_speech_features.mfcc的winstep=0.01，视频帧率是25）
        2. 通过MFCC获得声音的特征，直接获得人嘴画面特征
        3. 根据durationSet循环多次使用ASD模型推理score，正数在说话，负数没有在说话
        4. 找出最大的人脸score

        
        Returns:
            dict: Mapping of track_id to (confidence, timestamp) for active speakers
        """
        current_time = time.perf_counter()
        results = {}
        allScores = {}
        
        # Skip if no audio or video data
        if not self.audio_buffer or not self.video_buffer:
            return results
        
        audio_features = self._preprocess_audio()
        
        durationSet = {1,1,1,2,2,2,3,3,4,5,6}
        # Process each track
        for track_id, face_frames in list(self.video_buffer.items()):
            if not face_frames:
                continue

            video_features = self._preprocess_video(face_frames)

            # 每秒有多少个audio_feature
            audio_feature_rate = int(1.0 / self.audio_stride)
            length = min((audio_features.shape[0] - audio_features.shape[0] % 4) / audio_feature_rate, video_features.shape[0] / self.video_frame_rate)
            print(f"length ======> {length}")
            audio_features = audio_features[:int(round(length * audio_feature_rate)),:]
            video_features = video_features[:int(round(length * self.video_frame_rate)),:,:]


            allScore = [] # Evaluation use TalkNet
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(audio_features[i * duration * audio_feature_rate:(i+1) * duration * audio_feature_rate,:]).unsqueeze(0).to(self.device)
                        inputV = torch.FloatTensor(video_features[i * duration * self.video_frame_rate: (i+1) * duration * self.video_frame_rate,:,:]).unsqueeze(0).to(self.device)
                        embedA = self.model.model.forward_audio_frontend(inputA)
                        embedV = self.model.model.forward_visual_frontend(inputV)
                        out = self.model.model.forward_audio_visual_backend(embedA, embedV)
                        
                        score = self.model.lossAV.forward(out, labels = None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
            # s = allScore[max(fidx - 2, 0): min(fidx + 3, len(allScore) - 1)] # average smoothing
            s = np.mean(allScore)
            allScores[track_id] = s
        print(f"allScores ======> {allScores}")
        return allScores
    
    def reset(self):
        """Reset the detector state"""
        self.audio_buffer.clear()
        self.video_buffer.clear()
        self.last_face_timestamps.clear()
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0

