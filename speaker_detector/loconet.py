import os
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
import time
from loconet_asd.loconet import loconet
from loconet_asd.configs.defaults import get_cfg_defaults
import cv2
import math
import python_speech_features
from utils.image_helper import save_image_to_tmp
import yaml
from omegaconf import OmegaConf


class LoCoNetSpeakerDetector:
    """Active Speaker Detection using LoCoNet ASD model"""
    
    def __init__(self, **kwargs):
        """
        Initialize the LoCoNet speaker detector.
        
        Args:
            model_path (str, optional): Path to pretrained LoCoNet model weights
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.video_frame_rate = kwargs.get('video_frame_rate', 25)
        self.audio_sample_rate = kwargs.get('audio_sample_rate', 16000)
        # Get device from kwargs or use default
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set model path and load LoCoNet model
        model_path = kwargs.get('model_path', 'loconet_asd/loconet_ava_best.model')
        config_path = kwargs.get('config_path', 'loconet_asd/configs/ava_small_v1.0_1gpu_e2e.yaml')
        
        # Load config
        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_path)
        cfg.freeze()
        self.cfg = cfg
        
        # Initialize and load model
        self.model = loconet(cfg, rank=None)
        if os.path.exists(model_path):
            self.model.loadParameters(model_path)
        else:
            print(f"Warning: Model weights not found at {model_path}")
        self.model.eval()
        self.model = self.model.to(self.device)
        
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
        """Convert raw audio to MFCC features for LoCoNet"""
        # Convert to numpy array if needed
        audio_data = list(self.audio_buffer)
        audio_data = np.array(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC features
        mfcc = python_speech_features.mfcc(
            audio_data,
            samplerate=self.audio_sample_rate,
            winlen=self.audio_window,
            winstep=self.audio_stride,
            numcep=13,
            nfilt=26,
            nfft=512,
            preemph=0.97,
            appendEnergy=True
        )
        
        # Add delta and delta-delta features
        delta = python_speech_features.delta(mfcc, 2)
        delta_delta = python_speech_features.delta(delta, 2)
        features = np.concatenate((mfcc, delta, delta_delta), axis=1)
        
        # Convert to tensor and add batch dimension
        features = torch.FloatTensor(features).unsqueeze(0)  # [1, T, 39]
        return features
    
    def _preprocess_video(self, face_frames):
        """Convert list of face frames to model input for LoCoNet"""
        # Extract mouth images and convert to tensor
        mouth_imgs = [frame[0] for frame in face_frames]  # List of [H, W, 3] images
        
        # Convert to numpy array and normalize to [0, 1]
        video_features = np.array(mouth_imgs, dtype=np.float32) / 255.0  # [T, H, W, 3]
        
        # Convert to tensor and permute to [T, C, H, W]
        video_tensor = torch.FloatTensor(video_features).permute(0, 3, 1, 2)  # [T, 3, H, W]
        
        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)  # [1, T, 3, H, W]
        
        return video_tensor
    
    def evaluate(self):
        """
        Evaluate the current state and return active speaker predictions using LoCoNet.
        
        Returns:
            dict: Mapping of track_id to confidence score (higher means more likely to be speaking)
        """
        current_time = time.perf_counter()
        results = {}
        
        # Skip if no audio or video data
        if not self.audio_buffer or not self.video_buffer:
            return results
        
        # Preprocess audio and video features
        audio_features = self._preprocess_audio()
        
        # Process each track
        for track_id, face_frames in list(self.video_buffer.items()):
            if not face_frames:
                continue
                
            # Preprocess video features for this track
            video_features = self._preprocess_video(face_frames)
            
            # Move tensors to device
            audio_tensor = audio_features.to(self.device)
            video_tensor = video_features.to(self.device)
            
            # Get sequence lengths
            audio_len = audio_tensor.size(1)  # [1, T, 39]
            video_len = video_tensor.size(1)  # [1, T, 3, H, W]
            
            # Pad sequences if needed to match lengths
            max_len = max(audio_len, video_len)
            if audio_len < max_len:
                padding = torch.zeros(1, max_len - audio_len, 39, device=self.device)
                audio_tensor = torch.cat([audio_tensor, padding], dim=1)
            elif video_len < max_len:
                padding = torch.zeros(1, max_len - video_len, 3, 112, 112, device=self.device)
                video_tensor = torch.cat([video_tensor, padding], dim=1)
            
            # Add batch dimension for number of speakers (1 in this case)
            audio_tensor = audio_tensor.unsqueeze(1)  # [1, 1, T, 39]
            video_tensor = video_tensor.unsqueeze(1)  # [1, 1, T, 3, 112, 112]
            
            # Create dummy labels and masks (not used in inference)
            b, s, t = 1, 1, max_len
            labels = torch.zeros((b, s, t), device=self.device).long()
            masks = torch.ones((b, s, t), device=self.device).float()
            
            with torch.no_grad():
                # Forward pass through LoCoNet
                audio_embed = self.model.model.forward_audio_frontend(audio_tensor.squeeze(1))  # [1, T, C]
                visual_embed = self.model.model.forward_visual_frontend(video_tensor.view(-1, *video_tensor.shape[3:]))  # [T, C, H, W]
                
                # Reshape visual features
                visual_embed = visual_embed.view(b, s, t, -1).permute(0, 1, 3, 2)  # [1, 1, C, T]
                audio_embed = audio_embed.permute(0, 2, 1)  # [1, C, T]
                
                # Cross attention
                audio_embed = audio_embed.repeat(s, 1, 1)  # [s, C, T]
                visual_embed = visual_embed.squeeze(0).permute(0, 2, 1)  # [s, T, C]
                
                audio_embed, visual_embed = self.model.model.forward_cross_attention(
                    audio_embed, visual_embed
                )
                
                # Get final predictions
                outsAV = self.model.model.forward_audio_visual_backend(audio_embed, visual_embed, b, s)
                
                # Get speaking scores (convert to probability with sigmoid)
                scores = torch.sigmoid(outsAV).mean().item()
                
                # Store result
                results[track_id] = scores
        
        return results
    
    def reset(self):
        """Reset the detector state"""
        self.audio_buffer.clear()
        self.video_buffer.clear()
        self.last_face_timestamps.clear()
        self.last_video_timestamp = 0.0
        self.last_audio_timestamp = 0.0

