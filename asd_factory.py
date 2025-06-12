from asd_interface import ASDInterface
import torch
import traceback
import sys
import os


class ASDDetectorFactory:
    def __init__(self, type: str, **kwargs):
        """
        使用指定的 类型创建 图像质量 实例。
        
        :param type: 'TalkNet' 等可选的 活跃说话人模型 类型。
        """
        self.type = type
        self.kwargs = kwargs

    def asd_detector(self)-> ASDInterface:
        """
        根据 type 获得图像检测器，。
        
        :return: ASDInterface 实例
        """
        try:
            if self.type == 'TalkNet':
                # 获取当前文件的路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                submodule_path = os.path.join(current_dir, 'TalkNet_ASD')

                # 将子模块路径添加到 sys.path
                sys.path.insert(0, submodule_path)

                from speaker_detector.talknet import TalkNetSpeakerDetector
                video_fps = self.kwargs.get('video_fps', 30)
                audio_sample_rate = self.kwargs.get('audio_sample_rate', 16000)
                device = self.kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                model_path = self.kwargs.get('model_path')
                detector = TalkNetSpeakerDetector(video_fps=video_fps, audio_sample_rate=audio_sample_rate, device=device, model_path=model_path)
                return detector
            elif self.type == 'LoCoNet':
                pass
            elif self.type == 'EASEE-50':
                pass
            elif self.type == 'Light-ASD':
                pass
            elif self.type == 'LR-ASD':
                # 获取当前文件的路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                submodule_path = os.path.join(current_dir, 'LR_ASD')

                # 将子模块路径添加到 sys.path
                sys.path.insert(0, submodule_path)

                from speaker_detector.lrasd import LRASDSpeakerDetector
                video_fps = self.kwargs.get('video_fps', 30)
                audio_sample_rate = self.kwargs.get('audio_sample_rate', 16000)
                device = self.kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                model_path = self.kwargs.get('model_path')
                detector = LRASDSpeakerDetector(video_fps=video_fps, audio_sample_rate=audio_sample_rate, device=device, model_path=model_path)
                return detector
        except Exception as e:
            traceback.print_exc()
        return None

