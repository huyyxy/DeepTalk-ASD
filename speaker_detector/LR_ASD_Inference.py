from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import sys
from pathlib import Path
import numpy as np
from LR_ASD.loss import lossAV, lossV  # 导入自定义的损失函数
from LR_ASD.model.Model import ASD_Model  # 导入ASD模型结构


class ASDInferenceForONNX(nn.Module):
    """
    仅包含推理计算的子图，用于导出 ONNX（不含 .numpy() 等不可导/设备相关操作）。
    输入: audioFeature [B, A, T], visualFeature [B, T, H, W]
    输出: score [B*T] 或 [N]，表示每帧的说话人分数（正类 logit 或概率）。
    """
    def __init__(self, asd_model: ASD_Model, loss_av: lossAV):
        super().__init__()
        self.model = asd_model
        self.lossAV = loss_av

    def forward(self, audioFeature: torch.Tensor, visualFeature: torch.Tensor) -> torch.Tensor:
        audioEmbed = self.model.forward_audio_frontend(audioFeature)
        visualEmbed = self.model.forward_visual_frontend(visualFeature)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        # 推理分支：只做 FC + 取正类分数，保持为 tensor
        x = outsAV.squeeze(1)
        x = self.lossAV.FC(x)
        score = x[:, 1]
        score = score.view(-1)
        return score


class ASDInference(nn.Module):
    """
    音频-视觉说话人检测(ASD)推理类
    用于加载训练好的模型并进行推理预测
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        初始化ASD推理模型
        
        Args:
            **kwargs: 可接受的关键字参数
                - device (str, optional): 指定运行设备('cuda'或'cpu'), 默认为'cuda'(如果可用)
        """
        super(ASDInference, self).__init__()
        # 设置运行设备(优先使用GPU如果可用)
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化ASD模型并移动到指定设备
        self.model: ASD_Model = ASD_Model().to(self.device)
        # 初始化音频-视觉损失函数
        self.lossAV: lossAV = lossAV().to(self.device)
        # 初始化视觉损失函数
        self.lossV: lossV = lossV().to(self.device)

    def inference(self, audioFeature: torch.Tensor, visualFeature: torch.Tensor) -> np.ndarray:
        """
        执行推理过程
        
        Args:
            audioFeature (torch.Tensor): 音频特征张量，形状为 [batch_size, audio_feature_dim, time_steps]
            visualFeature (torch.Tensor): 视觉特征张量，形状为 [batch_size, visual_feature_dim, time_steps]
            
        Returns:
            np.ndarray: 预测分数numpy数组，形状为 [batch_size,]，表示每个样本是目标说话人的概率
        """
        # 前向传播音频特征
        audioEmbed: torch.Tensor = self.model.forward_audio_frontend(audioFeature.to(self.device))
        # 前向传播视觉特征
        visualEmbed: torch.Tensor = self.model.forward_visual_frontend(visualFeature.to(self.device))
        # 融合音频和视觉特征
        outsAV: torch.Tensor = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
        # 计算预测分数(不需要标签，因为这是推理阶段)
        score = self.lossAV.forward(outsAV, labels=None)
        # 提取正类(说话人)的概率分数，并转换为numpy数组
        # predScore: np.ndarray = predScore[:, 1].detach().cpu().numpy()
        return score

    def loadParameters(self, path: Union[str, Path]) -> None:
        """
        加载预训练模型参数
        
        Args:
            path (Union[str, Path]): 预训练模型参数文件路径
            
        Note:
            1. 会自动处理模型参数名称中的'module.'前缀
            2. 会跳过不匹配的参数
            3. 加载完成后会将模型设置为评估模式
            
        Raises:
            FileNotFoundError: 当指定的模型文件不存在时抛出
            RuntimeError: 当加载模型参数出错时抛出
        """
        # 检查文件是否存在
        if not Path(path).exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
            
        try:
            # 获取当前模型的状态字典
            selfState: Dict[str, torch.Tensor] = self.state_dict()
            # 加载预训练参数
            loadedState: Dict[str, torch.Tensor] = torch.load(path, map_location=self.device)
            
            # 遍历加载的参数
            for name, param in loadedState.items():
                origName: str = name  # 保存原始参数名
                # 如果参数名不在当前模型中，尝试移除'module.'前缀
                if name not in selfState:
                    name = name.replace("module.", "")
                    if name not in selfState:
                        print(f"警告: 参数 {origName} 不在模型中，已跳过。")
                        continue
                # 检查参数形状是否匹配
                if selfState[name].size() != loadedState[origName].size():
                    sys.stderr.write(
                        f"参数维度不匹配: {origName}, "
                        f"模型维度: {selfState[name].size()}, "
                        f"加载维度: {loadedState[origName].size()}\n"
                    )
                    continue
                # 复制参数值
                selfState[name].copy_(param)
                
            # 将模型设置为评估模式
            self.eval()
            
        except Exception as e:
            raise RuntimeError(f"加载模型参数时出错: {str(e)}") from e

    def export_onnx(
        self,
        path: Union[str, Path],
        *,
        audio_time_steps: int = 100,
        video_time_steps: int = 25,
        audio_ceps: int = 13,
        visual_h: int = 112,
        visual_w: int = 112,
        opset_version: int = 14,
        dynamic: bool = True,
    ) -> None:
        """
        将当前模型导出为 ONNX，便于用 onnxruntime 等做推理。

        Args:
            path: 输出 .onnx 文件路径
            audio_time_steps: 导出时音频时间步（动态时可为任意）
            video_time_steps: 导出时视频帧数（动态时可为任意）
            audio_ceps: MFCC 维度，默认 13
            visual_h, visual_w: 口部图像高宽，默认 112
            opset_version: ONNX opset 版本
            dynamic: 是否对时间维使用动态轴（推荐 True）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        wrapper = ASDInferenceForONNX(self.model, self.lossAV)
        wrapper.eval()

        B = 1
        dummy_audio = torch.randn(B, audio_ceps, audio_time_steps, device=self.device)
        dummy_visual = torch.randn(B, video_time_steps, visual_h, visual_w, device=self.device)

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "audioFeature": {0: "batch", 2: "audio_time"},
                "visualFeature": {0: "batch", 1: "video_time"},
                "score": {0: "num_frames"},
            }

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_audio, dummy_visual),
                str(path),
                input_names=["audioFeature", "visualFeature"],
                output_names=["score"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
            )
