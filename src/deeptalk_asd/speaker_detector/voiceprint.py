"""
文件名: voiceprint.py
描述:
    Speaker Embedding 提取器组件，基于 sherpa_onnx 封装。
    用于从传入音频序列中提取单个说话人特征，并进行特征相似度计算。
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

from ..deeptalk_logger import DeepTalkLogger
logger = DeepTalkLogger(__name__)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    计算两个声纹向量的余弦相似度
    
    Args:
        embedding1: 第一个声纹向量
        embedding2: 第二个声纹向量
        
    Returns:
        余弦相似度 (范围 -1 到 1，越接近 1 表示越相似)
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class SpeakerEmbeddingExtractor:
    """声纹特征提取器封装类"""

    def __init__(
        self,
        model_dir: str,
        model_name: str,
        num_threads: int = 2,
        provider: str = "cpu",
        debug: bool = False
    ):
        """
        初始化声纹提取器
        
        Args:
            model_dir: 模型文件所在目录
            model_name: 模型文件名
            num_threads: 计算线程数
            provider: 计算后端 ("cpu", "cuda", "coreml")
            debug: 是否开启调试模式
        """
        if sherpa_onnx is None:
            raise ImportError(
                "sherpa-onnx library is not installed. "
                "Please install it via: pip install sherpa-onnx"
            )

        model_path = str(Path(model_dir) / model_name)
        self.model_path = model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
        )
        
        if not config.validate():
            raise ValueError(f"无效的配置: {config}")
        
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self.extractor.dim
        
        logger.info(f"声纹提取器初始化成功, 模型: {model_path}, 维度: {self.dim}")

    def extract_from_samples(
        self, 
        samples: np.ndarray, 
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        从音频样本提取声纹向量
        
        Args:
            samples: 音频样本数组 (float32, 单声道)
            sample_rate: 采样率 (默认 16kHz)
            
        Returns:
            声纹向量 (numpy array) 或 None
        """
        if len(samples) == 0:
            return None

        # 确保数据类型为 float32
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        # 创建流
        stream = self.extractor.create_stream()
        
        # 输入音频数据
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        
        # 检查是否就绪
        if not self.extractor.is_ready(stream):
            logger.debug("音频数据不足，无法提取稳定声纹")
            return None
        
        # 计算声纹向量
        embedding = self.extractor.compute(stream)
        return np.array(embedding)
