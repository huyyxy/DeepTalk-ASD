import onnxruntime as ort
import numpy as np
import time


CHUNK = 512                     # Silero VAD 需要 512 samples
SUPPORTED_SAMPLE_RATES = [8000, 16000]

# 定期重置 VAD 状态
MODEL_RESET_STATES_TIME = 5.0

class SileroVAD:
    """Silero VAD ONNX 封装 (支持 8kHz/16kHz 及其整数倍采样率，自动降采样和分块处理)"""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def _validate_input(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """验证并预处理输入音频
        
        Args:
            x: 输入音频数据 (1D 或 2D array)
            sr: 采样率
            
        Returns:
            处理后的音频数据和目标采样率 (16kHz)
        """
        # 确保是 2D: (1, samples)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk: {x.ndim}")
        
        # 自动降采样：如果采样率是 16000 的整数倍
        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]  # 每 step 个样本取一个
            sr = 16000
        
        if sr not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Supported sampling rates: {SUPPORTED_SAMPLE_RATES} (or multiples of 16000)")
        
        return x, sr

    def _prob_chunk(self, chunk_f32: np.ndarray) -> float:
        """计算单个 512 samples 块的语音概率（内部方法）"""
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"Expected {CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64)
        }
        out, self._state = self.session.run(None, ort_inputs)

        self._context = x[:, -self.context_size:]
        return float(out[0][0])

    def prob(self, audio_f32: np.ndarray, sr: int = 16000) -> float:
        """计算语音概率（支持可变长度输入）
        
        Args:
            audio_f32: 输入音频数据 (float32 格式，范围 [-1, 1])
            sr: 采样率（默认 16000，支持 8000/16000 及其整数倍）
            
        Returns:
            语音概率值 (0.0 ~ 1.0)，如果输入包含多个块，返回最大概率值
        """
        # 验证并预处理输入
        x, sr = self._validate_input(audio_f32, sr)
        
        num_samples = x.shape[1]
        
        # 如果样本数不足一个块，进行零填充
        if num_samples < CHUNK:
            pad_size = CHUNK - num_samples
            x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0.0)
            num_samples = CHUNK
        
        # 如果样本数不是 CHUNK 的整数倍，进行零填充
        if num_samples % CHUNK != 0:
            pad_size = CHUNK - (num_samples % CHUNK)
            x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0.0)
        
        # 分块处理
        probs = []
        for i in range(0, x.shape[1], CHUNK):
            chunk = x[0, i:i+CHUNK]
            prob = self._prob_chunk(chunk)
            probs.append(prob)
        
        self.maybe_reset()
        
        # 返回最大概率值（只要有一个块检测到语音就认为有语音）
        return max(probs) if probs else 0.0
