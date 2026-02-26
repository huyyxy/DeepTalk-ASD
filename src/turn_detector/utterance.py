

from typing import Iterable, List, Union
from ..audio_frame import AudioFrame
import numpy as np
from enum import IntEnum, auto, unique
from dataclasses import dataclass


@unique
class TurnState(IntEnum):
    """Turn 检测状态"""
    IDLE = auto()            # 空闲，等待语音
    TURN_START = auto()      # 确定 Turn 开始
    TURN_CONTINUE = auto()   # Turn 继续
    TURN_CONFIRMED = auto()  # Turn 已确认为真实语音段
    TURN_SILENCE = auto()    # Turn 中检测到静音
    TURN_END = auto()        # 正常结束（已确认的语音段结束）
    TURN_REJECTED = auto()   # 拒绝结束（未通过验证，伪语音段）

@dataclass
class Utterance:
    """
    表示一个语音段，由多个 AudioFrame 组成。
    通常用于 VAD 检测到的一个完整的语音片段。
    """
    face_id: int
    turn_state: TurnState
    frames: List[AudioFrame]
    
    @property
    def data(self) -> np.ndarray:
        """
        获取音频数据。
        """
        return np.concatenate([frame.data for frame in self.frames])

    @property
    def sample_rate(self) -> int:
        """
        获取音频采样率。
        """
        return self.frames[0].sample_rate

    @property
    def num_channels(self) -> int:
        """
        获取音频声道数。
        """
        return self.frames[0].num_channels

    @property
    def samples_per_channel(self) -> int:
        """
        获取每个声道的样本数。
        """
        return self.frames[0].samples_per_channel

    def get_loudness_percentile_95(self) -> float:
        """
        计算该语音段的95分位音量。
        注意：
        >>> import numpy as np
        >>> np.abs(np.int16(-32768))
        -32768   # 静默返回错误结果，没有任何警告！
        所以结果有小概率略微偏小，但影响不大。
        """
        y = np.ctypeslib.as_array(self.data)

        # 在 int16 上计算绝对值的百分位数（只分配一个 int16 数组）
        percentile_95 = np.percentile(np.abs(y), 95)
        
        # 只对标量结果进行归一化
        return float(percentile_95) / 32768.0
