from abc import ABC, abstractmethod
from enum import Enum, unique
from ..audio_frame import AudioFrame


@unique
class AudioResamplerQuality(str, Enum):
    QUICK = "quick"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AudioResampler(ABC):
    """
    音频重采样器的抽象基类。

    `AudioResampler` 定义了将音频数据从一个采样率重采样到另一个采样率的接口。
    它支持多声道、可配置的重采样质量以及声道转换。
    """

    def __init__(
        self,
        output_rate: int,
        *,
        input_rate: int | None = None,
        output_num_channels: int = 1,
        input_num_channels: int | None = None,
        quality: AudioResamplerQuality = AudioResamplerQuality.MEDIUM,
        output_frame_duration_ms: int | None = None,
    ) -> None:
        """
        初始化一个用于重采样音频数据的 `AudioResampler` 实例。

        Args:
            output_rate (int): 输出音频数据的期望采样率（单位：Hz）。
            input_rate (int | None, optional): 输入音频数据的采样率（单位：Hz）。
                如果为 None，则在 push 时从 AudioFrame 中获取。当 push bytearray 时必须提供。
                默认为 None。
            output_num_channels (int, optional): 输出音频的声道数（例如，1 为单声道，2 为立体声）。
                默认为 1。
            input_num_channels (int | None, optional): 输入音频的声道数。
                如果为 None，则在 push 时从 AudioFrame 中获取。当 push bytearray 时必须提供。
                默认为 None。
            quality (AudioResamplerQuality, optional): 重采样器的质量设置。可以是 `AudioResamplerQuality` 
                枚举值之一：`QUICK`、`LOW`、`MEDIUM`、`HIGH`、`VERY_HIGH`。更高的质量设置会带来更好的音频质量，
                但需要更多的处理能力。默认为 `AudioResamplerQuality.MEDIUM`。
            output_frame_duration_ms (int | None, optional): 输出 AudioFrame 的时长（单位：毫秒）。
                如果设置，重采样后的音频数据将被切分成指定时长的帧。如果为 None，则每次 push 
                返回的帧时长与输入数据对应的重采样结果一致。默认为 None。

        Raises:
            Exception: 如果创建重采样器时出错。
        """
        self._output_rate = output_rate
        self._input_rate = input_rate
        self._output_num_channels = output_num_channels
        self._input_num_channels = input_num_channels
        self._quality = quality
        self._output_frame_duration_ms = output_frame_duration_ms
        
        # 计算每帧输出的采样数
        if output_frame_duration_ms is not None:
            self._output_samples_per_frame = int(output_rate * output_frame_duration_ms / 1000)
        else:
            self._output_samples_per_frame = None

    @property
    def input_rate(self) -> int | None:
        """返回输入采样率，如果未设置则返回 None。"""
        return self._input_rate

    @property
    def output_rate(self) -> int:
        """返回输出采样率。"""
        return self._output_rate

    @property
    def input_num_channels(self) -> int | None:
        """返回输入声道数，如果未设置则返回 None。"""
        return self._input_num_channels

    @property
    def output_num_channels(self) -> int:
        """返回输出声道数。"""
        return self._output_num_channels

    @property
    def quality(self) -> AudioResamplerQuality:
        """返回重采样质量。"""
        return self._quality

    @property
    def output_frame_duration_ms(self) -> int | None:
        """返回输出帧时长（毫秒），如果未设置则返回 None。"""
        return self._output_frame_duration_ms

    @property
    def output_samples_per_frame(self) -> int | None:
        """返回每帧输出的采样数，如果未设置 output_frame_duration_ms 则返回 None。"""
        return self._output_samples_per_frame

    @abstractmethod
    def push(self, data: bytearray | AudioFrame) -> list[AudioFrame]:
        """
        将音频数据推送到重采样器中，并获取所有可用的重采样数据。

        此方法接受音频数据，根据配置的输入和输出采样率对其进行重采样，
        并返回处理输入后所有可用的重采样数据。

        当输入为 AudioFrame 时，会自动从中获取 sample_rate 和 num_channels。
        当输入为 bytearray 时，必须在初始化时提供 input_rate 和 input_num_channels。

        Args:
            data (bytearray | AudioFrame): 要重采样的音频数据。可以是包含 int16le 格式原始音频字节的
                `bytearray`，也可以是 `AudioFrame` 对象。

        Returns:
            list[AudioFrame]: 包含重采样音频数据的 `AudioFrame` 对象列表。
                如果当前没有可用的输出数据，列表可能为空。

        Raises:
            ValueError: 当输入为 bytearray 但未设置 input_rate 或 input_num_channels 时。
            Exception: 如果在重采样过程中出错。
        """
        pass

    @abstractmethod
    def flush(self) -> list[AudioFrame]:
        """
        刷新重采样器中所有剩余的音频数据，并获取重采样后的数据。

        当不再提供输入数据时，应调用此方法以确保所有内部缓冲区都被处理，
        并且所有重采样数据都已输出。

        Returns:
            list[AudioFrame]: 包含刷新后剩余重采样音频数据的 `AudioFrame` 对象列表。
                如果没有剩余的输出数据，列表可能为空。

        Raises:
            Exception: 如果在刷新过程中出错。
        """
        pass

    @staticmethod
    def _convert_channels(data, input_channels: int, output_channels: int):
        """
        转换音频数据的声道数。

        Args:
            data (np.ndarray): 输入音频数据。对于多声道，形状应为 (samples, channels)。
            input_channels (int): 输入声道数。
            output_channels (int): 输出声道数。

        Returns:
            np.ndarray: 转换后的音频数据。
        """
        import numpy as np
        
        if input_channels == output_channels:
            return data
        
        if input_channels == 1 and output_channels == 2:
            # 单声道转立体声：复制到两个声道
            if data.ndim == 1:
                return np.column_stack([data, data])
            else:
                return np.column_stack([data, data])
        elif input_channels == 2 and output_channels == 1:
            # 立体声转单声道：取两个声道的平均值
            if data.ndim == 1:
                # 数据已经是交错格式，需要重塑
                data = data.reshape(-1, 2)
            return np.mean(data, axis=1).astype(data.dtype)
        elif input_channels > output_channels:
            # 多声道转少声道：取前 output_channels 个声道的平均值
            if data.ndim == 1:
                data = data.reshape(-1, input_channels)
            return np.mean(data[:, :output_channels], axis=1).astype(data.dtype) if output_channels == 1 else data[:, :output_channels]
        else:
            # 少声道转多声道：复制第一个声道到所有输出声道
            if data.ndim == 1:
                if input_channels > 1:
                    data = data.reshape(-1, input_channels)
                else:
                    # 单声道数据，复制到所有输出声道
                    return np.column_stack([data] * output_channels)
            # 复制现有声道并填充
            result = np.zeros((data.shape[0] if data.ndim > 1 else len(data), output_channels), dtype=data.dtype)
            if data.ndim == 1:
                result[:, 0] = data
                for i in range(1, output_channels):
                    result[:, i] = data
            else:
                for i in range(input_channels):
                    result[:, i] = data[:, i]
                for i in range(input_channels, output_channels):
                    result[:, i] = data[:, 0]  # 用第一个声道填充
            return result

