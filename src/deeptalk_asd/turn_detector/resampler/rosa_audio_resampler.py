from typing import Any
import numpy as np
import librosa

from ..audio_frame import AudioFrame
from .audio_resampler import AudioResampler, AudioResamplerQuality


class RosaAudioResampler(AudioResampler):
    """
    使用 librosa 库实现的音频重采样器。

    librosa 是一个用于音频和音乐分析的 Python 库，提供基于 scipy 的高质量重采样功能。
    注意：librosa 的重采样是非流式的，每次 push 都会处理完整的输入数据。
    支持动态输入采样率和声道数（从 AudioFrame 获取），以及声道转换。

    注意：当输入为 AudioFrame 时，会自动将输入帧的 userdata 复制到输出帧。
    当涉及多帧合并或切分时，使用最后一个输入帧的 userdata。
    """

    # librosa 使用 res_type 参数控制重采样质量
    _QUALITY_MAP = {
        AudioResamplerQuality.QUICK: "linear",      # 线性插值，最快
        AudioResamplerQuality.LOW: "fft",           # FFT 方法
        AudioResamplerQuality.MEDIUM: "soxr_mq",    # SoX 中等质量
        AudioResamplerQuality.HIGH: "soxr_hq",      # SoX 高质量
        AudioResamplerQuality.VERY_HIGH: "soxr_vhq", # SoX 最高质量
    }

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
        初始化 RosaAudioResampler。

        Args:
            output_rate (int): 输出音频数据的期望采样率（单位：Hz）。
            input_rate (int | None, optional): 输入音频数据的采样率（单位：Hz）。
                如果为 None，则在 push 时从 AudioFrame 中获取。当 push bytearray 时必须提供。
                默认为 None。
            output_num_channels (int, optional): 输出音频的声道数。默认为 1。
            input_num_channels (int | None, optional): 输入音频的声道数。
                如果为 None，则在 push 时从 AudioFrame 中获取。当 push bytearray 时必须提供。
                默认为 None。
            quality (AudioResamplerQuality, optional): 重采样质量。默认为 MEDIUM。
            output_frame_duration_ms (int | None, optional): 输出帧时长（毫秒）。默认为 None。
        """
        super().__init__(
            output_rate,
            input_rate=input_rate,
            output_num_channels=output_num_channels,
            input_num_channels=input_num_channels,
            quality=quality,
            output_frame_duration_ms=output_frame_duration_ms,
        )

        self._res_type = self._QUALITY_MAP.get(quality, "soxr_mq")
        self._buffer: bytes = b""
        
        # 记录当前缓冲区的输入参数（用于 flush）
        self._buffer_input_rate: int | None = None
        self._buffer_input_num_channels: int | None = None
        
        # 输出缓冲区，用于按指定帧时长切分输出
        self._output_buffer: np.ndarray | None = None
        
        # 保存最后一个输入帧的 userdata，用于复制到输出帧
        self._last_input_userdata: dict[str, Any] = {}

    def push(self, data: bytearray | AudioFrame) -> list[AudioFrame]:
        """
        将音频数据推送到重采样器中，并获取所有可用的重采样数据。

        注意：librosa 的重采样是批处理模式，不是真正的流式处理。

        Args:
            data (bytearray | AudioFrame): 要重采样的音频数据。

        Returns:
            list[AudioFrame]: 包含重采样音频数据的 `AudioFrame` 对象列表。

        Raises:
            ValueError: 当输入为 bytearray 但未设置 input_rate 或 input_num_channels 时。
        """
        if isinstance(data, AudioFrame):
            input_rate = data.sample_rate
            input_num_channels = data.num_channels
            bdata = bytes(data.data.cast("b"))
            # 保存输入帧的 userdata
            self._last_input_userdata = data.userdata.copy()
        else:
            # bytearray 输入，必须使用预设的 input_rate 和 input_num_channels
            if self._input_rate is None:
                raise ValueError("push bytearray 时必须在初始化时提供 input_rate")
            if self._input_num_channels is None:
                raise ValueError("push bytearray 时必须在初始化时提供 input_num_channels")
            input_rate = self._input_rate
            input_num_channels = self._input_num_channels
            bdata = bytes(data)

        if not bdata:
            return []

        # 如果输入参数变化，先处理现有缓冲区
        if (self._buffer and 
            (self._buffer_input_rate != input_rate or 
             self._buffer_input_num_channels != input_num_channels)):
            # 输入参数变化，先处理旧的缓冲区
            old_frames = self._process_buffer()
            self._buffer = bdata
            self._buffer_input_rate = input_rate
            self._buffer_input_num_channels = input_num_channels
            new_frames = self._process_buffer()
            return old_frames + new_frames

        # 累积数据到缓冲区
        self._buffer += bdata
        self._buffer_input_rate = input_rate
        self._buffer_input_num_channels = input_num_channels

        return self._process_buffer()

    def _process_buffer(self) -> list[AudioFrame]:
        """
        处理当前缓冲区中的数据。

        Returns:
            list[AudioFrame]: 重采样后的 AudioFrame 列表。
        """
        if not self._buffer:
            return []

        input_rate = self._buffer_input_rate
        input_num_channels = self._buffer_input_num_channels
        output_num_channels = self._output_num_channels

        if input_rate is None or input_num_channels is None:
            return []

        # 将字节数据转换为 numpy 数组 (int16)
        input_array = np.frombuffer(self._buffer, dtype=np.int16)

        # 转换为浮点数 [-1.0, 1.0] 范围 (librosa 需要浮点输入)
        input_float = input_array.astype(np.float32) / 32768.0

        # 如果是多声道，需要重塑数组
        if input_num_channels > 1:
            input_float = input_float.reshape(-1, input_num_channels).T  # librosa 期望 (channels, samples)

        # 使用 librosa 进行重采样
        if input_num_channels > 1:
            # 对每个声道分别重采样
            resampled_channels = []
            for ch in range(input_num_channels):
                resampled_ch = librosa.resample(
                    input_float[ch],
                    orig_sr=input_rate,
                    target_sr=self._output_rate,
                    res_type=self._res_type,
                )
                resampled_channels.append(resampled_ch)
            output_float = np.stack(resampled_channels, axis=0)
            output_float = output_float.T  # 转回 (samples, channels)
        else:
            output_float = librosa.resample(
                input_float,
                orig_sr=input_rate,
                target_sr=self._output_rate,
                res_type=self._res_type,
            )

        if output_float.size == 0:
            self._buffer = b""
            return []

        # 转换回 int16
        output_int16 = (output_float * 32768.0).clip(-32768, 32767).astype(np.int16)

        # 清空输入缓冲区
        self._buffer = b""

        # 声道转换（如果需要）
        if input_num_channels != output_num_channels:
            output_int16 = self._convert_channels(output_int16, input_num_channels, output_num_channels)

        return self._pack_output_frames(output_int16)

    def _pack_output_frames(self, output_array: np.ndarray) -> list[AudioFrame]:
        """
        将重采样后的数据打包成 AudioFrame 列表。
        
        如果设置了 output_frame_duration_ms，则按指定时长切分输出；
        否则将所有数据打包成单个 AudioFrame。

        Args:
            output_array (np.ndarray): 重采样后的音频数据数组。

        Returns:
            list[AudioFrame]: AudioFrame 列表。
        """
        output_num_channels = self._output_num_channels

        # 如果未设置输出帧时长，直接返回单个帧
        if self._output_samples_per_frame is None:
            output_bytes = output_array.tobytes()
            samples_per_channel = len(output_array) if output_num_channels == 1 else output_array.shape[0]

            frame = AudioFrame(
                data=output_bytes,
                sample_rate=self._output_rate,
                num_channels=output_num_channels,
                samples_per_channel=samples_per_channel,
            )
            # 复制输入帧的 userdata 到输出帧
            if self._last_input_userdata:
                frame.update_userdata(self._last_input_userdata)
            return [frame]

        # 将新数据添加到输出缓冲区
        if self._output_buffer is None:
            self._output_buffer = output_array
        else:
            self._output_buffer = np.concatenate([self._output_buffer, output_array])

        frames: list[AudioFrame] = []
        samples_per_frame = self._output_samples_per_frame

        # 从缓冲区中提取完整的帧
        while True:
            buffer_samples = len(self._output_buffer) if output_num_channels == 1 else self._output_buffer.shape[0]
            if buffer_samples < samples_per_frame:
                break

            # 提取一帧数据
            if output_num_channels == 1:
                frame_data = self._output_buffer[:samples_per_frame]
                self._output_buffer = self._output_buffer[samples_per_frame:]
            else:
                frame_data = self._output_buffer[:samples_per_frame, :]
                self._output_buffer = self._output_buffer[samples_per_frame:, :]

            frame = AudioFrame(
                data=frame_data.tobytes(),
                sample_rate=self._output_rate,
                num_channels=output_num_channels,
                samples_per_channel=samples_per_frame,
            )
            # 复制输入帧的 userdata 到输出帧
            if self._last_input_userdata:
                frame.update_userdata(self._last_input_userdata)
            frames.append(frame)

        # 如果缓冲区为空，重置为 None
        buffer_samples = len(self._output_buffer) if output_num_channels == 1 else self._output_buffer.shape[0]
        if buffer_samples == 0:
            self._output_buffer = None

        return frames

    def flush(self) -> list[AudioFrame]:
        """
        刷新重采样器中所有剩余的音频数据。

        Returns:
            list[AudioFrame]: 包含刷新后剩余重采样音频数据的 `AudioFrame` 对象列表。
        """
        frames: list[AudioFrame] = []
        output_num_channels = self._output_num_channels
        
        # 处理输入缓冲区中剩余的数据
        if self._buffer:
            result = self._process_buffer()
            frames.extend(result)
            self._buffer = b""

        # 输出缓冲区中剩余的数据（不足一帧的部分）
        if self._output_buffer is not None:
            buffer_samples = len(self._output_buffer) if output_num_channels == 1 else self._output_buffer.shape[0]
            if buffer_samples > 0:
                frame = AudioFrame(
                    data=self._output_buffer.tobytes(),
                    sample_rate=self._output_rate,
                    num_channels=output_num_channels,
                    samples_per_channel=buffer_samples,
                )
                # 复制输入帧的 userdata 到输出帧
                if self._last_input_userdata:
                    frame.update_userdata(self._last_input_userdata)
                frames.append(frame)
            self._output_buffer = None

        return frames

