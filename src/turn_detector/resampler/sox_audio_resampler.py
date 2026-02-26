from typing import Any
import numpy as np
import soxr

from ..audio_frame import AudioFrame
from .audio_resampler import AudioResampler, AudioResamplerQuality


class SoxAudioResampler(AudioResampler):
    """
    使用 soxr 库实现的音频重采样器。

    soxr 是 SoX Resampler 库的 Python 绑定，提供高质量的音频重采样功能。
    支持动态输入采样率和声道数（从 AudioFrame 获取），以及声道转换。

    注意：当输入为 AudioFrame 时，会自动将输入帧的 userdata 复制到输出帧。
    当涉及多帧合并或切分时，使用最后一个输入帧的 userdata。
    """

    # soxr 质量映射
    _QUALITY_MAP = {
        AudioResamplerQuality.QUICK: soxr.QQ,      # Quick Quality
        AudioResamplerQuality.LOW: soxr.LQ,        # Low Quality
        AudioResamplerQuality.MEDIUM: soxr.MQ,     # Medium Quality
        AudioResamplerQuality.HIGH: soxr.HQ,       # High Quality
        AudioResamplerQuality.VERY_HIGH: soxr.VHQ, # Very High Quality
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
        初始化 SoxAudioResampler。

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

        self._soxr_quality = self._QUALITY_MAP.get(quality, soxr.MQ)
        
        # 重采样器缓存：key 为 (input_rate, input_num_channels)
        self._resamplers: dict[tuple[int, int], soxr.ResampleStream] = {}
        
        # 如果 input_rate 和 input_num_channels 都已知，预创建重采样器
        if input_rate is not None and input_num_channels is not None:
            self._resamplers[(input_rate, input_num_channels)] = self._create_resampler(
                input_rate, input_num_channels
            )
        
        # 输出缓冲区，用于按指定帧时长切分输出
        self._output_buffer: np.ndarray | None = None
        
        # 保存最后一个输入帧的 userdata，用于复制到输出帧
        self._last_input_userdata: dict[str, Any] = {}

    def _create_resampler(self, input_rate: int, input_num_channels: int) -> soxr.ResampleStream:
        """
        创建一个 soxr 重采样器实例。

        Args:
            input_rate (int): 输入采样率。
            input_num_channels (int): 输入声道数。

        Returns:
            soxr.ResampleStream: 重采样器实例。
        """
        # soxr 使用输入声道数进行重采样，声道转换在后处理中进行
        return soxr.ResampleStream(
            input_rate,
            self._output_rate,
            input_num_channels,
            dtype=np.int16,
            quality=self._soxr_quality,
        )

    def _get_or_create_resampler(self, input_rate: int, input_num_channels: int) -> soxr.ResampleStream:
        """
        获取或创建对应参数的重采样器。

        Args:
            input_rate (int): 输入采样率。
            input_num_channels (int): 输入声道数。

        Returns:
            soxr.ResampleStream: 重采样器实例。
        """
        key = (input_rate, input_num_channels)
        if key not in self._resamplers:
            self._resamplers[key] = self._create_resampler(input_rate, input_num_channels)
        return self._resamplers[key]

    def push(self, data: bytearray | AudioFrame) -> list[AudioFrame]:
        """
        将音频数据推送到重采样器中，并获取所有可用的重采样数据。

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

        # 获取或创建对应的重采样器
        resampler = self._get_or_create_resampler(input_rate, input_num_channels)

        # 将字节数据转换为 numpy 数组 (int16)
        input_array = np.frombuffer(bdata, dtype=np.int16)

        # 如果是多声道，需要重塑数组
        if input_num_channels > 1:
            input_array = input_array.reshape(-1, input_num_channels)

        # 使用 soxr 进行重采样
        output_array = resampler.resample_chunk(input_array)

        if output_array.size == 0:
            return []

        # 声道转换（如果需要）
        if input_num_channels != self._output_num_channels:
            output_array = self._convert_channels(output_array, input_num_channels, self._output_num_channels)

        return self._pack_output_frames(output_array)

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

        # 刷新所有重采样器
        for (input_rate, input_num_channels), resampler in self._resamplers.items():
            # 创建一个空的 numpy 数组作为最后的输入
            if input_num_channels > 1:
                empty_input = np.array([], dtype=np.int16).reshape(0, input_num_channels)
            else:
                empty_input = np.array([], dtype=np.int16)

            # 使用 last=True 标志刷新缓冲区
            output_array = resampler.resample_chunk(empty_input, last=True)

            # 处理重采样器的输出
            if output_array.size > 0:
                # 声道转换（如果需要）
                if input_num_channels != output_num_channels:
                    output_array = self._convert_channels(output_array, input_num_channels, output_num_channels)

                if self._output_samples_per_frame is None:
                    # 未设置帧时长，直接输出
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
                    frames.append(frame)
                else:
                    # 设置了帧时长，添加到缓冲区并切分
                    frames.extend(self._pack_output_frames(output_array))

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

