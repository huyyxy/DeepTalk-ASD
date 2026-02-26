from .audio_resampler import AudioResampler, AudioResamplerQuality


class AudioResamplerFactory:
    """
    音频重采样器工厂类。

    用于根据指定的重采样器类型创建相应的 AudioResampler 实例。
    """

    def __init__(
        self,
        resampler_type: str,
        output_rate: int,
        *,
        input_rate: int | None = None,
        output_num_channels: int = 1,
        input_num_channels: int | None = None,
        quality: AudioResamplerQuality = AudioResamplerQuality.MEDIUM,
        output_frame_duration_ms: int | None = None,
        **kwargs,
    ):
        """
        初始化音频重采样器工厂。

        Args:
            resampler_type (str): 重采样器类型，可选值：
                - 'sox' 或 'soxr': 使用 soxr 库
                - 'rosa' 或 'librosa': 使用 librosa 库
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
            **kwargs: 其他参数。
        """
        self.resampler_type = resampler_type.lower()
        self.output_rate = output_rate
        self.input_rate = input_rate
        self.output_num_channels = output_num_channels
        self.input_num_channels = input_num_channels
        self.quality = quality
        self.output_frame_duration_ms = output_frame_duration_ms
        self.kwargs = kwargs

    def create(self) -> AudioResampler:
        """
        根据配置创建并返回相应的 AudioResampler 实例。

        Returns:
            AudioResampler: 音频重采样器实例。

        Raises:
            ValueError: 如果指定的重采样器类型不支持。
        """
        if self.resampler_type in ("sox", "soxr"):
            from .sox_audio_resampler import SoxAudioResampler

            return SoxAudioResampler(
                self.output_rate,
                input_rate=self.input_rate,
                output_num_channels=self.output_num_channels,
                input_num_channels=self.input_num_channels,
                quality=self.quality,
                output_frame_duration_ms=self.output_frame_duration_ms,
            )
        elif self.resampler_type in ("rosa", "librosa"):
            from .rosa_audio_resampler import RosaAudioResampler

            return RosaAudioResampler(
                self.output_rate,
                input_rate=self.input_rate,
                output_num_channels=self.output_num_channels,
                input_num_channels=self.input_num_channels,
                quality=self.quality,
                output_frame_duration_ms=self.output_frame_duration_ms,
            )
        else:
            raise ValueError(
                f"不支持的重采样器类型: {self.resampler_type}。"
                f"可选值: 'sox', 'soxr', 'rosa', 'librosa'"
            )

    @staticmethod
    def create_resampler(
        resampler_type: str,
        output_rate: int,
        *,
        input_rate: int | None = None,
        output_num_channels: int = 1,
        input_num_channels: int | None = None,
        quality: AudioResamplerQuality = AudioResamplerQuality.MEDIUM,
        output_frame_duration_ms: int | None = None,
    ) -> AudioResampler:
        """
        静态方法：直接创建并返回 AudioResampler 实例。

        这是一个便捷方法，无需实例化工厂即可创建重采样器。

        Args:
            resampler_type (str): 重采样器类型。
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

        Returns:
            AudioResampler: 音频重采样器实例。
        """
        factory = AudioResamplerFactory(
            resampler_type,
            output_rate,
            input_rate=input_rate,
            output_num_channels=output_num_channels,
            input_num_channels=input_num_channels,
            quality=quality,
            output_frame_duration_ms=output_frame_duration_ms,
        )
        return factory.create()

