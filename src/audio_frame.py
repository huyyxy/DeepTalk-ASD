import ctypes
from typing import Any, Union


class AudioFrame:
    """
    表示音频数据帧的类，具有特定属性，如采样率、声道数和每声道采样数。

    音频数据的格式为16位有符号整数（int16），按声道交错排列。
    """

    def __init__(
        self,
        data: Union[bytes, bytearray, memoryview],
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ) -> None:
        """
        初始化 AudioFrame 实例。

        Args:
            data (Union[bytes, bytearray, memoryview]): 原始音频数据，长度必须至少为
                `num_channels * samples_per_channel * sizeof(int16)` 字节。
            sample_rate (int): 音频的采样率，单位为 Hz。
            num_channels (int): 音频声道数（例如，1 表示单声道，2 表示立体声）。
            samples_per_channel (int): 每个声道的采样数。

        Raises:
            ValueError: 如果 `data` 的长度小于所需大小。
        """
        data = memoryview(data).cast("B")

        if len(data) < num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16):
            raise ValueError(
                "data length must be >= num_channels * samples_per_channel * sizeof(int16)"
            )

        if len(data) % ctypes.sizeof(ctypes.c_int16) != 0:
            # 当数据长度大于所需长度时可能发生
            raise ValueError("data length must be a multiple of sizeof(int16)")

        n = len(data) // ctypes.sizeof(ctypes.c_int16)
        self._data = (ctypes.c_int16 * n).from_buffer_copy(data)

        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._samples_per_channel = samples_per_channel
        self._userdata: dict[str, Any] = {}

    @staticmethod
    def create(sample_rate: int, num_channels: int, samples_per_channel: int) -> "AudioFrame":
        """
        创建一个新的空 AudioFrame 实例，指定采样率、声道数和每声道采样数。

        Args:
            sample_rate (int): 音频的采样率，单位为 Hz。
            num_channels (int): 音频声道数（例如，1 表示单声道，2 表示立体声）。
            samples_per_channel (int): 每个声道的采样数。

        Returns:
            AudioFrame: 一个新的 AudioFrame 实例，数据未初始化（零填充）。
        """
        size = num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        data = bytearray(size)
        return AudioFrame(data, sample_rate, num_channels, samples_per_channel)

    @property
    def data(self) -> memoryview:
        """
        返回音频数据的内存视图，作为16位有符号整数。

        Returns:
            memoryview: 音频数据的内存视图。
        """
        return memoryview(self._data).cast("B").cast("h")

    @property
    def sample_rate(self) -> int:
        """
        返回音频帧的采样率。

        Returns:
            int: 采样率，单位为 Hz。
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """
        返回音频帧的声道数。

        Returns:
            int: 音频声道数（例如，1 表示单声道，2 表示立体声）。
        """
        return self._num_channels

    @property
    def samples_per_channel(self) -> int:
        """
        返回每个声道的采样数。

        Returns:
            int: 每个声道的采样数。
        """
        return self._samples_per_channel

    @property
    def duration_ms(self) -> int:
        """
        返回音频帧的时长（毫秒）。

        Returns:
            int: 时长，单位为毫秒。
        """
        return int(self.samples_per_channel * 1000 / self.sample_rate)

    @property
    def userdata(self) -> dict[str, Any]:
        """
        返回音频帧的外带用户数据字典。

        Returns:
            dict[str, Any]: 用户数据字典。
        """
        return self._userdata

    def get_userdata(self, key: str, default: Any = None) -> Any:
        """
        获取指定键的用户数据。

        Args:
            key (str): 数据键名。
            default (Any, optional): 键不存在时的默认值。默认为 None。

        Returns:
            Any: 对应键的值，如果键不存在则返回默认值。
        """
        return self._userdata.get(key, default)

    def set_userdata(self, key: str, value: Any) -> None:
        """
        设置指定键的用户数据。

        Args:
            key (str): 数据键名。
            value (Any): 要设置的值。
        """
        self._userdata[key] = value

    def update_userdata(self, data: dict[str, Any]) -> None:
        """
        批量更新用户数据。

        Args:
            data (dict[str, Any]): 要更新的用户数据字典。
        """
        self._userdata.update(data)

    def clear_userdata(self) -> None:
        """
        清空用户数据。
        """
        self._userdata.clear()

    def copy_userdata_from(self, other: "AudioFrame") -> None:
        """
        从另一个 AudioFrame 复制用户数据。

        Args:
            other (AudioFrame): 源音频帧。
        """
        self._userdata = other._userdata.copy()

    def to_wav_bytes(self) -> bytes:
        """
        将音频帧数据转换为 WAV 格式的字节流。

        Returns:
            bytes: WAV 格式编码的音频数据。
        """
        import wave
        import io

        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wav:
                wav.setnchannels(self.num_channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(self._data)

            return wav_file.getvalue()

    def __repr__(self) -> str:
        return (
            f"AudioFrame(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"samples_per_channel={self.samples_per_channel}, "
            f"duration_ms={self.duration_ms})"
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, *_: Any):
        from pydantic_core import core_schema
        import base64

        def validate_audio_frame(value: Any) -> "AudioFrame":
            if isinstance(value, AudioFrame):
                return value

            if isinstance(value, tuple):
                value = value[0]

            if isinstance(value, dict):
                return AudioFrame(
                    data=base64.b64decode(value["data"]),
                    sample_rate=value["sample_rate"],
                    num_channels=value["num_channels"],
                    samples_per_channel=value["samples_per_channel"],
                )

            raise TypeError("Invalid type for AudioFrame")

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.model_fields_schema(
                        {
                            "data": core_schema.model_field(core_schema.str_schema()),
                            "sample_rate": core_schema.model_field(core_schema.int_schema()),
                            "num_channels": core_schema.model_field(core_schema.int_schema()),
                            "samples_per_channel": core_schema.model_field(
                                core_schema.int_schema()
                            ),
                        },
                    ),
                    core_schema.no_info_plain_validator_function(validate_audio_frame),
                ]
            ),
            python_schema=core_schema.no_info_plain_validator_function(validate_audio_frame),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: {
                    "data": base64.b64encode(instance.data).decode("utf-8"),
                    "sample_rate": instance.sample_rate,
                    "num_channels": instance.num_channels,
                    "samples_per_channel": instance.samples_per_channel,
                }
            ),
        )

