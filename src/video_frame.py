from typing import Union
from typing import List, Optional
from typing import Any
from enum import IntEnum


class VideoCodec(IntEnum):
    VP8 = 0
    H264 = 1
    AV1 = 2
    VP9 = 3
    H265 = 4

class VideoRotation(IntEnum):
    VIDEO_ROTATION_0 = 0
    VIDEO_ROTATION_90 = 1
    VIDEO_ROTATION_180 = 2
    VIDEO_ROTATION_270 = 3

class VideoBufferType(IntEnum):
    RGBA = 0
    ABGR = 1
    ARGB = 2
    BGRA = 3
    RGB24 = 4
    I420 = 5
    I420A = 6
    I422 = 7
    I444 = 8
    I010 = 9
    NV12 = 10

class VideoStreamType(IntEnum):
    VIDEO_STREAM_NATIVE = 0
    VIDEO_STREAM_WEBGL = 1
    VIDEO_STREAM_HTML = 2

class VideoFrame:
    """
    表示一个包含相关元数据和像素数据的视频帧。

    该类提供访问视频帧属性的方法，如宽度、高度和像素格式，
    以及操作和转换视频帧的方法。
    """

    def __init__(
        self,
        width: int,
        height: int,
        type: VideoBufferType,
        data: Union[bytes, bytearray, memoryview],
    ) -> None:
        """
        初始化一个新的 VideoFrame 实例。

        Args:
            width (int): 视频帧的宽度（以像素为单位）。
            height (int): 视频帧的高度（以像素为单位）。
            type (VideoBufferType): 视频帧数据的格式类型
                （例如，RGBA、BGRA、RGB24 等）。
            data (Union[bytes, bytearray, memoryview]): 视频帧的原始像素数据。
        """
        self._width = width
        self._height = height
        self._type = type
        self._data = bytearray(data)

    @property
    def width(self) -> int:
        """
        返回视频帧的宽度（以像素为单位）。

        Returns:
            int: 视频帧的宽度。
        """
        return self._width

    @property
    def height(self) -> int:
        """
        返回视频帧的高度（以像素为单位）。

        Returns:
            int: 视频帧的高度。
        """
        return self._height

    @property
    def type(self) -> VideoBufferType:
        """
        返回视频帧的格式类型。

        Returns:
            VideoBufferType: 视频帧的格式类型。
        """
        return self._type

    @property
    def data(self) -> memoryview:
        """
        返回视频帧原始像素数据的 memoryview。

        Returns:
            memoryview: 视频帧的原始像素数据，作为 memoryview 对象返回。
        """
        return memoryview(self._data)

    def get_plane(self, plane_nth: int) -> Optional[memoryview]:
        """
        根据索引返回视频帧中特定平面的 memoryview。

        某些视频格式（例如，I420、NV12）包含多个平面（Y、U、V 通道）。
        此方法允许通过索引访问各个平面。

        Args:
            plane_nth (int): 要检索的平面的索引（从 0 开始）。

        Returns:
            Optional[memoryview]: 指定平面数据的 memoryview，如果索引超出格式范围则返回 None。
        """
        pass

    def convert(
        self, type: VideoBufferType, *, flip_y: bool = False
    ) -> "VideoFrame":
        """
        将当前视频帧转换为不同的格式类型，可选择垂直翻转帧。

        Args:
            type (VideoBufferType): 要转换到的目标格式类型
                （例如，RGBA、I420）。
            flip_y (bool, optional): 如果为 True，帧将垂直翻转。默认为 False。

        Returns:
            VideoFrame: 指定格式的新 VideoFrame 对象。

        Raises:
            Exception: 如果不支持该转换。

        Example:
            将帧从 RGBA 格式转换为 I420 格式：

            >>> frame = VideoFrame(width=1920, height=1080, type=VideoBufferType.RGBA, data=raw_data)
            >>> converted_frame = frame.convert(VideoBufferType.I420)
            >>> print(converted_frame.type)
            VideoBufferType.I420

        Example:
            将帧从 BGRA 格式转换为 RGB24 格式并垂直翻转：

            >>> frame = VideoFrame(width=1280, height=720, type=VideoBufferType.BGRA, data=raw_data)
            >>> converted_frame = frame.convert(VideoBufferType.RGB24, flip_y=True)
            >>> print(converted_frame.type)
            VideoBufferType.RGB24
            >>> print(converted_frame.width, converted_frame.height)
            1280 720
        """
        pass

    def __repr__(self) -> str:
        return f"VideoFrame(width={self.width}, height={self.height}, type={self.type})"
