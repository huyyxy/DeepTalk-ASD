#!/usr/bin/env python3
"""
realtime_asd_demo.py

实时活动说话者检测 (Active Speaker Detection) 演示程序。
从摄像头采集视频帧、麦克风采集音频帧，通过 DeepTalk-ASD 判断视频画面中
哪个人脸在说话，并用 OpenCV 实时显示检测结果。

用法:
    python3 demo/realtime_asd_demo.py [OPTIONS]

    # 使用缺省摄像头和麦克风:
    python3 demo/realtime_asd_demo.py

    # 指定摄像头和麦克风:
    python3 demo/realtime_asd_demo.py --camera 1 --microphone 2

    # 自定义 ASD 组件:
    python3 demo/realtime_asd_demo.py --face-detector inspireface --speaker-detector LR-ASD-ONNX
"""

import os
import sys
import time
import argparse
import threading
import struct
import traceback

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from deeptalk_asd import ASDDetectorFactory, VideoFrame, VideoBufferType, AudioFrame, TurnState


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
AUDIO_CHUNK_MS = 30          # 每次采集的音频时长（毫秒）
SPEAKING_PERSIST_SEC = 0.5   # 说话者绿框持续时间（秒）
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2
COLOR_SPEAKING = (0, 255, 0)     # 绿色 (BGR)
COLOR_NOT_SPEAKING = (0, 0, 255) # 红色 (BGR)
COLOR_TEXT_BG = (0, 0, 0)        # 文字背景色
COLOR_TEXT_FG = (255, 255, 255)  # 文字前景色


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DeepTalk-ASD 实时活动说话者检测演示"
    )

    # 设备参数
    parser.add_argument(
        "--camera", type=int, default=0,
        help="摄像头设备索引 (默认: 0)"
    )
    parser.add_argument(
        "--camera-width", type=int, default=640,
        help="摄像头采集宽度 (默认: 640)"
    )
    parser.add_argument(
        "--camera-height", type=int, default=480,
        help="摄像头采集高度 (默认: 480)"
    )
    parser.add_argument(
        "--microphone", type=int, default=None,
        help="麦克风设备索引 (默认: 系统缺省设备)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="音频采样率 (默认: 16000)"
    )

    # ASD 组件参数
    parser.add_argument(
        "--face-detector", type=str, default="inspireface",
        help="人脸检测器类型 (默认: inspireface)"
    )
    parser.add_argument(
        "--turn-detector", type=str, default="silero-vad",
        help="轮次检测器类型 (默认: silero-vad)"
    )
    parser.add_argument(
        "--speaker-detector", type=str, default="LR-ASD-ONNX",
        help="说话者检测器类型 (默认: LR-ASD-ONNX)"
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=None,
        help="ONNX 权重目录 (默认: <project>/weights)"
    )
    parser.add_argument(
        "--vad-model-path", type=str, default=None,
        help="VAD 模型文件路径 (可选)"
    )

    return parser.parse_args()


class SpeakingStateTracker:
    """
    线程安全的说话状态管理器。

    记录每个 track_id 最后一次被判定为说话者的时间戳，
    并提供查询接口判断该 track_id 是否仍处于"说话"状态（0.5 秒内）。
    """

    def __init__(self, persist_sec: float = SPEAKING_PERSIST_SEC):
        self._speaking_faces: dict[int, float] = {}  # track_id -> timestamp
        self._lock = threading.Lock()
        self._persist_sec = persist_sec

    def update_speakers(self, speaker_scores: dict):
        """
        更新说话者状态。

        参数:
            speaker_scores: evaluate() 返回的 {track_id: score, ...}
        """
        if not speaker_scores:
            return
        now = time.perf_counter()
        with self._lock:
            for track_id, score in speaker_scores.items():
                if score > 0:
                    self._speaking_faces[track_id] = now

    def is_speaking(self, track_id: int) -> bool:
        """判断某个 track_id 是否仍在"说话"状态"""
        now = time.perf_counter()
        with self._lock:
            last_time = self._speaking_faces.get(track_id)
            if last_time is None:
                return False
            if now - last_time > self._persist_sec:
                del self._speaking_faces[track_id]
                return False
            return True

    def cleanup(self):
        """清理过期的说话状态"""
        now = time.perf_counter()
        with self._lock:
            expired = [
                tid for tid, ts in self._speaking_faces.items()
                if now - ts > self._persist_sec
            ]
            for tid in expired:
                del self._speaking_faces[tid]


class AudioCaptureThread(threading.Thread):
    """
    音频采集线程。

    使用 PyAudio 从麦克风采集音频，调用 ASD 的 append_audio / evaluate 方法，
    并将说话者结果更新到 SpeakingStateTracker。
    """

    def __init__(self, asd, state_tracker: SpeakingStateTracker,
                 sample_rate: int = 16000, device_index: int = None):
        super().__init__(daemon=True)
        self._asd = asd
        self._state_tracker = state_tracker
        self._sample_rate = sample_rate
        self._device_index = device_index
        self._running = False
        self._chunk_samples = int(sample_rate * AUDIO_CHUNK_MS / 1000)  # 每 chunk 采样数

    def run(self):
        """音频采集主循环"""
        import pyaudio

        pa = pyaudio.PyAudio()
        self._running = True

        # 打印可用麦克风设备
        print("\n--- 可用音频输入设备 ---")
        for i in range(pa.get_device_count()):
            dev_info = pa.get_device_info_by_index(i)
            if dev_info["maxInputChannels"] > 0:
                marker = " <-- 当前" if self._device_index is not None and i == self._device_index else ""
                if self._device_index is None and i == pa.get_default_input_device_info()['index']:
                    marker = " <-- 缺省"
                print(f"  [{i}] {dev_info['name']} (输入通道: {dev_info['maxInputChannels']}){marker}")
        print("---\n")

        stream_kwargs = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": self._sample_rate,
            "input": True,
            "frames_per_buffer": self._chunk_samples,
        }
        if self._device_index is not None:
            stream_kwargs["input_device_index"] = self._device_index

        try:
            stream = pa.open(**stream_kwargs)
        except Exception as e:
            print(f"[错误] 无法打开麦克风: {e}")
            traceback.print_exc()
            pa.terminate()
            return

        print(f"[音频] 麦克风已开启 (采样率: {self._sample_rate}, chunk: {self._chunk_samples} samples)")

        try:
            while self._running:
                try:
                    raw_data = stream.read(self._chunk_samples, exception_on_overflow=False)
                except Exception as e:
                    print(f"[警告] 音频读取异常: {e}")
                    continue

                create_time = time.perf_counter()

                # 封装为 AudioFrame
                audio_frame = AudioFrame(
                    data=raw_data,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    samples_per_channel=self._chunk_samples,
                )

                # 调用 ASD append_audio 进行 VAD 检测
                utterance = self._asd.append_audio(audio_frame, create_time)

                if utterance is not None and utterance.turn_state == TurnState.TURN_END:
                    # VAD 检测到说话结束，评估活动说话者
                    speaker_scores = self._asd.evaluate()
                    if speaker_scores:
                        self._state_tracker.update_speakers(speaker_scores)
                        print(f"[ASD] 说话者评估结果: {speaker_scores}")

        except Exception as e:
            print(f"[错误] 音频线程异常: {e}")
            traceback.print_exc()
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("[音频] 麦克风已关闭")

    def stop(self):
        """停止音频采集"""
        self._running = False


def draw_face_overlay(frame: np.ndarray, face_profile, is_speaking: bool):
    """
    在视频帧上绘制人脸框和信息标注。

    参数:
        frame: OpenCV BGR 图像 (numpy.ndarray)
        face_profile: FaceProfile 实例
        is_speaking: 该人脸是否正在说话
    """
    rect = face_profile.face_rectangle
    x = int(rect.x)
    y = int(rect.y)
    w = int(rect.width)
    h = int(rect.height)

    # 选择边框颜色
    color = COLOR_SPEAKING if is_speaking else COLOR_NOT_SPEAKING

    # 绘制人脸边框
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

    # 构建信息文本
    info_parts = [f"ID:{face_profile.track_id}"]
    if face_profile.gender:
        info_parts.append(face_profile.gender)
    if face_profile.age:
        info_parts.append(face_profile.age)
    if face_profile.emotion:
        info_parts.append(face_profile.emotion)
    if is_speaking:
        info_parts.append("🔊")

    info_text = " | ".join(info_parts)

    # 计算文字大小和位置
    (text_w, text_h), baseline = cv2.getTextSize(info_text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_x = x
    text_y = y - 8  # 文字在边框上方

    # 绘制文字背景
    cv2.rectangle(
        frame,
        (text_x, text_y - text_h - baseline),
        (text_x + text_w, text_y + baseline),
        COLOR_TEXT_BG, -1
    )

    # 绘制文字
    cv2.putText(
        frame, info_text,
        (text_x, text_y),
        FONT, FONT_SCALE, COLOR_TEXT_FG, FONT_THICKNESS, cv2.LINE_AA
    )


def create_asd(args):
    """
    根据命令行参数创建 ASD 实例。

    参数:
        args: argparse 命名空间

    返回:
        ASDInterface 实例
    """
    # 确定 ONNX 目录
    onnx_dir = args.onnx_dir
    if onnx_dir is None:
        onnx_dir = os.path.join(PROJECT_ROOT, "weights")

    # 人脸检测器配置
    face_detector_config = {
        "type": args.face_detector,
    }

    # 轮次检测器配置
    vad_model_path = args.vad_model_path
    if vad_model_path is None:
        vad_model_path = os.path.join(PROJECT_ROOT, "weights", "silero_vad.onnx")
    turn_detector_config = {
        "type": args.turn_detector,
        "model_path": vad_model_path,
    }

    # 说话者检测器配置
    speaker_detector_config = {
        "type": args.speaker_detector,
        "onnx_dir": onnx_dir,
    }

    print(f"[ASD] 创建 ASD 实例...")
    print(f"  人脸检测器: {face_detector_config}")
    print(f"  轮次检测器: {turn_detector_config}")
    print(f"  说话者检测器: {speaker_detector_config}")

    factory = ASDDetectorFactory(
        face_detector=face_detector_config,
        turn_detector=turn_detector_config,
        speaker_detector=speaker_detector_config,
    )
    asd = factory.create()
    if asd is None:
        raise RuntimeError("ASD 实例创建失败，请检查组件配置和模型文件")
    print("[ASD] 创建成功")
    return asd


def main():
    args = parse_args()

    # 1. 创建 ASD 实例
    asd = create_asd(args)

    # 2. 创建说话状态跟踪器
    state_tracker = SpeakingStateTracker()

    # 3. 启动音频采集线程
    audio_thread = AudioCaptureThread(
        asd=asd,
        state_tracker=state_tracker,
        sample_rate=args.sample_rate,
        device_index=args.microphone,
    )
    audio_thread.start()

    # 4. 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.camera}")
        audio_thread.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    # 等待摄像头预热（某些虚拟摄像头插件需要初始化时间）
    print("[视频] 等待摄像头预热...")
    time.sleep(2.0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[视频] 摄像头已开启: 设备 {args.camera}, 分辨率 {actual_w}x{actual_h}")
    print("[提示] 按 'q' 键退出\n")

    window_name = "DeepTalk-ASD 实时演示"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    read_fail_count = 0
    MAX_READ_FAILURES = 30  # 最大连续读取失败次数

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_fail_count += 1
                if read_fail_count >= MAX_READ_FAILURES:
                    print(f"[错误] 连续 {MAX_READ_FAILURES} 次无法读取摄像头帧，退出")
                    break
                print(f"[警告] 无法读取摄像头帧 (重试 {read_fail_count}/{MAX_READ_FAILURES})")
                time.sleep(0.1)
                continue
            read_fail_count = 0  # 读取成功，重置计数器

            create_time = time.perf_counter()

            # 将 OpenCV BGR 帧转换为 RGB24 格式的 VideoFrame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            video_frame = VideoFrame(
                width=w,
                height=h,
                type=VideoBufferType.RGB24,
                data=bytes(frame_rgb),
            )

            # 调用 ASD 处理视频帧
            face_profiles = asd.append_video(video_frame, create_time)

            # 清理过期的说话状态
            state_tracker.cleanup()

            # 绘制人脸框和信息
            if face_profiles:
                for profile in face_profiles:
                    is_speaking = state_tracker.is_speaking(profile.track_id)
                    draw_face_overlay(frame, profile, is_speaking)

            # 显示帧
            cv2.imshow(window_name, frame)

            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                break

            # 检查窗口是否已关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[信息] 检测到 Ctrl+C，正在退出...")
    finally:
        # 清理资源
        audio_thread.stop()
        audio_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[信息] 程序已退出")


if __name__ == "__main__":
    main()
