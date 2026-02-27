#!/usr/bin/env python3
"""
video_asd_demo.py

从视频文件中检测活动说话者 (Active Speaker Detection) 的演示程序。
读取视频文件，提取音频和视频帧，通过 DeepTalk-ASD 判断画面中哪个人脸在说话，
并将检测结果渲染到输出视频中。

用法:
    python3 demo/video_asd_demo.py --input demo/demo.mp4
    python3 demo/video_asd_demo.py --input demo/demo.mp4 --output demo/demo_result.mp4
    python3 demo/video_asd_demo.py --input demo/demo.mp4 --display
"""

import os
import sys
import time
import argparse
import subprocess

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from deeptalk_asd import ASDDetectorFactory, VideoFrame, VideoBufferType, AudioFrame


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_MS = 30
AUDIO_CHUNK_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)
EVAL_INTERVAL_SEC = 1.0

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
COLOR_SPEAKING = (0, 255, 0)
COLOR_NOT_SPEAKING = (0, 0, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT_FG = (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepTalk-ASD 视频文件活动说话者检测演示")
    parser.add_argument("--input", type=str, default=os.path.join(PROJECT_ROOT, "demo", "demo.mp4"),
                        help="输入视频文件路径 (默认: demo/demo.mp4)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频文件路径 (默认: 输入文件名_asd_result.mp4)")
    parser.add_argument("--display", action="store_true",
                        help="实时显示处理画面 (按 q 退出)")
    parser.add_argument("--face-detector", type=str, default="inspireface")
    parser.add_argument("--turn-detector", type=str, default="silero-vad")
    parser.add_argument("--speaker-detector", type=str, default="LR-ASD-ONNX")
    parser.add_argument("--onnx-dir", type=str, default=None)
    parser.add_argument("--vad-model-path", type=str, default=None)
    parser.add_argument("--abs-amplitude-threshold", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=float, default=EVAL_INTERVAL_SEC,
                        help="说话者评估间隔，单位秒 (默认: 1.0)")
    return parser.parse_args()


def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """使用 ffmpeg 从视频中提取音频为 PCM int16 单声道数据"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 提取音频失败:\n{stderr_text}")
    audio_data = np.frombuffer(proc.stdout, dtype=np.int16)
    print(f"[音频] 提取完成: {len(audio_data)} 采样点, "
          f"时长 {len(audio_data) / sample_rate:.2f}s")
    return audio_data


class SpeakingStateTracker:
    """说话状态追踪器，记录每个 track_id 最新的说话评分"""

    def __init__(self):
        self._scores: dict[int, float] = {}

    def update_speakers(self, speaker_scores: dict):
        if not speaker_scores:
            return
        self._scores = dict(speaker_scores)

    def is_speaking(self, track_id: int) -> bool:
        return self._scores.get(track_id, 0) > 0


def draw_face_overlay(frame: np.ndarray, face_profile, is_speaking: bool):
    """在视频帧上绘制人脸框和信息标注"""
    rect = face_profile.face_rectangle
    x, y = int(rect.x), int(rect.y)
    w, h = int(rect.width), int(rect.height)

    color = COLOR_SPEAKING if is_speaking else COLOR_NOT_SPEAKING
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

    parts = [f"ID:{face_profile.track_id}"]
    if is_speaking:
        parts.append("SPEAKING")

    info_text = " | ".join(parts)
    (text_w, text_h), baseline = cv2.getTextSize(info_text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_x = x
    text_y = y - 8

    cv2.rectangle(frame, (text_x, text_y - text_h - baseline),
                  (text_x + text_w, text_y + baseline), COLOR_TEXT_BG, -1)
    cv2.putText(frame, info_text, (text_x, text_y),
                FONT, FONT_SCALE, COLOR_TEXT_FG, FONT_THICKNESS, cv2.LINE_AA)


def create_asd(args):
    """根据命令行参数创建 ASD 实例"""
    onnx_dir = args.onnx_dir or os.path.join(PROJECT_ROOT, "weights")
    vad_model_path = args.vad_model_path or os.path.join(PROJECT_ROOT, "weights", "silero_vad.onnx")

    face_detector_config = {"type": args.face_detector}
    turn_detector_config = {
        "type": args.turn_detector,
        "model_path": vad_model_path,
        "abs_amplitude_threshold": args.abs_amplitude_threshold,
    }
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

    if not os.path.isfile(args.input):
        print(f"[错误] 输入视频文件不存在: {args.input}")
        sys.exit(1)

    # 确定输出路径
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_asd_result.mp4"

    # 1. 创建 ASD 实例
    asd = create_asd(args)

    # 2. 提取音频
    print(f"[信息] 正在从视频中提取音频...")
    audio_data = extract_audio_from_video(args.input, AUDIO_SAMPLE_RATE)

    # 3. 打开视频
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频文件: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"[视频] 已打开: {args.input}")
    print(f"  分辨率: {frame_w}x{frame_h}, FPS: {fps:.2f}, "
          f"总帧数: {total_frames}, 时长: {duration:.2f}s")

    # 4. 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频: {args.output}")
        cap.release()
        sys.exit(1)

    # 5. 初始化状态追踪
    state_tracker = SpeakingStateTracker()
    eval_interval = args.eval_interval
    audio_cursor = 0
    last_eval_video_time = -eval_interval  # 保证首次满足间隔即触发

    if args.display:
        window_name = "DeepTalk-ASD 视频分析"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    process_start = time.perf_counter()

    print(f"[信息] 评估间隔: {eval_interval}s")
    print(f"\n[信息] 开始处理视频...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps
            create_time = process_start + video_time

            # ── 送入音频块（追赶到当前视频时间） ──
            target_audio_pos = int(video_time * AUDIO_SAMPLE_RATE)
            while audio_cursor < target_audio_pos and audio_cursor < len(audio_data):
                chunk_end = min(audio_cursor + AUDIO_CHUNK_SAMPLES, len(audio_data))
                chunk = audio_data[audio_cursor:chunk_end]

                chunk_time = process_start + audio_cursor / AUDIO_SAMPLE_RATE
                audio_frame = AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=AUDIO_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                asd.append_audio(audio_frame, chunk_time)
                audio_cursor = chunk_end

            # ── 送入视频帧 ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            video_frame = VideoFrame(
                width=w, height=h,
                type=VideoBufferType.RGB24,
                data=bytes(frame_rgb),
            )
            face_profiles = asd.append_video(video_frame, create_time)

            # ── 定时评估说话者 ──
            if video_time - last_eval_video_time >= eval_interval:
                eval_end = create_time
                eval_start = eval_end - eval_interval
                speaker_scores = asd.evaluate(eval_start, eval_end)
                if speaker_scores:
                    state_tracker.update_speakers(speaker_scores)
                    print(f"  [{video_time:.2f}s] 说话者评估: {speaker_scores}")
                last_eval_video_time = video_time

            # ── 绘制标注：说话绿框，不说话红框 ──
            if face_profiles:
                for profile in face_profiles:
                    is_speaking = state_tracker.is_speaking(profile.track_id)
                    draw_face_overlay(frame, profile, is_speaking)

            writer.write(frame)

            # 进度显示
            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                elapsed = time.perf_counter() - process_start
                progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                speed = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  进度: {frame_idx}/{total_frames} ({progress:.1f}%), "
                      f"速度: {speed:.1f} fps, 已耗时: {elapsed:.1f}s")

            # 实时显示
            if args.display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[信息] 用户中断")
                    break

    except KeyboardInterrupt:
        print("\n[信息] 检测到 Ctrl+C，正在退出...")
    finally:
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - process_start
        print(f"\n[完成] 输出视频: {args.output}")
        print(f"  处理了 {frame_idx} 帧, 总耗时 {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
