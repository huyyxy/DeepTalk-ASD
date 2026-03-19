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
    python3 demo/video_asd_demo.py --input demo/demo.mp4 --turn-detector pvad
"""

import os
import sys
import time
import argparse
import subprocess

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from deeptalk_asd import ASDDetectorFactory, VideoFrame, VideoBufferType, AudioFrame, TurnState


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_MS = 30
AUDIO_CHUNK_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)
SPEAKING_PERSIST_SEC = 0.5   # 说话者绿框持续时间（秒）
# 与 Silero VAD 一致：utterance 开头为前置静音、结尾为触发结束的静音
EVAL_PREFIX_TRIM_SEC = 0.1
EVAL_SUFFIX_TRIM_SEC = 0.1
EVAL_MIN_CORE_SEC = 0.6

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
COLOR_SPEAKING = (0, 255, 0)         # 绿色 - 主说话人
COLOR_SPEAKING_SECONDARY = (255, 165, 0)  # 蓝色 - 次说话人 (BGR)
COLOR_NOT_SPEAKING = (0, 0, 255)     # 红色
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT_FG = (255, 255, 255)
COLOR_TURN_STATE = (0, 255, 0)       # 绿色 - TurnState 显示

TURN_STATE_DISPLAY_MS = 300  # TurnState 显示持续时间（毫秒）


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
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型文件目录 (默认: <project>/weights)")
    parser.add_argument("--vad-model-name", type=str, default=None,
                        help="VAD 模型文件名 (默认: silero_vad.onnx)")
    parser.add_argument("--voiceprint-model-name", type=str, default=None,
                        help="声纹模型文件名 (例如: wespeaker_zh_cnceleb_resnet34.onnx)")
    parser.add_argument("--abs-amplitude-threshold", type=float, default=0.01)
    return parser.parse_args()


def get_video_start_time(video_path: str) -> float:
    """使用 ffprobe 获取视频流的 start_time（秒），用于修正音画对齐"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=start_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    start_str = result.stdout.strip()
    if start_str and start_str != "N/A":
        return float(start_str)
    return 0.0


def extract_audio_from_video(video_path: str, sample_rate: int = 16000,
                             start_time: float = 0) -> np.ndarray:
    """使用 ffmpeg 从视频中提取音频为 PCM int16 单声道数据，可指定起始时间"""
    cmd = ["ffmpeg"]
    if start_time > 0:
        cmd += ["-ss", f"{start_time:.6f}"]
    cmd += [
        "-i", video_path,
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
    """说话状态追踪器，支持持久说话人和超时说话人，区分主/次说话人"""

    def __init__(self, persist_sec: float = SPEAKING_PERSIST_SEC):
        self._speaking_faces: dict[int, float] = {}  # track_id -> timestamp (超时机制)
        self._persistent_faces: set[int] = set()     # 持久说话人 (不超时)
        self._primary_speaker: int = -1               # 分数最高的说话人 track_id
        self._persist_sec = persist_sec

    def _update_primary(self, speaker_scores: dict):
        """从 speaker_scores 中找出分数最高的说话人"""
        best_tid, best_score = -1, 0
        for tid, score in speaker_scores.items():
            if score > best_score:
                best_tid, best_score = tid, score
        self._primary_speaker = best_tid

    def update_speakers(self, speaker_scores: dict):
        """更新说话者状态（0.5 秒超时）"""
        if not speaker_scores:
            return
        now = time.perf_counter()
        for track_id, score in speaker_scores.items():
            if score > 0:
                self._speaking_faces[track_id] = now
        self._update_primary(speaker_scores)

    def set_persistent_speakers(self, speaker_scores: dict):
        """设置持久说话人（绿框不超时消失，直到手动清除）"""
        self._persistent_faces = {
            tid for tid, score in speaker_scores.items() if score > 0
        }
        self._update_primary(speaker_scores)

    def clear_persistent_speakers(self):
        """清除持久说话人状态"""
        self._persistent_faces.clear()
        self._primary_speaker = -1

    def has_persistent_speakers(self) -> bool:
        """查询当前是否有持久说话人"""
        return len(self._persistent_faces) > 0

    def get_speaking_level(self, track_id: int) -> str:
        """
        获取说话级别：
            'primary'   - 分数最高的说话人（绿色）
            'secondary' - 其他说话人（蓝色）
            None        - 未说话（红色）
        """
        # 检查持久说话人
        if track_id in self._persistent_faces:
            return 'primary' if track_id == self._primary_speaker else 'secondary'
        # 检查超时说话人
        now = time.perf_counter()
        last_time = self._speaking_faces.get(track_id)
        if last_time is not None:
            if now - last_time <= self._persist_sec:
                return 'primary' if track_id == self._primary_speaker else 'secondary'
            else:
                del self._speaking_faces[track_id]
        return None


def draw_face_overlay(frame: np.ndarray, face_profile, speaking_level: str):
    """
    在视频帧上绘制人脸框和信息标注。
    speaking_level: 'primary' (绿), 'secondary' (蓝), None (红)
    """
    rect = face_profile.face_rectangle
    x, y = int(rect.x), int(rect.y)
    w, h = int(rect.width), int(rect.height)

    if speaking_level == 'primary':
        color = COLOR_SPEAKING
    elif speaking_level == 'secondary':
        color = COLOR_SPEAKING_SECONDARY
    else:
        color = COLOR_NOT_SPEAKING
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

    parts = [f"ID:{face_profile.id}"]
    if speaking_level == 'primary':
        parts.append("SPEAKING")

    info_text = " | ".join(parts)
    (text_w, text_h), baseline = cv2.getTextSize(info_text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_x = x
    text_y = y - 8

    cv2.rectangle(frame, (text_x, text_y - text_h - baseline),
                  (text_x + text_w, text_y + baseline), COLOR_TEXT_BG, -1)
    cv2.putText(frame, info_text, (text_x, text_y),
                FONT, FONT_SCALE, COLOR_TEXT_FG, FONT_THICKNESS, cv2.LINE_AA)


class TurnStateDisplay:
    """TurnState 显示管理器，在画面右上角显示状态并保留指定时间"""

    def __init__(self, display_ms: float = TURN_STATE_DISPLAY_MS):
        self._current_state: TurnState = None
        self._expire_time: float = 0
        self._display_ms = display_ms / 1000.0  # 转换为秒

    def set_state(self, state: TurnState):
        """设置当前显示的 TurnState，并设置过期时间"""
        self._current_state = state
        self._expire_time = time.perf_counter() + self._display_ms

    def get_current_state(self) -> TurnState:
        """获取当前应该显示的 TurnState，如果已过期则返回 None"""
        if self._current_state is None:
            return None
        if time.perf_counter() > self._expire_time:
            self._current_state = None
            return None
        return self._current_state


def draw_turn_state_overlay(frame: np.ndarray, turn_state: TurnState):
    """
    在视频帧右上角绘制 TurnState 状态。
    """
    if turn_state is None:
        return

    state_text = turn_state.name
    (text_w, text_h), baseline = cv2.getTextSize(state_text, FONT, FONT_SCALE * 1.2, FONT_THICKNESS + 1)

    # 右上角位置（留 20px 边距）
    frame_h, frame_w = frame.shape[:2]
    text_x = frame_w - text_w - 20
    text_y = text_h + 20

    # 绘制背景
    cv2.rectangle(frame,
                  (text_x - 10, text_y - text_h - baseline - 5),
                  (text_x + text_w + 10, text_y + baseline + 5),
                  COLOR_TEXT_BG, -1)

    # 绘制文字（绿色）
    cv2.putText(frame, state_text, (text_x, text_y),
                FONT, FONT_SCALE * 1.2, COLOR_TURN_STATE, FONT_THICKNESS + 1, cv2.LINE_AA)


def create_asd(args):
    """根据命令行参数创建 ASD 实例"""
    model_dir = args.model_dir or os.path.join(PROJECT_ROOT, "weights")

    face_detector_config = {"type": args.face_detector}

    vad_params = {
        "model_dir": model_dir,
        "prefix_padding_ms": int(EVAL_PREFIX_TRIM_SEC * 1000),
        "abs_amplitude_threshold": args.abs_amplitude_threshold,
        "silence_duration_ms": int(EVAL_SUFFIX_TRIM_SEC * 1000),
    }
    if args.vad_model_name:
        vad_params["model_name"] = args.vad_model_name

    if args.turn_detector == "pvad":
        turn_detector_config = {
            "type": "pvad",
            "model_dir": model_dir,
            "vad": {"type": "silero-vad", **vad_params},
        }
    else:
        turn_detector_config = {"type": args.turn_detector, **vad_params}

    speaker_detector_config = {
        "type": args.speaker_detector,
        "model_dir": model_dir,
    }
    if args.voiceprint_model_name:
        speaker_detector_config["voiceprint_model_name"] = args.voiceprint_model_name

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

    # 2. 获取视频流 start_time 并提取音频（从视频起始时刻开始，保证音画对齐）
    video_start_time = get_video_start_time(args.input)
    if video_start_time > 0:
        print(f"[信息] 视频流 start_time={video_start_time:.3f}s，音频将从此时刻开始提取")

    print(f"[信息] 正在从视频中提取音频...")
    audio_data = extract_audio_from_video(args.input, AUDIO_SAMPLE_RATE,
                                          start_time=video_start_time)

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

    # 4. 启动 ffmpeg 子进程：通过管道接收原始帧，同时从原视频取音频，一步完成编码输出
    #    用 -ss 跳过原视频中视频流起始之前的音频，保证音画对齐
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if video_start_time > 0:
        ffmpeg_cmd += ["-ss", f"{video_start_time:.6f}"]
    ffmpeg_cmd += [
        "-i", args.input,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        args.output,
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 5. 初始化状态追踪
    state_tracker = SpeakingStateTracker()
    turn_state_display = TurnStateDisplay()
    audio_cursor = 0
    confirmed_has_speaker = False  # 本轮 utterance 是否已在 TURN_CONFIRMED 检测到说话人

    if args.display:
        window_name = "DeepTalk-ASD 视频分析"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    process_start = time.perf_counter()

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
                utterance = asd.append_audio(audio_frame, chunk_time)

                # ── TURN_START / TURN_REJECTED：显示状态 ──
                if utterance is not None and utterance.turn_state in (TurnState.TURN_START, TurnState.TURN_REJECTED):
                    turn_state_display.set_state(utterance.turn_state)

                # ── TURN_CONFIRMED：检测说话人，有则持久绿框 ──
                if utterance is not None and utterance.turn_state == TurnState.TURN_CONFIRMED:
                    turn_state_display.set_state(utterance.turn_state)
                    full_duration = utterance.duration_seconds()
                    eval_end = chunk_time
                    eval_start = eval_end - full_duration
                    speaker_scores = asd.evaluate(eval_start, eval_end)
                    if speaker_scores:
                        has_speaker = any(score > 0 for score in speaker_scores.values())
                        if has_speaker:
                            state_tracker.set_persistent_speakers(speaker_scores)
                            confirmed_has_speaker = True
                        else:
                            confirmed_has_speaker = False
                        print(f"  [{video_time:.2f}s] TURN_CONFIRMED 说话者评估: {speaker_scores}")
                    else:
                        confirmed_has_speaker = False

                # ── SPEAKER_CHANGE：pVAD 检测到说话人切换 ──
                elif utterance is not None and utterance.turn_state == TurnState.SPEAKER_CHANGE:
                    turn_state_display.set_state(utterance.turn_state)
                    eval_end = chunk_time
                    eval_start = eval_end - 1.0
                    speaker_scores = asd.evaluate(eval_start, eval_end)
                    if speaker_scores:
                        has_speaker = any(score > 0 for score in speaker_scores.values())
                        if has_speaker:
                            state_tracker.set_persistent_speakers(speaker_scores)
                            confirmed_has_speaker = True
                        else:
                            confirmed_has_speaker = False
                        print(f"  [{video_time:.2f}s] SPEAKER_CHANGE 说话人切换: {speaker_scores}")

                # ── TURN_END：清除持久绿框或补充检测 ──
                elif utterance is not None and utterance.turn_state == TurnState.TURN_END:
                    turn_state_display.set_state(utterance.turn_state)
                    if confirmed_has_speaker:
                        state_tracker.clear_persistent_speakers()
                        print(f"  [{video_time:.2f}s] TURN_END: 清除持久说话人")
                        
                    # 无论此前是否确认过说话人，都在句子结束时对整个 core_duration 进行一次 evaluate。
                    # 这样可以累积足够时长（>0.5s）的音频以提取并更新声纹档案
                    full_duration = utterance.duration_seconds()
                    core_duration = full_duration - EVAL_PREFIX_TRIM_SEC - EVAL_SUFFIX_TRIM_SEC
                    if core_duration >= EVAL_MIN_CORE_SEC:
                        eval_end = chunk_time - EVAL_SUFFIX_TRIM_SEC
                        eval_start = eval_end - core_duration
                    else:
                        eval_end = chunk_time
                        eval_start = eval_end - full_duration
                        
                    speaker_scores = asd.evaluate(eval_start, eval_end)
                    if speaker_scores:
                        if not confirmed_has_speaker:
                            state_tracker.update_speakers(speaker_scores)
                            print(f"  [{video_time:.2f}s] TURN_END 补充检测: {speaker_scores}")
                        else:
                            print(f"  [{video_time:.2f}s] TURN_END 声纹验证和更新: {speaker_scores}")
                            
                    confirmed_has_speaker = False

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

            # ── 绘制标注：主说话人绿框，次说话人蓝框，不说话红框 ──
            if face_profiles:
                for profile in face_profiles:
                    speaking_level = state_tracker.get_speaking_level(profile.id)
                    draw_face_overlay(frame, profile, speaking_level)

            # ── 绘制 TurnState 到右上角 ──
            current_turn_state = turn_state_display.get_current_state()
            if current_turn_state is not None:
                draw_turn_state_overlay(frame, current_turn_state)

            ffmpeg_proc.stdin.write(frame.tobytes())

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
        if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
            ffmpeg_proc.stdin.close()
        if args.display:
            cv2.destroyAllWindows()

        print(f"\n[信息] 正在等待 ffmpeg 编码完成...")
        ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
            stderr_text = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
            print(f"[警告] ffmpeg 编码失败:\n{stderr_text}")
        else:
            print(f"[信息] 视频编码完成")

        total_elapsed = time.perf_counter() - process_start
        print(f"\n[完成] 输出视频: {args.output}")
        print(f"  处理了 {frame_idx} 帧, 总耗时 {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
