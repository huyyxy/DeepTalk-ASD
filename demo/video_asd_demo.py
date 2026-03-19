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
SPEAKING_PERSIST_SEC = 0.5

# 与 Silero VAD 一致：utterance 开头为前置静音、结尾为触发结束的静音
EVAL_PREFIX_TRIM_SEC = 0.1
EVAL_SUFFIX_TRIM_SEC = 0.1
EVAL_MIN_CORE_SEC = 0.6

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
COLOR_SPEAKING = (0, 255, 0)
COLOR_SPEAKING_SECONDARY = (255, 165, 0)
COLOR_NOT_SPEAKING = (0, 0, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT_FG = (255, 255, 255)
COLOR_TURN_STATE = (0, 255, 0)

TURN_STATE_DISPLAY_MS = 300


# ──────────────────────────────────────────────
# 命令行解析
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# 音视频工具函数
# ──────────────────────────────────────────────

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
    """使用 ffmpeg 从视频中提取 PCM int16 单声道音频数据"""
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


def start_ffmpeg_encoder(output_path: str, frame_w: int, frame_h: int,
                         fps: float, input_path: str,
                         video_start_time: float) -> subprocess.Popen:
    """启动 ffmpeg 编码子进程，通过 stdin 接收原始帧，从原视频取音频"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if video_start_time > 0:
        cmd += ["-ss", f"{video_start_time:.6f}"]
    cmd += [
        "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ──────────────────────────────────────────────
# ASD 工厂
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# 说话状态追踪
# ──────────────────────────────────────────────

class SpeakingStateTracker:
    """追踪各人脸的说话状态，区分主说话人 / 次说话人 / 未说话"""

    def __init__(self, persist_sec: float = SPEAKING_PERSIST_SEC):
        self._speaking_faces: dict[int, float] = {}  # track_id -> 最后说话时间戳
        self._persistent_faces: set[int] = set()      # 持久说话人（不超时）
        self._primary_speaker: int = -1                # 分数最高的说话人
        self._persist_sec = persist_sec

    def _update_primary(self, speaker_scores: dict):
        best_tid, best_score = -1, 0
        for tid, score in speaker_scores.items():
            if score > best_score:
                best_tid, best_score = tid, score
        self._primary_speaker = best_tid

    def update_speakers(self, speaker_scores: dict):
        """更新说话者状态（带超时机制）"""
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
        self._persistent_faces.clear()
        self._primary_speaker = -1

    def has_persistent_speakers(self) -> bool:
        return len(self._persistent_faces) > 0

    def get_speaking_level(self, track_id: int) -> str | None:
        """
        返回说话级别：
            'primary'   - 分数最高的说话人
            'secondary' - 其他说话人
            None        - 未说话
        """
        if track_id in self._persistent_faces:
            return 'primary' if track_id == self._primary_speaker else 'secondary'

        now = time.perf_counter()
        last_time = self._speaking_faces.get(track_id)
        if last_time is not None:
            if now - last_time <= self._persist_sec:
                return 'primary' if track_id == self._primary_speaker else 'secondary'
            else:
                del self._speaking_faces[track_id]
        return None


# ──────────────────────────────────────────────
# TurnState 右上角显示
# ──────────────────────────────────────────────

class TurnStateDisplay:
    """管理 TurnState 在画面右上角的显示与自动过期"""

    def __init__(self, display_ms: float = TURN_STATE_DISPLAY_MS):
        self._current_state: TurnState = None
        self._expire_time: float = 0
        self._display_sec = display_ms / 1000.0

    def set_state(self, state: TurnState):
        self._current_state = state
        self._expire_time = time.perf_counter() + self._display_sec

    def get_current_state(self) -> TurnState | None:
        if self._current_state is None:
            return None
        if time.perf_counter() > self._expire_time:
            self._current_state = None
            return None
        return self._current_state


# ──────────────────────────────────────────────
# 画面绘制
# ──────────────────────────────────────────────

def draw_face_overlay(frame: np.ndarray, face_profile, speaking_level: str | None):
    """在帧上绘制人脸框和标注"""
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
    text_x, text_y = x, y - 8
    cv2.rectangle(frame, (text_x, text_y - text_h - baseline),
                  (text_x + text_w, text_y + baseline), COLOR_TEXT_BG, -1)
    cv2.putText(frame, info_text, (text_x, text_y),
                FONT, FONT_SCALE, COLOR_TEXT_FG, FONT_THICKNESS, cv2.LINE_AA)


def draw_turn_state_overlay(frame: np.ndarray, turn_state: TurnState):
    """在帧右上角绘制 TurnState 状态文字"""
    if turn_state is None:
        return

    state_text = turn_state.name
    scale = FONT_SCALE * 1.2
    thickness = FONT_THICKNESS + 1
    (text_w, text_h), baseline = cv2.getTextSize(state_text, FONT, scale, thickness)

    frame_h, frame_w = frame.shape[:2]
    text_x = frame_w - text_w - 20
    text_y = text_h + 20

    cv2.rectangle(frame,
                  (text_x - 10, text_y - text_h - baseline - 5),
                  (text_x + text_w + 10, text_y + baseline + 5),
                  COLOR_TEXT_BG, -1)
    cv2.putText(frame, state_text, (text_x, text_y),
                FONT, scale, COLOR_TURN_STATE, thickness, cv2.LINE_AA)


def draw_annotations(frame: np.ndarray, face_profiles, state_tracker: SpeakingStateTracker,
                     turn_state_display: TurnStateDisplay):
    """在帧上绘制所有标注（人脸框 + TurnState）"""
    if face_profiles:
        for profile in face_profiles:
            speaking_level = state_tracker.get_speaking_level(profile.id)
            draw_face_overlay(frame, profile, speaking_level)

    current_turn_state = turn_state_display.get_current_state()
    if current_turn_state is not None:
        draw_turn_state_overlay(frame, current_turn_state)


# ──────────────────────────────────────────────
# 核心处理流水线
# ──────────────────────────────────────────────

class VideoASDPipeline:
    """
    视频 ASD 处理流水线。

    处理流程:
        1. 初始化 ASD 引擎、提取音频、打开视频
        2. 逐帧循环:
           a. 将音频块喂入 ASD（追赶到当前视频时间）
           b. 根据轮次事件更新说话者状态
           c. 将视频帧喂入 ASD，获取人脸信息
           d. 在帧上绘制标注
           e. 将帧写入输出视频
        3. 关闭资源、等待编码完成
    """

    def __init__(self, args):
        self.args = args
        self.asd = None
        self.audio_data: np.ndarray = None
        self.audio_cursor = 0

        self.state_tracker = SpeakingStateTracker()
        self.turn_state_display = TurnStateDisplay()
        self.confirmed_has_speaker = False

        self.cap: cv2.VideoCapture = None
        self.fps = 0.0
        self.total_frames = 0
        self.frame_w = 0
        self.frame_h = 0
        self.video_start_time = 0.0

        self.ffmpeg_proc: subprocess.Popen = None
        self.process_start = 0.0

    # ── 初始化阶段 ──

    def _init_asd(self):
        self.asd = create_asd(self.args)

    def _extract_audio(self):
        self.video_start_time = get_video_start_time(self.args.input)
        if self.video_start_time > 0:
            print(f"[信息] 视频流 start_time={self.video_start_time:.3f}s，"
                  f"音频将从此时刻开始提取")

        print(f"[信息] 正在从视频中提取音频...")
        self.audio_data = extract_audio_from_video(
            self.args.input, AUDIO_SAMPLE_RATE, start_time=self.video_start_time)

    def _open_video(self):
        self.cap = cv2.VideoCapture(self.args.input)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开视频文件: {self.args.input}")
            sys.exit(1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"[视频] 已打开: {self.args.input}")
        print(f"  分辨率: {self.frame_w}x{self.frame_h}, FPS: {self.fps:.2f}, "
              f"总帧数: {self.total_frames}, 时长: {duration:.2f}s")

    def _start_encoder(self):
        self.ffmpeg_proc = start_ffmpeg_encoder(
            self.args.output, self.frame_w, self.frame_h,
            self.fps, self.args.input, self.video_start_time)

    # ── 音频与轮次事件处理 ──

    def _feed_audio_until(self, video_time: float):
        """将音频块喂入 ASD，追赶到当前视频时间，并处理产生的轮次事件"""
        target_audio_pos = int(video_time * AUDIO_SAMPLE_RATE)

        while self.audio_cursor < target_audio_pos and self.audio_cursor < len(self.audio_data):
            chunk_end = min(self.audio_cursor + AUDIO_CHUNK_SAMPLES, len(self.audio_data))
            chunk = self.audio_data[self.audio_cursor:chunk_end]
            chunk_time = self.process_start + self.audio_cursor / AUDIO_SAMPLE_RATE

            audio_frame = AudioFrame(
                data=chunk.tobytes(),
                sample_rate=AUDIO_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            utterance = self.asd.append_audio(audio_frame, chunk_time)

            if utterance is not None:
                self._handle_turn_event(utterance, chunk_time, video_time)

            self.audio_cursor = chunk_end

    def _handle_turn_event(self, utterance, chunk_time: float, video_time: float):
        """根据轮次事件类型分派到对应的处理方法"""
        state = utterance.turn_state
        self.turn_state_display.set_state(state)

        handlers = {
            TurnState.TURN_CONFIRMED: self._on_turn_confirmed,
            TurnState.SPEAKER_CHANGE: self._on_speaker_change,
            TurnState.TURN_END: self._on_turn_end,
        }

        handler = handlers.get(state)
        if handler:
            handler(utterance, chunk_time, video_time)

    def _on_turn_confirmed(self, utterance, chunk_time: float, video_time: float):
        """TURN_CONFIRMED：对整段语音做说话人检测，有结果则设置持久绿框"""
        eval_end = chunk_time
        eval_start = eval_end - utterance.duration_seconds()
        speaker_scores = self.asd.evaluate(eval_start, eval_end)

        if not speaker_scores:
            self.confirmed_has_speaker = False
            return

        has_speaker = any(score > 0 for score in speaker_scores.values())
        if has_speaker:
            self.state_tracker.set_persistent_speakers(speaker_scores)
            self.confirmed_has_speaker = True
        else:
            self.confirmed_has_speaker = False

        print(f"  [{video_time:.2f}s] TURN_CONFIRMED 说话者评估: {speaker_scores}")

    def _on_speaker_change(self, utterance, chunk_time: float, video_time: float):
        """SPEAKER_CHANGE：pVAD 检测到说话人切换，重新评估最近 1 秒"""
        eval_end = chunk_time
        eval_start = eval_end - 1.0
        speaker_scores = self.asd.evaluate(eval_start, eval_end)

        if not speaker_scores:
            return

        has_speaker = any(score > 0 for score in speaker_scores.values())
        if has_speaker:
            self.state_tracker.set_persistent_speakers(speaker_scores)
            self.confirmed_has_speaker = True
        else:
            self.confirmed_has_speaker = False

        print(f"  [{video_time:.2f}s] SPEAKER_CHANGE 说话人切换: {speaker_scores}")

    def _on_turn_end(self, utterance, chunk_time: float, video_time: float):
        """TURN_END：清除持久绿框，并对完整语段做最终评估以更新声纹"""
        if self.confirmed_has_speaker:
            self.state_tracker.clear_persistent_speakers()
            print(f"  [{video_time:.2f}s] TURN_END: 清除持久说话人")

        eval_start, eval_end = self._compute_turn_end_eval_range(utterance, chunk_time)
        speaker_scores = self.asd.evaluate(eval_start, eval_end)

        if speaker_scores:
            if not self.confirmed_has_speaker:
                self.state_tracker.update_speakers(speaker_scores)
                print(f"  [{video_time:.2f}s] TURN_END 补充检测: {speaker_scores}")
            else:
                print(f"  [{video_time:.2f}s] TURN_END 声纹验证和更新: {speaker_scores}")

        self.confirmed_has_speaker = False

    def _compute_turn_end_eval_range(self, utterance, chunk_time: float) -> tuple[float, float]:
        """计算 TURN_END 时的评估时间范围，尽量去掉前后静音段"""
        full_duration = utterance.duration_seconds()
        core_duration = full_duration - EVAL_PREFIX_TRIM_SEC - EVAL_SUFFIX_TRIM_SEC

        if core_duration >= EVAL_MIN_CORE_SEC:
            eval_end = chunk_time - EVAL_SUFFIX_TRIM_SEC
            eval_start = eval_end - core_duration
        else:
            eval_end = chunk_time
            eval_start = eval_end - full_duration

        return eval_start, eval_end

    # ── 视频帧处理 ──

    def _process_video_frame(self, frame: np.ndarray, create_time: float) -> list:
        """将视频帧送入 ASD 并返回检测到的人脸列表"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        video_frame = VideoFrame(
            width=w, height=h,
            type=VideoBufferType.RGB24,
            data=bytes(frame_rgb),
        )
        return self.asd.append_video(video_frame, create_time)

    # ── 进度显示 ──

    def _print_progress(self, frame_idx: int):
        if frame_idx % 100 == 0 or frame_idx == self.total_frames:
            elapsed = time.perf_counter() - self.process_start
            progress = frame_idx / self.total_frames * 100 if self.total_frames > 0 else 0
            speed = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  进度: {frame_idx}/{self.total_frames} ({progress:.1f}%), "
                  f"速度: {speed:.1f} fps, 已耗时: {elapsed:.1f}s")

    # ── 资源清理 ──

    def _cleanup(self, frame_idx: int):
        self.cap.release()

        if self.ffmpeg_proc.stdin and not self.ffmpeg_proc.stdin.closed:
            self.ffmpeg_proc.stdin.close()

        if self.args.display:
            cv2.destroyAllWindows()

        print(f"\n[信息] 正在等待 ffmpeg 编码完成...")
        self.ffmpeg_proc.wait()

        if self.ffmpeg_proc.returncode != 0:
            stderr_text = self.ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
            print(f"[警告] ffmpeg 编码失败:\n{stderr_text}")
        else:
            print(f"[信息] 视频编码完成")

        total_elapsed = time.perf_counter() - self.process_start
        print(f"\n[完成] 输出视频: {self.args.output}")
        print(f"  处理了 {frame_idx} 帧, 总耗时 {total_elapsed:.1f}s")

    # ── 主循环 ──

    def _process_loop(self):
        """逐帧处理循环"""
        if self.args.display:
            window_name = "DeepTalk-ASD 视频分析"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        frame_idx = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                video_time = frame_idx / self.fps
                create_time = self.process_start + video_time

                self._feed_audio_until(video_time)
                face_profiles = self._process_video_frame(frame, create_time)
                draw_annotations(frame, face_profiles, self.state_tracker, self.turn_state_display)

                self.ffmpeg_proc.stdin.write(frame.tobytes())

                frame_idx += 1
                self._print_progress(frame_idx)

                if self.args.display:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("[信息] 用户中断")
                        break

        except KeyboardInterrupt:
            print("\n[信息] 检测到 Ctrl+C，正在退出...")
        finally:
            self._cleanup(frame_idx)

    # ── 公开入口 ──

    def run(self):
        """
        执行完整的视频 ASD 处理流水线:
            1. 创建 ASD 引擎
            2. 提取音频
            3. 打开视频
            4. 启动输出编码器
            5. 逐帧处理
        """
        self._init_asd()
        self._extract_audio()
        self._open_video()
        self._start_encoder()

        self.process_start = time.perf_counter()
        print(f"\n[信息] 开始处理视频...")
        self._process_loop()


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[错误] 输入视频文件不存在: {args.input}")
        sys.exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_asd_result.mp4"

    pipeline = VideoASDPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
