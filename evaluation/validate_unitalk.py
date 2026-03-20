#!/usr/bin/env python3
"""
validate_unitalk.py — 使用 UniTalk 中文数据集验证 DeepTalk-ASD 全流水线性能

功能:
1. 逐场景处理视频：音视频精确对齐 → 同步喂入 ASD → 收集语音段和逐帧检测结果
2. 将检测结果与 UniTalk 标注对比：通过 bbox IoU 匹配 track_id 到 entity_id
3. 计算 per-frame ASD 指标（准确率、精确率、召回率、F1）
4. 输出"哪个人脸说了哪段话"的详细结果
5. 生成 markdown 格式验证报告

用法:
    python3 demo/validate_unitalk.py --split val
    python3 demo/validate_unitalk.py --split val --max-videos 5
    python3 demo/validate_unitalk.py --split train --output-report train_report.md
"""

import os
import sys
import time
import bisect
import argparse
import subprocess
import csv
import re
import traceback
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from deeptalk_asd import (
    ASDDetectorFactory,
    VideoFrame,
    VideoBufferType,
    AudioFrame,
    TurnState,
)

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_MS = 30
AUDIO_CHUNK_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)

IOU_MATCH_THRESHOLD = 0.3
FRAME_TIME_TOLERANCE = 0.2  # ±1 frame at 25 fps default is 0.04，Sweet Spot is 0.2


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class Annotation:
    video_id: str
    frame_timestamp: float
    entity_id: str
    bbox: Tuple[float, float, float, float]  # normalised (x1, y1, x2, y2)
    label: str  # SPEAKING_AUDIBLE | NOT_SPEAKING


@dataclass
class SpeechSegment:
    """A detected speech turn — records *who* said *what* and *when*."""
    video_id: str
    scene_name: str
    track_id: int
    start_time: float       # scene-local
    end_time: float          # scene-local
    original_start: float    # in original video
    original_end: float      # in original video
    confidence: float
    matched_entity_id: Optional[str] = None


@dataclass
class FrameDetection:
    """Per-face detection within a single frame."""
    track_id: int
    bbox_norm: Tuple[float, float, float, float]
    is_speaking: bool
    confidence: float


@dataclass
class FrameRecord:
    """All detections for a single frame."""
    frame_idx: int
    scene_time: float
    original_time: float
    detections: List[FrameDetection] = field(default_factory=list)


@dataclass
class SceneResult:
    video_id: str
    scene_name: str
    video_path: str
    scene_start: float
    scene_end: float
    duration: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    speech_segments: List[SpeechSegment] = field(default_factory=list)
    frame_records: List[FrameRecord] = field(default_factory=list)
    processing_time: float = 0.0
    av_offset: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    # ASD metrics — only on IoU-matched faces
    asd_tp: int = 0
    asd_fp: int = 0
    asd_tn: int = 0
    asd_fn: int = 0
    # End-to-end metrics — all annotations
    e2e_tp: int = 0
    e2e_fp: int = 0
    e2e_tn: int = 0
    e2e_fn: int = 0
    # Coverage
    total_annotations: int = 0
    matched_annotations: int = 0
    total_speaking: int = 0
    total_not_speaking: int = 0

    def _safe_div(self, a, b):
        return a / b if b > 0 else 0.0

    def asd_accuracy(self):
        t = self.asd_tp + self.asd_fp + self.asd_tn + self.asd_fn
        return self._safe_div(self.asd_tp + self.asd_tn, t)

    def asd_precision(self):
        return self._safe_div(self.asd_tp, self.asd_tp + self.asd_fp)

    def asd_recall(self):
        return self._safe_div(self.asd_tp, self.asd_tp + self.asd_fn)

    def asd_f1(self):
        p, r = self.asd_precision(), self.asd_recall()
        return self._safe_div(2 * p * r, p + r)

    def e2e_accuracy(self):
        t = self.e2e_tp + self.e2e_fp + self.e2e_tn + self.e2e_fn
        return self._safe_div(self.e2e_tp + self.e2e_tn, t)

    def e2e_precision(self):
        return self._safe_div(self.e2e_tp, self.e2e_tp + self.e2e_fp)

    def e2e_recall(self):
        return self._safe_div(self.e2e_tp, self.e2e_tp + self.e2e_fn)

    def e2e_f1(self):
        p, r = self.e2e_precision(), self.e2e_recall()
        return self._safe_div(2 * p * r, p + r)

    def face_match_rate(self):
        return self._safe_div(self.matched_annotations, self.total_annotations)

    def accumulate(self, other: "EvalMetrics"):
        for f in (
            "asd_tp", "asd_fp", "asd_tn", "asd_fn",
            "e2e_tp", "e2e_fp", "e2e_tn", "e2e_fn",
            "total_annotations", "matched_annotations",
            "total_speaking", "total_not_speaking",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))


# ──────────────────────────────────────────────
# Annotation Parser
# ──────────────────────────────────────────────

class AnnotationParser:
    def __init__(self, csv_path: str):
        self.annotations: Dict[str, List[Annotation]] = defaultdict(list)
        self._load(csv_path)

    def _load(self, csv_path: str):
        if not os.path.exists(csv_path):
            print(f"[警告] CSV 文件不存在: {csv_path}")
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ann = Annotation(
                        video_id=row["video_id"],
                        frame_timestamp=float(row["frame_timestamp"]),
                        entity_id=row["entity_id"],
                        bbox=(
                            float(row["entity_box_x1"]),
                            float(row["entity_box_y1"]),
                            float(row["entity_box_x2"]),
                            float(row["entity_box_y2"]),
                        ),
                        label=row["label"],
                    )
                    self.annotations[ann.video_id].append(ann)
                except (KeyError, ValueError):
                    pass
        for vid in self.annotations:
            self.annotations[vid].sort(key=lambda a: a.frame_timestamp)
        total = sum(len(v) for v in self.annotations.values())
        print(f"[标注] 加载完成: {len(self.annotations)} 个视频, {total} 条标注")

    def get_in_range(self, video_id: str, start: float, end: float) -> List[Annotation]:
        return [
            a
            for a in self.annotations.get(video_id, [])
            if start <= a.frame_timestamp <= end
        ]


# ──────────────────────────────────────────────
# Audio-Video Alignment
# ──────────────────────────────────────────────

class AVAligner:
    """Handles the A/V stream start-time offset that commonly appears in MP4 containers."""

    @staticmethod
    def _probe_start(video_path: str, stream: str) -> float:
        sel = "v:0" if stream == "v" else "a:0"
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", sel,
            "-show_entries", "stream=start_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        s = r.stdout.strip()
        return float(s) if s and s != "N/A" else 0.0

    @staticmethod
    def extract_aligned_audio(
        video_path: str, sr: int = AUDIO_SAMPLE_RATE
    ) -> Tuple[np.ndarray, float]:
        """Return (audio_data, av_offset) where audio_data[0] aligns with video frame 0."""
        audio_start = AVAligner._probe_start(video_path, "a")
        video_start = AVAligner._probe_start(video_path, "v")
        offset = video_start - audio_start  # >0 means video starts later

        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
            "-f", "s16le", "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 音频提取失败: {proc.stderr.decode()[:200]}")
        raw = np.frombuffer(proc.stdout, dtype=np.int16)

        shift = int(abs(offset) * sr)
        if offset > 0.001:
            aligned = raw[shift:] if shift < len(raw) else raw
        elif offset < -0.001:
            aligned = np.concatenate([np.zeros(shift, dtype=np.int16), raw])
        else:
            aligned = raw

        return aligned, offset


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def parse_scene_filename(name: str) -> Tuple[float, float]:
    """Extract (start, end) from *_scene_NNN_start-end.mp4.

    These times are the BUFFERED cut times, meaning scene-time 0 corresponds
    to original-time ``start``.
    """
    m = re.search(r"_scene_\d+_(\d+\.?\d*)-(\d+\.?\d*)\.mp4$", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def bbox_iou(a: Tuple, b: Tuple) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────
# Scene Processor
# ──────────────────────────────────────────────

class SceneProcessor:
    """Feed a single scene video through the full DeepTalk-ASD pipeline.

    For every frame, the processor records which faces are detected and whether
    they are currently in "speaking" state (based on turn events from VAD).
    Speech segments are recorded at TURN_END / SPEAKER_CHANGE boundaries.
    """

    def __init__(self, args):
        self.args = args

    def _create_asd(self):
        model_dir = os.path.join(PROJECT_ROOT, "weights")
        face_cfg = {"type": "inspireface"}
        vad_params = {
            "model_dir": model_dir,
            "prefix_padding_ms": 300,
            "min_voice_duration_ms": 200,
            "silence_duration_ms": 500,
            "abs_amplitude_threshold": 0.01,
        }
        turn_cfg = {"type": self.args.turn_detector, **vad_params}
        speaker_cfg = {"type": self.args.speaker_detector, "model_dir": model_dir}

        factory = ASDDetectorFactory(
            face_detector=face_cfg,
            turn_detector=turn_cfg,
            speaker_detector=speaker_cfg,
        )
        asd = factory.create()
        if asd is None:
            raise RuntimeError("ASD 实例创建失败")
        return asd

    # ── public entry ──

    def process(self, video_path: str, video_id: str, scene_name: str) -> SceneResult:
        scene_start, scene_end = parse_scene_filename(scene_name)
        result = SceneResult(
            video_id=video_id,
            scene_name=scene_name,
            video_path=video_path,
            scene_start=scene_start,
            scene_end=scene_end,
        )

        t0 = time.perf_counter()
        try:
            self._run(result)
        except Exception as e:
            result.error = str(e)
            traceback.print_exc()
        result.processing_time = time.perf_counter() - t0

        n_seg = len(result.speech_segments)
        if result.error is None:
            print(
                f"  [完成] 耗时 {result.processing_time:.1f}s, "
                f"{n_seg} 个语音段, {result.total_frames} 帧"
            )
        return result

    # ── core loop ──

    def _run(self, result: SceneResult):
        asd = self._create_asd()

        # 1. Extract aligned audio
        audio_data, av_offset = AVAligner.extract_aligned_audio(result.video_path)
        result.av_offset = av_offset
        print(
            f"  [音频] {len(audio_data)} 采样 "
            f"({len(audio_data) / AUDIO_SAMPLE_RATE:.2f}s), "
            f"AV 偏移 {av_offset:.3f}s"
        )

        # 2. Open video
        cap = cv2.VideoCapture(result.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {result.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.fps = fps
        result.total_frames = total_frames
        result.duration = total_frames / fps if fps > 0 else 0
        print(
            f"  [视频] {frame_w}x{frame_h}, {fps:.1f} fps, "
            f"{total_frames} 帧, {result.duration:.2f}s"
        )

        # 3. Frame-by-frame processing
        process_start = time.perf_counter()
        audio_cursor = 0
        active_speakers: Dict[int, float] = {}
        turn_start_scene: Optional[float] = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps
            create_time = process_start + video_time

            # ── Feed audio up to current video time ──
            target_pos = int(video_time * AUDIO_SAMPLE_RATE)
            while audio_cursor < target_pos and audio_cursor < len(audio_data):
                chunk_end = min(audio_cursor + AUDIO_CHUNK_SAMPLES, len(audio_data))
                chunk = audio_data[audio_cursor:chunk_end]
                chunk_time = process_start + audio_cursor / AUDIO_SAMPLE_RATE
                audio_scene_t = audio_cursor / AUDIO_SAMPLE_RATE

                audio_frame = AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=AUDIO_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                utterance = asd.append_audio(audio_frame, chunk_time)

                self._handle_turn(
                    utterance, asd, result,
                    chunk_time, audio_scene_t,
                    active_speakers, turn_start_scene,
                )
                # _handle_turn may mutate active_speakers / turn_start_scene
                # via the mutable dict; turn_start_scene is returned
                turn_start_scene = self._last_turn_start  # see _handle_turn

                audio_cursor = chunk_end

            # ── Feed video frame ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            vf = VideoFrame(
                width=w, height=h,
                type=VideoBufferType.RGB24,
                data=bytes(frame_rgb),
            )
            face_profiles = asd.append_video(vf, create_time)

            # Record per-frame detections
            record = FrameRecord(
                frame_idx=frame_idx,
                scene_time=video_time,
                original_time=result.scene_start + video_time,
            )
            if face_profiles:
                for fp in face_profiles:
                    rect = fp.face_rectangle
                    nx1 = rect.x / w
                    ny1 = rect.y / h
                    nx2 = (rect.x + rect.width) / w
                    ny2 = (rect.y + rect.height) / h
                    is_spk = fp.id in active_speakers
                    conf = active_speakers.get(fp.id, 0.0)
                    record.detections.append(
                        FrameDetection(
                            track_id=fp.id,
                            bbox_norm=(nx1, ny1, nx2, ny2),
                            is_speaking=is_spk,
                            confidence=conf,
                        )
                    )
            result.frame_records.append(record)

            frame_idx += 1
            if frame_idx % 200 == 0:
                pct = frame_idx / total_frames * 100
                print(
                    f"    进度: {frame_idx}/{total_frames} ({pct:.0f}%), "
                    f"语音段 {len(result.speech_segments)}"
                )

        cap.release()
        asd.reset()

    # ── turn event handling ──

    _last_turn_start: Optional[float] = None

    @staticmethod
    def _pick_top_speaker(scores: Dict[int, float]) -> Dict[int, float]:
        """Return only the track with the highest positive score.

        Standard ASD evaluation assumes one active speaker per turn.
        Secondary positive scores are usually lip-movement noise.
        """
        if not scores:
            return {}
        best_tid = max(scores, key=scores.get)
        if scores[best_tid] > 0:
            return {best_tid: scores[best_tid]}
        return {}

    def _handle_turn(
        self,
        utterance,
        asd,
        result: SceneResult,
        chunk_time: float,
        audio_scene_t: float,
        active_speakers: Dict[int, float],
        turn_start_scene: Optional[float],
    ):
        """Dispatch turn events and update speaking state / speech segments."""
        self._last_turn_start = turn_start_scene
        state = utterance.turn_state

        if state == TurnState.TURN_CONFIRMED:
            eval_end = chunk_time
            eval_start = eval_end - utterance.duration_seconds()
            scores = asd.evaluate(eval_start, eval_end)
            if scores:
                active_speakers.clear()
                active_speakers.update(self._pick_top_speaker(scores))
                if active_speakers and self._last_turn_start is None:
                    self._last_turn_start = max(
                        0, audio_scene_t - utterance.duration_seconds()
                    )

        elif state == TurnState.SPEAKER_CHANGE:
            eval_end = chunk_time
            eval_start = eval_end - 1.0
            scores = asd.evaluate(eval_start, eval_end)
            if scores:
                old_ids = set(active_speakers.keys())
                new_active = self._pick_top_speaker(scores)
                for tid in old_ids - set(new_active.keys()):
                    self._record_segment(
                        result, tid,
                        self._last_turn_start, audio_scene_t,
                        active_speakers.get(tid, 0),
                    )
                active_speakers.clear()
                active_speakers.update(new_active)
                if active_speakers:
                    self._last_turn_start = audio_scene_t

        elif state == TurnState.TURN_END:
            eval_end = chunk_time
            eval_start = eval_end - utterance.duration_seconds()
            scores = asd.evaluate(eval_start, eval_end)
            seg_start = self._last_turn_start
            if seg_start is None:
                seg_start = max(0, audio_scene_t - utterance.duration_seconds())
            if scores:
                top = self._pick_top_speaker(scores)
                for tid, s in top.items():
                    self._record_segment(
                        result, tid, seg_start, audio_scene_t, s
                    )
            active_speakers.clear()
            self._last_turn_start = None

        elif state == TurnState.TURN_REJECTED:
            active_speakers.clear()
            self._last_turn_start = None

    def _record_segment(
        self,
        result: SceneResult,
        track_id: int,
        start_scene: Optional[float],
        end_scene: float,
        confidence: float,
    ):
        if start_scene is None:
            start_scene = max(0, end_scene - 0.5)
        result.speech_segments.append(
            SpeechSegment(
                video_id=result.video_id,
                scene_name=result.scene_name,
                track_id=track_id,
                start_time=start_scene,
                end_time=end_scene,
                original_start=result.scene_start + start_scene,
                original_end=result.scene_start + end_scene,
                confidence=confidence,
            )
        )


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────

class Evaluator:
    """Match ASD detections to UniTalk annotations and compute metrics."""

    def __init__(self, parser: AnnotationParser):
        self.parser = parser

    def evaluate_scene(self, result: SceneResult) -> EvalMetrics:
        if result.error:
            return EvalMetrics()

        gt = self.parser.get_in_range(
            result.video_id, result.scene_start, result.scene_end
        )
        if not gt:
            return EvalMetrics()

        frame_lookup: Dict[float, FrameRecord] = {
            r.scene_time: r for r in result.frame_records
        }
        frame_times = sorted(frame_lookup.keys())

        metrics = EvalMetrics()

        for ann in gt:
            metrics.total_annotations += 1
            is_spk_gt = ann.label == "SPEAKING_AUDIBLE"
            if is_spk_gt:
                metrics.total_speaking += 1
            else:
                metrics.total_not_speaking += 1

            scene_t = ann.frame_timestamp - result.scene_start
            closest = self._closest_frame(scene_t, frame_times, frame_lookup)
            if closest is None:
                # No matching frame — treat as undetected
                if is_spk_gt:
                    metrics.e2e_fn += 1
                else:
                    metrics.e2e_tn += 1
                continue

            best_det, best_iou = self._match_face(ann.bbox, closest.detections)

            if best_det is None or best_iou < IOU_MATCH_THRESHOLD:
                if is_spk_gt:
                    metrics.e2e_fn += 1
                else:
                    metrics.e2e_tn += 1
                continue

            metrics.matched_annotations += 1

            if is_spk_gt:
                if best_det.is_speaking:
                    metrics.asd_tp += 1
                    metrics.e2e_tp += 1
                else:
                    metrics.asd_fn += 1
                    metrics.e2e_fn += 1
            else:
                if best_det.is_speaking:
                    metrics.asd_fp += 1
                    metrics.e2e_fp += 1
                else:
                    metrics.asd_tn += 1
                    metrics.e2e_tn += 1

        return metrics

    def match_segments_to_entities(self, result: SceneResult):
        """Assign the best-matching entity_id to each speech segment."""
        if not result.frame_records:
            return
        gt = self.parser.get_in_range(
            result.video_id, result.scene_start, result.scene_end
        )
        if not gt:
            return

        frame_lookup = {r.scene_time: r for r in result.frame_records}
        frame_times = sorted(frame_lookup.keys())

        # Accumulate IoU votes: track_id → {entity_id → total_iou}
        votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for ann in gt:
            scene_t = ann.frame_timestamp - result.scene_start
            rec = self._closest_frame(scene_t, frame_times, frame_lookup)
            if rec is None:
                continue
            det, iou = self._match_face(ann.bbox, rec.detections)
            if det is not None and iou >= IOU_MATCH_THRESHOLD:
                votes[det.track_id][ann.entity_id] += iou

        mapping: Dict[int, str] = {}
        for tid, ent_votes in votes.items():
            mapping[tid] = max(ent_votes, key=ent_votes.get)

        for seg in result.speech_segments:
            seg.matched_entity_id = mapping.get(seg.track_id)

    # ── helpers ──

    @staticmethod
    def _closest_frame(
        t: float, times: List[float], lookup: Dict[float, FrameRecord]
    ) -> Optional[FrameRecord]:
        if not times:
            return None
        idx = bisect.bisect_left(times, t)
        best_t, best_d = None, float("inf")
        for i in (idx - 1, idx):
            if 0 <= i < len(times):
                d = abs(times[i] - t)
                if d < best_d:
                    best_d = d
                    best_t = times[i]
        if best_t is not None and best_d <= FRAME_TIME_TOLERANCE:
            return lookup[best_t]
        return None

    @staticmethod
    def _match_face(
        ann_bbox: Tuple, detections: List[FrameDetection]
    ) -> Tuple[Optional[FrameDetection], float]:
        best_det, best_iou = None, 0.0
        for det in detections:
            iou = bbox_iou(ann_bbox, det.bbox_norm)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        return best_det, best_iou


# ──────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────

class ReportGenerator:
    def __init__(self, args):
        self.args = args

    def generate(
        self,
        results: List[SceneResult],
        metrics: EvalMetrics,
        per_video: Dict[str, EvalMetrics],
    ) -> str:
        L = []

        # ── Title ──
        L.append("# DeepTalk-ASD UniTalk 中文数据集验证报告\n")
        L.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ── 1. Configuration ──
        L.append("## 1. 验证配置\n")
        L.append("| 配置项 | 值 |")
        L.append("|--------|----|")
        L.append(f"| 数据集分割 | {self.args.split} |")
        L.append(f"| 处理场景数 | {len(results)} |")
        L.append(f"| 人脸检测器 | inspireface |")
        L.append(f"| 轮次检测器 | {self.args.turn_detector} |")
        L.append(f"| 说话者检测器 | {self.args.speaker_detector} |")
        L.append(f"| IoU 匹配阈值 | {IOU_MATCH_THRESHOLD} |")
        L.append(f"| 帧时间容差 | {FRAME_TIME_TOLERANCE}s |")
        L.append("")

        # ── 2. Overall Metrics ──
        L.append("## 2. 总体性能指标\n")
        L.append("### 2.1 ASD 性能（仅在人脸匹配成功的标注上评估）\n")
        L.append("| 指标 | 数值 |")
        L.append("|------|------|")
        L.append(f"| 评估标注总数 | {metrics.total_annotations} |")
        L.append(
            f"| 人脸匹配成功 | {metrics.matched_annotations} "
            f"({metrics.face_match_rate() * 100:.1f}%) |"
        )
        L.append(f"| 说话标注 / 不说话标注 | {metrics.total_speaking} / {metrics.total_not_speaking} |")
        L.append(f"| **准确率 (Accuracy)** | **{metrics.asd_accuracy():.4f}** |")
        L.append(f"| **精确率 (Precision)** | **{metrics.asd_precision():.4f}** |")
        L.append(f"| **召回率 (Recall)** | **{metrics.asd_recall():.4f}** |")
        L.append(f"| **F1 分数** | **{metrics.asd_f1():.4f}** |")
        L.append("")

        L.append("### 2.2 端到端性能（含人脸检测缺失）\n")
        L.append("| 指标 | 数值 |")
        L.append("|------|------|")
        L.append(f"| **准确率** | **{metrics.e2e_accuracy():.4f}** |")
        L.append(f"| **精确率** | **{metrics.e2e_precision():.4f}** |")
        L.append(f"| **召回率** | **{metrics.e2e_recall():.4f}** |")
        L.append(f"| **F1 分数** | **{metrics.e2e_f1():.4f}** |")
        L.append("")

        # ── 2.3 Confusion Matrix ──
        L.append("### 2.3 混淆矩阵\n")
        L.append("**ASD（人脸匹配成功的标注）:**\n")
        L.append("```")
        L.append(f"                 预测:说话  预测:不说话")
        L.append(f"实际:说话      {metrics.asd_tp:8d}    {metrics.asd_fn:8d}")
        L.append(f"实际:不说话    {metrics.asd_fp:8d}    {metrics.asd_tn:8d}")
        L.append("```\n")
        L.append("**端到端（全部标注）:**\n")
        L.append("```")
        L.append(f"                 预测:说话  预测:不说话")
        L.append(f"实际:说话      {metrics.e2e_tp:8d}    {metrics.e2e_fn:8d}")
        L.append(f"实际:不说话    {metrics.e2e_fp:8d}    {metrics.e2e_tn:8d}")
        L.append("```\n")

        # ── 3. Speech Segments ──
        L.append("## 3. 语音段检测结果（哪个人脸说了哪段话）\n")
        by_video: Dict[str, List[SpeechSegment]] = defaultdict(list)
        for sr in results:
            for seg in sr.speech_segments:
                by_video[sr.video_id].append(seg)

        if not by_video:
            L.append("*未检测到任何语音段。*\n")
        else:
            for vid in sorted(by_video.keys()):
                segs = sorted(by_video[vid], key=lambda s: s.original_start)
                L.append(f"### 视频: {vid}\n")
                L.append(f"共检测到 **{len(segs)}** 个语音段：\n")
                L.append(
                    "| # | 场景 | 原始视频时间 | 时长 | Track ID | "
                    "匹配 Entity | 置信度 |"
                )
                L.append("|---|------|-------------|------|----------|------------|--------|")
                for i, seg in enumerate(segs, 1):
                    ent = seg.matched_entity_id or "-"
                    dur = seg.original_end - seg.original_start
                    L.append(
                        f"| {i} | {seg.scene_name} | "
                        f"{seg.original_start:.2f}s – {seg.original_end:.2f}s | "
                        f"{dur:.2f}s | {seg.track_id} | {ent} | "
                        f"{seg.confidence:.3f} |"
                    )
                L.append("")

        # ── 4. Per-video Metrics ──
        L.append("## 4. 各视频详细指标\n")
        if per_video:
            L.append(
                "| 视频 ID | 标注数 | 匹配数 | 匹配率 | "
                "ASD 准确率 | ASD 精确率 | ASD 召回率 | ASD F1 |"
            )
            L.append(
                "|---------|--------|--------|--------|"
                "-----------|-----------|-----------|--------|"
            )
            for vid in sorted(per_video.keys()):
                m = per_video[vid]
                L.append(
                    f"| {vid} | {m.total_annotations} | "
                    f"{m.matched_annotations} | "
                    f"{m.face_match_rate() * 100:.1f}% | "
                    f"{m.asd_accuracy():.4f} | {m.asd_precision():.4f} | "
                    f"{m.asd_recall():.4f} | {m.asd_f1():.4f} |"
                )
            L.append("")

        # ── 5. Scene Processing Details ──
        L.append("## 5. 场景处理统计\n")
        L.append(
            "| 场景 | 视频 ID | 时长 | FPS | 帧数 | 语音段 | "
            "处理耗时 | AV 偏移 | 状态 |"
        )
        L.append(
            "|------|---------|------|-----|------|--------|"
            "---------|---------|------|"
        )
        for sr in results:
            status = "成功" if sr.error is None else f"失败: {sr.error[:30]}"
            L.append(
                f"| {sr.scene_name} | {sr.video_id} | "
                f"{sr.duration:.2f}s | {sr.fps:.1f} | {sr.total_frames} | "
                f"{len(sr.speech_segments)} | {sr.processing_time:.1f}s | "
                f"{sr.av_offset:.3f}s | {status} |"
            )
        L.append("")

        # ── 6. Summary ──
        total_time = sum(sr.processing_time for sr in results)
        total_vid = sum(sr.duration for sr in results)
        total_segs = sum(len(sr.speech_segments) for sr in results)
        ok = sum(1 for sr in results if sr.error is None)
        fail = len(results) - ok
        speed = total_vid / total_time if total_time > 0 else 0

        L.append("## 6. 总结\n")
        L.append(f"- 处理场景: {len(results)} 个（成功 {ok}，失败 {fail}）")
        L.append(f"- 视频总时长: {total_vid:.1f}s")
        L.append(f"- 处理总耗时: {total_time:.1f}s（速度比: {speed:.2f}x）")
        L.append(f"- 检测到语音段: {total_segs} 个")
        L.append(
            f"- 标注: 说话帧 {metrics.total_speaking}，"
            f"不说话帧 {metrics.total_not_speaking}"
        )
        L.append(f"- 人脸匹配率: {metrics.face_match_rate() * 100:.1f}%")
        L.append("")

        L.append("> **注意**: 由于 VAD 存在 ~200ms 确认延迟和 ~500ms 尾部静音容忍，")
        L.append("> 语音段的起止时间与标注会有少量系统性偏差，这是 turn-based 检测的固有特性。")
        L.append("")

        f1 = metrics.asd_f1()
        if f1 >= 0.8:
            verdict = "ASD F1 >= 0.8，模型在 UniTalk 中文数据集上表现 **优秀**。"
        elif f1 >= 0.6:
            verdict = "ASD F1 >= 0.6，模型表现 **良好**，仍有改进空间。"
        elif f1 > 0:
            verdict = "ASD F1 < 0.6，模型需要 **进一步优化**。"
        else:
            verdict = "无法计算有效的 F1 分数，请检查数据和配置。"
        L.append(f"**结论**: {verdict}\n")

        L.append("\n---\n*本报告由 DeepTalk-ASD 验证系统自动生成*\n")
        return "\n".join(L)

    def save(self, report: str, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[报告] 已保存到: {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepTalk-ASD UniTalk 数据集验证")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJECT_ROOT, "UniTalk_cn/data/videos/scenes"),
    )
    p.add_argument(
        "--csv-dir", default=os.path.join(PROJECT_ROOT, "UniTalk_cn/csv")
    )
    p.add_argument("--output-report", default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--turn-detector", default="silero-vad")
    p.add_argument("--speaker-detector", default="LR-ASD-ONNX")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_report is None:
        args.output_report = f"validation_report_{args.split}.md"

    print("=" * 70)
    print("DeepTalk-ASD UniTalk 数据集验证")
    print("=" * 70)

    # 1. Load annotations
    csv_path = os.path.join(args.csv_dir, f"{args.split}_orig.csv")
    print(f"\n[步骤 1] 加载 CSV 标注: {csv_path}")
    annotations = AnnotationParser(csv_path)

    # 2. Scan scene videos
    scenes_dir = os.path.join(args.dataset_dir, args.split)
    if not os.path.exists(scenes_dir):
        print(f"[错误] 场景目录不存在: {scenes_dir}")
        return

    print(f"\n[步骤 2] 扫描视频: {scenes_dir}")
    video_files: List[Tuple[str, str, str]] = []
    for video_id in sorted(os.listdir(scenes_dir)):
        video_dir = os.path.join(scenes_dir, video_id)
        if not os.path.isdir(video_dir):
            continue
        for scene_file in sorted(os.listdir(video_dir)):
            if scene_file.endswith(".mp4"):
                video_files.append(
                    (video_id, scene_file, os.path.join(video_dir, scene_file))
                )
    print(f"  找到 {len(video_files)} 个场景视频")

    if args.max_videos:
        video_files = video_files[: args.max_videos]
        print(f"  限制处理前 {args.max_videos} 个")

    # 3. Process scenes
    print(f"\n[步骤 3] 批量处理")
    processor = SceneProcessor(args)
    scene_results: List[SceneResult] = []

    for idx, (vid, sname, spath) in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] {vid}/{sname}")
        result = processor.process(spath, vid, sname)
        scene_results.append(result)

    # 4. Evaluate
    print(f"\n[步骤 4] 评估检测结果")
    evaluator = Evaluator(annotations)

    for sr in scene_results:
        if sr.error is None:
            evaluator.match_segments_to_entities(sr)

    overall = EvalMetrics()
    per_video: Dict[str, EvalMetrics] = {}

    for sr in scene_results:
        if sr.error is not None:
            continue
        sm = evaluator.evaluate_scene(sr)
        vid = sr.video_id
        if vid not in per_video:
            per_video[vid] = EvalMetrics()
        per_video[vid].accumulate(sm)
        overall.accumulate(sm)

    # 5. Generate report
    print(f"\n[步骤 5] 生成验证报告")
    rg = ReportGenerator(args)
    report = rg.generate(scene_results, overall, per_video)
    rg.save(report, args.output_report)

    # 6. Print summary
    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)
    m = overall
    print(
        f"标注: {m.total_annotations} 条 "
        f"(说话 {m.total_speaking}, 不说话 {m.total_not_speaking})"
    )
    print(
        f"人脸匹配: {m.matched_annotations}/{m.total_annotations} "
        f"({m.face_match_rate() * 100:.1f}%)"
    )
    print(
        f"ASD  — Acc {m.asd_accuracy():.4f}, "
        f"Prec {m.asd_precision():.4f}, "
        f"Recall {m.asd_recall():.4f}, "
        f"F1 {m.asd_f1():.4f}"
    )
    print(
        f"E2E  — Acc {m.e2e_accuracy():.4f}, "
        f"Prec {m.e2e_precision():.4f}, "
        f"Recall {m.e2e_recall():.4f}, "
        f"F1 {m.e2e_f1():.4f}"
    )
    print(f"报告已保存: {args.output_report}")


if __name__ == "__main__":
    main()
