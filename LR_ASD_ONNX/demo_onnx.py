"""
LR-ASD ONNX 推理示例

使用 export_onnx.py 导出的 3 个 ONNX 模型进行主动说话人检测。
不依赖 PyTorch，仅需 onnxruntime + opencv + python_speech_features + numpy。

用法:
    # 从视频文件推理（需包含人脸画面和音频）
    python demo_onnx.py --video path/to/face_clip.avi --onnx_dir onnx_output

    # 从分离的音频和视频帧目录推理
    python demo_onnx.py --audio path/to/audio.wav --frames_dir path/to/face_frames/ --onnx_dir onnx_output

    # 使用随机数据快速验证模型能否正常运行
    python demo_onnx.py --onnx_dir onnx_output --dummy

输入要求:
    - 视频: 包含单个人脸的裁剪视频片段（25fps），或人脸帧图片目录
    - 音频: 16kHz 单声道 WAV，或直接从视频中提取
    - 人脸尺寸: 脚本会自动 resize 到 112x112 灰度图
"""

import os
import argparse
import math
import numpy as np
import onnxruntime as ort


def load_sessions(onnx_dir, use_gpu=False):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    sess_audio = ort.InferenceSession(os.path.join(onnx_dir, "audio_frontend.onnx"), providers=providers)
    sess_visual = ort.InferenceSession(os.path.join(onnx_dir, "visual_frontend.onnx"), providers=providers)
    sess_backend = ort.InferenceSession(os.path.join(onnx_dir, "av_backend.onnx"), providers=providers)

    print(f"ONNX 模型加载完成 (provider: {sess_audio.get_providers()[0]})")
    return sess_audio, sess_visual, sess_backend


def extract_audio_feature(audio_path, num_video_frames, fps=25):
    """从 WAV 文件提取 MFCC 特征并对齐到视频帧数"""
    import python_speech_features
    from scipy.io import wavfile

    sr, audio = wavfile.read(audio_path)
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.int16)
        sr = 16000

    mfcc = python_speech_features.mfcc(audio, sr, numcep=13, winlen=0.025, winstep=0.010)

    target_len = num_video_frames * 4
    if mfcc.shape[0] < target_len:
        shortage = target_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, shortage), (0, 0)), "wrap")
    mfcc = mfcc[:target_len, :]

    return mfcc.astype(np.float32)


def extract_video_frames(video_path, face_size=112):
    """
    从视频文件提取灰度人脸帧序列。
    与 Columbia_test.py 推理逻辑一致：先 resize 到 224x224，再中心裁剪 112x112，
    使模型更聚焦于人脸中下部区域（鼻子-嘴巴-下巴）。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray[56:168, 56:168]  # 中心裁剪 112x112
        frames.append(gray)
    cap.release()

    return np.array(frames, dtype=np.float32)


def load_frames_from_dir(frames_dir, face_size=112):
    """
    从图片目录加载人脸帧。
    与 Columbia_test.py 推理逻辑一致：先 resize 到 224x224，再中心裁剪 112x112。
    """
    import cv2

    files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    if not files:
        raise FileNotFoundError(f"在 {frames_dir} 中未找到图片文件")

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray[56:168, 56:168]  # 中心裁剪 112x112
        frames.append(gray)

    return np.array(frames, dtype=np.float32)


def run_inference(sess_audio, sess_visual, sess_backend, audio_feature, video_feature):
    """
    单次推理。

    参数:
        audio_feature: (T*4, 13) MFCC 特征
        video_feature: (T, 112, 112) 灰度人脸帧, 像素值 0-255

    返回:
        scores: (T,) 每帧的说话概率分数
    """
    audio_input = audio_feature[np.newaxis, ...]   # (1, T*4, 13)
    visual_input = video_feature[np.newaxis, ...]   # (1, T, 112, 112)

    audio_embed = sess_audio.run(None, {"audio_feature": audio_input})[0]
    visual_embed = sess_visual.run(None, {"visual_feature": visual_input})[0]
    scores = sess_backend.run(None, {"audio_embed": audio_embed, "visual_embed": visual_embed})[0]

    return scores


def run_multi_duration(sess_audio, sess_visual, sess_backend, audio_feature, video_feature):
    """
    多时长滑窗推理，与原始 Columbia_test.py 中的 evaluate_network 逻辑一致。
    对同一段音视频用不同时长切片推理后取平均，结果更稳定。

    参数:
        audio_feature: (N_audio, 13) 完整 MFCC 特征
        video_feature: (N_video, 112, 112) 完整灰度人脸帧

    返回:
        scores: (T,) 每帧平均说话概率分数
    """
    length = min(
        (audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100,
        video_feature.shape[0] / 25,
    )
    num_frames = int(round(length * 25))
    audio_feature = audio_feature[: int(round(length * 100)), :]
    video_feature = video_feature[: num_frames, :, :]

    duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    all_scores = []

    for duration in duration_set:
        batch_size = int(math.ceil(length / duration))
        scores = []
        for i in range(batch_size):
            a_start = i * duration * 100
            a_end = (i + 1) * duration * 100
            v_start = i * duration * 25
            v_end = (i + 1) * duration * 25

            audio_chunk = audio_feature[a_start:a_end, :]
            video_chunk = video_feature[v_start:v_end, :, :]

            if audio_chunk.shape[0] == 0 or video_chunk.shape[0] == 0:
                continue

            actual_v_frames = video_chunk.shape[0]
            target_a_len = actual_v_frames * 4
            if audio_chunk.shape[0] < target_a_len:
                audio_chunk = np.pad(
                    audio_chunk,
                    ((0, target_a_len - audio_chunk.shape[0]), (0, 0)),
                    "wrap",
                )
            audio_chunk = audio_chunk[:target_a_len, :]

            chunk_scores = run_inference(
                sess_audio, sess_visual, sess_backend, audio_chunk, video_chunk
            )
            scores.extend(chunk_scores.tolist())

        if len(scores) >= num_frames:
            all_scores.append(scores[:num_frames])

    if not all_scores:
        return np.zeros(num_frames)

    return np.round(np.mean(np.array(all_scores), axis=0), 1).astype(float)


def demo_dummy(sess_audio, sess_visual, sess_backend):
    """使用随机数据验证 ONNX 模型推理流程"""
    print("\n--- 随机数据测试 ---")
    num_frames = 25

    audio_feature = np.random.randn(num_frames * 4, 13).astype(np.float32)
    video_feature = np.random.randint(0, 255, (num_frames, 112, 112)).astype(np.float32)

    scores = run_inference(sess_audio, sess_visual, sess_backend, audio_feature, video_feature)

    print(f"输入: 音频 ({num_frames * 4}, 13), 视频 ({num_frames}, 112, 112)")
    print(f"输出: 每帧分数 shape={scores.shape}")
    print(f"分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"分数均值: {scores.mean():.4f}")
    print("推理流程验证通过!")


def demo_video(video_path, sess_audio, sess_visual, sess_backend):
    """从视频文件进行推理"""
    import subprocess
    import tempfile

    print(f"\n--- 视频文件推理: {video_path} ---")

    video_feature = extract_video_frames(video_path)
    print(f"视频帧数: {len(video_feature)}")

    audio_path = os.path.splitext(video_path)[0] + ".wav"
    tmp_audio = None
    if not os.path.exists(audio_path):
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_audio.close()
        audio_path = tmp_audio.name
        cmd = f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 -vn "{audio_path}" -loglevel panic'
        subprocess.call(cmd, shell=True)
        print(f"已从视频提取音频: {audio_path}")

    audio_feature = extract_audio_feature(audio_path, len(video_feature))
    print(f"MFCC 特征: {audio_feature.shape}")

    scores = run_multi_duration(sess_audio, sess_visual, sess_backend, audio_feature, video_feature)

    print(f"\n推理结果 (每帧说话概率, 共 {len(scores)} 帧):")
    for i, s in enumerate(scores):
        bar = "█" * int(s * 20) if s > 0 else ""
        label = "说话" if s > 0 else "未说话"
        print(f"  帧 {i:4d} | 分数: {s:+6.1f} | {label} | {bar}")

    speaking_ratio = np.mean(np.array(scores) > 0) * 100
    print(f"\n总结: {len(scores)} 帧中有 {speaking_ratio:.1f}% 被判定为说话状态")

    if tmp_audio:
        os.unlink(tmp_audio.name)

    return scores


def demo_separate(audio_path, frames_dir, sess_audio, sess_visual, sess_backend):
    """从分离的音频文件和帧目录进行推理"""
    print(f"\n--- 分离输入推理 ---")
    print(f"  音频: {audio_path}")
    print(f"  帧目录: {frames_dir}")

    video_feature = load_frames_from_dir(frames_dir)
    print(f"  视频帧数: {len(video_feature)}")

    audio_feature = extract_audio_feature(audio_path, len(video_feature))
    print(f"  MFCC 特征: {audio_feature.shape}")

    scores = run_multi_duration(sess_audio, sess_visual, sess_backend, audio_feature, video_feature)

    print(f"\n推理结果 (每帧说话概率, 共 {len(scores)} 帧):")
    for i, s in enumerate(scores):
        label = "说话" if s > 0 else "未说话"
        print(f"  帧 {i:4d} | 分数: {s:+6.1f} | {label}")

    speaking_ratio = np.mean(np.array(scores) > 0) * 100
    print(f"\n总结: {len(scores)} 帧中有 {speaking_ratio:.1f}% 被判定为说话状态")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="LR-ASD ONNX 推理示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--onnx_dir", type=str, default="weights", help="ONNX 模型目录")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径 (含人脸裁剪片段)")
    parser.add_argument("--audio", type=str, default=None, help="输入音频路径 (16kHz WAV)")
    parser.add_argument("--frames_dir", type=str, default=None, help="人脸帧图片目录")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 推理")
    parser.add_argument("--dummy", action="store_true", help="使用随机数据测试推理流程")
    args = parser.parse_args()

    sess_audio, sess_visual, sess_backend = load_sessions(args.onnx_dir, use_gpu=args.gpu)

    if args.dummy:
        demo_dummy(sess_audio, sess_visual, sess_backend)
    elif args.video:
        demo_video(args.video, sess_audio, sess_visual, sess_backend)
    elif args.audio and args.frames_dir:
        demo_separate(args.audio, args.frames_dir, sess_audio, sess_visual, sess_backend)
    else:
        print("请指定输入方式:")
        print("  --dummy                         随机数据测试")
        print("  --video path/to/clip.avi        视频文件推理")
        print("  --audio a.wav --frames_dir dir/ 分离输入推理")
        print("\n运行 --help 查看详细用法")


if __name__ == "__main__":
    main()
