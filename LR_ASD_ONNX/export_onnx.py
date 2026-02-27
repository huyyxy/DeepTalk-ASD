# ===== 修复 beartype 与 PyTorch ONNX 导出的兼容性问题 =====
# PyTorch 2.2.x 内部的 ONNX 导出代码集成了 @beartype 装饰器，
# 会导致某些模型导出时产生类型检查错误，这里在 import torch 之前将其禁用。
import beartype

_orig = beartype.beartype

def _noop_beartype(*args, **kwargs):
    """替代 beartype.beartype 的空装饰器"""
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def _wrapper(func):
        return func
    return _wrapper

beartype.beartype = _noop_beartype
# ============================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.Model import ASD_Model
from loss import lossAV


class OnnxAudioEncoder(nn.Module):
    """
    ONNX 兼容的音频编码器。
    原始 audio_encoder 使用 MaxPool3d 处理 4D 输入（PyTorch 隐式支持），
    但导出 ONNX 时会产生不兼容的 MaxPool 算子，这里替换为等效的 MaxPool2d。
    """

    def __init__(self, original_encoder):
        super().__init__()
        self.block1 = original_encoder.block1
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block2 = original_encoder.block2
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block3 = original_encoder.block3

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)
        return x


class AudioFrontend(nn.Module):
    """音频前端：MFCC -> 音频嵌入"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = OnnxAudioEncoder(encoder)

    def forward(self, x):
        # x: (B, T*4, 13)
        x = x.unsqueeze(1).transpose(2, 3)  # (B, 1, 13, T*4)
        x = self.encoder(x)  # (B, T, 128)
        return x


class VisualFrontend(nn.Module):
    """视觉前端：灰度人脸序列 -> 视觉嵌入"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # x: (B, T, 112, 112)
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255.0 - 0.4161) / 0.1688
        x = self.encoder(x)  # (B, T, 128)
        return x


class AVBackend(nn.Module):
    """音视频后端：音频嵌入 + 视觉嵌入 -> 说话人 logit（与原始 LR-ASD 一致，不做 softmax；>0 判为说话）"""

    def __init__(self, fusion, detector, fc):
        super().__init__()
        self.fusion = fusion
        self.detector = detector
        self.fc = fc

    def forward(self, audio_embed, visual_embed):
        # audio_embed: (B, T, 128), visual_embed: (B, T, 128)
        x = self.fusion(audio_embed, visual_embed)  # (B, T, 256)
        x = self.detector(x)  # (B, T, 128)
        x = torch.reshape(x, (-1, 128))  # (B*T, 128)
        x = self.fc(x)  # (B*T, 2)
        return x[:, 1]  # (B*T,) 说话 logit，与原始 LR-ASD 一致；>0 判为说话


def load_model(weight_path, device):
    model = ASD_Model()
    loss_av = lossAV()

    state_dict = torch.load(weight_path, map_location=device)
    model_state = {}
    loss_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
        elif k.startswith("lossAV."):
            loss_state[k[len("lossAV."):]] = v

    model.load_state_dict(model_state)
    loss_av.load_state_dict(loss_state)

    model.to(device).eval()
    loss_av.to(device).eval()
    return model, loss_av


def export_audio_frontend(model, output_dir, device, num_frames):
    print(f"[1/3] 导出 audio_frontend (T={num_frames}) ...")
    wrapper = AudioFrontend(model.audioEncoder).to(device).eval()

    dummy_audio = torch.randn(1, num_frames * 4, 13, device=device)

    path = os.path.join(output_dir, "audio_frontend.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        path,
        input_names=["audio_feature"],
        output_names=["audio_embed"],
        dynamic_axes={
            "audio_feature": {0: "batch", 1: "time_x4"},
            "audio_embed": {0: "batch", 1: "time"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  -> 已保存: {path}")
    return path


def export_visual_frontend(model, output_dir, device, num_frames):
    print(f"[2/3] 导出 visual_frontend (T={num_frames}) ...")
    wrapper = VisualFrontend(model.visualEncoder).to(device).eval()

    dummy_visual = torch.randn(1, num_frames, 112, 112, device=device)

    path = os.path.join(output_dir, "visual_frontend.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_visual,),
        path,
        input_names=["visual_feature"],
        output_names=["visual_embed"],
        dynamic_axes={
            "visual_feature": {0: "batch", 1: "time"},
            "visual_embed": {0: "batch", 1: "time"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  -> 已保存: {path}")
    return path


def export_av_backend(model, loss_av, output_dir, device, num_frames):
    print(f"[3/3] 导出 av_backend (T={num_frames}) ...")
    wrapper = AVBackend(
        model.fusion, model.detector, loss_av.FC
    ).to(device).eval()

    dummy_a = torch.randn(1, num_frames, 128, device=device)
    dummy_v = torch.randn(1, num_frames, 128, device=device)

    path = os.path.join(output_dir, "av_backend.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_a, dummy_v),
        path,
        input_names=["audio_embed", "visual_embed"],
        output_names=["score"],
        dynamic_axes={
            "audio_embed": {0: "batch", 1: "time"},
            "visual_embed": {0: "batch", 1: "time"},
            "score": {0: "batch_x_time"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  -> 已保存: {path}")
    return path


def verify_onnx(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ {os.path.basename(onnx_path)} 验证通过")
        return True
    except ImportError:
        print("  (跳过验证: 未安装 onnx 库)")
        return True
    except Exception as e:
        print(f"  ✗ {os.path.basename(onnx_path)} 验证失败: {e}")
        return False


def verify_with_onnxruntime(model, loss_av, onnx_paths, device, num_frames):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("\n(跳过数值验证: 未安装 onnxruntime)")
        return

    print("\n数值一致性验证...")

    audio_input = torch.randn(1, num_frames * 4, 13, device=device)
    visual_input = torch.randn(1, num_frames, 112, 112, device=device)

    with torch.no_grad():
        pt_audio_embed = model.forward_audio_frontend(audio_input)
        pt_visual_embed = model.forward_visual_frontend(visual_input)
        pt_av_out = model.forward_audio_visual_backend(pt_audio_embed, pt_visual_embed)
        pt_score = loss_av.FC(pt_av_out)[:, 1]  # logit，与导出一致

    audio_np = audio_input.cpu().numpy()
    visual_np = visual_input.cpu().numpy()

    providers = ["CPUExecutionProvider"]

    sess_audio = ort.InferenceSession(onnx_paths[0], providers=providers)
    ort_audio_embed = sess_audio.run(None, {"audio_feature": audio_np})[0]

    sess_visual = ort.InferenceSession(onnx_paths[1], providers=providers)
    ort_visual_embed = sess_visual.run(None, {"visual_feature": visual_np})[0]

    sess_backend = ort.InferenceSession(onnx_paths[2], providers=providers)
    ort_score = sess_backend.run(
        None, {"audio_embed": ort_audio_embed, "visual_embed": ort_visual_embed}
    )[0]

    pt_score_np = pt_score.cpu().numpy()
    max_diff = np.max(np.abs(ort_score - pt_score_np))
    print(f"  PyTorch vs ONNX Runtime 最大误差: {max_diff:.6e}")
    if max_diff < 1e-4:
        print("  ✓ 数值一致性验证通过")
    else:
        print(f"  ⚠ 误差较大，请检查 (阈值 1e-4)")


def main():
    parser = argparse.ArgumentParser(description="LR-ASD 模型导出为 ONNX 格式")
    parser.add_argument(
        "--weight", type=str, default="weights/finetuning_TalkSet.model",
        help="模型权重路径 (默认: weights/finetuning_TalkSet.model)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="weights",
        help="ONNX 文件输出目录 (默认: weights)",
    )
    parser.add_argument(
        "--num_frames", type=int, default=25,
        help="用于 tracing 的帧数 T (默认: 25, 即 1 秒视频)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="运行设备 (默认: cpu)",
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="跳过 ONNX 验证和数值一致性检查",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"加载权重: {args.weight}")
    model, loss_av = load_model(args.weight, device)
    print(f"模型加载完成, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    onnx_paths = []
    onnx_paths.append(export_audio_frontend(model, args.output_dir, device, args.num_frames))
    onnx_paths.append(export_visual_frontend(model, args.output_dir, device, args.num_frames))
    onnx_paths.append(export_av_backend(model, loss_av, args.output_dir, device, args.num_frames))

    if not args.skip_verify:
        print("\nONNX 模型验证...")
        for p in onnx_paths:
            verify_onnx(p)
        verify_with_onnxruntime(model, loss_av, onnx_paths, device, args.num_frames)

    print(f"\n导出完成! ONNX 文件保存在: {args.output_dir}/")
    print("  - audio_frontend.onnx  : MFCC特征 (B, T*4, 13) -> 音频嵌入 (B, T, 128)")
    print("  - visual_frontend.onnx : 灰度人脸 (B, T, 112, 112) -> 视觉嵌入 (B, T, 128)")
    print("  - av_backend.onnx      : 音视频嵌入 -> 说话人概率分数 (B*T,)")


if __name__ == "__main__":
    main()
