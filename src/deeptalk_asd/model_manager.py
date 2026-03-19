"""
模型管理器：自动下载、缓存、离线模式支持。

环境变量:
    DEEPTALK_ASD_OFFLINE: 设为 1/true/yes 启用离线模式，禁止所有网络下载
    DEEPTALK_ASD_CACHE_DIR: 自定义模型缓存目录，默认 ~/.cache/deeptalk_asd/
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path

from .deeptalk_logger import DeepTalkLogger

logger = DeepTalkLogger(__name__)

# GitHub Release 基础 URL
_RELEASE_BASE = "https://github.com/huyyxy/DeepTalk-ASD/releases/download/v0.2.1"

# 模型注册表
MODEL_REGISTRY = {
    "audio_frontend.onnx": {
        "url": f"{_RELEASE_BASE}/audio_frontend.onnx",
        "sha256": "a1b55df7105fb730196b527f3e5c8d39f4ab38013a21d01c15404e870a8a5cb3",
        "size_mb": 0.9,
        "description": "LR-ASD 音频前端",
    },
    "visual_frontend.onnx": {
        "url": f"{_RELEASE_BASE}/visual_frontend.onnx",
        "sha256": "af8e213b148d573008c14068ebb6255779b2cce1fef56416e276bd5f49ca6e29",
        "size_mb": 1.5,
        "description": "LR-ASD 视觉前端",
    },
    "av_backend.onnx": {
        "url": f"{_RELEASE_BASE}/av_backend.onnx",
        "sha256": "13ec6c03885e104b7bca0f30b0574f0e3ffc896ca6a7b9d71e75ca413c75eb55",
        "size_mb": 0.8,
        "description": "LR-ASD 音视频融合后端",
    },
    "silero_vad.onnx": {
        "url": f"{_RELEASE_BASE}/silero_vad.onnx",
        "sha256": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        "size_mb": 2.2,
        "description": "Silero VAD 语音轮次检测",
    },
    "Pikachu": {
        "url": f"{_RELEASE_BASE}/Pikachu",
        "sha256": "5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431",
        "size_mb": 16.0,
        "description": "InspireFace 人脸检测资源包",
    },
    "wespeaker_zh_cnceleb_resnet34.onnx": {
        "url": f"{_RELEASE_BASE}/wespeaker_zh_cnceleb_resnet34.onnx",
        "sha256": "f86cd6c509f331f0e20b07bd48d1b2eb7de54202643401c4e84695ac861a0e5a",
        "size_mb": 25.3,
        "description": "WeSpeaker 声纹特征提取",
    },
}


def is_offline_mode() -> bool:
    """检查是否为离线模式"""
    val = os.environ.get("DEEPTALK_ASD_OFFLINE", "").lower()
    return val in ("1", "true", "yes")


def get_model_cache_dir() -> Path:
    """获取模型缓存目录"""
    custom = os.environ.get("DEEPTALK_ASD_CACHE_DIR")
    if custom:
        return Path(custom)

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "deeptalk_asd"

    return Path.home() / ".cache" / "deeptalk_asd"


def _verify_hash(file_path: Path, expected_sha256: str) -> bool:
    """校验文件 SHA256 哈希值"""
    if not expected_sha256:
        return True
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def _download_with_progress(url: str, dest: Path):
    """带进度显示的文件下载"""
    try:
        from tqdm import tqdm

        req = urllib.request.urlopen(url)
        total = int(req.headers.get("Content-Length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            file=sys.stderr,
        ) as pbar:
            while True:
                chunk = req.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except ImportError:
        # 没有 tqdm 时使用简单的下载
        logger.info(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, str(dest))


def ensure_model(name: str, cache_dir: Path = None) -> Path:
    """确保模型文件存在。联网模式下不存在则自动下载；离线模式下不存在则抛异常。

    Args:
        name: 模型注册表中的名称
        cache_dir: 缓存目录，默认使用 get_model_cache_dir()

    Returns:
        模型文件的绝对路径
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可用模型: {list(MODEL_REGISTRY.keys())}")

    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODEL_REGISTRY[name]
    local_path = cache_dir / name

    # 已存在且校验通过 → 直接返回
    if local_path.exists() and _verify_hash(local_path, model_info.get("sha256")):
        return local_path

    # ── 离线模式：不下载，直接报错并打印指引 ──
    if is_offline_mode():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"[DeepTalk-ASD] 离线模式下未找到模型: {name}\n"
            f"缓存目录: {cache_dir}\n"
            f"\n"
            f"请通过以下方式之一获取模型文件:\n"
            f"  1. 在有网络的环境执行预下载:\n"
            f"     python3 -m deeptalk_asd download-models --cache-dir {cache_dir}\n"
            f"  2. 手动下载并放入缓存目录:\n"
            f"     URL: {model_info['url']}\n"
            f"     目标路径: {local_path}\n"
            f"  3. 取消离线模式（取消设置 DEEPTALK_ASD_OFFLINE 环境变量）\n"
            f"{'='*60}"
        )

    # ── 联网模式：自动下载 ──
    logger.info(f"Downloading {name} ({model_info['size_mb']:.1f} MB)...")
    try:
        _download_with_progress(model_info["url"], local_path)
    except Exception as e:
        # 下载失败时清理不完整文件
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(
            f"[DeepTalk-ASD] 下载模型 {name} 失败: {e}\n"
            f"URL: {model_info['url']}\n"
            f"你可以手动下载该文件并放置到: {local_path}"
        ) from e

    # 校验下载后的文件
    if not _verify_hash(local_path, model_info.get("sha256")):
        local_path.unlink()
        raise RuntimeError(
            f"[DeepTalk-ASD] 模型 {name} 下载后校验失败，文件可能已损坏。\n"
            f"请重试下载或手动获取: {model_info['url']}"
        )

    logger.info(f"Model {name} downloaded to {local_path}")
    return local_path


def ensure_all_models(cache_dir: Path = None):
    """下载所有注册的模型文件"""
    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    print(f"[DeepTalk-ASD] Model cache directory: {cache_dir}")
    for name in MODEL_REGISTRY:
        try:
            path = ensure_model(name, cache_dir)
            print(f"  ✓ {name} -> {path}")
        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)


def print_model_info(cache_dir: Path = None):
    """打印模型缓存状态信息"""
    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    print(f"[DeepTalk-ASD] Model Info")
    print(f"  Cache directory : {cache_dir}")
    print(f"  Offline mode    : {'ON' if is_offline_mode() else 'OFF'}")
    print()

    total_size = 0
    for name, info in MODEL_REGISTRY.items():
        local_path = cache_dir / name
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            valid = _verify_hash(local_path, info.get("sha256"))
            status = "✓" if valid else "⚠ hash mismatch"
            print(f"  {status} {name:45s} {size_mb:7.1f} MB  ({info['description']})")
        else:
            print(f"  ✗ {name:45s}   MISSING  ({info['description']})")

    print(f"\n  Total cached: {total_size:.1f} MB")
