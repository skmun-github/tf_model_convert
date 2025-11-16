import os

# TensorFlow가 RTX 5090 GPU를 절대 건드리지 않게 GPU를 숨김 (CPU-only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from repvit_kws import KWSRepViT_Torch, KWSRepViT_TF


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------

BATCH_SIZE = 2
TIME_STEPS = 100
MEL_BINS = 40
NUM_CLASSES = 2
SEED = 0


# ---------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------

def set_global_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    """PyTorch (N, C, H, W)를 TF 형식 (N, H, W, C)로 변환."""
    if x.ndim == 4:
        return np.transpose(x, (0, 2, 3, 1))
    return x


# ---------------------------------------------------------
# PyTorch: 전체 0 초기화 + 검증
# ---------------------------------------------------------

def init_torch_all_zero(model: nn.Module) -> None:
    """
    PyTorch 모델의 모든 floating-point parameter 및 buffer를 0으로 설정.
    BatchNorm running_mean / running_var 등도 포함.
    """
    # trainable parameters
    for p in model.parameters():
        if p is not None and p.data.dtype.is_floating_point:
            p.data.zero_()

    # buffers (BN running stats, 등)
    for buf in model.buffers():
        if buf is not None and buf.dtype.is_floating_point:
            buf.zero_()


def check_torch_all_zero(model: nn.Module, tol: float = 1e-8) -> None:
    """
    PyTorch 모델의 모든 floating-point parameter/buffer가 tol 이하인지 확인.
    """
    max_abs = 0.0

    for p in model.parameters():
        if p is not None and p.data.dtype.is_floating_point:
            v = float(p.data.abs().max().item())
            if v > max_abs:
                max_abs = v

    for buf in model.buffers():
        if buf is not None and buf.dtype.is_floating_point:
            v = float(buf.abs().max().item())
            if v > max_abs:
                max_abs = v

    print(f"[Check] PyTorch max |param/buffer| after zero init: {max_abs:.3e}")
    if max_abs > tol:
        raise AssertionError(
            f"PyTorch parameters/buffers are not zero (max_abs={max_abs:.3e} > {tol})."
        )


# ---------------------------------------------------------
# TensorFlow: 전체 0 초기화 + 검증
# ---------------------------------------------------------

def init_tf_all_zero(model: tf.keras.Model) -> None:
    """
    TensorFlow Keras 모델의 모든 weight (trainable + non-trainable)를 0으로 설정.
    Conv/Dense kernel, bias, BN gamma/beta/moving stats 등 모두 포함.
    """
    for w in model.weights:
        w.assign(tf.zeros_like(w))


def check_tf_all_zero(model: tf.keras.Model, tol: float = 1e-8) -> None:
    """
    TensorFlow 모델의 모든 weight가 tol 이하인지 확인.
    """
    max_abs = 0.0

    for w in model.weights:
        arr = w.numpy()
        if arr.size == 0:
            continue
        v = float(np.max(np.abs(arr)))
        if v > max_abs:
            max_abs = v

    print(f"[Check] TF      max |weight|       after zero init: {max_abs:.3e}")
    if max_abs > tol:
        raise AssertionError(
            f"TensorFlow weights are not zero (max_abs={max_abs:.3e} > {tol})."
        )


# ---------------------------------------------------------
# 중간 feature 수집 (PyTorch)
# ---------------------------------------------------------

def collect_torch_features(
    model: KWSRepViT_Torch,
    x_np: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    PyTorch KWSRepViT에서 backbone + head 중간 출력 수집.

    반환: [(name, feature_numpy), ...]
      - "patch_embed"
      - "block_1" ~ "block_N"
      - "global_pool"
      - "logits"
    """
    model.eval()
    feats: list[tuple[str, np.ndarray]] = []

    with torch.no_grad():
        x = torch.from_numpy(x_np)  # (B, T, F)
        if x.ndim == 3:
            x = x.unsqueeze(1)      # (B, 1, T, F)

        # features[0]: patch_embed
        h = model.backbone.features[0](x)
        feats.append(("patch_embed", h.detach().cpu().numpy()))

        # features[1:]: blocks
        for i, block in enumerate(model.backbone.features[1:], start=1):
            h = block(h)
            feats.append((f"block_{i}", h.detach().cpu().numpy()))

        # global_pool & head
        pooled = model.global_pool(h).flatten(1)      # (B, C)
        feats.append(("global_pool", pooled.detach().cpu().numpy()))

        logits = model.head(pooled)                   # (B, num_classes)
        feats.append(("logits", logits.detach().cpu().numpy()))

    return feats


# ---------------------------------------------------------
# 중간 feature 수집 (TensorFlow)
# ---------------------------------------------------------

def collect_tf_features(
    model: KWSRepViT_TF,
    x_np: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    TF KWSRepViT에서 backbone + head 중간 출력 수집.

    반환: [(name, feature_numpy), ...]
      - "patch_embed"
      - "block_1" ~ "block_N"
      - "global_pool"
      - "logits"
    """
    feats: list[tuple[str, np.ndarray]] = []

    x = tf.convert_to_tensor(x_np)  # (B, T, F)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, axis=-1)  # (B, T, F, 1)

    # patch_embed
    h = model.backbone.patch_embed(x, training=False)
    feats.append(("patch_embed", h.numpy()))

    # blocks
    for i, block in enumerate(model.backbone.blocks, start=1):
        h = block(h, training=False)
        feats.append((f"block_{i}", h.numpy()))

    # global_pool & classifier
    pooled = model.global_pool(h)
    feats.append(("global_pool", pooled.numpy()))

    logits = model.classifier(pooled)
    feats.append(("logits", logits.numpy()))

    return feats


# ---------------------------------------------------------
# layer-wise 비교
# ---------------------------------------------------------

def compare_layerwise(
    torch_feats: list[tuple[str, np.ndarray]],
    tf_feats: list[tuple[str, np.ndarray]],
    atol: float = 1e-7,
) -> None:
    """
    각 레이어별로 PyTorch / TF 출력을 엄격하게 비교.
    - 4D 텐서: Torch NCHW → NHWC로 변환 후 비교
    - 2D 텐서: 그대로 비교
    - rtol=0, atol=1e-7 기준으로 allclose 통과해야 함.
    """
    if len(torch_feats) != len(tf_feats):
        raise AssertionError(
            f"Number of feature tensors differ: torch={len(torch_feats)}, tf={len(tf_feats)}"
        )

    print("\n==== Layer-wise activation comparison (ZERO-initialized models) ====")
    all_ok = True

    for (name_t, ft_t), (name_f, ft_f) in zip(torch_feats, tf_feats):
        if name_t != name_f:
            raise AssertionError(
                f"Layer name mismatch: torch='{name_t}' vs tf='{name_f}'"
            )

        name = name_t
        arr_t = ft_t
        arr_f = ft_f

        # Conv feature: NCHW → NHWC
        if arr_t.ndim == 4 and arr_f.ndim == 4:
            arr_t = nchw_to_nhwc(arr_t)

        if arr_t.shape != arr_f.shape:
            raise AssertionError(
                f"[{name}] shape mismatch: torch={arr_t.shape}, tf={arr_f.shape}"
            )

        diff = arr_t - arr_f
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))

        print(
            f"{name:12s} | shape={arr_t.shape}, "
            f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
        )

        if not np.allclose(arr_t, arr_f, rtol=0.0, atol=atol):
            print(f"  -> allclose failed for layer '{name}' (atol={atol:g})")
            all_ok = False

    if not all_ok:
        raise AssertionError("Some intermediate outputs differ beyond tolerance.")
    else:
        print("✓ All intermediate outputs match within strict tolerance.")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main() -> None:
    set_global_seeds(SEED)

    # 1) 모델 생성
    torch_model = KWSRepViT_Torch(num_classes=NUM_CLASSES, in_channels=1)
    tf_model = KWSRepViT_TF(num_classes=NUM_CLASSES, in_channels=1)

    # 2) TF 모델 build (weights 생성)
    dummy_x = tf.zeros((1, TIME_STEPS, MEL_BINS), dtype=tf.float32)
    _ = tf_model(dummy_x, training=False)

    # 3) 두 모델 모두 0으로 초기화
    print("Initializing PyTorch model (all floating parameters/buffers → 0)...")
    init_torch_all_zero(torch_model)

    print("Initializing TensorFlow model (all weights → 0)...")
    init_tf_all_zero(tf_model)

    # 4) 초기화 검증
    check_torch_all_zero(torch_model, tol=1e-8)
    check_tf_all_zero(tf_model, tol=1e-8)

    # 5) 동일 입력으로 중간 feature 수집
    x_np = np.random.randn(BATCH_SIZE, TIME_STEPS, MEL_BINS).astype("float32")

    print("Collecting intermediate features from PyTorch...")
    torch_feats = collect_torch_features(torch_model, x_np)

    print("Collecting intermediate features from TensorFlow...")
    tf_feats = collect_tf_features(tf_model, x_np)

    # 6) layer-wise 비교
    compare_layerwise(torch_feats, tf_feats, atol=1e-7)


if __name__ == "__main__":
    main()
