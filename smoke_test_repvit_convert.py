import os

# 이 테스트 스크립트에서는 TensorFlow와 PyTorch 모두 CPU만 사용하도록 GPU를 숨긴다.
# (RTX 5090 + TF GPU 이슈를 완전히 우회하기 위함)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import tensorflow as tf

from repvit_kws import KWSRepViT_Torch, KWSRepViT_TF


def set_global_seeds(seed: int = 0):
    """Numpy / Torch / TF 난수 시드 고정."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def count_torch_params(model: torch.nn.Module):
    """PyTorch 모델 파라미터 수 (trainable 기준)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_tf_params(model: tf.keras.Model):
    """TensorFlow 모델 파라미터 수 (trainable / non-trainable 분리)."""
    trainable = int(
        sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    )
    non_trainable = int(
        sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
    )
    total = trainable + non_trainable
    return total, trainable, non_trainable


def forward_check(
    torch_model: torch.nn.Module,
    tf_model: tf.keras.Model,
    batch_size: int = 2,
    time_steps: int = 100,
    mel_bins: int = 40,
    num_classes: int = 2,
    n_trials: int = 3,
):
    """여러 번 랜덤 입력으로 forward를 돌려서 shape / NaN 여부 체크."""
    for trial in range(n_trials):
        x_np = np.random.randn(batch_size, time_steps, mel_bins).astype("float32")

        # ----- PyTorch -----
        x_torch = torch.from_numpy(x_np)
        torch_model.eval()
        with torch.no_grad():
            y_torch = torch_model(x_torch)

        if y_torch.shape != (batch_size, num_classes):
            raise RuntimeError(
                f"[PyTorch] 출력 shape mismatch: 기대 {(batch_size, num_classes)}, "
                f"실제 {tuple(y_torch.shape)}"
            )
        if not torch.isfinite(y_torch).all():
            raise RuntimeError("[PyTorch] 출력에 NaN 또는 Inf가 포함되어 있습니다.")

        # ----- TensorFlow -----
        x_tf = tf.convert_to_tensor(x_np)
        y_tf = tf_model(x_tf, training=False)

        if tuple(y_tf.shape) != (batch_size, num_classes):
            raise RuntimeError(
                f"[TensorFlow] 출력 shape mismatch: 기대 {(batch_size, num_classes)}, "
                f"실제 {tuple(y_tf.shape)}"
            )
        if not bool(tf.reduce_all(tf.math.is_finite(y_tf)).numpy()):
            raise RuntimeError("[TensorFlow] 출력에 NaN 또는 Inf가 포함되어 있습니다.")


def gradient_smoke_test_torch(
    torch_model: torch.nn.Module,
    batch_size: int = 2,
    time_steps: int = 100,
    mel_bins: int = 40,
    num_classes: int = 2,
):
    """PyTorch에서 backward가 실제로 잘 도는지 최소한의 체크."""
    torch_model.train()
    torch_model.zero_grad(set_to_none=True)

    x_np = np.random.randn(batch_size, time_steps, mel_bins).astype("float32")
    x_torch = torch.from_numpy(x_np)
    x_torch.requires_grad_()

    logits = torch_model(x_torch)
    if logits.shape != (batch_size, num_classes):
        raise RuntimeError(
            f"[PyTorch] gradient 체크 중 출력 shape mismatch: {tuple(logits.shape)}"
        )

    loss = logits.mean()
    loss.backward()

    nonzero_grad_params = 0
    for p in torch_model.parameters():
        if p.grad is None:
            continue
        if p.grad.abs().sum().item() > 0.0:
            nonzero_grad_params += 1

    if nonzero_grad_params == 0:
        raise RuntimeError(
            "[PyTorch] gradient 체크 실패: 모든 gradient가 0 또는 None 입니다."
        )

    return nonzero_grad_params


def gradient_smoke_test_tf(
    tf_model: tf.keras.Model,
    batch_size: int = 2,
    time_steps: int = 100,
    mel_bins: int = 40,
    num_classes: int = 2,
):
    """TensorFlow에서 GradientTape로 gradient가 잘 나오는지 체크."""
    x_np = np.random.randn(batch_size, time_steps, mel_bins).astype("float32")
    x_tf = tf.convert_to_tensor(x_np)

    with tf.GradientTape() as tape:
        logits = tf_model(x_tf, training=True)
        if tuple(logits.shape) != (batch_size, num_classes):
            raise RuntimeError(
                f"[TensorFlow] gradient 체크 중 출력 shape mismatch: {tuple(logits.shape)}"
            )
        loss = tf.reduce_mean(logits)

    grads = tape.gradient(loss, tf_model.trainable_weights)
    nonzero_grad_vars = 0
    for g in grads:
        if g is None:
            continue
        if tf.reduce_any(tf.not_equal(g, 0.0)):
            nonzero_grad_vars += 1

    if nonzero_grad_vars == 0:
        raise RuntimeError(
            "[TensorFlow] gradient 체크 실패: 모든 gradient가 0 또는 None 입니다."
        )

    return nonzero_grad_vars


def inspect_torch_backbone_shapes():
    """PyTorch 백본(RepViT) 각 stage의 feature map shape을 출력."""
    model = KWSRepViT_Torch(num_classes=2, in_channels=1)
    x = torch.randn(1, 100, 40)
    x = x.unsqueeze(1)  # (1, 1, 100, 40)

    print("\n[PyTorch] Backbone feature map shapes:")
    for i, layer in enumerate(model.backbone.features):
        x = layer(x)
        print(f"  features[{i}]: {tuple(x.shape)}")


def inspect_tf_backbone_shapes():
    """TensorFlow 백본(RepViT) 각 stage의 feature map shape을 출력."""
    model = KWSRepViT_TF(num_classes=2, in_channels=1)
    x = tf.random.normal((1, 100, 40, 1))

    print("\n[TensorFlow] Backbone feature map shapes:")
    x = model.backbone.patch_embed(x, training=False)
    print(f"  patch_embed: {tuple(x.shape)}")
    for i, block in enumerate(model.backbone.blocks):
        x = block(x, training=False)
        print(f"  block[{i}]: {tuple(x.shape)}")


def main():
    batch_size = 2
    time_steps = 100
    mel_bins = 40
    num_classes = 2

    set_global_seeds(0)

    # 모델 생성
    torch_model = KWSRepViT_Torch(num_classes=num_classes, in_channels=1)
    tf_model = KWSRepViT_TF(num_classes=num_classes, in_channels=1)

    # TensorFlow 모델은 한 번 호출해서 weights를 materialize해야 count_params가 정확함
    x_init = np.random.randn(batch_size, time_steps, mel_bins).astype("float32")
    _ = tf_model(tf.convert_to_tensor(x_init), training=False)

    # ------ 파라미터 수 체크 ------
    pt_total, pt_trainable = count_torch_params(torch_model)
    tf_total, tf_trainable, tf_non_trainable = count_tf_params(tf_model)

    print("==== Parameter counts ====")
    print(f"PyTorch trainable params:       {pt_trainable}")
    print(f"PyTorch total (동일):           {pt_total}")
    print(f"TensorFlow trainable params:    {tf_trainable}")
    print(f"TensorFlow non-trainable:       {tf_non_trainable}")
    print(f"TensorFlow total params:        {tf_total}")

    if pt_trainable == tf_trainable:
        print("✓ Trainable parameter counts match between PyTorch and TensorFlow.")
    else:
        print("⚠ WARNING: Trainable parameter counts differ between PyTorch and TensorFlow.")

    # ------ Forward / shape / NaN 체크 ------
    print("\n==== Forward shape & finite checks ====")
    forward_check(
        torch_model,
        tf_model,
        batch_size=batch_size,
        time_steps=time_steps,
        mel_bins=mel_bins,
        num_classes=num_classes,
        n_trials=3,
    )
    print("✓ Forward passes succeeded on both frameworks (shapes match, no NaNs/Infs).")

    # ------ Gradient 체크 ------
    print("\n==== Gradient smoke tests ====")
    n_pt = gradient_smoke_test_torch(
        torch_model, batch_size, time_steps, mel_bins, num_classes
    )
    n_tf = gradient_smoke_test_tf(
        tf_model, batch_size, time_steps, mel_bins, num_classes
    )
    print(f"PyTorch:    {n_pt} parameters with non-zero gradients.")
    print(f"TensorFlow: {n_tf} variables with non-zero gradients.")
    print("✓ Backprop works in both frameworks.")

    # ------ Backbone feature map shape 확인 ------
    inspect_torch_backbone_shapes()
    inspect_tf_backbone_shapes()


if __name__ == "__main__":
    main()
