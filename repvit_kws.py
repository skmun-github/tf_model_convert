import torch
import torch.nn as nn
from torch.nn.init import constant_
import tensorflow as tf


# ---------------------------------------------------------------------------
# 공통 유틸 함수
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    """Make channel number divisible by `divisor` (MobileNet-style)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Variant used for SE reduction channels."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def _is_stride_one(stride):
    if isinstance(stride, int):
        return stride == 1
    if isinstance(stride, (tuple, list)):
        return all(s == 1 for s in stride)
    return False


# ---------------------------------------------------------------------------
# PyTorch 구현
# ---------------------------------------------------------------------------

class SEModule(nn.Module):
    """Squeeze-and-Excitation module (PyTorch)."""

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio,
                rd_divisor,
                round_limit=0.0,
            )
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * torch.sigmoid(x_se)


class Conv2D_BN(nn.Sequential):
    """Conv2d + BatchNorm2d block used in RepViT (PyTorch)."""

    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0,
        resolution=-10000,
    ):
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                a,
                b,
                ks,
                stride,
                pad,
                dilation,
                groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(b))
        constant_(self.bn.weight, bn_weight_init)
        constant_(self.bn.bias, 0.0)

    @torch.no_grad()
    def fuse(self):
        """Fuse Conv+BN into a single Conv2d (for deployment)."""
        c, bn = self._modules.values()
        w_scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        w = c.weight * w_scale[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            bias=True,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """Residual wrapper with optional stochastic depth (PyTorch)."""

    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m
        self.drop = float(drop)

    def forward(self, x):
        if self.training and self.drop > 0.0:
            keep_prob = 1.0 - self.drop
            shape = (x.size(0), 1, 1, 1)
            random_tensor = keep_prob + torch.rand(shape, device=x.device)
            binary_mask = random_tensor.floor()
            return x + self.m(x) * binary_mask / keep_prob
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        """Fusing logic kept for completeness; not used in tests."""
        if isinstance(self.m, Conv2D_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(
                m.weight.shape[0],
                m.weight.shape[1],
                1,
                1,
                device=m.weight.device,
            )
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(
                m.weight.shape[0],
                m.weight.shape[1],
                1,
                1,
                device=m.weight.device,
            )
            identity = nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity
            return m
        else:
            return self


class RepVGGDW(nn.Module):
    """Depthwise RepVGG-style block (PyTorch)."""

    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv2D_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(
            ed,
            ed,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=ed,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        # (conv3x3 + conv1x1 + identity) 후 BN
        return self.bn((self.conv(x) + self.conv1(x)) + x)


class RepViTBlock(nn.Module):
    """RepViT block (PyTorch MetaFormer-style)."""

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        self.identity = _is_stride_one(stride) and inp == oup
        assert hidden_dim == 2 * inp

        if not _is_stride_one(stride):
            # downsampling / non-identity block
            self.token_mixer = nn.Sequential(
                Conv2D_BN(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=inp,
                ),
                SEModule(inp, 0.25) if use_se else nn.Identity(),
                Conv2D_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2D_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2D_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0.0),
                )
            )
        else:
            # stride == 1, 채널 동일 → identity 가능
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SEModule(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2D_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2D_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0.0),
                )
            )

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):
    """RepViT backbone (PyTorch)."""

    def __init__(self, cfgs, in_channels=1, out_indices=None):
        super().__init__()
        self.cfgs = cfgs

        # patch embedding
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2D_BN(in_channels, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2D_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]

        # RepViT blocks
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                RepViTBlock(
                    input_channel,
                    exp_size,
                    output_channel,
                    k,
                    s,
                    use_se,
                    use_hs,
                )
            )
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.out_indices = out_indices
        if out_indices is not None:
            self.out_channels = [self.cfgs[idx - 1][2] for idx in out_indices]
        else:
            self.out_channels = self.cfgs[-1][2]

    def forward(self, x):
        for f in self.features:
            x = f(x)
        return x


# 약 20만 파라미터 수준 KWS RepViT 설정
KWS_TINY_CFGS = [
    # k, t, c, SE, HS, s
    [3, 2, 24, 1, 0, 1],
    [3, 2, 24, 0, 0, 1],
    [3, 2, 24, 0, 0, 1],
    [3, 2, 48, 0, 1, (2, 1)],
    [3, 2, 48, 1, 1, 1],
    [3, 2, 48, 0, 1, 1],
    [3, 2, 48, 1, 1, 1],
    [3, 2, 48, 0, 1, 1],
    [3, 2, 48, 1, 1, 1],
    [3, 2, 48, 0, 1, 1],
    [3, 2, 96, 0, 1, (2, 1)],
    [3, 2, 96, 1, 1, 1],
    [3, 2, 96, 0, 1, 1],
]


class KWSRepViT_Torch(nn.Module):
    """KWS classifier using RepViT backbone (PyTorch).

    Input:  (B, 100, 40)  [time, mel]
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=2, in_channels=1, cfgs=None):
        super().__init__()
        if cfgs is None:
            cfgs = KWS_TINY_CFGS
        self.backbone = RepViT(cfgs, in_channels=in_channels, out_indices=None)
        last_channels = _make_divisible(cfgs[-1][2], 8)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# TensorFlow 2.x 구현
# ---------------------------------------------------------------------------

class TF_SEModule(tf.keras.layers.Layer):
    """Squeeze-and-Excitation module (TensorFlow)."""

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        act_layer=tf.keras.layers.ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio,
                rd_divisor,
                round_limit=0.0,
            )
        self.fc1 = tf.keras.layers.Conv2D(
            rd_channels,
            kernel_size=1,
            use_bias=True,
        )
        self.act = act_layer()
        self.fc2 = tf.keras.layers.Conv2D(
            channels,
            kernel_size=1,
            use_bias=True,
        )

    def call(self, x, training=None):
        x_se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * tf.nn.sigmoid(x_se)


class TF_Conv2D_BN(tf.keras.layers.Layer):
    """Conv2D + BatchNormalization block (TensorFlow, channels_last)."""

    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0,
        resolution=-10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_pad = pad is not None and pad != 0
        if isinstance(pad, int):
            pad_tuple = (pad, pad)
        elif isinstance(pad, (tuple, list)):
            pad_tuple = tuple(pad)
        else:
            pad_tuple = (0, 0)

        self.pad_layer = None
        if self.has_pad:
            self.pad_layer = tf.keras.layers.ZeroPadding2D(padding=pad_tuple)

        self.conv = tf.keras.layers.Conv2D(
            filters=b,
            kernel_size=ks,
            strides=stride,
            padding="valid" if self.has_pad else "same",
            dilation_rate=dilation,
            groups=groups,
            use_bias=False,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.Constant(bn_weight_init),
            beta_initializer="zeros",
        )

    def call(self, x, training=None):
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class TF_Residual(tf.keras.layers.Layer):
    """Residual wrapper with optional stochastic depth (TensorFlow)."""

    def __init__(self, m, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.drop = float(drop)

    def call(self, x, training=None):
        if training and self.drop > 0.0:
            keep_prob = 1.0 - self.drop
            shape = (tf.shape(x)[0], 1, 1, 1)
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
            binary_mask = tf.floor(random_tensor)
            return x + self.m(x, training=training) * binary_mask / keep_prob
        else:
            return x + self.m(x, training=training)


class TF_RepVGGDW(tf.keras.layers.Layer):
    """Depthwise RepVGG-style block (TensorFlow)."""

    def __init__(self, ed, **kwargs):
        super().__init__(**kwargs)
        self.conv = TF_Conv2D_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=ed,
            kernel_size=1,
            strides=1,
            padding="same",
            groups=ed,
            use_bias=True,
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.conv(x, training=training) + self.conv1(x) + x
        x = self.bn(x, training=training)
        return x


class TF_RepViTBlock(tf.keras.layers.Layer):
    """RepViT block (TensorFlow)."""

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, **kwargs):
        super().__init__(**kwargs)
        self.identity = _is_stride_one(stride) and inp == oup
        assert hidden_dim == 2 * inp

        if not _is_stride_one(stride):
            token_mixer_layers = [
                TF_Conv2D_BN(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=inp,
                ),
            ]
            if use_se:
                token_mixer_layers.append(TF_SEModule(inp, 0.25))
            token_mixer_layers.append(
                TF_Conv2D_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.token_mixer = tf.keras.Sequential(
                token_mixer_layers, name="token_mixer"
            )

            channel_layers = [
                TF_Conv2D_BN(oup, 2 * oup, 1, 1, 0),
                tf.keras.layers.Activation("gelu"),
                TF_Conv2D_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0.0),
            ]
            self.channel_mixer = TF_Residual(
                tf.keras.Sequential(
                    channel_layers, name="channel_mixer_inner"
                )
            )
        else:
            assert self.identity
            token_mixer_layers = [TF_RepVGGDW(inp)]
            if use_se:
                token_mixer_layers.append(TF_SEModule(inp, 0.25))
            self.token_mixer = tf.keras.Sequential(
                token_mixer_layers, name="token_mixer"
            )

            channel_layers = [
                TF_Conv2D_BN(inp, hidden_dim, 1, 1, 0),
                tf.keras.layers.Activation("gelu"),
                TF_Conv2D_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0.0),
            ]
            self.channel_mixer = TF_Residual(
                tf.keras.Sequential(
                    channel_layers, name="channel_mixer_inner"
                )
            )

    def call(self, x, training=None):
        x = self.token_mixer(x, training=training)
        x = self.channel_mixer(x, training=training)
        return x


class TF_RepViT(tf.keras.Model):
    """RepViT backbone (TensorFlow)."""

    def __init__(self, cfgs, in_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.cfgs = cfgs

        input_channel = self.cfgs[0][2]
        self.patch_embed = tf.keras.Sequential(
            [
                TF_Conv2D_BN(in_channels, input_channel // 2, 3, 2, 1),
                tf.keras.layers.Activation("gelu"),
                TF_Conv2D_BN(input_channel // 2, input_channel, 3, 2, 1),
            ],
            name="patch_embed",
        )

        self.blocks = []
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            block = TF_RepViTBlock(
                input_channel,
                exp_size,
                output_channel,
                k,
                s,
                use_se,
                use_hs,
            )
            self.blocks.append(block)
            input_channel = output_channel

    def call(self, x, training=None):
        x = self.patch_embed(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x


class KWSRepViT_TF(tf.keras.Model):
    """KWS classifier using RepViT backbone (TensorFlow).

    Input:  (B, 100, 40) or (B, 100, 40, 1)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=2, in_channels=1, cfgs=None, **kwargs):
        super().__init__(**kwargs)
        if cfgs is None:
            cfgs = KWS_TINY_CFGS
        self.backbone = TF_RepViT(cfgs, in_channels=in_channels)
        last_channels = _make_divisible(cfgs[-1][2], 8)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=None):
        # x: (B, T, F) or (B, T, F, 1)
        if x.shape.rank == 3:
            x = tf.expand_dims(x, axis=-1)
        x = self.backbone(x, training=training)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # 간단한 PyTorch 전방향 테스트만 수행 (GPU/TF 의존성 없이)
    batch_size = 2
    time_steps = 100
    mel_bins = 40
    num_classes = 2

    model = KWSRepViT_Torch(num_classes=num_classes, in_channels=1)
    x = torch.randn(batch_size, time_steps, mel_bins)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch smoke test: output shape={tuple(y.shape)}, params={params}")
