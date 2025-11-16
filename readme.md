
# RepViT KWS: PyTorch ↔ TensorFlow Structure Verification Project

This repository implements a **small RepViT-based KWS (Keyword Spotting) model** in  
both

- **PyTorch**, and  
- **TensorFlow 2 (tf.keras)**

and provides a **smoke test** to check, in a fairly strict way, that

> “Are these two implementations really the **same model**?”

---

## 1. Repository Layout

```text
.
├── repvit_kws.py                # PyTorch / TF model definitions (RepViT KWS)
└── smoke_test_repvit_convert.py # Structure / gradient / shape verification smoke test
````

### 1.1 `repvit_kws.py`

* **PyTorch side**

  * `KWSRepViT_Torch`

    * Input: `(B, 100, 40)` = (batch, time, mel)
    * Internals:

      * `RepViT` backbone (MetaFormer-style, depthwise/group conv + SE)
      * `AdaptiveAvgPool2d(1)` + `Linear` classifier
    * Output: `(B, 2)` → binary KWS logits (yes/no)
  * Designed with a small config (`KWS_TINY_CFGS`) so that it has about
    **214,222 trainable parameters**.

* **TensorFlow side**

  * `KWSRepViT_TF`

    * Input: `(B, 100, 40)` or `(B, 100, 40, 1)` (adds channel dim automatically)
    * Internals:

      * `TF_RepViT` backbone (channels_last, uses `Conv2D(groups=...)`)
      * `GlobalAveragePooling2D` + `Dense(num_classes=2)`
    * Output: `(B, 2)` logits
  * Based on the same `KWS_TINY_CFGS` and designed so that **structure and parameter
    counts match the PyTorch model**.

> ⚠️ **Important**
> This file is for **architecture definitions**.
> A **PyTorch → TF weight transfer script is not included yet**.
> The current stage focuses on strictly verifying that “the structures are the same”.

---

## 2. Requirements & Installation

### 2.1 Required packages

* Python 3.11+ (tested with Python 3.12)
* PyTorch 2.x (only CPU mode is used here)
* TensorFlow 2.x (tested with 2.19.0, run in **CPU mode**)
* Numpy

The TensorFlow implementation uses `Conv2D(groups=...)`, which is
**officially supported from TensorFlow 2.4 onwards**.

### 2.2 Example: lightweight test environment

```bash
# 1) Create a new virtual environment (example: conda)
conda create -n repvit_convert python=3.11 -y
conda activate repvit_convert

# 2) Install PyTorch (CPU) – choose a version suitable for your system
pip install "torch==2.3.0"

# 3) Install TensorFlow (CPU-only execution in this project)
pip install "tensorflow==2.19.0"

# 4) Others
pip install numpy
```

> ⚠️ **About RTX 50 series (5090 etc.) + TensorFlow GPU**
> As of 2025, TensorFlow 2.19/2.20 installed via pip does **not include CUDA
> kernels for Compute Capability 12.0 (Blackwell)**, and on RTX 5090/5080/5070
> GPU execution often fails with
> `CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_INVALID_HANDLE` and similar errors
> reported in TensorFlow issue trackers.

Because of this, all smoke tests in this project are designed so that
**TensorFlow always runs on CPU only**.

---

## 3. PyTorch / TensorFlow Model Structure Summary

### 3.1 Shared design idea

Using the small RepViT configuration `KWS_TINY_CFGS`:

* Patch embedding:

  * Two `stride=2` convolutions → time: 100 → 50 → 25, freq: 40 → 20 → 10
* Some later blocks use `stride=(2, 1)` to **downsample only the time axis**
  (25 → 13 → 7), while keeping the frequency axis (40→20→10) fixed.
* Channel widths grow as 24 → 48 → 96.

Final feature map:

* PyTorch: `(B, 96, 7, 10)`
* TensorFlow: `(B, 7, 10, 96)`

Global average pooling gives `(B, 96)`, followed by `Dense/Linear(2)` for binary
classification.

### 3.2 PyTorch implementation (`KWSRepViT_Torch`)

* Input `(B, 100, 40)` → `unsqueeze(1)` → `(B, 1, 100, 40)`
* `RepViT` backbone:

  * `Conv2D_BN` (1→12, ks=3, s=2, p=1)
  * `Conv2D_BN` (12→24, ks=3, s=2, p=1)
  * Then 13 `RepViTBlock`s
* `AdaptiveAvgPool2d(1)` → `(B, 96, 1, 1)` → flatten → `Linear(96→2)`

BatchNorm uses PyTorch defaults (`eps=1e-5`, `momentum=0.1`).

### 3.3 TensorFlow implementation (`KWSRepViT_TF`)

* Input `(B, 100, 40)` → if needed, `expand_dims` → `(B, 100, 40, 1)`
* `TF_RepViT` backbone:

  * `TF_Conv2D_BN` (1→12, ks=3, s=2, pad=1)
  * `TF_Conv2D_BN` (12→24, ks=3, s=2, pad=1)
  * Then 13 `TF_RepViTBlock`s

    * depthwise / group conv is implemented via `tf.keras.layers.Conv2D(groups=...)`
* `GlobalAveragePooling2D()` → `(B, 96)` → `Dense(2)`

BatchNorm uses Keras defaults (`epsilon=1e-3`, `momentum=0.99`).

> 💡 **Important practical point**
>
> * The number of **trainable parameters** is exactly the same for PyTorch and
>   TensorFlow.
>   (Verified in the smoke test as 214,222 trainable params.)
> * Because BatchNorm epsilon/momentum defaults differ between frameworks, if
>   your goal is **bit‑exact PyTorch pretrained weight → TF porting**, you may
>   want to align these hyperparameters later.
>   However, for **training each model independently** for KWS, the current
>   defaults are practically fine.

---

## 4. Smoke Test: `smoke_test_repvit_convert.py`

This script automatically performs the following:

1. **Force TensorFlow to run on CPU only** (hide GPUs)
2. Fix random seeds
3. Build both PyTorch & TF models and compare parameter counts
4. Run multiple forward passes to check shapes & NaN/Inf
5. Verify that gradients actually flow (backprop) in both frameworks
6. Print backbone feature map shapes at each stage

### 4.1 Disabling GPU (TensorFlow is CPU-only in this project)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

* With `CUDA_VISIBLE_DEVICES=-1`, TensorFlow treats the system as having no
  CUDA-capable GPU and runs on CPU only.
* In actual run logs, you will see messages like:

  ```text
  CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
  env: CUDA_VISIBLE_DEVICES="-1"
  CUDA_VISIBLE_DEVICES is set to -1 - this hides all GPUs from CUDA
  XLA service ... initialized for platform Host
  ```

  → This means **all GPUs are hidden and only the Host (CPU) platform is used**,
  which is exactly what the script intends.

> Why do this?
>
> * On RTX 50 series (5090 etc.) with the current TensorFlow pip builds,
>   CUDA kernels for Compute Capability 12.0 are missing, and GPU execution often
>   produces errors such as `CUDA_ERROR_INVALID_PTX`,
>   `CUDA_ERROR_INVALID_HANDLE`.
> * Since this project’s goal is **structural / conversion verification**, the
>   tests are designed to completely avoid this issue and run **TensorFlow on
>   CPU only** for stability.

### 4.2 Global seed setup

```python
def set_global_seeds(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

* This does not make PyTorch and TF initializations bit‑identical, but it does
  improve basic reproducibility of the tests.

### 4.3 Comparing parameter counts

```python
pt_total, pt_trainable = count_torch_params(torch_model)
tf_total, tf_trainable, tf_non_trainable = count_tf_params(tf_model)
```

* PyTorch:

  * uses `model.parameters()`; here **all parameters are trainable**.
* TensorFlow:

  * sums over `trainable_weights` and `non_trainable_weights` separately.

Example log:

```text
==== Parameter counts ====
PyTorch trainable params:       214222
PyTorch total (동일):           214222
TensorFlow trainable params:    214222
TensorFlow non-trainable:       6888
TensorFlow total params:        221110
✓ Trainable parameter counts match between PyTorch and TensorFlow.
```

**Interpretation:**

* The number of **trainable parameters** is **exactly the same** →
  strong evidence that **layer structure and channel layout are matched across
  frameworks**.
* The 6,888 non‑trainable params on the TF side come from internal
  `BatchNormalization` statistics (e.g. `moving_mean`, `moving_variance`) and
  are an expected framework difference.

### 4.4 Forward shape & NaN/Inf checks

```python
forward_check(
    torch_model,
    tf_model,
    batch_size=2,
    time_steps=100,
    mel_bins=40,
    num_classes=2,
    n_trials=3,
)
```

For each trial:

* PyTorch:

  * `x_np ~ N(0,1)` → shape `(B, T, F)`
  * `(B, T, F)` → `torch_model` → `(B, 2)`
  * Check for NaN / Inf
* TensorFlow:

  * same `x_np` → `tf.convert_to_tensor`
  * `tf_model(x_tf, training=False)` → `(B, 2)`
  * Check for NaN / Inf

Example output:

```text
==== Forward shape & finite checks ====
✓ Forward passes succeeded on both frameworks (shapes match, no NaNs/Infs).
```

**Interpretation:**

* The input/output interface is identical for both frameworks:
  `(B, 100, 40) -> (B, 2)`.
* Multiple random inputs produce no NaN/Inf, indicating there are no obviously
  unstable operations in deeper parts (RepVGGDW, SE, group conv, BN, etc.).

### 4.5 Gradient smoke test

```python
n_pt = gradient_smoke_test_torch(...)
n_tf = gradient_smoke_test_tf(...)
```

* PyTorch:

  * `logits = model(x)` (with `requires_grad=True` on input)
  * `loss = logits.mean()` → `loss.backward()`
  * Count parameters whose gradients are non-zero.
* TensorFlow:

  * `with tf.GradientTape():`
  * `logits = model(x, training=True)`
  * `loss = reduce_mean(logits)`
  * `tape.gradient(loss, trainable_weights)` and count variables whose gradients
    are non-zero.

Example log:

```text
==== Gradient smoke tests ====
PyTorch:    143 parameters with non-zero gradients.
TensorFlow: 143 variables with non-zero gradients.
✓ Backprop works in both frameworks.
```

**Interpretation:**

* In both frameworks, a large number of parameters/variables receive non-zero
  gradients.
* If the structure were broken or if some module were blocking gradients, we
  would expect very few (or zero) non-zero gradients; instead we see 143 on both
  sides.
* Although the loss is simple (`mean(logits)`), this is a solid **smoke test
  that the computational graph is intact and trainable** for both frameworks.

### 4.6 Backbone feature map shapes

PyTorch:

```text
[PyTorch] Backbone feature map shapes:
  features[0]: (1, 24, 25, 10)
  features[1]: (1, 24, 25, 10)
  features[2]: (1, 24, 25, 10)
  features[3]: (1, 24, 25, 10)
  features[4]: (1, 48, 13, 10)
  ...
  features[11]: (1, 96, 7, 10)
  features[12]: (1, 96, 7, 10)
  features[13]: (1, 96, 7, 10)
```

TensorFlow:

```text
[TensorFlow] Backbone feature map shapes:
  patch_embed: (1, 25, 10, 24)
  block[0]: (1, 25, 10, 24)
  block[1]: (1, 25, 10, 24)
  block[2]: (1, 25, 10, 24)
  block[3]: (1, 13, 10, 48)
  ...
  block[10]: (1, 7, 10, 96)
  block[11]: (1, 7, 10, 96)
  block[12]: (1, 7, 10, 96)
```

**Interpretation:**

* PyTorch uses `(B, C, H, W)`, while TF uses `(B, H, W, C)`, but
  the values for H/W/C match exactly at every stage:

  * **time axis**: 100 → 50 → 25 → 13 → 7
  * **frequency axis**: 40 → 20 → 10 (then kept constant)
  * **channels**: 24 → 48 → 96
* This means that **patch embedding, stride (2,1) blocks, and channel expansion
  are all aligned 1:1 across the two frameworks**, and the RepViT design is
  faithfully reflected in both.

---

## 5. How to Run

### 5.1 Running the full smoke test

```bash
cd /home/skmoon/codes/251117_tf_convert  # or your cloned repo path
python smoke_test_repvit_convert.py
```

Key things to look for in the output:

* Parameter counts match:

  ```text
  PyTorch trainable params:       214222
  TensorFlow trainable params:    214222
  ✓ Trainable parameter counts match between PyTorch and TensorFlow.
  ```

* Forward & NaN checks pass:

  ```text
  ==== Forward shape & finite checks ====
  ✓ Forward passes succeeded on both frameworks (shapes match, no NaNs/Infs).
  ```

* Gradient smoke tests pass:

  ```text
  ==== Gradient smoke tests ====
  PyTorch:    143 parameters with non-zero gradients.
  TensorFlow: 143 variables with non-zero gradients.
  ✓ Backprop works in both frameworks.
  ```

* Backbone feature shapes:

  * Check that PyTorch `features[...]` and TF `patch_embed` / `block[...]` have
    matching H/W/C patterns (format differences aside).

### 5.2 If you only want a simple PyTorch model test

```bash
python repvit_kws.py
```

* The `__main__` block in `repvit_kws.py` creates a PyTorch model, runs a single
  forward pass on random input, and prints output shape and parameter count.

---

## 6. Limitations & Next Steps

### 6.1 What is already done

* Implemented a small RepViT-based KWS network in both PyTorch and TensorFlow.
* For both implementations:

  * **Trainable parameter counts match exactly (214,222)**
  * Multiple forward passes show **matching shapes and no NaNs/Infs**
  * Both PyTorch and TF have **broad, non-zero gradient coverage** over
    parameters
  * Backbone feature map shapes **match perfectly at each stage**
* TensorFlow runs in CPU-only mode; RTX 5090 + TF GPU compatibility issues are
  intentionally avoided.

### 6.2 What is not yet done (possible future work)

1. **PyTorch → TensorFlow weight mapping script**

   * Load a PyTorch `state_dict` and copy weights into corresponding TF layers
     with appropriate transposes.
   * Then perform a **numerical equivalence test** on identical inputs
     (e.g. L2 / max absolute difference between outputs).

2. **Aligning BatchNorm hyperparameters (optional)**

   * Use TF `BatchNormalization(epsilon=1e-5, momentum=0.9)` and
     PyTorch `BatchNorm2d(eps=1e-5, momentum=0.1)`
     so that running statistics behave more similarly across frameworks.

3. **Add training/evaluation scripts on a real KWS dataset**

   * Example: Google Speech Commands or your own KWS dataset.
   * Train the same architecture in both Torch and TF and compare performance.

---

## 7. Conclusion

This repository focuses on

> **“When we implement a RepViT KWS model in both PyTorch and TensorFlow,
> are these two implementations structurally the same model?”**

based on the smoke test results so far:

* Structure (layers / channels / strides)
* Trainable parameter counts
* Forward numerical stability
* Backward gradient flow

are all **consistent between the two frameworks**, and TensorFlow runs stably in
CPU-only mode.

Therefore:

> **“Is this architecture ready to be used as a base for real training/experiments?”**

From a **structure / conversion** standpoint,
**yes — it is at a level that can be used in practice**.
(For actual weight porting or TF GPU usage, the additional steps noted above are
recommended.)

```
```
