
# RepViT KWS: PyTorch ↔ TensorFlow Structure Verification Project

यह रिपोज़िटरी एक **small RepViT‑based KWS (Keyword Spotting) model** को  
दोनों फ्रेमवर्क में इम्प्लीमेंट करती है:

- **PyTorch**  
- **TensorFlow 2 (tf.keras)**  

और एक **smoke test** प्रदान करती है, जो काफ़ी सख़्ती से यह जाँचने के लिए बनाया गया है कि:

> “क्या इन दोनों implementations की **model structure वास्तव में समान** है?”

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

      * `RepViT` backbone (MetaFormer‑style, depthwise/group conv + SE)
      * `AdaptiveAvgPool2d(1)` + `Linear` classifier
    * Output: `(B, 2)` → binary KWS logits (yes/no)
  * यह मॉडल `KWS_TINY_CFGS` नाम के small config को उपयोग करता है,
    ताकि कुल लगभग **214,222 trainable parameters** हों।

* **TensorFlow side**

  * `KWSRepViT_TF`

    * Input: `(B, 100, 40)` या `(B, 100, 40, 1)`
      (दूसरे केस में channel dimension अपने आप जोड़ लिया जाता है)
    * Internals:

      * `TF_RepViT` backbone (channels_last, `Conv2D(groups=...)` का उपयोग)
      * `GlobalAveragePooling2D` + `Dense(num_classes=2)`
    * Output: `(B, 2)` logits
  * वही `KWS_TINY_CFGS` config पर आधारित है, और इस तरह डिज़ाइन किया गया है कि
    **structure और parameter counts PyTorch model के साथ match करें**।

> ⚠️ **Important**
> यह फ़ाइल केवल **architecture definitions** के लिए है।
> **PyTorch → TF weight transfer script** अभी शामिल नहीं है।
> अभी तक का फ़ोकस यह है कि “structure वास्तव में समान है या नहीं” इसे सख़्ती से verify किया जाए।

---

## 2. Requirements & Installation

### 2.1 Required packages

* Python 3.11+ (Python 3.12 पर टेस्ट किया गया)
* PyTorch 2.x (यह प्रोजेक्ट केवल CPU मोड का उपयोग करता है)
* TensorFlow 2.x (2.19.0 पर टेस्ट किया गया, और **CPU mode** में चलाया गया)
* Numpy

TensorFlow implementation `Conv2D(groups=...)` का उपयोग करता है,
जो **TensorFlow 2.4 से officially supported** है।

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
> 2025 तक pip से इंस्टॉल किए गए TensorFlow 2.19/2.20 में
> **Compute Capability 12.0 (Blackwell)** के लिए CUDA kernels शामिल नहीं हैं,
> और RTX 5090/5080/5070 पर GPU execution के दौरान अक्सर
> `CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_INVALID_HANDLE` जैसे errors रिपोर्ट हुए हैं,
> जो TensorFlow issue trackers में documented हैं।

इसी वजह से, इस प्रोजेक्ट के सभी smoke tests इस तरह डिज़ाइन किए गए हैं कि
**TensorFlow हमेशा CPU-only mode में ही चले**।

---

## 3. PyTorch / TensorFlow Model Structure Summary

### 3.1 Shared design idea

छोटे RepViT configuration `KWS_TINY_CFGS` का उपयोग करते हुए:

* Patch embedding:

  * दो `stride=2` convolution layers
    → time: 100 → 50 → 25
    → freq: 40 → 20 → 10
* बाद के कुछ blocks `stride=(2, 1)` इस्तेमाल करते हैं, जो
  **सिर्फ़ time axis को downsample** करते हैं (25 → 13 → 7),
  और frequency axis (40→20→10) को stable रखते हैं।
* Channels: 24 → 48 → 96 तक grow होती हैं।

Final feature map:

* PyTorch: `(B, 96, 7, 10)`
* TensorFlow: `(B, 7, 10, 96)`

Global average pooling से `(B, 96)` मिलता है, जिसके बाद `Dense/Linear(2)` से
binary classification किया जाता है।

### 3.2 PyTorch implementation (`KWSRepViT_Torch`)

* Input: `(B, 100, 40)` → `unsqueeze(1)` → `(B, 1, 100, 40)`
* `RepViT` backbone:

  * `Conv2D_BN` (1→12, kernel_size=3, stride=2, padding=1)
  * `Conv2D_BN` (12→24, kernel_size=3, stride=2, padding=1)
  * इसके बाद 13 `RepViTBlock` sequentially
* `AdaptiveAvgPool2d(1)` → `(B, 96, 1, 1)` → flatten → `Linear(96→2)`

BatchNorm PyTorch defaults (`eps=1e-5`, `momentum=0.1`) उपयोग करता है।

### 3.3 TensorFlow implementation (`KWSRepViT_TF`)

* Input: `(B, 100, 40)`; अगर rank 3 है तो `expand_dims` से `(B, 100, 40, 1)` बनाया जाता है।
* `TF_RepViT` backbone:

  * `TF_Conv2D_BN` (1→12, ks=3, stride=2, pad=1)
  * `TF_Conv2D_BN` (12→24, ks=3, stride=2, pad=1)
  * इसके बाद 13 `TF_RepViTBlock`

    * depthwise / group convolution `tf.keras.layers.Conv2D(groups=...)` से implement की गई है
* `GlobalAveragePooling2D()` → `(B, 96)` → `Dense(2)`

BatchNorm Keras defaults (`epsilon=1e-3`, `momentum=0.99`) उपयोग करता है।

> 💡 **Important practical point**
>
> * PyTorch और TensorFlow दोनों के लिए **trainable parameters की संख्या बिल्कुल समान** है।
>   (smoke test में यह 214,222 trainable params के रूप में verify किया गया है।)
> * BatchNorm के epsilon/momentum defaults दोनों फ्रेमवर्क में अलग हैं।
>   अगर आपका लक्ष्य **bit-exact PyTorch pretrained weight → TF porting** है,
>   तो आप बाद में इन hyperparameters को align करना चाहेंगे।
>   लेकिन अगर आप **KWS के लिए दोनों models को स्वतंत्र रूप से train** कर रहे हैं,
>   तो current defaults practical रूप से काफ़ी ठीक हैं।

---

## 4. Smoke Test: `smoke_test_repvit_convert.py`

यह script स्वतः निम्न चीज़ें करता है:

1. **TensorFlow को CPU-only पर मजबूर करना** (GPU को छिपा कर)
2. Global seeds fix करना
3. PyTorch और TF दोनों models बनाकर parameter counts compare करना
4. Multiple forward passes चलाकर shapes और NaN/Inf की जाँच करना
5. दोनों frameworks में gradients सही से flow हो रहे हैं या नहीं, यह verify करना
6. Backbone के हर stage पर feature map shapes प्रिंट करना

### 4.1 Disabling GPU (TensorFlow is CPU-only in this project)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

* `CUDA_VISIBLE_DEVICES=-1` होने पर TensorFlow सिस्टम को
  **कोई CUDA-capable GPU नहीं** मानता है और केवल CPU पर run करता है।
* रन लॉग में आपको कुछ ऐसे messages दिखेंगे:

  ```text
  CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
  env: CUDA_VISIBLE_DEVICES="-1"
  CUDA_VISIBLE_DEVICES is set to -1 - this hides all GPUs from CUDA
  XLA service ... initialized for platform Host
  ```

  → इसका मतलब है कि **सभी GPUs छुपाए जा चुके हैं और केवल Host (CPU) platform उपयोग में है**,
  जो ठीक वही behavior है जिसे यह script target कर रहा है।

> Why do this?
>
> * RTX 50 series (5090 आदि) + current TensorFlow pip builds के साथ
>   Compute Capability 12.0 के लिए CUDA kernels missing हैं, और GPU execution के दौरान
>   `CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_INVALID_HANDLE` जैसे errors अक्सर देखे गए हैं।
> * इस project का goal **structure / conversion verification** है,
>   इसलिए tests को जानबूझकर ऐसे डिज़ाइन किया गया है कि TensorFlow **केवल CPU पर** चले,
>   ताकि environment stable रहे।

### 4.2 Global seed setup

```python
def set_global_seeds(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

* यह PyTorch और TF की initializations को bit‑identical नहीं बनाता,
  लेकिन tests की basic reproducibility बेहतर होती है।

### 4.3 Comparing parameter counts

```python
pt_total, pt_trainable = count_torch_params(torch_model)
tf_total, tf_trainable, tf_non_trainable = count_tf_params(tf_model)
```

* PyTorch:

  * `model.parameters()` के ज़रिये parameters count करता है;
    इस implementation में सभी parameters trainable हैं।
* TensorFlow:

  * `trainable_weights` और `non_trainable_weights` को अलग-अलग sum करता है।

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

* **Trainable parameters** की संख्या दोनों frameworks में बिल्कुल समान है →
  यह मज़बूत संकेत है कि **layer structure और channel layout** PyTorch और TF दोनों में
  1:1 तरीके से match कर रहे हैं।
* TF के 6,888 non‑trainable params
  `BatchNormalization` layers की internal statistics (`moving_mean`, `moving_variance` आदि) हैं,
  जो कि framework-level expected difference है।

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

हर trial में:

* **PyTorch**:

  * `x_np ~ N(0, 1)` → `(B, T, F)`
  * `(B, T, F)` → `torch_model` → `(B, 2)`
  * Output में NaN/Inf मौजूद है या नहीं, यह चेक किया जाता है।
* **TensorFlow**:

  * वही `x_np` → `tf.convert_to_tensor`
  * `tf_model(x_tf, training=False)` → `(B, 2)`
  * Output में NaN/Inf check किया जाता है।

Example output:

```text
==== Forward shape & finite checks ====
✓ Forward passes succeeded on both frameworks (shapes match, no NaNs/Infs).
```

**Interpretation:**

* Input/output interface दोनों frameworks में पूरी तरह consistent है:
  `(B, 100, 40) -> (B, 2)`.
* कई random inputs पर भी NaN/Inf नहीं दिखते,
  जिसका अर्थ है कि deep layers (RepVGGDW, SE, group conv, BN आदि) में
  कोई स्पष्ट numerical instability नहीं है।

### 4.5 Gradient smoke test

```python
n_pt = gradient_smoke_test_torch(...)
n_tf = gradient_smoke_test_tf(...)
```

* **PyTorch**:

  * Input पर `requires_grad=True` के साथ `logits = model(x)`
  * `loss = logits.mean()` → `loss.backward()`
  * Non-zero gradients वाले parameters की संख्या count की जाती है।
* **TensorFlow**:

  * `with tf.GradientTape():`
  * `logits = model(x, training=True)`
  * `loss = reduce_mean(logits)`
  * `tape.gradient(loss, trainable_weights)` के बाद,
    non-zero gradients वाले variables count किए जाते हैं।

Example log:

```text
==== Gradient smoke tests ====
PyTorch:    143 parameters with non-zero gradients.
TensorFlow: 143 variables with non-zero gradients.
✓ Backprop works in both frameworks.
```

**Interpretation:**

* दोनों frameworks में बड़ी संख्या में parameters/variables को non-zero gradients मिलते हैं।
* अगर structure गलत होता या कोई module gradient को block कर रहा होता,
  तो non-zero gradients की count बहुत कम या 0 होती;
  लेकिन यहाँ PyTorch और TensorFlow दोनों में 143 दिख रहे हैं।
* Loss भले ही simple है (`mean(logits)`), लेकिन यह एक अच्छा **smoke test** है
  कि computational graph दोनों frameworks में intact और trainable है।

### 4.6 Backbone feature map shapes

**PyTorch:**

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

**TensorFlow:**

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

* PyTorch `(B, C, H, W)` format उपयोग करता है, जबकि TensorFlow `(B, H, W, C)`।
  Format अलग है लेकिन H/W/C के numeric मान हर stage पर match कर रहे हैं:

  * **time axis**: 100 → 50 → 25 → 13 → 7
  * **frequency axis**: 40 → 20 → 10 (उसके बाद stable)
  * **channels**: 24 → 48 → 96
* इसका मतलब है कि **patch embedding, stride (2,1) वाले blocks, और channel expansion**
  दोनों implementations में 1:1 aligned हैं,
  और RepViT design को faithful तरीके से PyTorch और TensorFlow दोनों में reproduce किया गया है।

---

## 5. How to Run

### 5.1 Running the full smoke test

```bash
cd /home/skmoon/codes/251117_tf_convert  # या आपका clone किया हुआ repo path
python smoke_test_repvit_convert.py
```

Output में आपको मुख्य रूप से ये चीज़ें देखनी चाहिए:

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

  * PyTorch के `features[...]` और TF के `patch_embed` / `block[...]` के
    H/W/C patterns (format differences छोड़कर) आपस में match होने चाहिए।

### 5.2 If you only want a simple PyTorch model test

```bash
python repvit_kws.py
```

* `repvit_kws.py` की `__main__` block एक PyTorch model बनाती है,
  random input पर एक forward pass चलाती है,
  और output shape व parameter count print करती है।

---

## 6. Limitations & Next Steps

### 6.1 What is already done

* एक small RepViT‑based KWS नेटवर्क PyTorch और TensorFlow दोनों में implement किया गया है।
* दोनों implementations के लिए:

  * **Trainable parameter counts बिल्कुल समान (214,222)**
  * Multiple forward passes में **shapes match और NaN/Inf नहीं**
  * PyTorch और TF दोनों में parameters/variables पर
    **काफ़ी व्यापक non‑zero gradient coverage**
  * Backbone feature map shapes हर stage पर **perfectly match** करते हैं।
* TensorFlow CPU‑only mode में चलता है; RTX 5090 + TF GPU compatibility issues
  जानबूझकर avoid किए गए हैं।

### 6.2 What is not yet done (possible future work)

1. **PyTorch → TensorFlow weight mapping script**

   * PyTorch `state_dict` को load करके,
     सही axis transpose के साथ संबंधित TF layers में weights copy करना।
   * फिर identical inputs पर **numerical equivalence test** चलाना
     (जैसे कि outputs के बीच L2 / max absolute difference मापना)।

2. **Aligning BatchNorm hyperparameters (optional)**

   * TF में `BatchNormalization(epsilon=1e-5, momentum=0.9)`
     और PyTorch में `BatchNorm2d(eps=1e-5, momentum=0.1)` जैसा setup उपयोग करके,
     running statistics दोनों frameworks में और भी ज़्यादा similar बनाए जा सकते हैं।

3. **Add training/evaluation scripts on a real KWS dataset**

   * उदाहरण: Google Speech Commands या आपका खुद का KWS dataset।
   * उसी architecture को Torch और TF दोनों में train करके
     performance की तुलना की जा सकती है।

---

## 7. Conclusion

यह repository मुख्य रूप से इस सवाल पर focus करती है:

> **“जब हम RepViT KWS model को PyTorch और TensorFlow दोनों में implement करते हैं,
> तो क्या ये दोनों implementations structurally उसी एक model को represent करते हैं?”**

अब तक के smoke test results के आधार पर:

* Structure (layers / channels / strides)
* Trainable parameter counts
* Forward numerical stability
* Backward gradient flow

ये सब दोनों frameworks के बीच **काफ़ी उच्च स्तर पर consistent** हैं,
और TensorFlow CPU-only mode में stable तरीके से चल रहा है।

इसलिए:

> **“क्या यह architecture वास्तविक training/experiments के base के रूप में
> उपयोग करने के लिए तैयार है?”**

**Structure / conversion** के दृष्टिकोण से,
**हाँ — यह practical उपयोग के लिए पर्याप्त mature स्थिति में है**।
(Actual weight porting या TF GPU usage के लिए, ऊपर बताए गए अतिरिक्त steps
follow करने की सिफ़ारिश की जाती है।)

```
```
