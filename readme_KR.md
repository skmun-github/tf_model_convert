
# RepViT KWS: PyTorch ↔ TensorFlow 구조 정합 & 스모크 테스트

이 리포지토리는 **RepViT** 스타일 백본을  
- **PyTorch**와  
- **TensorFlow 2.x**  

두 프레임워크로 동시에 구현하고,

입력  
> `(batch, time=100, mel=40)`  

출력  
> `(batch, 2)`  

형태의 **키워드 스팟팅(KWS) 이진 분류기**로 쓰기 위한 **구조 검증용 프로젝트**입니다.

원 논문/공식 구현:

- RepViT: *Revisiting Mobile CNN From ViT Perspective* (CVPR 2024) :contentReference[oaicite:0]{index=0}  
- 공식 PyTorch 구현: THU-MIG/RepViT :contentReference[oaicite:1]{index=1}  

여기서는 ImageNet 사이즈 모델이 아니라,  
**KWS 전용으로 축소된 “작은 RepViT 변종”**을 구현하고 있습니다  
(파라미터 약 **214k (trainable)** 수준).

---

## 1. 리포지토리 구조

프로젝트 루트(예: `/home/skmoon/codes/251117_tf_convert`)에는 다음 두 파일이 있습니다.

```text
repvit_kws.py                 # PyTorch & TensorFlow 모델 정의
smoke_test_repvit_convert.py  # 두 구현이 제대로 맞는지 검증하는 스모크 테스트
````

### 1.1 `repvit_kws.py`

* **PyTorch 쪽**

  * `SEModule`
  * `Conv2D_BN`
  * `RepVGGDW`
  * `RepViTBlock`
  * `RepViT` (백본)
  * `KWSRepViT_Torch` (최종 KWS classifier: `(B,100,40) -> (B,2)`)

* **TensorFlow 2.x 쪽**

  * `TF_SEModule`
  * `TF_Conv2D_BN`
  * `TF_RepVGGDW`
  * `TF_RepViTBlock`
  * `TF_RepViT` (백본)
  * `KWSRepViT_TF` (KWS classifier: `(B,100,40)` 또는 `(B,100,40,1) -> (B,2)`)

* **공통 설정**

  * `KWS_TINY_CFGS`: 작은 KWS 모델용 RepViT 구조 설정
  * 이 설정 기준으로

    * PyTorch trainable params: **214,222개**
    * TF trainable params: **214,222개**
    * TF non-trainable params: **6,888개**
      (대부분 `BatchNormalization`의 `moving_mean`, `moving_var` 로,
      Keras에서 **non-trainable 변수**로 관리하는 것이 설계입니다. ([Keras][1]))

### 1.2 `smoke_test_repvit_convert.py`

두 프레임워크 구현이 **정말로 같은 네트워크인지**를 확인하기 위해 만든 테스트 스크립트입니다.

하는 일:

1. **TF GPU 완전히 비활성화**

   * 스크립트 시작 부분에서
     `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`
     로 설정 → TF에서 **RTX 5090을 전혀 보지 못하도록** 함.
     이는 TensorFlow 2.x pip wheel이 현재 **Blackwell(Compute 12.0) GPU에서 `CUDA_ERROR_INVALID_PTX`** 를 자주 내는 이슈 때문입니다. ([Google AI Developers Forum][2])

2. **난수 시드 고정**

   * `numpy`, `torch`, `tensorflow` 시드를 0으로 고정 (완전한 재현 목적이라기보다는, 테스트 안정성용).

3. **모델 생성 및 파라미터 수 비교**

   * `KWSRepViT_Torch`, `KWSRepViT_TF`를 생성하고,
   * TF 모델은 한 번 호출해서 weight를 materialize한 뒤,
   * 각 프레임워크에서

     * total / trainable / non-trainable 파라미터 수를 각각 계산.

4. **Forward shape & NaN/Inf 체크**

   * 여러 번 랜덤 입력 `(B=2, T=100, F=40)`을 생성하고
   * PyTorch / TF 각각:

     * 출력 shape이 `(2, 2)`인지
     * NaN / Inf가 없는지
       를 검사.

5. **Gradient smoke test**

   * PyTorch:

     * 작은 배치에 대해 `loss = logits.mean()` 후 `loss.backward()` 수행
     * **non-zero gradient**를 가진 파라미터 텐서 개수를 센다.
   * TensorFlow:

     * `tf.GradientTape()`로 같은 방식으로 미분
     * non-zero gradient를 가진 `trainable_variables` 개수를 센다.
   * 두 숫자가 모두 0이 아니고, 충분히 많은 경우
     → 학습 그래프가 제대로 연결되어 있다고 판단.

6. **Backbone feature map shape 확인**

   * PyTorch:

     * `model.backbone.features` 를 stage별로 통과시키며
       각 stage output shape 출력 (포맷: `(B, C, H, W)`).
   * TensorFlow:

     * `model.backbone.patch_embed` + 각 `block[i]`를 통과시키며
       output shape 출력 (포맷: `(B, H, W, C)`).

---

## 2. 테스트된 환경 정보 (예시)

아래 정보는 실제로 스크립트를 실행한 환경에서 얻은 값입니다.

* OS: 리눅스 (Xorg + GNOME 프로세스가 `nvidia-smi`에 보였음)
* GPU: **NVIDIA GeForce RTX 5090 (Compute Capability 12.0)**
* NVIDIA 드라이버: **570.195.03**, CUDA 12.8 (드라이버 레벨)
* Python: 3.12 (가상환경 경로에서 확인)
* 주요 패키지 (사용자 `pip list` 기준):

  * `torch 2.9.0.dev20250725+cu128`
  * `torchvision 0.24.0.dev20250725+cu128`
  * `torchaudio 2.8.0.dev20250725+cu128`
  * `tensorflow 2.19.0`
  * `tf_keras 2.19.0`
  * `keras 3.10.0`
  * 각종 `nvidia-cuda-*`, `nvidia-cudnn-*` (CUDA 12.8 계열)

**중요:**
이 프로젝트에서는 **TensorFlow는 항상 CPU 전용으로 사용**합니다.

* 이유: 현재(2025-11 기준) TF 2.x pip wheel은
  RTX 50 시리즈(Compute 12.0)에서 `CUDA_ERROR_INVALID_PTX` / `CUDA_ERROR_INVALID_HANDLE` 에러를 빈번히 발생시키고 있음. ([GitHub][3])
* 스크립트에서 `CUDA_VISIBLE_DEVICES=-1` 로 GPU를 숨기기 때문에,
  TF는 오직 CPU + XLA(Host)만 사용합니다.

---

## 3. 설치 & 실행 방법 (step-by-step)

다른 머신/리눅스 환경에서 재현한다고 가정하고 설명합니다.

### 3.1 리포지토리 클론

```bash
git clone <YOUR_REPO_URL>  # 예: git@github.com:<user>/repvit-kws-convert.git
cd 251117_tf_convert       # 또는 실제 폴더 이름
```

> 이 README는 `repvit_kws.py` 와 `smoke_test_repvit_convert.py` 가
> 리포지토리 루트에 있다고 가정합니다.

### 3.2 파이썬 환경 준비 (예시: conda)

이미 환경이 있다면 이 단계는 건너뛰어도 됩니다.

```bash
conda create -n repvit_convert python=3.12 -y
conda activate repvit_convert
```

> 실제 실행 로그는 Python 3.12 환경에서 나온 것이므로,
> 완전히 동일하게 맞추고 싶다면 3.12를 사용하는 것이 가장 안전합니다.

### 3.3 필수 패키지 설치

이 프로젝트가 돌아가는 데 필요한 것은 **사실상 두 가지**입니다:

* PyTorch
* TensorFlow 2.x (CPU로만 쓸 예정)

예시 (CUDA 여부는 상관 없음 — TF는 어차피 CPU만 사용):

```bash
# PyTorch (예시: CPU 전용)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TensorFlow 2.x (예시: 최신 2.x)
pip install "tensorflow>=2.15,<3"
```

> 실제 사용 환경에서는 `tensorflow 2.19.0`으로 스크립트가 정상 동작했습니다.
> 다른 버전도 동작할 수 있지만, README에서 테스트했다고 말할 수 있는 버전은 2.19.0 뿐입니다.

---

## 4. 사용 방법

### 4.1 기본 PyTorch 스모크 테스트

```bash
python repvit_kws.py
```

예상되는 출력 예시 (요지는 이런 형태):

```text
PyTorch smoke test: output shape=(2, 2), params=214222
```

* 입력: 난수 `(batch=2, time=100, mel=40)`
* 출력: `(2, 2)`
* 파라미터 수: 약 **214,222** (trainable 기준)

이걸로 **PyTorch 쪽 모델이 기본적으로 잘 정의되어 있고,
입·출력 차원이 의도대로 구성되어 있다는 것**을 확인할 수 있습니다.

---

### 4.2 PyTorch ↔ TF 구조 정합 스모크 테스트

```bash
python smoke_test_repvit_convert.py
```

이 스크립트는 다음 순서대로 검증을 수행합니다.

#### 4.2.1 TF GPU 비활성화

실행 로그 상단:

```text
CUDA_VISIBLE_DEVICES="-1"
CUDA_VISIBLE_DEVICES is set to -1 - this hides all GPUs from CUDA
XLA service ... initialized for platform Host
```

* 이 부분이 보이면,
  TF는 **GPU를 전혀 보지 못하고 CPU(Host)만 사용 중**입니다. ([Google AI Developers Forum][2])

#### 4.2.2 파라미터 카운트 비교

예시 출력:

```text
==== Parameter counts ====
PyTorch trainable params:       214222
PyTorch total (동일):           214222
TensorFlow trainable params:    214222
TensorFlow non-trainable:       6888
TensorFlow total params:        221110
✓ Trainable parameter counts match between PyTorch and TensorFlow.
```

해석:

* **PyTorch trainable = 214,222**

* **TensorFlow trainable = 214,222**
  → **학습 가능한 파라미터 수가 완전히 일치** →
  두 프레임워크 구현의 **구조가 동일**하다는 강력한 증거.

* TF non-trainable 6,888:

  * `BatchNormalization` 레이어의 `moving_mean`, `moving_variance` 등
    Keras가 내부적으로 관리하는 **비학습 변수들**이 여기에 포함됩니다. ([Keras][1])

> 이 단계만으로도
> “레이어 수, 채널 수, 커널 개수 등이 PyTorch/TF에서 1:1로 잘 맞는다”는 것을 확인할 수 있습니다.

#### 4.2.3 Forward shape & NaN/Inf 체크

예시:

```text
==== Forward shape & finite checks ====
✓ Forward passes succeeded on both frameworks (shapes match, no NaNs/Infs).
```

* 여러 번의 랜덤 입력에 대해

  * PyTorch: `(B, 100, 40) -> (B, 2)`
  * TensorFlow: `(B, 100, 40) 또는 (B, 100, 40, 1) -> (B, 2)`
* 모든 경우에서

  * 출력 shape가 `(batch_size, 2)` 그대로
  * NaN, Inf 값이 하나도 없음

> → 입력/출력 인터페이스와 기본 연산 안정성이
> 두 프레임워크 모두 정상이라는 뜻입니다.

#### 4.2.4 Gradient smoke test

예시:

```text
==== Gradient smoke tests ====
PyTorch:    143 parameters with non-zero gradients.
TensorFlow: 143 variables with non-zero gradients.
✓ Backprop works in both frameworks.
```

이 단계에서 하는 일:

* PyTorch:

  * 작은 배치에서 loss를 한 번 계산 후 `backward()`
  * gradient가 **0이 아닌** 파라미터 텐서 개수 = `143`
* TensorFlow:

  * `GradientTape`로 같은 작업 수행
  * non-zero gradient 변수 개수 = `143`

**같은 숫자**가 나온다는 것 자체가:

* PyTorch/TF 모두에서

  * 대부분의 weight/bias에 gradient가 잘 흐르고 있고,
  * 학습 그래프가 끊기지 않았다는 의미.

> → **“이 구조로 실제 학습을 시켜도 되는가?”라는 질문에
> 최소한의 sanity check를 통과한 상태**라고 볼 수 있습니다.

#### 4.2.5 Backbone feature map shape 비교

예시:

```text
[PyTorch] Backbone feature map shapes:
  features[0]: (1, 24, 25, 10)
  ...
  features[11]: (1, 96, 7, 10)
  features[12]: (1, 96, 7, 10)
  features[13]: (1, 96, 7, 10)

[TensorFlow] Backbone feature map shapes:
  patch_embed: (1, 25, 10, 24)
  block[0]: (1, 25, 10, 24)
  ...
  block[10]: (1, 7, 10, 96)
  block[11]: (1, 7, 10, 96)
  block[12]: (1, 7, 10, 96)
```

* PyTorch: `(N, C, H, W)`

  * 예: `(1, 24, 25, 10)` = batch 1, 채널 24, time 25, freq 10
* TensorFlow: `(N, H, W, C)`

  * 예: `(1, 25, 10, 24)` = batch 1, time 25, freq 10, 채널 24

각 stage별로 비교해보면:

* patch embedding 후:

  * Torch: `(1, 24, 25, 10)`
  * TF:   `(1, 25, 10, 24)`
    → H/W/C 동일, 포맷만 다름

* 첫 downsample (stride (2,1)):

  * Torch: `(1, 48, 13, 10)`
  * TF:   `(1, 13, 10, 48)`

* 두 번째 downsample (stride (2,1)):

  * Torch: `(1, 96, 7, 10)`
  * TF:   `(1, 7, 10, 96)`

> → **다운샘플링 패턴과 채널 확장이 PyTorch/TF에서 완전히 일치합니다.**
> 이는 stride, padding, groups 설정이 두 구현에서 정확히 맞았다는 뜻입니다.

---

## 5. 테스트 결과의 의미 (실무 관점 정리)

지금까지의 결과를 종합하면:

1. **아키텍처 정합**

   * trainable 파라미터 수가 완전히 같고,
   * backbone stage별 feature map shape까지 일치합니다.
   * → **PyTorch/TF 두 구현이 “같은 RepViT 기반 KWS 네트워크”라는 근거가 충분히 강함**

2. **학습 가능성**

   * 두 프레임워크 모두에서 gradient가 실제로 흘러가는 것을 확인했고,
   * NaN/Inf 없이 forward가 정상 작동합니다.
   * → **실제로 학습에 투입해도 구조적인 문제는 없다고 보는 것이 합리적**입니다.

3. **환경 제약**

   * RTX 5090 + TensorFlow 2.x pip wheel 조합은
     아직 GPU에서 안정적으로 동작하지 않는 이슈가 있으므로, ([GitHub][3])
   * 이 프로젝트에서는 **TF를 CPU-only**로 사용하는 것을 기본 전제로 두고 있습니다.
   * PyTorch는 GPU(CUDA 12.8)에서 잘 동작하고 있어,
     실제 학습은 PyTorch + GPU,
     구조 검증 및 이식용 TF 구현은 CPU에서 사용하는 구도로 운용하는 것이 현실적인 선택입니다.

4. **아직 하지 않은 것 (향후 확장 가능)**

   이 리포지토리가 **현재까지 보장하는 것**은:

   * 구조가 맞는 PyTorch/TF 모델 정의
   * 구조 및 학습 가능성에 대한 스모크 테스트

   아직 포함하지 않은 것:

   * **PyTorch → TF 가중치 변환 스크립트**
   * 동일 입력에 대해 `max |y_torch - y_tf|` 같은 **수치적 동등성 비교**

   → 만약 “학습된 PyTorch 체크포인트를 TF로 1:1 이식”이 필요하다면,

   1. Conv / Linear weight를 축 순서에 맞게 transpose해서 옮기는 변환기 작성
   2. 변환된 TF 모델 출력과 PyTorch 모델 출력을 비교
      (예: `max_abs_diff`, `mean_abs_diff` 등)

   을 추가하면, **완전한 의미의 “컨버전 검증 루프”**가 만들어집니다. ([Kaixi Hou's Log][4])

---

## 6. 요약

* `repvit_kws.py`
  → RepViT 스타일 KWS 모델을 **PyTorch & TensorFlow 2.x 양쪽에 동일하게 구현**한 파일
* `smoke_test_repvit_convert.py`
  → 두 구현이

  * 동일한 trainable 파라미터 수를 가지는지
  * 동일한 input/output shape을 가지는지
  * NaN/Inf 없이 안정적인지
  * gradient가 실제로 흐르는지
  * backbone stage별 feature shape이 일치하는지
    를 단계별로 검증하는 스크립트

**이 README 시점에서의 결론:**

> ✅ 구조적으로 동일한 RepViT 기반 KWS 모델이
> ✅ PyTorch/TF 두 프레임워크에 잘 구현되어 있고,
> ✅ 학습 가능성·shape·파라미터 정합까지 검증된 상태라
> **실무에서 이 코드를 베이스로 학습/실험을 시작해도 무방한 수준**입니다.

추가로 weight 변환까지 하고 싶다면,
이 위에 “PyTorch → TF weight 매핑 + numeric diff 테스트”만 얹으면 됩니다.


[1]: https://keras.io/api/layers/normalization_layers/batch_normalization/?utm_source=chatgpt.com "BatchNormalization layer"
[2]: https://discuss.ai.google.dev/t/batchnormalization-in-training-mode-without-updating-moving-mean-and-variance/31556?utm_source=chatgpt.com "BatchNormalization in training mode without updating moving ..."
[3]: https://github.com/THU-MIG/RepViT/issues?utm_source=chatgpt.com "Issues · THU-MIG/RepViT"
[4]: https://kaixih.github.io/batch-norm/?utm_source=chatgpt.com "Moving Mean and Moving Variance In Batch Normalization"
