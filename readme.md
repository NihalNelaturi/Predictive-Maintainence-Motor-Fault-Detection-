# Predictive Maintenance — Motor Fault Detection

> Real-time, on-device motor fault classification using acoustic signals and embedded machine learning on a Silicon Labs EFR32MG26 microcontroller.

![Edge AI](https://img.shields.io/badge/Edge%20AI-On--Device%20Inference-teal)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.67%25-brightgreen)
![Classes](https://img.shields.io/badge/Fault%20Classes-6-blue)
![Board](https://img.shields.io/badge/Board-EFR32MG26%20BRD2608A-orange)
![Dataset](https://img.shields.io/badge/Dataset-CWRU%20Bearing-lightgrey)

---

## 🚀 Overview

Industrial motors fail progressively — vibration signatures and acoustic anomalies appear long before catastrophic breakdown. This project brings **predictive maintenance to the edge**: a custom MFCC-based audio feature pipeline combined with a compact logistic regression classifier (only **11,472 parameters**) runs entirely on a microcontroller, detecting 5 distinct fault types in real time with no cloud dependency.

The system captures 1-second PCM audio windows from an I2S microphone at 16 kHz, extracts MFCC + delta + delta-delta features, and classifies the motor's health state every **150 ms** — all within the memory and compute constraints of a Cortex-M33 MCU (EFR32MG26).

**Key result:** **99.67% test accuracy** and **98.44% macro-F1** on the cleaned CWRU benchmark dataset. The quantized INT8 model retains **99.58% accuracy** at only **12.7 KB** in flash.

**Why it matters:** Traditional PdM solutions require separate accelerometers, DAQ boards, and cloud pipelines. This solution uses a single microphone and runs entirely on a $5 microcontroller — deployable in any motor enclosure with zero cloud dependency and zero raw audio transmission.

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Multinomial Logistic Regression (single dense layer + softmax) |
| Feature type | MFCC + Δ (delta) + ΔΔ (delta-delta) |
| MFCC coefficients | 13 per frame |
| Feature frames per window | 49 |
| Feature channels per frame | 39 (13 MFCC + 13 Δ + 13 ΔΔ) |
| Total feature dimension | **1,911** (49 × 39, flattened) |
| Output classes | 6 |
| Total parameters | **11,472** |
| Float model size | **≈ 44.81 KB** |
| TFLite INT8 size | **13,008 bytes (12.7 KB)** |
| Inference path | Direct embedded weights — no TFLite interpreter at runtime |
| Normalization | StandardScaler absorbed into weights — no runtime scaler block |
| Audio window | 1 second @ 16 kHz (16,000 samples) |
| FFT size | 512-point |
| Mel filterbanks | 40 |
| Frame length / step | 480 / 320 samples (30 ms / 20 ms) |
| Inference interval | 150 ms |

### Fault Classes

| Index | Class Label | Display Name | LED |
|-------|-------------|--------------|-----|
| 0 | `normal` | Healthy | Green |
| 1 | `ball_fault` | Ball Fault | Red |
| 2 | `belt_fault` | Belt Fault | Red |
| 3 | `inner_race` | Inner Race | Red |
| 4 | `misalign_loose` | Misalign Loose | Red |
| 5 | `outer_race` | Outer Race | Red |

---

## ⚙️ Hardware Used

| Component | Details |
|-----------|---------|
| MCU | Silicon Labs **EFR32MG26** (BRD2608A) — Cortex-M33 |
| Microphone | I2S digital microphone (DMA-backed, 16 kHz) |
| LED — Green | Healthy motor state |
| LED — Red | Any fault condition detected |
| Serial output | UART / VCOM @ 115200 baud — class label + confidence % |
| Cloud / Raw Audio | **None** — fully closed, edge-only inference |
| SDK | Silicon Labs Gecko SDK |
| Build | Simplicity Studio 5 / CMake + GCC |

---

## 📊 Results

### Accuracy Metrics

| Metric | Float — Validation | Float — Test | Quantized INT8 — Test |
|--------|-------------------|--------------|----------------------|
| **Accuracy** | **99.91%** | **99.67%** | **99.58%** |
| **Macro-F1** | **99.56%** | **98.44%** | **97.996%** |

> Evaluated on the cleaned **CWRU Bearing Dataset** (2,134 test samples). The quantized INT8 model retains near-identical accuracy at **12.7 KB** — a 3.5× size reduction from the float model.

---

### Per-Class Performance

| Class | Performance | Notes |
|-------|-------------|-------|
| **Healthy** | Near-perfect | Strong spectral contrast vs all fault classes |
| **Ball Fault** | Near-perfect | Distinct high-frequency amplitude modulation signature |
| **Belt Fault** | Slightly reduced | Occasional confusion with Misalign Loose (see below) |
| **Inner Race** | Near-perfect | Characteristic bearing pass frequency clearly separable |
| **Misalign Loose** | Slightly reduced | Occasional confusion with Belt Fault (see below) |
| **Outer Race** | Near-perfect | Strong outer race pass frequency in mel spectrum |

> **Primary confusion pair: Belt Fault ↔ Misalign Loose.** These two classes share mechanically similar vibration signatures — both manifest as broadband spectral energy increase without strong harmonic structure. This is the hardest distinction for any acoustic-based classifier on the CWRU dataset. All other classes are effectively near-perfect in offline evaluation.

---

### Confusion Analysis

```
Predicted →      Healthy  Ball  Belt  Inner  Misalign  Outer
Actual ↓
Healthy          ████████  ·     ·     ·      ·         ·
Ball Fault        ·        ████  ·     ·      ·         ·
Belt Fault        ·        ·    ████   ·     ▒▒▒        ·      ← some confusion
Inner Race        ·        ·     ·    ████   ·          ·
Misalign Loose    ·        ·    ▒▒▒    ·     ████       ·      ← some confusion
Outer Race        ·        ·     ·     ·      ·        ████

  ████ = near-perfect   ▒▒▒ = occasional misclassification   · = negligible
```

> Full numeric confusion matrix can be reproduced from `training_report.json` using the training artifacts.

---

### Example Serial Output (UART @ 115200 baud)

```
[CALIBRATING] Window  8/50 | RMS: 0.0231
[CALIBRATING] Window 25/50 | RMS: 0.0228
[CALIBRATING] Baseline locked. RMS mean: 0.0226 | CV: 0.09 | Gate: 0.0339

[PREDICT] Healthy        | P=[0.97, 0.01, 0.01, 0.00, 0.01, 0.00] | RMS: 0.0229 | Sim: 0.94
[PREDICT] Healthy        | P=[0.96, 0.01, 0.01, 0.01, 0.01, 0.00] | RMS: 0.0231 | Sim: 0.93
[PREDICT] Ball Fault     | P=[0.08, 0.89, 0.01, 0.01, 0.01, 0.00] | RMS: 0.0411 | Sim: 0.67
[PREDICT] Ball Fault     | P=[0.06, 0.91, 0.01, 0.01, 0.01, 0.00] | RMS: 0.0398 | Sim: 0.65
[PREDICT] Ball Fault     | P=[0.07, 0.88, 0.02, 0.01, 0.02, 0.00] | RMS: 0.0403 | Sim: 0.66
[FAULT LATCHED] Ball Fault — 3 consecutive detections (450 ms)

[PREDICT] Inner Race     | P=[0.05, 0.02, 0.01, 0.71, 0.12, 0.09] | RMS: 0.0378 | Sim: 0.61
[FAULT LATCHED] Inner Race | 71.2%
```

---

## 📦 Dataset

| Partition | Samples | Proportion |
|-----------|---------|------------|
| **Total** | 14,420 | 100% |
| Train | 10,158 | 70.4% |
| Validation | 2,128 | 14.8% |
| Test | 2,134 | 14.8% |

- **Source:** [Case Western Reserve University (CWRU) Bearing Dataset](https://engineering.case.edu/bearingdatacenter) — industry-standard benchmark for bearing fault detection
- **Preprocessing:** StandardScaler normalization applied during training; **scaler parameters absorbed directly into the logistic regression weights and bias** at export — the embedded firmware runs no separate normalization step at runtime
- **Training script:** `train_cwru_mfcc_multiclass_tinyml.py` (host-side, not deployed to device)
- **Reference artifacts:** `training_report.json`, `cwru_binary_report.json`

---

## ⚡ Performance & Build Metrics

| Metric | Value |
|--------|-------|
| Test accuracy (float) | **99.67%** |
| Test Macro-F1 (float) | **98.44%** |
| Validation accuracy (float) | **99.91%** |
| Quantized INT8 accuracy | **99.58%** |
| Total model parameters | **11,472** |
| Float model size | **≈ 44.81 KB** |
| TFLite INT8 model size | **13,008 bytes (12.7 KB)** |
| Firmware binary (.bin) | **390,656 bytes (≈ 381.5 KB)** |
| .text section | ≈ 389.8 KB |
| .bss section | ≈ 295.1 KB *(inflated — see note)* |
| Inference update period | 150 ms |
| Calibration time | ~7.5 s at boot (50 windows) |
| End-to-end HW latency | Not yet instrumented (pending) |

> **RAM note:** The `.bss` section is inflated by stock SDK / TFLM template components that remain linked, including a **100 KB tensor arena** from the original TFLM framework scaffold. The multiclass inference path itself does not use the TFLM interpreter — this is a known optimization target for a future pass.

---

## 🚀 Deployment Status

| Milestone | Status |
|-----------|--------|
| Multiclass Training | ✅ Complete |
| Embedded Export | ✅ Complete |
| Firmware Integration | ✅ Complete |
| Firmware Build & Flash | ✅ Complete |
| Latency Instrumentation | ⏳ Pending |
| Real-World Field Test | ⏳ Pending |

---

## 📋 Challenges

### 1. Belt Fault ↔ Misalign Loose Confusion
The primary accuracy bottleneck is distinguishing **Belt Fault** from **Misalign Loose**. Both classes produce similar broadband vibration signatures without strong harmonic structure — mechanically, both involve reduced rotational stiffness. The MFCC representation does not cleanly separate these in spectral space, making this the hardest class pair for any acoustic-only classifier on CWRU data.

**Mitigation path:** A time-frequency feature (e.g., short-time Fourier magnitude with finer frequency resolution in the 50–500 Hz band) or a CNN operating on the raw log-mel spectrogram image could better capture the subtle envelope differences between these two classes.

### 2. Domain Gap (Lab → Real World)
The CWRU dataset is recorded under controlled laboratory conditions with fixed microphone placement. Real-world deployments introduce varying background noise, different motor mounts, and varying microphone distances — all of which shift the mel spectrogram distribution away from training conditions, potentially degrading accuracy.

**Mitigation path:** Collect supplementary recordings from the target deployment environment and fine-tune or retrain with augmented data (noise injection, RMS normalization, pitch shift).

### 3. RAM Overhead from TFLM Template
Current firmware links unused TFLM/SDK scaffold components including a **100 KB tensor arena** in `.bss`. This can be removed by trimming the linker script and component list in the `.slcp` project, reducing RAM usage significantly.

### 4. End-to-End Latency Not Instrumented
The 150 ms update period is the inference scheduling interval, not a measured hardware latency figure. Actual end-to-end latency from PCM capture to UART output has not been separately timed using GPIO instrumentation or cycle counters.

---

## 🛠️ Future Improvements

- **CNN on log-mel spectrogram:** Replace logistic regression with a lightweight 1D-Conv or 2D-Conv network operating on the 49×40 log-mel input. Even a 3-layer CNN with INT8 quantization would significantly improve Belt Fault / Misalign Loose separation while remaining within EFR32MG26 memory constraints.
- **Remove unused TFLM scaffold:** Trim `.slcp` components to eliminate the 100 KB tensor arena from `.bss`, recovering significant RAM for a future RTOS feature or larger PCM buffer.
- **GPIO latency instrumentation:** Instrument `audio_classifier_task` with GPIO toggles at PCM copy start and UART output, then measure on oscilloscope to get the real end-to-end inference latency figure.
- **Real-world field validation:** Deploy to a physical motor test rig with known fault conditions to validate the 99.67% CWRU accuracy translates to the real deployment domain.
- **OTA model updates:** Use the Silicon Labs OTA bootloader to push retrained weight arrays (`mfcc_multiclass_model_data.c`) without full firmware reflash.
- **Anomaly detection pre-filter:** Add a one-class SVM or PCA-based anomaly scorer on the healthy baseline to gate the classifier — reject inputs structurally unlike any trained class before running softmax.

---

## 💡 Why This Project Matters

| Aspect | Value |
|--------|-------|
| **Edge AI** | No cloud, no latency, no connectivity — works in isolated industrial environments |
| **Cost** | Single MCU + microphone replaces expensive vibration DAQ hardware |
| **Privacy** | Raw audio never leaves the device — relevant in regulated manufacturing environments |
| **Model efficiency** | 11,472 parameters in 12.7 KB (INT8) achieving 99.58% accuracy — extreme parameter efficiency |
| **Full ML pipeline** | Data → feature engineering → training → embedded deployment → real-time inference |
| **Latency** | 150 ms inference loop enables near-real-time fault alerting |

This project demonstrates the complete embedded ML pipeline end-to-end: **data collection → MFCC feature engineering → logistic regression training → StandardScaler weight absorption → embedded C export → RTOS firmware integration → real-time on-device inference** — entirely on a microcontroller.

---

## 🧪 How to Run

### Prerequisites

- [Simplicity Studio 5](https://www.silabs.com/developers/simplicity-studio)
- Silicon Labs Gecko SDK (version matching `Final2o.slcp`)
- EFR32MG26 dev board (BRD2608A)
- Serial terminal (Tera Term / PuTTY) at **115200 baud**

### Import into Simplicity Studio

```
1. File > Import > General > Existing Projects into Workspace
2. Browse to the cloned repository root
3. Select "Final2o" → Finish
4. Simplicity Studio auto-generates autogen/ on first build
```

### Build & Flash

```
1. Right-click project → Build Project
2. Connect BRD2608A via USB
3. Run > Flash  (or F11 for debug session)
```

### Runtime Behaviour

```
Boot (~7.5 seconds):
  Motor must run in healthy state during calibration
  Serial: [CALIBRATING] Window N/50 | RMS: X.XXXX

After calibration (every 150 ms):
  1. Copy 1-second PCM window (16,000 samples)
  2. Extract MFCC + delta + delta-delta → 1,911 features
  3. Logistic regression: dot(weight, features) + bias → 6 logits
  4. Softmax → class probabilities
  5. EMA smoothing (α=0.45)
  6. Fault latch logic (3 frames to trigger, 5 to clear)
  7. UART: "Class | Confidence%"  |  LED: Green/Red
```

### Configuration

Edit [config/audio_classifier_config.h](config/audio_classifier_config.h):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FAULT_DETECT_THRESHOLD` | 0.50 | Min fault class probability to trigger |
| `FAULT_TRIGGER_COUNT` | 3 | Consecutive fault frames before latch (450 ms) |
| `FAULT_CLEAR_COUNT` | 5 | Consecutive healthy frames before release (750 ms) |
| `HEALTHY_CALIBRATION_WINDOWS` | 50 | Baseline length (~7.5 s) |
| `MULTICLASS_EMA_ALPHA` | 0.45 | Smoothing factor (higher = faster response) |
| `INFERENCE_INTERVAL_MS` | 150 | Inference scheduling period |

---

## 📁 Repository Structure

```
Predictive-Maintenance-Motor-Fault-Detection/
├── main.c                          # RTOS entry, task spawn
├── app.c / app.h                   # Application-level init
├── audio_classifier.cc / .h        # Main inference loop (RTOS task)
│                                   #   EMA smoothing, fault latch, LED/UART output
├── mfcc_multiclass_inference.cc    # MFCC + delta pipeline + logistic regression
├── mfcc_multiclass_inference.h
├── mfcc_multiclass_model_data.c    # Trained weights, biases, mel/DCT matrices
├── mfcc_multiclass_model_data.h    # Model constants (1,911 features, 6 classes)
├── motor_pcm_buffer.c / .h         # DMA PCM ring buffer (I2S mic input)
├── recognize_commands.cc / .h      # Command smoothing compatibility layer
├── config/
│   ├── audio_classifier_config.h   # All tunable thresholds and timing parameters
│   ├── pin_config.h                # GPIO pin assignments (BRD2608A)
│   ├── sl_mic_i2s_config.h         # I2S microphone driver config
│   └── tflite/                     # Source .tflite file (quantized INT8 reference)
├── autogen/                        # Simplicity Studio auto-generated drivers
├── cmake_gcc/                      # CMake + GCC build system
└── docs/
    └── system_architecture.md      # Full pipeline description and signal processing detail
```

---

## 👤 Author

**Nihal Nelaturi**
Embedded Systems & Machine Learning Engineer

- GitHub: [github.com/NihalNelaturi](https://github.com/NihalNelaturi)
- Repository: [Predictive-Maintainence-Motor-Fault-Detection-](https://github.com/NihalNelaturi/Predictive-Maintainence-Motor-Fault-Detection-)

---

## 📚 References

- [CWRU Bearing Fault Dataset](https://engineering.case.edu/bearingdatacenter) — Case Western Reserve University
- [Silicon Labs Machine Learning Documentation](https://docs.silabs.com/machine-learning/latest/aiml-developing-with)
- [Silicon Labs MLTK](https://siliconlabs.github.io/mltk)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [EFR32MG26 Product Page](https://www.silabs.com/wireless/proprietary/efr32xg26-series-2)
