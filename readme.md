# Predictive Maintenance — Motor Fault Detection

> Real-time, on-device motor fault classification using acoustic signals and embedded machine learning on a Silicon Labs EFR32xG26 microcontroller.

---

## 🚀 Overview

Industrial motors fail progressively — vibration signatures and acoustic anomalies appear long before catastrophic breakdown. This project brings **predictive maintenance to the edge**: a custom MFCC-based audio feature pipeline combined with a lightweight linear classifier runs entirely on a microcontroller, detecting 5 distinct fault types in real time with no cloud dependency.

The system captures 1-second PCM audio windows from an I2S microphone at 16 kHz, extracts MFCC + delta + delta-delta features, and classifies the motor's health state every **150 ms** — all within the memory and compute constraints of a Cortex-M33 with the Silicon Labs MVP hardware accelerator.

**Why it matters:** Traditional vibration-based PdM solutions require separate accelerometers, ADC boards, and cloud pipelines. This solution uses a single microphone and runs entirely on a $5 microcontroller — deployable in any motor enclosure.

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Feature type | MFCC + Δ (delta) + ΔΔ (delta-delta) |
| MFCC coefficients | 13 per frame |
| Feature frames per window | 49 |
| Total feature dimension | **1,911** (13 × 3 channels × 49 frames) |
| Classifier | Multinomial Logistic Regression (linear layer + softmax) |
| Classes | 6 (see below) |
| Audio window | 1 second @ 16 kHz (16,000 samples) |
| FFT size | 512-point |
| Mel filterbanks | 40 |
| Frame length / step | 480 / 320 samples (30 ms / 20 ms) |
| Inference interval | 150 ms |
| Reported accuracy | ~85% (test set) |

### Fault Classes

| Index | Label | Description |
|-------|-------|-------------|
| 0 | **Healthy** | Normal motor operation (baseline) |
| 1 | **Ball Fault** | Defect on bearing rolling elements |
| 2 | **Belt Fault** | Belt slippage or wear |
| 3 | **Inner Race** | Inner bearing race defect |
| 4 | **Misalign / Loose** | Shaft misalignment or mechanical looseness |
| 5 | **Outer Race** | Outer bearing race defect |

### Runtime Decision Logic

- **EMA smoothing** (α = 0.45) over raw softmax probabilities to suppress frame-level noise
- **Fault latch**: requires 3 consecutive fault frames (450 ms) to trigger — prevents single-frame spikes
- **Fault clear**: requires 5 consecutive healthy frames (750 ms) to release — adds hysteresis
- **Healthy calibration**: 50-window baseline (~7.5 s) captured at startup; sets dynamic signal gate and similarity reference
- **Dynamic signal gate**: upper RMS threshold = `baseline_rms_mean × 1.5` — adapts to microphone placement and motor loudness

---

## ⚙️ Hardware Used

| Component | Details |
|-----------|---------|
| MCU | Silicon Labs **EFR32xG26** (BRD2608A) — Cortex-M33, MVP accelerator |
| Microphone | I2S digital microphone (via `sl_mic_i2s`) |
| LEDs | Red (LED0) = fault detected; Green (LED1) = healthy / activity |
| Interface | VCOM serial @ 115200 baud for classification output |
| SDK | Silicon Labs Gecko SDK + MLTK |
| Build | Simplicity Studio 5 / CMake + GCC |

---

## ⚡ Performance Metrics

| Metric | Value |
|--------|-------|
| Overall test accuracy | ~85% |
| Inference latency | < 150 ms per window |
| Calibration time | ~7.5 seconds at boot |
| Flash usage | ~430 KB (model weights dominant) |
| RAM usage | ~40 KB (feature buffers + stack) |
| Power mode | Active (continuous inference) |

> Note: Accuracy figures are from offline evaluation on the training dataset split. On-device latency measured on EFR32xG26 with MVP accelerator enabled.

---

## 📊 Challenges

### 1. Healthy vs. Fault Class Confusion
The most significant accuracy bottleneck was confusion between the **Healthy** class and fault classes with subtle spectral signatures (particularly Inner Race and Outer Race at low fault severity). Bearing fault signatures often manifest as narrow-band harmonics that can be masked by mechanical noise in the mel filterbank representation.

**Mitigation:** A dedicated healthy baseline calibration phase establishes a machine-specific reference. The similarity score (`machine_similarity`) computed as cosine distance from this baseline is used as an auxiliary gate — predictions are suppressed when the signal is too different from the motor's own healthy profile, reducing false positives from environmental noise.

### 2. Dataset Imbalance and Recording Conditions
Training data for fault classes (especially Belt Fault and Misalign/Loose) was harder to collect in controlled conditions, leading to uneven class representation. Variation in microphone placement across recordings also introduced spurious spectral differences unrelated to fault state.

### 3. Feature Sensitivity vs. Generalization
MFCC + delta + delta-delta captures temporal dynamics well but produces a 1,911-dimensional feature vector for a linear classifier — prone to overfitting on small datasets. L2 regularization was applied during training; the softmax + EMA pipeline on-device compensates for per-frame uncertainty.

---

## 🛠️ Future Improvements

- **CNN / 1D-Conv model**: Replace logistic regression with a lightweight convolutional network operating directly on the log-mel spectrogram. Even a 3-layer CNN would significantly improve nonlinear fault pattern separation without exceeding EFR32xG26 memory limits when quantized to INT8.
- **INT8 quantization via TFLM**: Deploy a proper TensorFlow Lite Micro model (`.tflite` + flatbuffer) to utilize the MVP hardware accelerator at full efficiency for matrix operations.
- **Larger and more diverse dataset**: Collect recordings across multiple motor types, load conditions, and microphone distances using the CWRU Bearing Dataset or equivalent as a supplement.
- **Anomaly detection pre-filter**: Use an autoencoder or one-class SVM trained only on healthy audio to gate the classifier — reject inputs that are too far from any known class before running the full feature pipeline.
- **Vibration sensor fusion**: Fuse I2S audio with accelerometer data (SPI/I2C) to improve fault type discrimination, particularly for misalignment vs. looseness.
- **OTA model updates**: Use Silicon Labs' OTA bootloader to push retrained model weights without reflashing the full firmware.

---

## 💡 Why This Project Matters

| Aspect | Value |
|--------|-------|
| **Edge AI** | No cloud, no latency, no connectivity requirement — works in isolated industrial environments |
| **Cost** | Single microcontroller + microphone replaces expensive vibration DAQ hardware |
| **Latency** | 150 ms inference loop enables near-real-time fault alerting vs. batch cloud pipelines (minutes) |
| **Privacy** | Audio never leaves the device — relevant in regulated manufacturing environments |
| **Scalability** | Firmware can be deployed to any EFR32xG2x board; model weights swappable via OTA |

This project demonstrates the full embedded ML pipeline: **data → features → training → deployment → real-time inference** — entirely on a microcontroller. It bridges the gap between data science (feature engineering, classifier training) and embedded systems (C/C++ firmware, RTOS tasks, hardware drivers).

---

## 🧪 How to Run

### Prerequisites

- [Simplicity Studio 5](https://www.silabs.com/developers/simplicity-studio)
- Silicon Labs Gecko SDK (version matching `Final2o.slcp`)
- EFR32xG26 dev board (BRD2608A) or compatible EFR32xG2x board
- Serial terminal (Tera Term / PuTTY / screen) at **115200 baud**

### Import into Simplicity Studio

```
1. File > Import > General > Existing Projects into Workspace
2. Browse to the cloned repository root
3. Select "Final2o" project → Finish
4. Simplicity Studio auto-generates autogen/ files on first build
```

### Build & Flash

```
1. Right-click project in Project Explorer → Build Project
2. Connect EFR32xG26 board via USB
3. Run > Flash (or press F11 for debug session)
```

### Runtime Behaviour

```
Boot:
  [CALIBRATING] Motor must run healthy for ~7.5 seconds
  Serial prints per-window RMS and calibration progress

After calibration:
  Every 150 ms:
    → Capture 1s PCM window
    → Extract MFCC + delta + delta-delta (1,911 features)
    → Run logistic regression + softmax
    → Apply EMA smoothing
    → Output: class label + probabilities on VCOM

LEDs:
  Green ON  = Healthy
  Red   ON  = Fault detected (latched for ≥450 ms)
```

### Configuration

Edit [config/audio_classifier_config.h](config/audio_classifier_config.h) to tune:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FAULT_DETECT_THRESHOLD` | 0.50 | Minimum fault class probability to trigger |
| `FAULT_TRIGGER_COUNT` | 3 | Consecutive frames before fault latches |
| `FAULT_CLEAR_COUNT` | 5 | Consecutive healthy frames before clearing |
| `HEALTHY_CALIBRATION_WINDOWS` | 50 | Baseline calibration length (~7.5 s) |
| `MULTICLASS_EMA_ALPHA` | 0.45 | Smoothing factor (higher = faster response) |
| `INFERENCE_INTERVAL_MS` | 150 | How often inference runs |

---

## 📁 Repository Structure

```
Predictive-Maintenance-Motor-Fault-Detection/
├── main.c                          # RTOS entry, task spawn
├── app.c / app.h                   # Application-level init
├── audio_classifier.cc / .h        # Main inference loop (RTOS task)
├── mfcc_multiclass_inference.cc    # MFCC + delta pipeline + logistic regression
├── mfcc_multiclass_inference.h
├── mfcc_multiclass_model_data.c    # Trained weights, biases, mel/DCT matrices
├── mfcc_multiclass_model_data.h    # Model constants and extern declarations
├── motor_pcm_buffer.c / .h         # DMA PCM ring buffer management
├── recognize_commands.cc / .h      # Command smoothing (compatibility layer)
├── config/
│   ├── audio_classifier_config.h   # All tunable thresholds and parameters
│   ├── pin_config.h                # GPIO pin assignments
│   ├── sl_mic_i2s_config.h         # I2S microphone driver config
│   └── tflite/                     # Source .tflite model file
├── autogen/                        # Simplicity Studio generated drivers
├── cmake_gcc/                      # CMake build system
└── docs/
    └── system_architecture.md      # System design and pipeline description
```

---

## 👤 Author

**Nihal Nelaturi**
Embedded Systems & Machine Learning Engineer

- GitHub: [github.com/NihalNelaturi](https://github.com/NihalNelaturi)
- Project: [Predictive Maintenance — Motor Fault Detection](https://github.com/NihalNelaturi/Predictive-Maintainence-Motor-Fault-Detection-)

---

## 📚 References

- [Silicon Labs Machine Learning Documentation](https://docs.silabs.com/machine-learning/latest/aiml-developing-with)
- [Silicon Labs MLTK](https://siliconlabs.github.io/mltk)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [CWRU Bearing Fault Dataset](https://engineering.case.edu/bearingdatacenter)
- [EFR32xG26 Product Page](https://www.silabs.com/wireless/proprietary/efr32xg26-series-2)
