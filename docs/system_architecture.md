# System Architecture — Motor Fault Detection

**Board:** BRD2608A / EFR32MG26 | **Model:** Multiclass Logistic Regression | **Test Accuracy:** 99.67%

## Overview

This document describes the end-to-end signal processing and inference pipeline running on the EFR32MG26 microcontroller for real-time motor fault classification.

---

## High-Level Pipeline

```
  I2S Microphone
       │
       ▼
 ┌─────────────────────────────┐
 │   DMA PCM Ring Buffer       │  motor_pcm_buffer.c
 │   16-bit @ 16 kHz           │
 └────────────┬────────────────┘
              │  Copy 16,000 samples (1s window)
              ▼
 ┌─────────────────────────────────────────────────────┐
 │   MFCC Feature Extraction Pipeline                  │  mfcc_multiclass_inference.cc
 │                                                     │
 │  For each of 49 frames (20 ms step, 30 ms window):  │
 │    1. Apply Hann window to 480 samples              │
 │    2. Zero-pad → 512-point real FFT (ARM DSP)       │
 │    3. Power spectrum (257 bins)                     │
 │    4. Mel filterbank (40 bands) → log-Mel           │
 │    5. DCT → 13 MFCC coefficients                   │
 │                                                     │
 │  Post-frame:                                        │
 │    6. Compute Δ  (delta)    over 49 frames          │
 │    7. Compute ΔΔ (delta²)   over 49 frames          │
 │                                                     │
 │  Flatten: [MFCC | Δ | ΔΔ] → 1,911-dim vector       │
 └────────────┬────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────┐
 │   Logistic Regression Classifier                    │  mfcc_multiclass_model_data.c
 │                                                     │
 │  logit[c] = bias[c] + weight[c] · feature_vector   │
 │  probabilities = softmax(logits)   (6 classes)      │
 └────────────┬────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────────┐
 │   Runtime Decision Logic                            │  audio_classifier.cc
 │                                                     │
 │  • EMA smoothing  (α = 0.45)                        │
 │  • Signal gate    (RMS threshold + dynamic gate)    │
 │  • Machine similarity check (cosine vs baseline)    │
 │  • Fault latch    (3 consecutive frames → trigger)  │
 │  • Fault clear    (5 consecutive healthy → release) │
 └────────────┬────────────────────────────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
  LED Output     VCOM Serial
  (Red/Green)   (115200 baud)
```

---

## MFCC Feature Pipeline Detail

### 1. Framing
The 1-second PCM window (16,000 samples) is divided into overlapping frames:
- **Frame length**: 480 samples (30 ms at 16 kHz)
- **Frame step**: 320 samples (20 ms)
- **Number of frames**: 49

### 2. Windowing
Each frame is multiplied by a **Hann window** to reduce spectral leakage at frame boundaries:
```
w[n] = 0.5 - 0.5 × cos(2π·n / N)
```

### 3. FFT & Power Spectrum
Zero-padded to 512 samples → 512-point real FFT using ARM CMSIS-DSP `arm_rfft_fast_f32`.
Power spectrum computed from complex FFT output → 257 unique bins.

### 4. Mel Filterbank
257 power spectrum bins projected onto 40 mel-spaced triangular filters.
Log applied: `log(mel_energy + 1e-6)` to compress dynamic range.

### 5. DCT → MFCC
Discrete Cosine Transform (DCT-II) applied to the 40 log-mel values → retain first **13 coefficients** (MFCC 0–12). The DCT matrix is pre-computed and stored in flash as `g_mfcc_multiclass_dct_matrix`.

### 6. Delta & Delta-Delta
First-order difference (Δ) and second-order difference (ΔΔ) computed across the 49 frames using central difference:
```
Δ[t] = 0.5 × (coeff[t+1] - coeff[t-1])
```
This captures temporal dynamics (onset/offset of spectral changes) which are critical for distinguishing fault types with similar steady-state spectra.

### 7. Feature Vector Assembly
```
feature_vector = [MFCC(13×49) | Δ(13×49) | ΔΔ(13×49)]
               = 1,911-dimensional float32 vector
```

---

## Classifier: Multinomial Logistic Regression

The classifier is a **single linear layer** — 6 weight vectors (one per class), each 1,911-dimensional, plus 6 bias scalars. Stored as flash constants:
- `g_mfcc_multiclass_weight[6 × 1,911]` — 11,466 float32 values
- `g_mfcc_multiclass_bias[6]`
- **Total parameters: 11,472 | Float model size: ≈ 44.81 KB | INT8 TFLite: 13,008 bytes (12.7 KB)**

Inference:
```
logit[c] = bias[c] + dot(weight[c], feature_vector)
probabilities = softmax(logits)
```

**Normalization note:** A StandardScaler was applied during training (zero-mean, unit-variance per feature). At export, the scaler parameters were **absorbed directly into the weight matrix and bias vector** — the embedded firmware performs no separate normalization step. This keeps the inference path minimal: one dot product + one softmax.

**Why logistic regression?** It is deterministic, requires no dynamic memory allocation, and fits entirely in flash. The MFCC + delta + delta-delta feature engineering is expressive enough that this model achieves **99.67% test accuracy** on the CWRU benchmark — matching or exceeding small neural networks at a fraction of the parameter count and inference cost.

---

## Healthy Baseline Calibration

On boot, before any fault detection begins, the system captures **50 windows** (~7.5 seconds) of audio with the motor running in its known-healthy state:

1. Per-window feature vectors are averaged → `healthy_baseline_mean[1911]`
2. Per-window RMS values are averaged and their variance computed
3. If coefficient of variation (stddev/mean) of RMS exceeds 0.30, a warning is printed — indicates unstable motor state during calibration
4. Post-calibration: `dynamic_signal_gate_high = baseline_rms_mean × 1.5`

This baseline is used at runtime to compute a **machine similarity score** (cosine similarity between current feature vector and baseline). Low similarity indicates the microphone is picking up something fundamentally different from the trained motor profile.

---

## Signal Gating

Before inference is considered valid, two gates are checked:

| Gate | Condition | Effect |
|------|-----------|--------|
| **RMS low gate** | `rms < 0.010` | Signal too quiet → skip inference (motor not running) |
| **RMS high gate** | `rms > dynamic_gate_high` | Signal clipping or impact noise → skip inference |
| **Similarity gate** | `machine_similarity < 0.45` | Feature distribution unrecognizable → report "unknown" |

---

## RTOS Task Structure

The application runs under **Micrium OS (µC/OS-III)**:

| Task | Priority | Stack | Role |
|------|----------|-------|------|
| `audio_classifier_task` | 20 | 6144 words | Inference loop (PCM copy → features → classify → output) |
| Silicon Labs kernel tasks | system | — | Sleeptimer, I2S DMA, IOSTREAM |

The inference task calls `sl_sleeptimer_delay_millisecond(INFERENCE_INTERVAL_MS)` between windows to yield the CPU.

---

## Memory Layout (measured from build)

| Region | Usage | Size |
|--------|-------|------|
| **Firmware binary (.bin)** | Total flash image | **381.5 KB** |
| .text section | Code + constants + model weights | ≈ 389.8 KB |
| .bss section | RAM (zero-init) | ≈ **295.1 KB** *(see note)* |
| Flash — model weights (float) | `g_mfcc_multiclass_weight` + bias | **44.81 KB** |
| Flash — INT8 TFLite model | `cwru_mfcc_multiclass_int8.tflite` | **12.7 KB** |
| RAM — PCM buffer | `pcm_window[16000]` (int16) | 32 KB |
| RAM — feature buffer | `latest_feature_vector[1911]` (float32) | 7.5 KB |
| RAM — FFT intermediate | `s_fft_input/output`, `s_power_spectrum` | ~6 KB |
| RAM — MFCC/delta arrays | `s_mfcc`, `s_delta`, `s_delta2` | ~8 KB |

> **RAM note:** The `.bss` section (295.1 KB) is inflated by stock SDK / TFLM template components that remain linked, including a **100 KB tensor arena** from the original TFLM framework scaffold. The custom inference path (`mfcc_multiclass_inference.cc`) does not use the TFLM interpreter and this arena is never used at runtime. Removing the TFLM component from `.slcp` is a known future optimization.

---

## Serial Output Format

Classification results are printed to VCOM at 115200 baud every inference cycle:

```
[CALIBRATING] Window 12/50 | RMS: 0.0234
[CALIBRATING] Baseline locked. RMS mean: 0.0221, CV: 0.12
[PREDICT] Healthy     | P=[0.82, 0.04, 0.03, 0.05, 0.03, 0.03] | RMS: 0.0219 | Sim: 0.91
[PREDICT] Ball Fault  | P=[0.11, 0.71, 0.05, 0.06, 0.04, 0.03] | RMS: 0.0287 | Sim: 0.73
[FAULT LATCHED] Ball Fault — 3 consecutive detections
```

---

## File Reference

| File | Purpose |
|------|---------|
| [main.c](../main.c) | RTOS kernel start, task creation |
| [app.c](../app.c) | Application init sequence |
| [audio_classifier.cc](../audio_classifier.cc) | Inference loop, EMA, fault latch logic |
| [mfcc_multiclass_inference.cc](../mfcc_multiclass_inference.cc) | FFT, Mel, MFCC, delta, logistic regression |
| [mfcc_multiclass_model_data.c](../mfcc_multiclass_model_data.c) | Trained weight/bias/matrix constants |
| [motor_pcm_buffer.c](../motor_pcm_buffer.c) | I2S DMA ring buffer |
| [config/audio_classifier_config.h](../config/audio_classifier_config.h) | All tunable parameters |
