# Predictive Maintenance - Motor Fault Detection

An embedded AI application for **real-time motor fault detection** using audio classification on Silicon Labs EFM32/EFR32 microcontrollers. Built with Simplicity Studio and TensorFlow Lite for Microcontrollers (TFLM).

---

## Description

This project runs an MFCC-based multiclass audio classification model directly on-device to detect motor faults from PCM audio input. It uses the Silicon Labs Machine Learning Toolkit (MLTK) pipeline and the MVP hardware accelerator (where available) for fast, low-power inference — enabling predictive maintenance without cloud connectivity.

---

## Features

- Real-time audio capture via I2S microphone
- MFCC feature extraction using the Silicon Labs Audio Feature Generator
- TensorFlow Lite Micro inference with a custom multiclass motor fault model
- LED indicators for detection and activity status
- VCOM serial output for classification results and debug logs
- Configurable detection threshold and sensitivity
- CMake + GCC build support via `cmake_gcc/`

---

## Project Structure

```
Final2o/
├── main.c                          # Application entry point
├── app.c / app.h                   # Application logic
├── audio_classifier.cc / .h        # Core audio classification loop
├── mfcc_multiclass_inference.cc    # MFCC inference pipeline
├── mfcc_multiclass_model_data.c    # Compiled TFLite model (C array)
├── motor_pcm_buffer.c / .h         # PCM audio buffer management
├── recognize_commands.cc / .h      # Post-processing and command recognition
├── config/                         # Hardware and ML configuration headers
├── autogen/                        # Auto-generated Simplicity Studio files
└── cmake_gcc/                      # CMake build system files
```

---

## How to Import and Run in Simplicity Studio

### Prerequisites

- [Simplicity Studio 5](https://www.silabs.com/developers/simplicity-studio) installed
- Silicon Labs Gecko SDK installed (matching version in `Final2o.slcp`)
- Compatible Silicon Labs board (EFR32xG24 or similar with MVP accelerator)

### Import Steps

1. Open **Simplicity Studio 5**.
2. Go to **File > Import**.
3. Select **More Import Options > General > Existing Projects into Workspace**.
4. Click **Browse** and navigate to the cloned repository folder.
5. Select the project and click **Finish**.

### Build and Flash

1. Right-click the project in **Project Explorer**.
2. Select **Build Project** — Simplicity Studio will auto-generate files in `autogen/`.
3. Connect your Silicon Labs board via USB.
4. Click the **Flash** (Run/Debug) button to program the device.

### Monitor Output

Open a serial terminal (e.g., Tera Term, PuTTY) at **115200 baud** on the VCOM port to view classification results.

---

## Configuration

Edit [`config/audio_classifier_config.h`](config/audio_classifier_config.h) to adjust:

- Detection LED assignment
- Activity LED assignment
- Detection threshold
- Sensitivity threshold
- Label filtering (underscore-prefixed labels are ignored by default)

---

## Model

The model uses MFCC features extracted from 1-second audio windows. It is trained using the [Silicon Labs MLTK](https://siliconlabs.github.io/mltk) and compiled into `mfcc_multiclass_model_data.c`. To replace with a retrained model, update the `.tflite` file in `config/tflite/` — Simplicity Studio will regenerate the C model data automatically.

---

## References

- [Silicon Labs Machine Learning Documentation](https://docs.silabs.com/machine-learning/latest/aiml-developing-with)
- [MLTK Documentation](https://siliconlabs.github.io/mltk)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
