/***************************************************************************//**
 * @file
 * @brief Audio multiclass classifier application config
 ******************************************************************************/

#ifndef AUDIO_CLASSIFIER_CONFIG_H
#define AUDIO_CLASSIFIER_CONFIG_H

// Compatibility macros retained because the stock recognize_commands module is
// still part of the generated build, even though this application no longer
// routes inference through the TFLM keyword path.
#define SMOOTHING_WINDOW_DURATION_MS 200
#define MINIMUM_DETECTION_COUNT 1
#define DETECTION_THRESHOLD 128
#define SUPPRESSION_TIME_MS 0
#define SENSITIVITY .5f
#define IGNORE_UNDERSCORE_LABELS 0

// On BRD2608A:
// sl_led_led0 = red LED
// sl_led_led1 = green LED
// Healthy = green, any fault class = red.
#define DETECTION_LED sl_led_led0
#define ACTIVITY_LED sl_led_led1

#define VERBOSE_MODEL_OUTPUT_LOGS 0
#define INFERENCE_INTERVAL_MS 150

#define MAX_CATEGORY_COUNT 6
#define MAX_RESULT_COUNT 8
#define TASK_STACK_SIZE 6144
#define TASK_PRIORITY 20

#define CATEGORY_LABELS { "Healthy", "Ball Fault", "Belt Fault", "Inner Race", "Misalign Loose", "Outer Race" }

// Multiclass smoothing and decision logic.
// EMA alpha reduced from 0.55 to 0.45: slightly smoother without slowing
// fault response enough to cause false negatives.
#define MULTICLASS_EMA_ALPHA           0.45f

#define FAULT_DETECT_THRESHOLD         0.50f
#define HEALTHY_SUPPRESS_THRESHOLD     0.60f
#define FAULT_CLEAR_THRESHOLD          0.76f  // was 0.74 — wider hysteresis gap

// 3 consecutive frames (450 ms) to latch a fault  — was 1.
// 5 consecutive frames (750 ms) to clear a fault  — was 2.
// Prevents single-frame spikes from latching, and brief dips from clearing.
#define FAULT_TRIGGER_COUNT            3
#define FAULT_CLEAR_COUNT              5

// Healthy baseline capture: 50 windows @ 150 ms = ~7.5 seconds (was 24 = 3.6 s).
// Motor should be running in a known-healthy state during this period.
#define HEALTHY_CALIBRATION_WINDOWS    50
#define CALIBRATION_MIN_RMS            0.010f

// Calibration quality warning: warn on serial if the coefficient of variation
// (stddev / mean) of per-window RMS exceeds this threshold.
// High CV means the motor was not at a stable operating point during calibration.
#define CALIBRATION_CV_WARN_THRESHOLD  0.30f

// Dynamic signal gate.
// After calibration the upper gate = baseline_rms_mean * CALIBRATION_RMS_GATE_SCALE.
// This adapts to mic placement / motor loudness instead of a fixed constant.
// The static SIGNAL_PRESENT_RMS_HIGH below is the pre-calibration fallback only.
#define CALIBRATION_RMS_GATE_SCALE     1.5f

#define SIGNAL_PRESENT_RMS_THRESHOLD   0.010f
#define SIGNAL_PRESENT_RMS_HIGH        0.045f  // fallback before calibration
#define MACHINE_PROFILE_CORR_LOW       0.45f
#define MACHINE_PROFILE_CORR_HIGH      0.88f

// Pipeline-health watchdog: attempt a soft audio re-init after this many
// consecutive frames where motor_pcm_copy_latest() or inference returns false.
#define MAX_PIPELINE_SKIP_COUNT        10

#define STATUS_LOG_INTERVAL_MS         150

#endif // AUDIO_CLASSIFIER_CONFIG_H
