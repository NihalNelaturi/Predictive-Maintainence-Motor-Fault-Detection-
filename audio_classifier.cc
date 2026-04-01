/***************************************************************************//**
 * @file
 * @brief Audio multiclass classifier application
 ******************************************************************************/
#include "os.h"
#include "sl_led.h"
#include "sl_simple_led_instances.h"
#include "sl_sleeptimer.h"
#include "sl_status.h"

#include "audio_classifier.h"
#include "config/audio_classifier_config.h"
#include "mfcc_multiclass_inference.h"
#include "motor_pcm_buffer.h"
#include "sl_ml_audio_feature_generation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if SL_SIMPLE_LED_COUNT < 2
#error "Sample application requires two leds"
#endif

static OS_TCB tcb;
static CPU_STK stack[TASK_STACK_SIZE];
static const char *const s_category_labels[MAX_CATEGORY_COUNT] = CATEGORY_LABELS;

int category_count = MAX_CATEGORY_COUNT;

static int16_t pcm_window[MFCC_MULTICLASS_WINDOW_SAMPLES];
static float latest_feature_vector[MFCC_MULTICLASS_FEATURE_DIM];
static float healthy_baseline_mean[MFCC_MULTICLASS_FEATURE_DIM];
static float latest_model_probabilities[MFCC_MULTICLASS_CLASS_COUNT];
static float latest_runtime_probabilities[MFCC_MULTICLASS_CLASS_COUNT];
static float smoothed_probabilities[MFCC_MULTICLASS_CLASS_COUNT];

static bool healthy_baseline_ready = false;
static uint16_t healthy_calibration_samples = 0U;
static float healthy_baseline_feature_mean = 0.0f;
static float healthy_baseline_centered_norm = 0.0f;

// RMS stats accumulated during calibration for dynamic signal gate and diagnostics.
static float calibration_rms_sum   = 0.0f;
static float calibration_rms_sumsq = 0.0f;
static float baseline_rms_mean     = 0.0f;

// Dynamic upper gate set at finalization; pre-calibration fallback = SIGNAL_PRESENT_RMS_HIGH.
static float dynamic_signal_gate_high = SIGNAL_PRESENT_RMS_HIGH;

static float latest_rms = 0.0f;
static float latest_machine_similarity = 0.0f;
static float latest_signal_gate = 0.0f;
static bool smoothed_probabilities_initialized = false;
static bool fault_latched = false;
static uint8_t consecutive_faults = 0U;
static uint8_t consecutive_healthy = 0U;
static uint32_t last_calibration_log_ms = 0U;
static uint32_t last_prediction_log_ms = 0U;
static int last_display_class_index = MFCC_MULTICLASS_HEALTHY_INDEX;

// Pipeline-health watchdog counter.
static uint8_t pipeline_skip_count = 0U;

static void audio_classifier_task(void *arg);

const char *get_category_label(int index)
{
  if ((index < 0) || (index >= category_count)) {
    return "unknown";
  }
  return s_category_labels[index];
}

static uint32_t get_current_time_ms(void)
{
  return sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
}

static float clamp_unit(float value)
{
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 1.0f) {
    return 1.0f;
  }
  return value;
}

static float max_float(float a, float b)
{
  return (a > b) ? a : b;
}

static void normalize_probabilities(float *probabilities)
{
  float sum = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_CLASS_COUNT; ++i) {
    probabilities[i] = max_float(probabilities[i], 0.0f);
    sum += probabilities[i];
  }

  if (sum <= 1.0e-6f) {
    for (int i = 0; i < MFCC_MULTICLASS_CLASS_COUNT; ++i) {
      probabilities[i] = (i == MFCC_MULTICLASS_HEALTHY_INDEX) ? 1.0f : 0.0f;
    }
    return;
  }

  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < MFCC_MULTICLASS_CLASS_COUNT; ++i) {
    probabilities[i] *= inv_sum;
  }
}

static int argmax_fault_class(const float *probabilities)
{
  int best_index = 1;
  float best_value = probabilities[1];
  for (int i = 2; i < MFCC_MULTICLASS_CLASS_COUNT; ++i) {
    if (probabilities[i] > best_value) {
      best_value = probabilities[i];
      best_index = i;
    }
  }
  return best_index;
}

static void set_led_state(bool calibrated, int active_class_index)
{
  if (!calibrated) {
    sl_led_turn_off(&DETECTION_LED);
    sl_led_turn_off(&ACTIVITY_LED);
    return;
  }

  if (active_class_index == MFCC_MULTICLASS_HEALTHY_INDEX) {
    sl_led_turn_off(&DETECTION_LED);
    sl_led_turn_on(&ACTIVITY_LED);
  } else {
    sl_led_turn_on(&DETECTION_LED);
    sl_led_turn_off(&ACTIVITY_LED);
  }
}

static void print_calibration_progress(uint32_t current_time_ms, bool force)
{
  if (healthy_baseline_ready) {
    return;
  }
  if (!force && ((current_time_ms - last_calibration_log_ms) < 1000U)) {
    return;
  }

  last_calibration_log_ms = current_time_ms;
  const float progress = (100.0f * static_cast<float>(healthy_calibration_samples))
                         / static_cast<float>(HEALTHY_CALIBRATION_WINDOWS);
  printf("Calibrating | %.0f%%\r\n", static_cast<double>(progress));
}

static void print_prediction(uint32_t current_time_ms, int display_class_index,
                             float confidence, bool force)
{
  if (!force && ((current_time_ms - last_prediction_log_ms) < STATUS_LOG_INTERVAL_MS)) {
    return;
  }

  last_prediction_log_ms = current_time_ms;
  printf("%s | %.1f%%\r\n",
         get_category_label(display_class_index),
         static_cast<double>(100.0f * clamp_unit(confidence)));
}

static void finalize_healthy_baseline(void)
{
  // Compute centred-norm of the accumulated MFCC feature baseline.
  healthy_baseline_feature_mean = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
    healthy_baseline_feature_mean += healthy_baseline_mean[i];
  }
  healthy_baseline_feature_mean /= static_cast<float>(MFCC_MULTICLASS_FEATURE_DIM);

  healthy_baseline_centered_norm = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
    const float centered = healthy_baseline_mean[i] - healthy_baseline_feature_mean;
    healthy_baseline_centered_norm += centered * centered;
  }
  healthy_baseline_centered_norm = std::sqrt(healthy_baseline_centered_norm);

  // Derive dynamic signal gate from calibration RMS statistics.
  const float n = static_cast<float>(healthy_calibration_samples);
  baseline_rms_mean = (n > 0.0f) ? (calibration_rms_sum / n) : 0.0f;

  float rms_variance = 0.0f;
  if (n > 1.0f) {
    const float mean_sq = calibration_rms_sum * calibration_rms_sum / n;
    rms_variance = (calibration_rms_sumsq - mean_sq) / (n - 1.0f);
  }
  const float rms_std = std::sqrt((rms_variance > 0.0f) ? rms_variance : 0.0f);

  // Upper gate = 1.5x calibration mean, clamped between a floor and the original
  // static ceiling. Never let the dynamic value exceed SIGNAL_PRESENT_RMS_HIGH:
  // a higher ceiling would suppress fault signals more than the original code did,
  // causing false negatives on louder motors.
  const float computed_gate = baseline_rms_mean * CALIBRATION_RMS_GATE_SCALE;
  const float gate_floor    = 2.0f * SIGNAL_PRESENT_RMS_THRESHOLD;
  const float gate_clamped  = (computed_gate > gate_floor) ? computed_gate : gate_floor;
  dynamic_signal_gate_high  = (gate_clamped < SIGNAL_PRESENT_RMS_HIGH) ? gate_clamped : SIGNAL_PRESENT_RMS_HIGH;

  // Calibration quality diagnostics printed to serial.
  if (baseline_rms_mean < (2.0f * CALIBRATION_MIN_RMS)) {
    printf("WARN | Calibration signal very weak (mean RMS=%.4f). "
           "Check microphone placement.\r\n",
           static_cast<double>(baseline_rms_mean));
  }

  const float cv = (baseline_rms_mean > 1.0e-6f) ? (rms_std / baseline_rms_mean) : 1.0f;
  if (cv > CALIBRATION_CV_WARN_THRESHOLD) {
    printf("WARN | Calibration signal unstable (CV=%.2f). "
           "Motor may not be at steady state.\r\n",
           static_cast<double>(cv));
  }

  printf("Calibration done | RMS mean=%.4f std=%.4f gate_high=%.4f\r\n",
         static_cast<double>(baseline_rms_mean),
         static_cast<double>(rms_std),
         static_cast<double>(dynamic_signal_gate_high));

  // Reset all runtime state for a clean start.
  healthy_baseline_ready = true;
  latest_rms                         = 0.0f;
  latest_machine_similarity          = 0.0f;
  latest_signal_gate                 = 0.0f;
  smoothed_probabilities_initialized = false;
  fault_latched                      = false;
  consecutive_faults                 = 0U;
  consecutive_healthy                = 0U;
  pipeline_skip_count                = 0U;
  last_display_class_index           = MFCC_MULTICLASS_HEALTHY_INDEX;
}

static void update_healthy_baseline(void)
{
  if (healthy_baseline_ready || (healthy_calibration_samples >= HEALTHY_CALIBRATION_WINDOWS)) {
    return;
  }

  if (latest_rms < CALIBRATION_MIN_RMS) {
    return;
  }

  ++healthy_calibration_samples;
  const float sample_count = static_cast<float>(healthy_calibration_samples);

  // Online Welford mean of the MFCC feature vector.
  for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
    const float delta = latest_feature_vector[i] - healthy_baseline_mean[i];
    healthy_baseline_mean[i] += delta / sample_count;
  }

  // Accumulate RMS stats for dynamic gate and diagnostics.
  calibration_rms_sum   += latest_rms;
  calibration_rms_sumsq += latest_rms * latest_rms;

  if (healthy_calibration_samples >= HEALTHY_CALIBRATION_WINDOWS) {
    finalize_healthy_baseline();
  }
}

static float compute_machine_similarity(void)
{
  if (!healthy_baseline_ready || (healthy_baseline_centered_norm <= 1.0e-6f)) {
    return 1.0f;
  }

  float current_feature_mean = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
    current_feature_mean += latest_feature_vector[i];
  }
  current_feature_mean /= static_cast<float>(MFCC_MULTICLASS_FEATURE_DIM);

  float dot = 0.0f;
  float current_norm_sq = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
    const float baseline_centered = healthy_baseline_mean[i] - healthy_baseline_feature_mean;
    const float current_centered  = latest_feature_vector[i] - current_feature_mean;
    dot           += baseline_centered * current_centered;
    current_norm_sq += current_centered * current_centered;
  }

  if (current_norm_sq <= 1.0e-6f) {
    return 0.0f;
  }

  const float corr = dot / (healthy_baseline_centered_norm * std::sqrt(current_norm_sq));
  return clamp_unit((corr - MACHINE_PROFILE_CORR_LOW)
                    / (MACHINE_PROFILE_CORR_HIGH - MACHINE_PROFILE_CORR_LOW));
}

static float compute_signal_gate(void)
{
  // Lower bound is a fixed noise floor; upper bound is dynamic (set at calibration).
  if (latest_rms <= SIGNAL_PRESENT_RMS_THRESHOLD) {
    return 0.0f;
  }
  if (latest_rms >= dynamic_signal_gate_high) {
    return 1.0f;
  }
  return clamp_unit((latest_rms - SIGNAL_PRESENT_RMS_THRESHOLD)
                    / (dynamic_signal_gate_high - SIGNAL_PRESENT_RMS_THRESHOLD));
}

static void apply_healthy_bias(void)
{
  const float similarity_weight = latest_machine_similarity
                                  * (0.5f + (0.5f * latest_machine_similarity));
  const float gate = clamp_unit(latest_signal_gate * similarity_weight);
  float fault_sum = 0.0f;

  latest_runtime_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX]
    = latest_model_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX]
      + ((1.0f - gate) * (1.0f - latest_model_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX]));

  for (int cls = 1; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
    latest_runtime_probabilities[cls] = latest_model_probabilities[cls] * gate;
    fault_sum += latest_runtime_probabilities[cls];
  }

  const float healthy_floor = max_float(1.0f - fault_sum, 0.0f);
  if (healthy_floor > latest_runtime_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX]) {
    latest_runtime_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX] = healthy_floor;
  }

  normalize_probabilities(latest_runtime_probabilities);
}

static void update_smoothed_probabilities(void)
{
  if (!smoothed_probabilities_initialized) {
    memcpy(smoothed_probabilities,
           latest_runtime_probabilities,
           sizeof(smoothed_probabilities));
    smoothed_probabilities_initialized = true;
  } else {
    for (int cls = 0; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
      smoothed_probabilities[cls] = (MULTICLASS_EMA_ALPHA * latest_runtime_probabilities[cls])
                                    + ((1.0f - MULTICLASS_EMA_ALPHA) * smoothed_probabilities[cls]);
    }
    normalize_probabilities(smoothed_probabilities);
  }
}

static int select_display_class(float *confidence_out)
{
  const int top_fault_index       = argmax_fault_class(smoothed_probabilities);
  const float top_fault_probability = smoothed_probabilities[top_fault_index];
  const float healthy_probability   = smoothed_probabilities[MFCC_MULTICLASS_HEALTHY_INDEX];

  if (!fault_latched) {
    if ((top_fault_probability >= FAULT_DETECT_THRESHOLD)
        && (healthy_probability <= HEALTHY_SUPPRESS_THRESHOLD)) {
      if (consecutive_faults < 255U) {
        ++consecutive_faults;
      }
    } else {
      consecutive_faults = 0U;
    }
    consecutive_healthy = 0U;

    if (consecutive_faults >= FAULT_TRIGGER_COUNT) {
      fault_latched      = true;
      consecutive_faults = 0U;
    }
  } else {
    if (healthy_probability >= FAULT_CLEAR_THRESHOLD) {
      if (consecutive_healthy < 255U) {
        ++consecutive_healthy;
      }
    } else {
      consecutive_healthy = 0U;
    }

    if (consecutive_healthy >= FAULT_CLEAR_COUNT) {
      fault_latched       = false;
      consecutive_healthy = 0U;
    }
  }

  if (fault_latched) {
    *confidence_out          = top_fault_probability;
    last_display_class_index = top_fault_index;
  } else {
    *confidence_out          = healthy_probability;
    last_display_class_index = MFCC_MULTICLASS_HEALTHY_INDEX;
  }

  return last_display_class_index;
}

void audio_classifier_init(void)
{
  RTOS_ERR err;
  char task_name[] = "audio classifier task";

  OSTaskCreate(&tcb,
               task_name,
               audio_classifier_task,
               DEF_NULL,
               TASK_PRIORITY,
               &stack[0],
               (TASK_STACK_SIZE / 10u),
               TASK_STACK_SIZE,
               0u,
               0u,
               DEF_NULL,
               (OS_OPT_TASK_STK_CLR),
               &err);

  EFM_ASSERT((RTOS_ERR_CODE_GET(err) == RTOS_ERR_NONE));
}

static void audio_classifier_task(void *arg)
{
  RTOS_ERR err;
  (void)arg;

  memset(healthy_baseline_mean, 0, sizeof(healthy_baseline_mean));
  motor_pcm_reset();

  if (!mfcc_multiclass_init()) {
    while (true) {
      set_led_state(false, MFCC_MULTICLASS_HEALTHY_INDEX);
      printf("ERROR | MFCC init failed\r\n");
      OSTimeDlyHMSM(0, 0, 1, 0, OS_OPT_TIME_HMSM_STRICT, &err);
    }
  }

  const sl_status_t feature_status = sl_ml_audio_feature_generation_init();
  if (feature_status != SL_STATUS_OK) {
    while (true) {
      set_led_state(false, MFCC_MULTICLASS_HEALTHY_INDEX);
      printf("ERROR | microphone init failed | code=%lu\r\n",
             static_cast<unsigned long>(feature_status));
      OSTimeDlyHMSM(0, 0, 1, 0, OS_OPT_TIME_HMSM_STRICT, &err);
    }
  }

  while (true) {
    OSTimeDlyHMSM(0, 0, 0, INFERENCE_INTERVAL_MS, OS_OPT_TIME_HMSM_STRICT, &err);

    // Pipeline-health watchdog: count consecutive failures and attempt a soft
    // audio re-init once MAX_PIPELINE_SKIP_COUNT is reached.
    if (!motor_pcm_copy_latest(pcm_window, MFCC_MULTICLASS_WINDOW_SAMPLES)) {
      ++pipeline_skip_count;
      if (pipeline_skip_count >= MAX_PIPELINE_SKIP_COUNT) {
        printf("ERROR | Audio pipeline stalled (%u consecutive skips). "
               "Re-initialising mic.\r\n",
               static_cast<unsigned int>(pipeline_skip_count));
        pipeline_skip_count = 0U;
        motor_pcm_reset();
        sl_ml_audio_feature_generation_init();
      }
      continue;
    }

    if (!mfcc_multiclass_extract_features_and_predict(pcm_window,
                                                      latest_feature_vector,
                                                      latest_model_probabilities,
                                                      &latest_rms)) {
      ++pipeline_skip_count;
      if (pipeline_skip_count >= MAX_PIPELINE_SKIP_COUNT) {
        printf("ERROR | Inference pipeline stalled (%u consecutive skips).\r\n",
               static_cast<unsigned int>(pipeline_skip_count));
        pipeline_skip_count = 0U;
      }
      continue;
    }

    pipeline_skip_count = 0U;  // reset on every successful frame

    const uint32_t current_time_ms = get_current_time_ms();
    if (!healthy_baseline_ready) {
      update_healthy_baseline();
      set_led_state(false, MFCC_MULTICLASS_HEALTHY_INDEX);
      print_calibration_progress(current_time_ms, false);
      continue;
    }

    latest_machine_similarity = compute_machine_similarity();
    latest_signal_gate        = compute_signal_gate();
    apply_healthy_bias();
    update_smoothed_probabilities();

    float confidence = 0.0f;
    const int display_class_index = select_display_class(&confidence);

    set_led_state(true, display_class_index);
    print_prediction(current_time_ms, display_class_index, confidence, false);
  }
}
