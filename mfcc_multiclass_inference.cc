#include "mfcc_multiclass_inference.h"

#include "arm_math.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

static arm_rfft_fast_instance_f32 s_rfft;
static bool s_initialized = false;
static constexpr float kPi = 3.14159265358979323846f;
static float s_hann_window[MFCC_MULTICLASS_FRAME_LENGTH];
static float s_fft_input[MFCC_MULTICLASS_FFT_LENGTH];
static float s_fft_output[MFCC_MULTICLASS_FFT_LENGTH];
static float s_power_spectrum[MFCC_MULTICLASS_SPECTROGRAM_BINS];
static float s_log_mel[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MEL_BINS];
static float s_mfcc[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MFCC_COUNT];
static float s_delta[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MFCC_COUNT];
static float s_delta2[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MFCC_COUNT];
static float s_logits[MFCC_MULTICLASS_CLASS_COUNT];

inline float pcm_to_float(int16_t sample)
{
  return static_cast<float>(sample) * (1.0f / 32768.0f);
}

void compute_delta(const float input[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MFCC_COUNT],
                   float output[MFCC_MULTICLASS_FEATURE_FRAMES][MFCC_MULTICLASS_MFCC_COUNT])
{
  for (int frame = 0; frame < MFCC_MULTICLASS_FEATURE_FRAMES; ++frame) {
    const int prev_frame = (frame == 0) ? 0 : (frame - 1);
    const int next_frame = (frame == (MFCC_MULTICLASS_FEATURE_FRAMES - 1))
                           ? (MFCC_MULTICLASS_FEATURE_FRAMES - 1)
                           : (frame + 1);
    for (int coeff = 0; coeff < MFCC_MULTICLASS_MFCC_COUNT; ++coeff) {
      output[frame][coeff] = 0.5f * (input[next_frame][coeff] - input[prev_frame][coeff]);
    }
  }
}

void compute_window_spectrum(const int16_t *pcm_window, int frame_index)
{
  const int sample_offset = frame_index * MFCC_MULTICLASS_FRAME_STEP;
  for (int i = 0; i < MFCC_MULTICLASS_FRAME_LENGTH; ++i) {
    s_fft_input[i] = pcm_to_float(pcm_window[sample_offset + i]) * s_hann_window[i];
  }
  for (int i = MFCC_MULTICLASS_FRAME_LENGTH; i < MFCC_MULTICLASS_FFT_LENGTH; ++i) {
    s_fft_input[i] = 0.0f;
  }

  arm_rfft_fast_f32(&s_rfft, s_fft_input, s_fft_output, 0);

  s_power_spectrum[0] = s_fft_output[0] * s_fft_output[0];
  s_power_spectrum[MFCC_MULTICLASS_SPECTROGRAM_BINS - 1] = s_fft_output[1] * s_fft_output[1];
  for (int bin = 1; bin < (MFCC_MULTICLASS_SPECTROGRAM_BINS - 1); ++bin) {
    const float real = s_fft_output[2 * bin];
    const float imag = s_fft_output[(2 * bin) + 1];
    s_power_spectrum[bin] = (real * real) + (imag * imag);
  }
}

void compute_log_mel_frame(int frame_index)
{
  float mel_values[MFCC_MULTICLASS_MEL_BINS] = { 0.0f };

  for (int bin = 0; bin < MFCC_MULTICLASS_SPECTROGRAM_BINS; ++bin) {
    const float power = s_power_spectrum[bin];
    const float *mel_row = &g_mfcc_multiclass_mel_matrix[bin * MFCC_MULTICLASS_MEL_BINS];
    for (int mel = 0; mel < MFCC_MULTICLASS_MEL_BINS; ++mel) {
      mel_values[mel] += power * mel_row[mel];
    }
  }

  for (int mel = 0; mel < MFCC_MULTICLASS_MEL_BINS; ++mel) {
    s_log_mel[frame_index][mel] = std::log(mel_values[mel] + 1.0e-6f);
  }
}

void compute_mfcc_frame(int frame_index)
{
  for (int coeff = 0; coeff < MFCC_MULTICLASS_MFCC_COUNT; ++coeff) {
    float sum = 0.0f;
    for (int mel = 0; mel < MFCC_MULTICLASS_MEL_BINS; ++mel) {
      sum += s_log_mel[frame_index][mel]
             * g_mfcc_multiclass_dct_matrix[(mel * MFCC_MULTICLASS_MFCC_COUNT) + coeff];
    }
    s_mfcc[frame_index][coeff] = sum;
  }
}

void softmax_logits(const float *logits, float *probabilities)
{
  float max_logit = logits[0];
  for (int cls = 1; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
    if (logits[cls] > max_logit) {
      max_logit = logits[cls];
    }
  }

  float sum = 0.0f;
  for (int cls = 0; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
    probabilities[cls] = std::exp(logits[cls] - max_logit);
    sum += probabilities[cls];
  }

  const float inv_sum = (sum > 1.0e-12f) ? (1.0f / sum) : 0.0f;
  for (int cls = 0; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
    probabilities[cls] *= inv_sum;
  }
}

} // namespace

bool mfcc_multiclass_init(void)
{
  if (s_initialized) {
    return true;
  }

  if (arm_rfft_fast_init_f32(&s_rfft, MFCC_MULTICLASS_FFT_LENGTH) != ARM_MATH_SUCCESS) {
    return false;
  }

  const float scale = 2.0f * kPi / static_cast<float>(MFCC_MULTICLASS_FRAME_LENGTH);
  for (int i = 0; i < MFCC_MULTICLASS_FRAME_LENGTH; ++i) {
    s_hann_window[i] = 0.5f - (0.5f * std::cos(scale * static_cast<float>(i)));
  }

  s_initialized = true;
  return true;
}

bool mfcc_multiclass_extract_features_and_predict(const int16_t *pcm_window,
                                                  float *feature_vector,
                                                  float *class_probabilities,
                                                  float *rms_out)
{
  if ((!s_initialized && !mfcc_multiclass_init())
      || (pcm_window == nullptr)
      || (feature_vector == nullptr)
      || (class_probabilities == nullptr)
      || (rms_out == nullptr)) {
    return false;
  }

  float rms_sum = 0.0f;
  for (int i = 0; i < MFCC_MULTICLASS_WINDOW_SAMPLES; ++i) {
    const float sample = pcm_to_float(pcm_window[i]);
    rms_sum += sample * sample;
  }
  *rms_out = std::sqrt(rms_sum / static_cast<float>(MFCC_MULTICLASS_WINDOW_SAMPLES));

  for (int frame = 0; frame < MFCC_MULTICLASS_FEATURE_FRAMES; ++frame) {
    compute_window_spectrum(pcm_window, frame);
    compute_log_mel_frame(frame);
    compute_mfcc_frame(frame);
  }

  compute_delta(s_mfcc, s_delta);
  compute_delta(s_delta, s_delta2);

  int out_index = 0;
  for (int coeff = 0; coeff < MFCC_MULTICLASS_MFCC_COUNT; ++coeff) {
    for (int frame = 0; frame < MFCC_MULTICLASS_FEATURE_FRAMES; ++frame) {
      feature_vector[out_index++] = s_mfcc[frame][coeff];
    }
  }
  for (int coeff = 0; coeff < MFCC_MULTICLASS_MFCC_COUNT; ++coeff) {
    for (int frame = 0; frame < MFCC_MULTICLASS_FEATURE_FRAMES; ++frame) {
      feature_vector[out_index++] = s_delta[frame][coeff];
    }
  }
  for (int coeff = 0; coeff < MFCC_MULTICLASS_MFCC_COUNT; ++coeff) {
    for (int frame = 0; frame < MFCC_MULTICLASS_FEATURE_FRAMES; ++frame) {
      feature_vector[out_index++] = s_delta2[frame][coeff];
    }
  }

  for (int cls = 0; cls < MFCC_MULTICLASS_CLASS_COUNT; ++cls) {
    const float *class_weight = &g_mfcc_multiclass_weight[cls * MFCC_MULTICLASS_FEATURE_DIM];
    float logit = g_mfcc_multiclass_bias[cls];
    for (int i = 0; i < MFCC_MULTICLASS_FEATURE_DIM; ++i) {
      logit += feature_vector[i] * class_weight[i];
    }
    s_logits[cls] = logit;
  }

  softmax_logits(s_logits, class_probabilities);
  return true;
}
