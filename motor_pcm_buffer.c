#include "motor_pcm_buffer.h"

#include "sl_core.h"

#define MOTOR_PCM_RING_SAMPLES 32768U

static int16_t s_ring_buffer[MOTOR_PCM_RING_SAMPLES];
static size_t s_write_index = 0U;
static size_t s_total_samples = 0U;

void motor_pcm_push_samples(const int16_t *samples, uint32_t n_frames)
{
  CORE_DECLARE_IRQ_STATE;
  CORE_ENTER_CRITICAL();

  for (uint32_t i = 0; i < n_frames; ++i) {
    s_ring_buffer[s_write_index] = samples[i];
    s_write_index = (s_write_index + 1U) % MOTOR_PCM_RING_SAMPLES;
    if (s_total_samples < MOTOR_PCM_RING_SAMPLES) {
      ++s_total_samples;
    }
  }

  CORE_EXIT_CRITICAL();
}

bool motor_pcm_copy_latest(int16_t *dest, size_t num_samples)
{
  CORE_DECLARE_IRQ_STATE;
  CORE_ENTER_CRITICAL();

  if (s_total_samples < num_samples) {
    CORE_EXIT_CRITICAL();
    return false;
  }

  const size_t start_index = (s_write_index + MOTOR_PCM_RING_SAMPLES - num_samples) % MOTOR_PCM_RING_SAMPLES;
  for (size_t i = 0; i < num_samples; ++i) {
    dest[i] = s_ring_buffer[(start_index + i) % MOTOR_PCM_RING_SAMPLES];
  }

  CORE_EXIT_CRITICAL();
  return true;
}

bool motor_pcm_has_window(size_t num_samples)
{
  CORE_DECLARE_IRQ_STATE;
  CORE_ENTER_CRITICAL();
  const bool ready = s_total_samples >= num_samples;
  CORE_EXIT_CRITICAL();
  return ready;
}

void motor_pcm_reset(void)
{
  CORE_DECLARE_IRQ_STATE;
  CORE_ENTER_CRITICAL();
  s_write_index = 0U;
  s_total_samples = 0U;
  for (size_t i = 0; i < MOTOR_PCM_RING_SAMPLES; ++i) {
    s_ring_buffer[i] = 0;
  }
  CORE_EXIT_CRITICAL();
}
