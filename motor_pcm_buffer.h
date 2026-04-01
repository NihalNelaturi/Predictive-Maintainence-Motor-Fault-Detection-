#ifndef MOTOR_PCM_BUFFER_H
#define MOTOR_PCM_BUFFER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void motor_pcm_push_samples(const int16_t *samples, uint32_t n_frames);
bool motor_pcm_copy_latest(int16_t *dest, size_t num_samples);
bool motor_pcm_has_window(size_t num_samples);
void motor_pcm_reset(void);

#ifdef __cplusplus
}
#endif

#endif // MOTOR_PCM_BUFFER_H
