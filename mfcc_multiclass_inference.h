#ifndef MFCC_MULTICLASS_INFERENCE_H
#define MFCC_MULTICLASS_INFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "mfcc_multiclass_model_data.h"

#ifdef __cplusplus
extern "C" {
#endif

bool mfcc_multiclass_init(void);
bool mfcc_multiclass_extract_features_and_predict(const int16_t *pcm_window,
                                                  float *feature_vector,
                                                  float *class_probabilities,
                                                  float *rms_out);

#ifdef __cplusplus
}
#endif

#endif // MFCC_MULTICLASS_INFERENCE_H
