/***************************************************************************//**
 * @file
 * @brief Configuration file for Audio Frontend
 ******************************************************************************/

#ifndef SL_ML_AUDIO_FEATURE_GENERATION_CONFIG_H
#define SL_ML_AUDIO_FEATURE_GENERATION_CONFIG_H

// The application uses the mic driver path directly for PCM capture, but this
// config still controls microphone initialization and should match the
// retrained deployment sample rate and framing.
#define SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE               8192
#define SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN                      1
#define SL_ML_AUDIO_FEATURE_GENERATION_MANUAL_CONFIG_ENABLE            1

#define SL_ML_FRONTEND_SAMPLE_RATE_HZ                            16000
#define SL_ML_FRONTEND_SAMPLE_LENGTH_MS                          1000
#define SL_ML_FRONTEND_WINDOW_SIZE_MS                            30
#define SL_ML_FRONTEND_WINDOW_STEP_MS                            20
#define SL_ML_FRONTEND_FFT_LENGTH                                512U
#define SL_ML_FRONTEND_FILTERBANK_N_CHANNELS                     40
#define SL_ML_FRONTEND_FILTERBANK_LOWER_BAND_LIMIT               20.0f
#define SL_ML_FRONTEND_FILTERBANK_UPPER_BAND_LIMIT               7600.0f

#define SL_ML_FRONTEND_NOISE_REDUCTION_ENABLE                    0
#define SL_ML_FRONTEND_NOISE_REDUCTION_SMOOTHING_BITS            5
#define SL_ML_FRONTEND_NOISE_REDUCTION_EVEN_SMOOTHING            0.004f
#define SL_ML_FRONTEND_NOISE_REDUCTION_ODD_SMOOTHING             0.004f
#define SL_ML_FRONTEND_NOISE_REDUCTION_MIN_SIGNAL_REMAINING      0.05f

#define SL_ML_FRONTEND_PCAN_ENABLE                               0
#define SL_ML_FRONTEND_PCAN_STRENGTH                             0.95f
#define SL_ML_FRONTEND_PCAN_OFFSET                               80.0f
#define SL_ML_FRONTEND_PCAN_GAIN_BITS                            21

#define SL_ML_FRONTEND_LOG_SCALE_ENABLE                          1
#define SL_ML_FRONTEND_LOG_SCALE_SHIFT                           6

#define SL_ML_AUDIO_FEATURE_GENERATION_QUANTIZE_DYNAMIC_SCALE_ENABLE   0

#endif // SL_ML_AUDIO_FEATURE_GENERATION_CONFIG_H
