#pragma once

#include "dsp/quants.h"

#ifndef restrict
#  define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1);
int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *activation, const __fp16 *permuted_weight, int m,
                                int k, int n);
int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *activation, const uint8_t *permuted_weight,
                                     int m, int k, int n, enum ggml_type weight_type);

#ifdef __cplusplus
}
#endif
