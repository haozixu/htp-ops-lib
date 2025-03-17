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

int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim);

int naive_flash_attn(float *restrict O, const float *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace op_utils {

int compare_result(const float *x, const float *y, int n_elems);

}

#endif
