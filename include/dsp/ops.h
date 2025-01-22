#pragma once

#ifndef restrict
#define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1);

#ifdef __cplusplus
}
#endif
