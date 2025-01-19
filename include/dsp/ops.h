#pragma once

#define restrict __restrict

int hvx_rms_norm_f32(float *restrict dst, const float *restrict src, int ne0, int ne1);
