#include <HAP_farf.h>

// std headers
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "dsp/hvx_math.h"
#include "dsp/ops.h"

namespace op_utils {

float compute_rmse(const float *x, const float *y, int n) {
  float squared_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = x[i] - y[i];
    squared_error += err * err;
  }
  float rmse = sqrtf(squared_error / n);
  return rmse;
}

int compare_result(const float *x, const float *y, int n_elems) {
  static int counter = 0;

  int layer = counter++ % 28;
  FARF(ALWAYS, "layer %d attention compare:", layer);

  // hard-coded constants
  constexpr int D = 128;
  constexpr int H = 12;  // 12 query heads
  for (int h = 0; h < n_elems / D; ++h) {
    int q    = h / H;
    int head = h % H;

    float rmse = compute_rmse(&x[h * D], &y[h * D], D);
    FARF(ALWAYS, "query %d head %d RMSE: %g", q, head, rmse);
  }
  return 0;
}

}  // namespace op_utils

namespace internal {

void test_int16_fp16_conversion() {
#if __HVX_ARCH__ < 73
  FARF(ALWAYS, "HVX native h <-> hf conversion not supported");
  return;
#endif

  static __fp16  input[64];
  static int16_t output[64];

  for (int i = 0; i < 64; ++i) {
    float x  = i * 0.25 - 8;
    input[i] = (__fp16) x;
  }

  vmemu(output) = Q6_Vh_equals_Vhf(vmemu(input));

  for (int i = 0; i < 64; ++i) {
    FARF(ALWAYS, "%s: x=%g y=%d", __func__, (float) input[i], output[i]);
  }
}

void test_fp16_exp2() {
  int    n    = 256;
  size_t size = n * sizeof(__fp16);

  __fp16 *input = nullptr;
  posix_memalign((void **) &input, VLEN, size);

  __fp16 *output = nullptr;
  posix_memalign((void **) &output, VLEN, size);

  __fp16 *output_ref = new __fp16[n];

  for (int i = 0; i < n; ++i) {
    float x       = -0.1 * i;
    input[i]      = (__fp16) x;
    output_ref[i] = (__fp16) exp2f(x);
  }

  auto in_vecs  = (HVX_Vector *) input;
  auto out_vecs = (HVX_Vector *) output;
  for (int i = 0; i < n / 64; ++i) {
    out_vecs[i] = hvx_my_exp2_vhf(in_vecs[i]);
  }

  for (int i = 0; i < n; ++i) {
    float x  = (float) input[i];
    float y0 = (float) output_ref[i];
    float y1 = (float) output[i];
    FARF(ALWAYS, "%s: i=%d, x=%g, my: %g, ref: %g", __func__, i, x, y1, y0);
  }

  delete[] output_ref;
  free(input);
  free(output);
}

}  // namespace internal

extern "C" {

void internal_op_tests();

void internal_op_tests() {
  using namespace internal;

  // test_int16_fp16_conversion();
  // test_fp16_exp2();
}
}
