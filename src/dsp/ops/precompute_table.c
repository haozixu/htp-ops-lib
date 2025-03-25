#include <HAP_farf.h>
#include <HAP_perf.h>

#include "dsp/hvx_math.h"
#include "dsp/vtcm_mgr.h"

static void precompute_safe_softmax_exp2_table() {
  uint8_t *table = (uint8_t *) vtcm_manager_reserve_area("safe_softmax::exp2_hf_qf16", 65536, 65536);
  if (!table) {
    FARF(ALWAYS, "%s: VTCM reservation failed", __func__);
    return;
  }

  const int n = 32768; // 32k fp16 elements in 64k area

  const int n_elems_per_vec = VLEN / sizeof(__fp16);
  const int n_vecs          = n / n_elems_per_vec;

  _Alignas(VLEN) uint16_t tmp[VLEN / sizeof(uint16_t)];

  HVX_Vector *pv_table = (HVX_Vector *) table;

  int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < n_vecs; ++i) {
    for (int j = 0; j < n_elems_per_vec; ++j) {
      int index = i * n_elems_per_vec + j;
      tmp[j]    = index | 0x8000;  // negative value
    }

    *pv_table++ = hvx_my_exp2_vhf_vqf16(vmem(tmp));
  }
  int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);

  FARF(ALWAYS, "%s: precompute table took %lld us", __func__, elapsed_us);
}

void init_precomputed_tables();

void init_precomputed_tables() {
  precompute_safe_softmax_exp2_table();
}
