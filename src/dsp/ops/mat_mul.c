#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

// debug & profile
#include "HAP_farf.h"
#include "HAP_perf.h"

#define WEIGHT_AREA_SIZE     (1 * 1048576)
#define ACTIVATION_AREA_SIZE (1 * 1048576)
#define OUTPUT_AREA_SIZE     (1 * 1048576)
#define SCRATCH_AREA_SIZE    (1 * 1048576)

static inline void swap_ptr(void **p1, void **p2) {
  void *t = *p1;
  *p1     = *p2;
  *p2     = t;
}

static inline size_t get_super_block_size(enum ggml_type weight_type) {
  switch (weight_type) {
    case GGML_TYPE_Q4_0:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return 0;
  }
}

static inline int dma_issue_load_from_ddr(dma_desc_1d_t *desc, void *vtcm_dst, const void *src, size_t size) {
  dma_wait_for_idle();

  desc->next       = 0;
  desc->length     = size;
  desc->type       = DMA_DESC_TYPE_1D;
  desc->src_bypass = 1;
  desc->dst_bypass = 0;
  desc->order      = 1;
  desc->dstate     = DMA_DESC_DSTATE_PENDING;
  desc->src        = (uint32_t) src;
  desc->dst        = (uint32_t) vtcm_dst;

  return dma_submit_one(desc);
}

static void find_chunk_size(size_t x_max, size_t y_max, size_t xy_max, size_t x_unit, size_t y_unit, size_t *x_out,
                            size_t *y_out) {
  int64_t best_xy = 0;
  size_t  best_x, best_y;

  for (size_t x = x_max; x > 0; x -= x_unit) {
    size_t  y  = smin(align_down(xy_max / x, y_unit), y_max);
    int64_t xy = x * y;
    if (best_xy < xy) {
      best_xy = xy;
      best_x = x, best_y = y;
    }
  }
  *x_out = best_x, *y_out = best_y;
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                                   int k) {
  // NOTE(hzx): prefetch two rows at a time. This needs improvement
  size_t prefetch_size  = k * 2 * sizeof(float);
  size_t input_row_size = k * sizeof(float);

  assert(k % HMX_FP16_TILE_N_COLS == 0);
  assert(input_row_size % VLEN == 0);
  int vecs_per_row = input_row_size / VLEN;

  for (int r = 0; r < n_rows; r += 2) {
    int prefetch_row_idx = r + 2;
    if (prefetch_row_idx < n_rows) {
      const float *prefetch_addr = src + prefetch_row_idx * k;
      l2fetch(prefetch_addr, prefetch_size, prefetch_size, 1, 0);
    }

    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    bool is_leftover = (n_rows - r) < 2;

    const HVX_Vector *pv_in0 = ((const HVX_Vector *) src) + r * vecs_per_row;
    const HVX_Vector *pv_in1 = pv_in0 + vecs_per_row;
    for (int vec_cnt = 0; vec_cnt < vecs_per_row; ++vec_cnt) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = is_leftover ? Q6_V_vzero() : *pv_in1++;  // next row

      HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      // compute output position
      int c        = vec_cnt * (VLEN / sizeof(float));  // 128/4=32
      int c0       = c / HMX_FP16_TILE_N_COLS;          // tile column index
      int tile_idx = r0 * (k / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                n_tot_chunks;
  int                n_chunks_per_task;
  int                k;
  __fp16            *dst;
  const __fp16      *src;
} permuted_weight_transfer_fp16_task_state_t;

static void transfer_permuted_weight_fp16_task(__fp16 *restrict vtcm_dst, const __fp16 *restrict src, int k,
                                               int n_col_tiles) {
  // transfer logical K*(32*n_col_tiles) weight block, direct copy, no extra computation
  size_t size   = k * n_col_tiles * HMX_FP16_TILE_N_COLS * sizeof(__fp16);
  int    n_vecs = size / VLEN;

  const size_t PREFETCH_SIZE   = 4096;
  const int    PREFETCH_N_VECS = PREFETCH_SIZE / VLEN;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        size_t prefetch_n_vecs = smin(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    *pv_out++ = *pv_in++;
  }
}

static void transfer_permuted_weight_fp16_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_transfer_fp16_task_state_t *state = (permuted_weight_transfer_fp16_task_state_t *) data;

  int k = state->k;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    int           c        = chunk_idx * HMX_FP16_TILE_N_COLS;
    __fp16       *vtcm_dst = state->dst + c * k;
    const __fp16 *src      = state->src + c * k;
    transfer_permuted_weight_fp16_task(vtcm_dst, src, k, chunk_size);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void transfer_permuted_weight_chunk_fp16(__fp16 *vtcm_dst, const __fp16 *src, int n_cols, int k) {
  // NOTE(hzx): weight matrix is already transposed. n_cols actually turns into n_rows
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const bool use_dma = true;

  if (use_dma) {
    size_t size = n_cols * k * sizeof(__fp16);

    dma_desc_1d_t desc;
    dma_issue_load_from_ddr(&desc, vtcm_dst, src, size);
    dma_wait_for_idle();

    return;
  }

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_cols / HMX_FP16_TILE_N_COLS;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);
  // size_t n_chunks_per_task = 1;

  permuted_weight_transfer_fp16_task_state_t state;
  state.task_id           = 0;                                          // task id counter
  state.n_tasks           = ceil_div(n_tot_chunks, n_chunks_per_task);  // old value: n_cols / HMX_FP16_TILE_N_COLS
  state.n_tot_chunks      = n_tot_chunks;
  state.n_chunks_per_task = n_chunks_per_task;
  state.k                 = k;
  state.dst               = vtcm_dst;
  state.src               = src;

  worker_pool_job_t job;
  job.fptr = transfer_permuted_weight_fp16_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  int                n_tot_chunks;  // number of total super-blocks
  int                n_chunks_per_task;
  __fp16            *dst;
  const void        *src;
  enum ggml_type     quant_type;
  bool               src_in_vtcm;
} permuted_weight_dequantize_qk_0_hvx_task_state_t;

#define EXPAND_QK_0_VEC_SCALES_COMPUTATION(blk, vs0_c, vs1_c, vs2_c, vs3_c) \
  do {                                                                      \
    __fp16 s0 = blk.scales[0];                                              \
    __fp16 s1 = blk.scales[1];                                              \
    __fp16 s2 = blk.scales[2];                                              \
    __fp16 s3 = blk.scales[3];                                              \
    __fp16 s4 = blk.scales[4];                                              \
    __fp16 s5 = blk.scales[5];                                              \
    __fp16 s6 = blk.scales[6];                                              \
    __fp16 s7 = blk.scales[7];                                              \
                                                                            \
    HVX_Vector vs0 = Q6_Vh_vsplat_R(fp16_to_bits(&s0));                     \
    HVX_Vector vs1 = Q6_Vh_vsplat_R(fp16_to_bits(&s1));                     \
    HVX_Vector vs2 = Q6_Vh_vsplat_R(fp16_to_bits(&s2));                     \
    HVX_Vector vs3 = Q6_Vh_vsplat_R(fp16_to_bits(&s3));                     \
    HVX_Vector vs4 = Q6_Vh_vsplat_R(fp16_to_bits(&s4));                     \
    HVX_Vector vs5 = Q6_Vh_vsplat_R(fp16_to_bits(&s5));                     \
    HVX_Vector vs6 = Q6_Vh_vsplat_R(fp16_to_bits(&s6));                     \
    HVX_Vector vs7 = Q6_Vh_vsplat_R(fp16_to_bits(&s7));                     \
                                                                            \
    vs0_c = Q6_V_valign_VVR(vs1, vs0, 64);                                  \
    vs1_c = Q6_V_valign_VVR(vs3, vs2, 64);                                  \
    vs2_c = Q6_V_valign_VVR(vs5, vs4, 64);                                  \
    vs3_c = Q6_V_valign_VVR(vs7, vs6, 64);                                  \
  } while (0)

static void dequantize_permuted_weight_q4_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst,
                                                             const my_block_q4_0 *restrict src, int n_blocks,
                                                             bool src_in_vtcm) {
  const int L2_PREFETCH_N_BLOCKS = 32;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
  };
  const HVX_Vector vlut_cvt = vmem(q4_0_to_fp16_lut);

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q4_0), sizeof(my_block_q4_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    HVX_Vector qs = vmemu(src[i].quants);

    HVX_Vector v_qs_lo = qs;  // no need to mask out high 4 bits in each byte since vlut will do that for us
    HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
    HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);

    // NOTE(hzx): the previous scalar->vector scales implementation is faster when src resides in DDR memory
    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q0), vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q0), vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q1), vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q1), vs3_c));
  }
}

static void dequantize_permuted_weight_q8_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst,
                                                             const my_block_q8_0 *restrict src, int n_blocks,
                                                             bool src_in_vtcm) {
  const int L2_PREFETCH_N_BLOCKS = 16;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q8_0), sizeof(my_block_q8_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    HVX_Vector vq0 = vmemu(src[i].quants);
    HVX_Vector vq1 = vmemu(src[i].quants + VLEN);

    HVX_VectorPair vp0 = Q6_Wh_vunpack_Vb(vq0);
    HVX_VectorPair vp1 = Q6_Wh_vunpack_Vb(vq1);

    HVX_Vector v0 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp0));
    HVX_Vector v1 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp0));
    HVX_Vector v2 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp1));
    HVX_Vector v3 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp1));

    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v1, vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v2, vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v3, vs3_c));
  }
}

static void dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_dequantize_qk_0_hvx_task_state_t *state = (permuted_weight_dequantize_qk_0_hvx_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    __fp16 *vtcm_dst = state->dst + chunk_idx * QK_K;

    if (state->quant_type == GGML_TYPE_Q4_0) {
      const my_block_q4_0 *src = ((const my_block_q4_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q4_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm);
    } else if (state->quant_type == GGML_TYPE_Q8_0) {
      const my_block_q8_0 *src = ((const my_block_q8_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q8_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm);
    }
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne,
                                                              enum ggml_type type, void *vtcm_scratch) {
  assert(ne % QK_K == 0);

  const bool src_in_vtcm = true;

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = ne / QK_K;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  permuted_weight_dequantize_qk_0_hvx_task_state_t state;
  state.task_id           = 0;
  state.n_tasks           = ceil_div(n_tot_chunks, n_chunks_per_task);
  state.n_tot_chunks      = n_tot_chunks;
  state.n_chunks_per_task = n_chunks_per_task;
  state.dst               = vtcm_dst;
  state.src               = src_in_vtcm ? vtcm_scratch : src;
  state.quant_type        = type;
  state.src_in_vtcm       = src_in_vtcm;

  worker_pool_job_t job;
  job.fptr = dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop;
  job.dptr = &state;

  // int64_t t0 = HAP_perf_get_qtimer_count();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  // int64_t e = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  // FARF(ALWAYS, "QK_0 dequantize: ne: %d time: %lld us", ne, e);
}

static void core_dot_chunk_fp16(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
      const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(out_tile);
    }
  }

  hmx_unit_release();
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows,
                                               int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;
    int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_src = ((const HVX_Vector *) tile)[r1 / 2];

      HVX_VectorPair vp = hvx_my_vhf_to_wsf(v_src);

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory

      *pv_out0 = Q6_V_lo_W(vp);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_V_hi_W(vp);
      }
    }
  }
}

int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, int m, int k, int n) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = WEIGHT_AREA_SIZE;

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = smin(m - mr, m_chunk_n_rows);

    // int64_t act_t0 = HAP_perf_get_qtimer_count();
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k);
    }
    // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

    // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = smin(n - nc, n_chunk_n_cols);

      // int64_t wei_t0 = HAP_perf_get_qtimer_count();
      {
        const __fp16 *permuted_weight_chunk = permuted_weight + nc * k;
        transfer_permuted_weight_chunk_fp16(vtcm_weight, permuted_weight_chunk, n_cols, k);
      }
      // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

      // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

      // int64_t core_t0 = HAP_perf_get_qtimer_count();
      {
        const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
      }
      // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

      // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

      // int64_t out_t0 = HAP_perf_get_qtimer_count();
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
      }
      // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

      // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n * sizeof(__fp16);
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}

int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *restrict activation,
                                     const uint8_t *restrict permuted_weight, int m, int k, int n,
                                     enum ggml_type weight_type) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  size_t super_block_size = get_super_block_size(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = WEIGHT_AREA_SIZE;
  const size_t scratch_area_size    = SCRATCH_AREA_SIZE;

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  assert(m_chunk_n_rows > 0 && n_chunk_n_cols > 0);

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = smin(m - mr, m_chunk_n_rows);

    // int64_t act_t0 = HAP_perf_get_qtimer_count();
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k);
    }
    // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

    // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

    void *buf_curr = vtcm_scratch0;
    void *buf_next = vtcm_scratch1;

    static dma_desc_1d_t desc
      __attribute__((aligned(64)));  // NOTE(hzx): make sure the DMA descriptor's lifetime is long enough

    // issue async DDR data transfer for the first weight chunk
    {
      const size_t n_cols_first            = smin(n, n_chunk_n_cols);
      const size_t first_weight_chunk_size = n_cols_first * k / QK_K * super_block_size;

      dma_issue_load_from_ddr(&desc, buf_curr, permuted_weight, first_weight_chunk_size);
    }

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = smin(n - nc, n_chunk_n_cols);

      // int64_t wei_t0 = HAP_perf_get_qtimer_count();
      {
        dma_wait_for_idle();  // wait until current weight chunk become ready

        const size_t nc_next = nc + n_chunk_n_cols;
        if (nc_next < n) {
          const size_t n_cols_next = smin(n - nc_next, n_chunk_n_cols);

          const size_t   next_weight_chunk_size = n_cols_next * k / QK_K * super_block_size;
          const uint8_t *next_weight_chunk      = permuted_weight + nc_next * k / QK_K * super_block_size;

          dma_issue_load_from_ddr(&desc, buf_next, next_weight_chunk, next_weight_chunk_size);
        }

        const uint8_t *permuted_weight_chunk = permuted_weight + (nc * k / QK_K) * super_block_size;
        dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, permuted_weight_chunk, n_cols * k, weight_type,
                                                          buf_curr);

        swap_ptr(&buf_curr, &buf_next);
      }
      // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

      // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

      // int64_t core_t0 = HAP_perf_get_qtimer_count();
      {
        const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
      }
      // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

      // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

      // int64_t out_t0 = HAP_perf_get_qtimer_count();
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
      }
      // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

      // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n / QK_K * super_block_size;
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}
