#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_internal.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

// debug & profile
#include "HAP_farf.h"
#include "HAP_perf.h"

#define WEIGHT_AREA_SIZE     (1 * 1048576)
#define ACTIVATION_AREA_SIZE (1 * 1048576)
#define OUTPUT_AREA_SIZE     (1 * 1048576)

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

  const HVX_Vector v_zero = Q6_V_vzero();

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
      HVX_Vector v1 = is_leftover ? v_zero : *pv_in1++;  // next row

      HVX_Vector v0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0, v_zero);
      HVX_Vector v1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1, v_zero);

      HVX_Vector v_out = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v1_qf32, v0_qf32));

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

static void transfer_permuted_weight_fp16_worker_loop(void *data) {
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

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_cols / HMX_FP16_TILE_N_COLS;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);
  // size_t n_chunks_per_task = 1;

  permuted_weight_transfer_fp16_task_state_t state;
  state.task_id           = 0;  // task id counter
  state.n_tasks           = ceil_div(n_tot_chunks, n_chunks_per_task); // old value: n_cols / HMX_FP16_TILE_N_COLS
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
      hmx_consume_accumlator_fp16(out_tile);
    }
  }

  hmx_unit_release();
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows,
                                               int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  const HVX_Vector v_zero    = Q6_V_vzero();
  const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
  const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
  const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;
    int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_src = ((const HVX_Vector *) tile)[r1 / 2];

      // converts fp16 to qf16
      v_src = Q6_Vqf16_vadd_VhfVhf(v_src, v_zero);

      // adapted from qhmath_hvx_vqf32_convert_vqf16 (in qhmath_hvx_convert.h)
      // extract packed exp & mantissa
      HVX_Vector exp_comp = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0x1f));    // exp component: low 5 bits
      HVX_Vector mantissa = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0xffe0));  // mantissa: bits 5~15

      // Convert qf16 biased exponent to qf32 biased exponent
      // new exp = exp + ( 127 (qf32 bias) -15(qf16 bias) ) = 112
      exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

      // elements index in v_src: [0, n, 1, n+1, ..., 31, n+31]
      // unpack into [0, 1, ..., 31], [n, n+1, ..., n+31]

      // unpack exp
      HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);  // keep low 16 bits
      HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);

      // unpack mantissa + convert qf16 mantissa to qf32 mantissa (left shift 16 bits)
      HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
      HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);  // keep high 16 bits

      // merge qf32 exp + mantissa
      HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
      HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory
      *pv_out0            = Q6_Vsf_equals_Vqf32(v0_qf32);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_Vsf_equals_Vqf32(v1_qf32);
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
  // FARF(ALWAYS, "    activation load: %ld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %ld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %ld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %ld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n * sizeof(__fp16);
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}
