#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

// for debug
#include <HAP_farf.h>
#include <HAP_perf.h>

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  // int                n_tot_chunks;
  // int                n_chunks_per_task;
  uint8_t           *vtcm_base;
  size_t             vtcm_size_per_thread;
  // params
  __fp16            *O;
  const __fp16      *Q, *K, *V, *mask;
  int                qo_len, kv_len, n_heads, n_kv_heads, head_dim;
} simple_fa_task_state_t;

static inline void swap_ptr(__fp16 **p0, __fp16 **p1) {
  __fp16 *t = *p0;
  *p0       = *p1;
  *p1       = t;
}

static inline void hvx_fill_uh(void *p, uint16_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  HVX_Vector  v_v    = Q6_Vh_vsplat_R(v);
  HVX_Vector *pv_out = (HVX_Vector *) p;
  for (int i = 0; i < size / VLEN; ++i) {
    *pv_out++ = v_v;
  }
}

size_t compute_vtcm_usage(int gqa_factor, int head_dim, int n_rows, int n_cols) {
  const size_t g_br = align_up(gqa_factor * n_rows, HMX_FP16_TILE_N_ROWS);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);    // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(n_cols * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * n_cols * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);        // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(n_cols * sizeof(__fp16), 256);

  size_t total = qo_tile_size * 3 /* Q, O0, O1 */ + kv_tile_size * 2 /* K, V */ + core_tile_size * 2 /* S, P */ +
                 d_tile_size /* D */ + col_vec_size * 4 + row_vec_size * 2 /* 2x row buffer */ +
                 512 /* HMX column scales */;
  return total;
}

#define MAX_G_BR        256
#define __vec_aligned__ __attribute__((aligned(VLEN)))

void find_chunk_size(size_t *blk_r, size_t *blk_c, int gqa_factor, int head_dim, int qo_len, int kv_len, size_t limit) {
  const size_t nr_unit = ceil_div(HMX_FP16_TILE_N_ROWS, gqa_factor);
  const size_t nc_unit = 64;

  size_t nr = nr_unit, nc = nc_unit;
  size_t nr_ok = nr, nc_ok = nc;
  assert(compute_vtcm_usage(gqa_factor, head_dim, nr, nc) <= limit);

  const size_t max_g_nr = MAX_G_BR;
  const size_t max_nr   = align_up(qo_len, nr_unit);
  const size_t max_nc   = align_up(kv_len, nc_unit);

  // increase Br first
  for (; nr <= max_nr && gqa_factor * nr <= max_g_nr; nr += nr_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr, nc) > limit) {
      break;
    }

    nr_ok = nr;
  }

  // then increase Bc
  for (; nc <= max_nc; nc += nc_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr_ok, nc) > limit) {
      break;
    }

    nc_ok = nc;
  }

  *blk_r = nr_ok, *blk_c = nc_ok;
}

// pre-assert: D is multiple of 64
void simple_flash_attn_core(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
                            const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                            const __fp16 *restrict qk_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                            int head_dim) {
  // "compile-time" configs
  // TODO: make them real compile-time constants (constexpr or template parameters)
  const int G = n_heads / n_kv_heads;  // GQA factor
  const int D = head_dim;

  const bool   qo_fp32_element = true;  // whether Q/O has fp32 elements
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  size_t blk_sz_r, blk_sz_c;  // Br, Bc
  find_chunk_size(&blk_sz_r, &blk_sz_c, G, head_dim, qo_len, kv_len, vtcm_limit - vtcm);
  assert(blk_sz_c % 64 == 0);

  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);  // Br'

  FARF(ALWAYS, "%s: Br=%d Bc=%d Br'=%d", __func__, blk_sz_r, blk_sz_c, g_br);

  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // compute tile/vector sizes
  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);      // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);          // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                  // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  const size_t kv_ld_blk_sz   = head_dim;               // no * element_size
  const size_t kv_ld_stride   = n_kv_heads * head_dim;  // no * element_size
  const size_t qo_ldst_blk_sz = G * head_dim;           // no * element_size
  const size_t qo_ldst_stride = n_heads * head_dim;     // no * element_size

  // begin VTCM allocation
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *v_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);

  // end VTCM allocation
  assert(vtcm_cur <= vtcm_limit);

  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;  // log2(e) = 1.44269504
  __fp16 qk_scale_hf = (__fp16) qk_scale;                     // NOTE: this conversion can be very slow

  // NOTE: there are 32 effective elements in scales, use 4 bytes splat (not Q6_Vh_vsplat_R)
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // prepare constants
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;  // range [0, 4096), two HMX tiles
  }

  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    // offsets within the first tile
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  /////////////// CORE LOGIC BEGIN

  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows*G, D] tile of Q into VTCM
    {
      // load block size: G*D elements
      const size_t q_ld_blk_sz_bytes = qo_ldst_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = qo_ldst_stride * qo_element_size;  // a.k.a. hidden_size

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      // FIXME(hzx): This L2 fetch may not be very useful
      // NOTE(hzx): what about prefetching in reverse order?
      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        // input positions
        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        // output positions
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0 = *pv_in0++;
            const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            // locate target dual-tile
            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;  // 16 * 128B = 2048B (1 tile)

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
        // clang-format on
      }
    }

    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);  // init to -65504 (-inf)
    hvx_fill_uh(mvec_l, 0, col_vec_size);       // init: 0

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);

    // inner loop over kv_len
    for (int jc = 0; jc < kv_len; jc += blk_sz_c) {
      const size_t n_cols        = smin(kv_len - jc, blk_sz_c);
      const size_t n_col_tiles   = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      const size_t n_col_vec_cnt = ceil_div(n_cols, 64);

      // load [Bc, D] tile of K^T into VTCM
      // TODO(hzx): use DMA? if DMA used, we should read from VTCM
      {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        // FIXME: Is this necessary?
        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 1);

        const HVX_Vector v_step         = Q6_V_vsplat_R(4);
        const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);

        // continuous fetch loop: [Bc/32, 32, D]
        for (int r0 = 0; r0 < n_col_tiles; ++r0) {
          __fp16 *out_base = k_tile + r0 * HMX_FP16_TILE_N_COLS * head_dim;  // transposed [D, 32] column chunk

          HVX_Vector v_offsets = v_offsets_base;                             // reset to base offsets

          for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
            int r = r0 * HMX_FP16_TILE_N_COLS + r1;
            if (r >= n_cols) {
              break;
            }

            const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + r * kv_ld_stride);

            // clang-format off
            #pragma unroll
            for (int d = 0; d < D / 64; ++d) {
              __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
              Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
            }
            // clang-format on

            v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
          }
        }
      }

      // issue L2 prefetch of V tile
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 0);
      }

      // compute dot product of tiles: dot(Q[Br', D], K[Bc, D]) ==> [Br', Bc]
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = k_tile + c * HMX_FP16_TILE_N_COLS * head_dim;

              // NOTE: we use `n_tiles_per_blk_c` instead of `n_col_tiles` here
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;
              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }

      // core softmax computation
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);  // fp16: -65504

        // prepare leftover masks
        const int            n_col_leftover  = n_cols % 64;
        const HVX_VectorPred q_leftover_mask = Q6_Q_vsetq_R(n_col_leftover);

        // read from S tile, process 2 rows at a time, generate P tile
        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          // vector registers, empty when initialized, fill in 2 rows at a time
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            // NOTE: make sure this match with S tile generation logic
            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            // decode 2 rows into row buffers
            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;  // 16 * 128B = 2048B (1 tile)

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            // mask out out-of-boundary values
            if (n_col_leftover > 0) {
              const int last_vec_idx    = n_cols / 64;
              row_buffer0[last_vec_idx] = Q6_V_vmux_QVV(q_leftover_mask, row_buffer0[last_vec_idx], v_neg_inf);
              row_buffer1[last_vec_idx] = Q6_V_vmux_QVV(q_leftover_mask, row_buffer1[last_vec_idx], v_neg_inf);
            }

            // compute rowmax(S)
            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            // reduction phase 1: inter-vector
            for (int j = 0; j < n_col_vec_cnt; ++j) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, row_buffer0[j]);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, row_buffer1[j]);
            }

            // clang-format off
            // reduction phase 2: intra-vector
            #pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }
            // clang-format on
            // now, v_s_rowmax0[63] = rowmax(S)_0, v_s_rowmax1[63] = rowmax(S)_1

            // shift rowmax(S_i^j) into v_s_rowmax_local
            HVX_Vector v_s_rowmax_pack2 =
              Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));    // highest 4 bytes are valid
            HVX_Vector v_s_rowmax_pack2_rot =
              Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));   // lowest 4 bytes valid
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));  // highest r*2 bytes valid
            v_s_rowmax_local = Q6_V_vlalign_VVR(v_s_rowmax_pack2_rot, v_s_rowmax_local_rot, r_vec_off * sizeof(__fp16));

            // compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            // broadcast new m_0^j and m_1^j to whole vectors using LUT
            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));  // lowest 4 bytes are valid
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            // compute rows of P_i^j = exp(S_i^j - m_i^j)
            // write permuted rows of P tile into VTCM
            // compute rowsum(P)
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;                                            // qf16
            HVX_Vector       v_p_rowsum1 = v_zero;                                            // qf16
            for (int c = 0; c < n_cols; c += 64) {
              HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);  // qf16
              HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);  // qf16

              HVX_Vector v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
              HVX_Vector v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);

              // handle qk_mask here
              int q_idx0 = ir + (r + 0) / G;
              int q_idx1 = ir + (r + 1) / G;
              int k_idx  = jc + c;

              // hope this won't cause out-of-bounds access
              HVX_Vector v_mask0 = vmemu(qk_mask + q_idx0 * kv_len + k_idx);
              HVX_Vector v_mask1 = vmemu(qk_mask + q_idx1 * kv_len + k_idx);

              const HVX_Vector v_fp16_mask_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16: -16.0
              HVX_VectorPred   q_mask_out0           = Q6_Q_vcmp_gt_VhfVhf(v_fp16_mask_threshold, v_mask0);
              HVX_VectorPred   q_mask_out1           = Q6_Q_vcmp_gt_VhfVhf(v_fp16_mask_threshold, v_mask1);

              v_p_row0_hf = Q6_V_vmux_QVV(q_mask_out0, v_zero, v_p_row0_hf);
              v_p_row1_hf = Q6_V_vmux_QVV(q_mask_out1, v_zero, v_p_row1_hf);

              // compute P tile output positions
              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;  // 16 * 128B = 2048B (1 tile)

              // write to P tile
              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              // rowsum(P) phase 1 reduction
              v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum0, v_p_row0_hf);
              v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum1, v_p_row1_hf);
            }

            // clang-format off
            // rowsum(P) phase 2 reduction
            #pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            // clang-format on
            // now, v_p_rowsum0[63] = rowsum(P)_0, v_p_rowsum1[63] = rowsum(P)_1

            // shift rowsum(P) into v_p_rowsum_local
            HVX_Vector v_p_rowsum_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_p_rowsum1, v_p_rowsum0, -2));
            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local = Q6_V_vlalign_VVR(v_p_rowsum_pack2_rot, v_p_rowsum_local_rot, r_vec_off * sizeof(__fp16));
          }

          // write local vector registers back to VTCM
          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
      }

      // load [Bc, D] tile of V into VTCM
      {
        // NOTE: at tile granularity, tile V's layout is column-major rather than row-major
        // because V tile is an RHS of matmul and HMX's dot RHS operands are column-major tiles
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        for (int r = 0; r < n_cols; r += 2) {
          const bool next_row_valid = (r + 1) < n_cols;

          const HVX_Vector *pv_in0 = (const HVX_Vector *) (v_ld_base + (r + 0) * kv_ld_stride);
          const HVX_Vector *pv_in1 = (const HVX_Vector *) (v_ld_base + (r + 1) * kv_ld_stride);

          // clang-format off
          #pragma unroll
          for (int c = 0; c < D; c += 64) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;
            int c0 = c / HMX_FP16_TILE_N_COLS;

            // transposed tile index: (c0, r0) => c0 * Bc/32 + r0
            int     tile_idx0  = (c0 + 0) * n_tiles_per_blk_c + r0;
            int     tile_idx1  = (c0 + 1) * n_tiles_per_blk_c + r0;
            __fp16 *tile_base0 = v_tile + tile_idx0 * HMX_FP16_TILE_N_ELMS;
            __fp16 *tile_base1 = v_tile + tile_idx1 * HMX_FP16_TILE_N_ELMS;

            HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r1 / 2;
            HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r1 / 2;
            *pv_out0            = Q6_V_lo_W(vp);
            *pv_out1            = Q6_V_hi_W(vp);
          }
          // clang-format on
        }
      }

      // issue L2 prefetch of the next K tile
      {
        int jc_next = jc + blk_sz_c;
        if (jc_next < kv_len) {
          const size_t n_cols_next = smin(kv_len - jc_next, blk_sz_c);

          const __fp16 *k_ld_base = K + jc_next * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
          l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols_next, 0);
        }
      }

      // NOTE: after the use of rowmax(S), store exp(m_i^{j-1} - m_i^j)^{-1} in the very same VTCM buffer
      HVX_Vector *mvec_inv_exp_m_diff = mvec_s_rowmax;

      // update rowmax vector m_i and vector l_i
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {  // i => r_vec_idx?
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);  // qf16

          HVX_Vector v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);    // fp16

          // l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);  // qf16
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vqf16(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          // compute exp(m_i^{j-1} - m_i^j)^{-1} = exp(m_i^j - m_i^{j-1})
          // HVX_Vector v_neg_m_diff = Q6_V_vxor_VV(v_m_diff, Q6_Vh_vsplat_R(0x8000));  // flip sign bit
          // mvec_inv_exp_m_diff[i]  = hvx_my_exp2_vhf_vqf16(v_neg_m_diff);             // fp16

          mvec_inv_exp_m_diff[i] = hvx_my_inv_vhf(v_exp_m_diff_hf);  // fp16
        }
      }

      // compute O_i^j = diag(exp(m_i^{j-1} - m_i^j))^{-1} O_i^{j-1} + P_i^j V_j
      {
        // generate D tile = diag(exp(m_i^j - m_i^{j-1}))
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_inv_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // D: [Br', Br']
              __fp16 *o_tile_in =
                o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // O: [Br', D] --T-> [D, Br']
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // P: [Br', Bc]
              __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // V: [Bc, D] --T-> [D, Bc]
              hmx_load_tiles_fp16(p_tile_in, v_tile_in, n_col_tiles);

              // NOTE: O's layout is also column-major as O is always on the RHS
              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
    }

    // generate final output: scale O_i = diag(l_i^{-1}) O_i
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;

            // NOTE: to simplify final output procedure, we turn final O into row-major layout
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }

    // store [n_rows*G, D] tile of O back to memory
    {
      const size_t o_st_blk_sz_bytes = qo_ldst_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = qo_ldst_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        HVX_Vector *pv_out0 =
          (HVX_Vector *) (o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * head_dim * qo_element_size);
        HVX_Vector *pv_out1 =
          (HVX_Vector *) (o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);
        
            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        } else {
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
  }
}

void simple_flash_attn_worker(void *data, int worker_index) {
  simple_fa_task_state_t *s = (simple_fa_task_state_t *) data;

  uint8_t *vtcm       = s->vtcm_base + worker_index * s->vtcm_size_per_thread;
  uint8_t *vtcm_limit = vtcm + s->vtcm_size_per_thread;

  hmx_manager_enable_execution();

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(s->task_id)) - 1;
    if (task_id >= s->n_tasks) {
      break;
    }

    int kv_head_idx = task_id;
    simple_flash_attn_core(kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->mask, s->qo_len, s->kv_len,
                           s->n_heads, s->n_kv_heads, s->head_dim);
  }

  hmx_manager_disable_execution();

  worker_pool_synctoken_jobdone(&(s->sync_ctx));
}

/**
 * Simple llama.cpp-style FlashAttention implementation
 *
 * batch_size dimension is omitted
 * 
 * Q: [qo_len, n_heads, head_dim], K/V: [kv_len, n_kv_heads, head_dim]
 * mask: [qo_len*, kv_len] broadcast to each head (first dimension maybe larger than qo_len)
 */
int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
  if (head_dim % 64 != 0 || n_heads % n_kv_heads != 0) {
    return -1;
  }

  const int    n_workers            = num_hvx128_contexts;
  const size_t vtcm_size_per_thread = 1024 * 1024;
  assert(n_workers * vtcm_size_per_thread < 6 * 1024 * 1024);  // don't use too much VTCM

  simple_fa_task_state_t state;
  state.O          = O;
  state.Q          = Q;
  state.K          = K;
  state.V          = V;
  state.mask       = mask;
  state.qo_len     = qo_len;
  state.kv_len     = kv_len;
  state.n_heads    = n_heads;
  state.n_kv_heads = n_kv_heads;
  state.head_dim   = head_dim;

  // TODO(hzx): parallelize along query_len x n_kv_heads dimension
  // size_t n_tot_chunks      = qo_len * n_kv_heads;
  // size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  state.task_id              = 0;
  state.n_tasks              = n_kv_heads;
  state.vtcm_base            = (uint8_t *) vtcm_manager_get_vtcm_base();
  state.vtcm_size_per_thread = vtcm_size_per_thread;

  worker_pool_job_t job;
  job.fptr = simple_flash_attn_worker;
  job.dptr = &state;

  int64_t t0 = HAP_perf_get_time_us();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  int64_t elapsed_us = HAP_perf_get_time_us() - t0;
  FARF(ALWAYS, "%s: %lld us, qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d", __func__, elapsed_us, qo_len,
       kv_len, n_heads, n_kv_heads, head_dim);

  return 0;
}

#define Br 32
#define Bc 256
#define D  128

int naive_flash_attn(float *restrict O, const float *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
  // 参数校验
  if (n_heads % n_kv_heads != 0) {
    return -1;
  }
  if (head_dim > D) {
    return -1;
  }

  const int gqa_factor = n_heads / n_kv_heads;

  const size_t qo_stride = n_heads * head_dim;
  const size_t kv_stride = n_kv_heads * head_dim;

  // 计算特征缩放因子
  const float qk_scale = 1.0f / sqrtf(head_dim);

  for (int h = 0; h < n_heads; ++h) {
    const int h_kv = h / gqa_factor;

    // 分块处理Q序列
    for (int i = 0; i < qo_len; i += Br) {
      const int q_start = i;
      const int q_end   = (i + Br) < qo_len ? (i + Br) : qo_len;
      const int br      = q_end - q_start;

      static float Qi[Br][D];
      static float Oi[Br][D];

      // 加载当前Q块 (fp32)
      const float *Q_src = Q + q_start * qo_stride + h * head_dim;
      for (int r = 0; r < br; ++r) {
        for (int d = 0; d < head_dim; ++d) {
          Qi[r][d] = Q_src[r * qo_stride + d];
        }
      }

      // 初始化输出块和中间状态
      static float mi[Br];
      static float li[Br];
      for (int r = 0; r < br; ++r) {
        mi[r] = -INFINITY;
        li[r] = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          Oi[r][d] = 0.0f;
        }
      }

      // 分块处理KV序列
      for (int j = 0; j < kv_len; j += Bc) {
        const int k_start = j;
        const int k_end   = (j + Bc) < kv_len ? (j + Bc) : kv_len;
        const int bc      = k_end - k_start;

        // 加载当前K块并转换为fp32
        static float  Kj[Bc][D];
        const __fp16 *K_src = K + k_start * kv_stride + h_kv * head_dim;
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Kj[c][d] = ((float) K_src[c * kv_stride + d]) * qk_scale;
          }
        }

        // 计算注意力分数块
        static float Sij[Br][Bc];
        for (int r = 0; r < br; ++r) {
          for (int c = 0; c < bc; ++c) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              sum += Qi[r][d] * Kj[c][d];
            }
            Sij[r][c] = sum;
          }
        }

        // 应用掩码（如果存在）
        if (mask != NULL) {
          for (int r = 0; r < br; ++r) {
            for (int c = 0; c < bc; ++c) {
              const int mask_idx = (q_start + r) * kv_len + (k_start + c);
              Sij[r][c] += (float) mask[mask_idx];
            }
          }
        }

        // 加载当前V块并转换为fp32
        static float  Vj[Bc][D];
        const __fp16 *V_src = V + k_start * kv_stride + h_kv * head_dim;
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Vj[c][d] = (float) V_src[c * kv_stride + d];
          }
        }

        // 在线softmax更新
        for (int r = 0; r < br; ++r) {
          // 当前块最大值
          float m_curr = -INFINITY;
          for (int c = 0; c < bc; ++c) {
            if (Sij[r][c] > m_curr) {
              m_curr = Sij[r][c];
            }
          }

          // 全局最大值
          const float m_new = fmaxf(mi[r], m_curr);

          // 计算指数和
          float exp_sum = 0.0f;
          float exp_values[Bc];
          for (int c = 0; c < bc; ++c) {
            exp_values[c] = expf(Sij[r][c] - m_new);
            exp_sum += exp_values[c];
          }

          // 更新累积因子
          const float l_prev = li[r] * expf(mi[r] - m_new);
          const float l_new  = l_prev + exp_sum;

          const float f = expf(m_new - mi[r]);
          // 更新输出值
          for (int d = 0; d < head_dim; ++d) {
            // 校正历史累积值
            Oi[r][d] *= f;

            // 累加新值
            float sum = 0.0f;
            for (int c = 0; c < bc; ++c) {
              sum += exp_values[c] * Vj[c][d];
            }
            Oi[r][d] += sum;
          }

          // 更新中间状态
          mi[r] = m_new;
          li[r] = l_new;
        }
      }

      // 写入最终输出
      float *O_dst = O + q_start * qo_stride + h * head_dim;
      for (int r = 0; r < br; ++r) {
        const float scale = 1.0f / li[r];
        for (int d = 0; d < head_dim; ++d) {
          O_dst[r * qo_stride + d] = Oi[r][d] * scale;
        }
      }
    }
  }
  return 0;
}

#undef Br
#undef Bc
#undef D
