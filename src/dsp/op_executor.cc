#include "dsp/op_executor.h"

#include <qurt_memory.h>

#include <vector>

#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "op_reg.h"

// debug
#include <HAP_farf.h>
#include <HAP_perf.h>

namespace {

size_t ggml_super_block_size(enum ggml_type type) {
  // TODO: more types
  switch (type) {
    case GGML_TYPE_Q4_0:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return -1;
  }
}

enum ggml_type matmul_op_to_weight_type(enum HtpOpsIndex op) {
  switch (op) {
    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      return GGML_TYPE_F16;
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
      return GGML_TYPE_Q4_0;
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
      return GGML_TYPE_Q8_0;
    default:
      return GGML_TYPE_COUNT;  // invalid type
  }
}

}  // namespace

extern "C" {

#define IN_PTR(i)  std::get<0>(in_bufs[i])
#define OUT_PTR(i) std::get<0>(out_bufs[i])

int execute_op_simple(struct OpComputeRequest *req) {
  // using FatPointer = std::pair<uint8_t *, size_t>;
  using Buffer = std::tuple<uint8_t *, size_t, bool>;
  std::vector<Buffer> in_bufs, out_bufs;

  auto add_buffer = [](std::vector<Buffer> &bufs, const RpcmemBufAddr &buf_addr, size_t size, bool cached = true) {
    auto base = reinterpret_cast<uint8_t *>(mmap_manager_get_map(buf_addr.fd));
    auto ptr  = base != nullptr ? base + buf_addr.offset : nullptr;
    bufs.push_back({ ptr, size, cached });
  };

  auto validate_in_bufs = [&]() {
    for (auto [ptr, size, cached] : in_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
      }
    }
  };

  auto validate_out_bufs = [&]() {
    for (auto [ptr, size, cached] : out_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
      }
    }
  };

  int ret = 0;
  switch (req->op) {
    case HTP_OPS_RMS_NORM_F32:
      {
        auto   params = reinterpret_cast<RmsNormF32Params *>(req->payload);
        size_t size   = params->ne0 * params->ne1 * sizeof(float);

        add_buffer(out_bufs, params->dst, size);
        add_buffer(in_bufs, params->src, size);

        validate_in_bufs();
        ret = hvx_rms_norm_f32((float *) OUT_PTR(0), (const float *) IN_PTR(0), params->ne0, params->ne1);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      {
        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n * sizeof(__fp16);

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size);

        validate_in_bufs();
        ret = hmx_mat_mul_permuted_w16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), (__fp16 *) IN_PTR(1), m, k, n);
        validate_out_bufs();
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
      {
        auto   weight_type      = matmul_op_to_weight_type(static_cast<HtpOpsIndex>(req->op));
        size_t super_block_size = ggml_super_block_size(weight_type);

        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n / QK_K * super_block_size;

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size, false);

        // int64_t t0 = HAP_perf_get_qtimer_count();
        validate_in_bufs();
        // int64_t t1 = HAP_perf_get_qtimer_count();
        ret =
          hmx_mat_mul_permuted_qk_0_d16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), IN_PTR(1), m, k, n, weight_type);
        // int64_t t2 = HAP_perf_get_qtimer_count();
        validate_out_bufs();
        // int64_t t3 = HAP_perf_get_qtimer_count();

        // int64_t mm_time_us  = HAP_perf_qtimer_count_to_us(t2 - t1);
        // int64_t tot_time_us = HAP_perf_qtimer_count_to_us(t3 - t0);

        // FARF(ALWAYS, "mm_time: %lld us, tot_time: %lld us, type: %d, (%d, %d, %d)", mm_time_us, tot_time_us,
        //      weight_type, m, k, n);
        // FARF(ALWAYS, "achieved weight load bandwidth: %.2f GB/s", 1e-3 * weight_size / mm_time_us);
      }
      break;

    default:
      break;
  }
  return ret;
}

#undef IN_PTR
#undef OUT_PTR
}
