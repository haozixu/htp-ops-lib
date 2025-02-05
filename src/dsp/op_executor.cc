#include "dsp/op_executor.h"

#include <qurt_memory.h>

#include <vector>

#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "op_reg.h"

extern "C" {

#define IN_PTR(i)  in_bufs[i].first
#define OUT_PTR(i) out_bufs[i].first

int execute_op_simple(struct OpComputeRequest *req) {
  using FatPointer = std::pair<uint8_t *, size_t>;
  std::vector<FatPointer> in_bufs, out_bufs;

  auto add_buffer = [](std::vector<FatPointer> &bufs, const RpcmemBufAddr &buf_addr, size_t size) {
    auto base = reinterpret_cast<uint8_t *>(mmap_manager_get_map(buf_addr.fd));
    auto ptr  = base != nullptr ? base + buf_addr.offset : nullptr;
    bufs.push_back({ ptr, size });
  };

  auto validate_in_bufs = [&]() {
    for (auto [ptr, size] : in_bufs) {
      if (ptr) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
      }
    }
  };

  auto validate_out_bufs = [&]() {
    for (auto [ptr, size] : out_bufs) {
      if (ptr) {
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
        auto params = reinterpret_cast<MatMulPermutedW16A32Params *>(req->payload);
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

    default:
      break;
  }
  return ret;
}

#undef IN_PTR
#undef OUT_PTR
}
