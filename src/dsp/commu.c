// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_mem.h>
#include <qurt_memory.h>
#include <stdint.h>

#include "dsp/ops.h"
#include "dsp/power.h"
#include "htp_ops.h"      // QAIC auto-generated header for FastRPC

static int dummy_handle;  // served as the global handle

// FastRPC interface
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle) {
  // We may keep this function simple and leave real initialization somewhere else

  *handle = (remote_handle64) &dummy_handle;

  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_close(remote_handle64 handle) {
  reset_power();

  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_init_backend(remote_handle64 handle) {
  FARF(ALWAYS, "init_backend called");

  setup_power();

  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_rms_norm_f32(remote_handle64 handle, int32 fd0, int32 offset0, int32 fd1, int32 offset1, int32 ne0,
                               int32 ne1) {

  int64_t t0 = HAP_perf_get_qtimer_count();

  // TODO(hzx): maybe we should cache fd -> address mapping
  uint8_t *p0, *p1;
  p0 = p1 = NULL;

  int err = HAP_mmap_get(fd0, (void **) &p0, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(fd1, (void **) &p1, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  int64_t t1 = HAP_perf_get_qtimer_count();

  const float *input = (float *) (p1 + offset1);
  size_t input_size = ne0 * ne1 * sizeof(float); // This can be inaccurate
  qurt_mem_cache_clean((qurt_addr_t) input, input_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  float *output = (float *) (p0 + offset0);
  err           = hvx_rms_norm_f32(output, input, ne0, ne1);
  if (err) {
    FARF(ALWAYS, "%s: bad input or alignment", __func__);
    goto bail;
  }

  // TODO(hzx): we need a smarter way to do this
  size_t output_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
  err                = qurt_mem_cache_clean((qurt_addr_t) output, output_size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

  int64_t t2 = HAP_perf_get_qtimer_count();

bail:
  if (p0) {
    HAP_mmap_put(fd0);
  }
  if (p1) {
    HAP_mmap_put(fd1);
  }

  int64_t elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  FARF(ALWAYS, "rms_norm_f32 (ne0=%d, ne1=%d) took %ld us", ne0, ne1, elapsed);
  FARF(ALWAYS, "    core + cache inv+flush: %ld us", HAP_perf_qtimer_count_to_us(t2 - t1));
  return err;
}
