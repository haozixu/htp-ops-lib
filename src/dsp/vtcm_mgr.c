#include "dsp/vtcm_mgr.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <string.h>

static void *vtcm_base       = 0;
static int   vtcm_mgr_ctx_id = 0;

void vtcm_manager_setup() {
  int err;

  unsigned int            avail_size, total_size;
  compute_res_vtcm_page_t avail_pages, total_pages;
  err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_query_VTCM failed with return code 0x%x", err);
    return;
  }
  FARF(ALWAYS, "available VTCM size: %d KiB, total VTCM size: %d KiB", avail_size / 1024, total_size / 1024);

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);

  // NOTE(hzx): here we try to request all VTCM memory in one page
  HAP_compute_res_attr_set_vtcm_param(&req, total_size, 1);

  vtcm_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // timeout 10ms
  if (vtcm_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    return;
  }

  vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&req);
  memset(vtcm_base, 0, total_size);
}

void vtcm_manager_reset() {
  if (vtcm_mgr_ctx_id) {
    HAP_compute_res_release(vtcm_mgr_ctx_id);
  }
}

void *vtcm_manager_get_vtcm_base() {
  return vtcm_base;
}
