#include "dsp/power.h"

#include <HAP_farf.h>
#include <HAP_power.h>
#include <string.h>

// TODO(hzx): maybe we should set params according to SoC model
void setup_power() {
  static int power_ctx;

  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;

  req.dcvs_v3.dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;

  req.dcvs_v3.set_latency = TRUE;
  req.dcvs_v3.latency     = 100;  // microseconds

  req.dcvs_v3.set_core_params           = TRUE;
  req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_TURBO_PLUS;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;

  req.dcvs_v3.set_bus_params           = TRUE;
  req.dcvs_v3.bus_params.min_corner    = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.bus_params.max_corner    = HAP_DCVS_VCORNER_TURBO_PLUS;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;

  int ret = HAP_power_set(&power_ctx, &req);
  if (ret != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set DCVS v3 failed");
  }
}

void reset_power() {
  static int power_ctx;

  HAP_power_request_t request;
  memset(&request, 0, sizeof(HAP_power_request_t));
  request.type                    = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_dcvs_enable = TRUE;
  request.dcvs_v3.set_latency     = TRUE;
  request.dcvs_v3.latency         = 65535;
  request.dcvs_v3.set_core_params = TRUE;
  request.dcvs_v3.set_bus_params  = TRUE;
  HAP_power_set(&power_ctx, &request);
}
