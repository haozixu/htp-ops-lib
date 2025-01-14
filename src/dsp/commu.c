// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <HAP_farf.h>

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
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_init_backend(remote_handle64 handle) {
  FARF(ALWAYS, "init_backend called");

  return AEE_SUCCESS;
}
