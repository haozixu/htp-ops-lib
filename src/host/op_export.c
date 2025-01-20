#include "host/op_export.h"

#include "host/session.h"
#include "htp_ops.h"

int htp_ops_rpc_rms_norm_f32(int dst_fd, int dst_offset, int src_fd, int src_offset, int ne0, int ne1) {
  return htp_ops_rms_norm_f32(get_global_handle(), dst_fd, dst_offset, src_fd, src_offset, ne0, ne1);
}
