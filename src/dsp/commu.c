// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <qurt_signal.h>
#include <qurt_thread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/mmap_mgr.h"
#include "dsp/op_executor.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "htp_ops.h"  // QAIC auto-generated header for FastRPC
#include "message.h"

static int dummy_handle;  // served as the global handle

struct MessageChannel {
  uint8_t      *msg;
  int           rpcmem_fd;
  size_t        max_msg_size;
  bool          msg_receiver_should_stop;
  qurt_signal_t msg_receiver_ready;
  qurt_thread_t msg_receiver_thread;
};

static struct MessageChannel global_msg_chan;

static void msg_receiver_loop(void *param) {
  struct MessageChannel *chan = (struct MessageChannel *) param;
  qurt_signal_set(&(chan->msg_receiver_ready), 1);

  const int SLEEP_TIME_US = 5;

  while (1) {
    if (chan->msg_receiver_should_stop) {
      break;
    }

    struct MessageHeader *msg_hdr = (struct MessageHeader *) chan->msg;
    if (msg_hdr == NULL) {
      qurt_sleep(SLEEP_TIME_US);  // wait until shared message buffer become available
    }

    // invalidate cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, chan->max_msg_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    // TODO(hzx): use more proper message state
    if (msg_hdr->state.v[0] == 0 || msg_hdr->state.v[1] != 0) {
      qurt_sleep(SLEEP_TIME_US);
      continue;
    }

    for (int i = 0; i < msg_hdr->n_reqs; ++i) {
      struct RequestHeader *req_hdr = message_header_get_request_ptr(msg_hdr, i);
      switch (req_hdr->type) {
        case REQUEST_TYPE_OP_COMPUTE:
          {
            // TODO(hzx): use separate thread (pool) to execute op
            struct OpComputeRequest *compute_req = (struct OpComputeRequest *) req_hdr->data;

            req_hdr->state = execute_op_simple(compute_req);
          }
          break;
        case REQUEST_TYPE_RPCMEM_MAP:
          {
            struct RpcmemMapRequest *map_req = (struct RpcmemMapRequest *) req_hdr->data;
            for (int j = 0; j < map_req->n_puts; ++j) {
              mmap_manager_put_map(map_req->fds[j]);
            }
            req_hdr->state = 0;
          }
          break;
        default:
          break;
      }
    }

    msg_hdr->state.v[1] = 1;
    // flush cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, message_total_size(msg_hdr), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    // TODO(hzx): estimate host's job completion time and sleep
    qurt_sleep(SLEEP_TIME_US);
  }
}

// init an empty (semantically unintialized) message channel
void message_channel_init(struct MessageChannel *chan) {
  chan->msg          = NULL;
  chan->rpcmem_fd    = -1;
  chan->max_msg_size = 0;

  chan->msg_receiver_should_stop = false;
}

bool message_channel_is_active(const struct MessageChannel *chan) {
  return chan->msg != NULL;
}

int message_channel_create(struct MessageChannel *chan, int rpcmem_fd, size_t max_msg_size) {
  uint8_t *p;
  int      err = HAP_mmap_get(rpcmem_fd, (void **) &p, NULL);
  if (err) {
    FARF(ALWAYS, "%s: HAP_mmap_get failed with %x", __func__, err);
    return -1;
  }

  // clear message state
  for (int i = 0; i < sizeof(struct MessageState); ++i) {
    p[i] = 0;
  }

  chan->msg_receiver_should_stop = false;
  qurt_signal_init(&(chan->msg_receiver_ready));

  const size_t stack_size = 8192;
  void *stack = memalign(4096, stack_size);
  if (!stack) {
    FARF(ALWAYS, "%s: failed to allocate memory for thread stack", __func__);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  // launch message receiver thread
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hops-msg-recv");
  qurt_thread_attr_set_priority(&attr, 64);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, stack_size);
  qurt_thread_attr_set_autostack(&attr, QURT_THREAD_AUTOSTACK_ENABLED);

  err = qurt_thread_create(&(chan->msg_receiver_thread), &attr, msg_receiver_loop, chan);
  if (err) {
    FARF(ALWAYS, "%s: qurt_thread_create failed with 0x%x", __func__, err);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  chan->msg          = p;
  chan->rpcmem_fd    = rpcmem_fd;
  chan->max_msg_size = max_msg_size;
  // wait until msg reciever thread is ready
  qurt_signal_wait_all(&(chan->msg_receiver_ready), 1);
  return 0;
}

int message_channel_destroy(struct MessageChannel *chan) {
  if (!message_channel_is_active(chan)) {
    return 0;
  }

  // signal message receiver thread to stop
  chan->msg_receiver_should_stop = true;

  int status;
  qurt_thread_join(chan->msg_receiver_thread, &status);
  qurt_signal_destroy(&(chan->msg_receiver_ready));
  HAP_mmap_put(chan->rpcmem_fd);

  message_channel_init(chan);
  return 0;
}

// FastRPC interface
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle) {
  // We may keep this function simple and leave real initialization somewhere else

  *handle = (remote_handle64) &dummy_handle;

  message_channel_init(&global_msg_chan);
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_close(remote_handle64 handle) {
  mmap_manager_release_all();
  message_channel_destroy(&global_msg_chan);
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
AEEResult htp_ops_create_channel(remote_handle64 handle, int32 fd, uint32 size) {
  if (message_channel_is_active(&global_msg_chan)) {
    return AEE_EALREADY;
  }

  return message_channel_create(&global_msg_chan, fd, size);
}

// FastRPC interface
AEEResult htp_ops_destroy_channel(remote_handle64 handle) {
  return message_channel_destroy(&global_msg_chan);
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

  const float *input      = (float *) (p1 + offset1);
  size_t       input_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
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
