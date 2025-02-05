#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vtcm_manager_setup();
void vtcm_manager_reset();

void *vtcm_manager_get_vtcm_base();

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  uint8_t *p = *vtcm_ptr;
  *vtcm_ptr += size;
  return p;
}

#ifdef __cplusplus
}
#endif
