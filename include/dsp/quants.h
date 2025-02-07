#pragma once

#include <stdint.h>

#define QK_K 256 // super-block size

#define QK4_0 32

typedef struct {
  __fp16  scales[8];
  uint8_t quants[8 * QK4_0/2];
} __attribute__((packed)) my_block_q4_0;
