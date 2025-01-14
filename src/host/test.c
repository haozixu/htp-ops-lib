#include <remote.h>
#include <stdio.h>

#include "host/session.h"

int main(int argc, char **argv) {
  int err = open_dsp_session(CDSP_DOMAIN_ID, 1);
  if (err != 0) {
    fprintf(stderr, "Open DSP session failed\n");
    return 1;
  }

  init_htp_backend();
  printf("init ok\n");

  close_dsp_session();
  return 0;
}
