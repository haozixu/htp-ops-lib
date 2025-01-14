#pragma once

int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();

void init_htp_backend();
