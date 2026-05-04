#ifndef VIT_H
#define VIT_H

#include <stdint.h>

#include "vit_params.h"
#include "vit_kernels.h"
#include "utils.h"

void vit_block(const int8_t *input, int8_t *output, int T_local, int E_local, const float IN_SCALE);
void run_model(const float *input_matrix, int32_t *out_net);

#endif