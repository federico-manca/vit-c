#ifndef VIT_PARAMS_H
#define VIT_PARAMS_H

#include <stddef.h>
#include <stdint.h>

enum {
    IN_CHANNELS = 1,
    HEIGHT = 28,
    WIDTH = 28,
    INPUT_DIM = IN_CHANNELS * HEIGHT * WIDTH,
    PATCH = 4,
    SEQUENCE_LENGTH = (HEIGHT / PATCH) * (WIDTH / PATCH),
    EMBED_DIM = 64,
    T = SEQUENCE_LENGTH,
    E = EMBED_DIM,
    F = E * 4,
    H = 4,
    OUT_CLS = 10
};

extern const float INPUT_SCALE;
extern const int32_t INPUT_ZERO_PT;

extern const float BN_SCALE;
extern const float BN0_OUT_SCALE;
extern const float BN1_SCALE;
extern const float MLP_SCALE;
extern const float RES2_SCALE;

extern const float SCALE_x;
extern const int32_t ZERO_x;
extern const float SCALE_x1;
extern const int32_t ZERO_x1;

extern const float ATTN_SCALE;

extern const int32_t PATCH_MULT;
extern const int32_t PATCH_SHIFT;
extern const int32_t PATCH_ZERO_PT;

extern const int32_t MULT_q;
extern const int32_t SHIFT_q;
extern const int32_t ZP_OUT_q;

extern const int32_t MULT_k;
extern const int32_t SHIFT_k;
extern const int32_t ZP_OUT_k;

extern const int32_t MULT_vproj;
extern const int32_t SHIFT_vproj;
extern const int32_t ZP_OUT_vproj;

extern const int32_t RATIO;
extern const int32_t RATIO1;

extern const int32_t MULT_a;
extern const int32_t SHIFT_a;
extern const int32_t ZP_OUT_a;

extern const int32_t MULT_v;
extern const int32_t SHIFT_v;
extern const int32_t ZP_OUT_v;

extern const int32_t MULT_h;
extern const int32_t SHIFT_h;
extern const int32_t ZP_OUT_h;

extern const int32_t MULT_fc1;
extern const int32_t SHIFT_fc1;
extern const int32_t ZP_OUT_fc1;

extern const int32_t MULT_fc2;
extern const int32_t SHIFT_fc2;
extern const int32_t ZP_OUT_fc2;

extern const int32_t HEAD_MULT;
extern const int32_t HEAD_SHIFT;
extern const int32_t HEAD_ZERO_PT;
extern const float HEAD_IN_SCALE;

extern const float rm_b0[E];
extern const float rs_b0[E];
extern const float scale_b0[E];
extern const float bias_b0[E];

extern const float rm_b1[E];
extern const float rs_b1[E];
extern const float scale_b1[E];
extern const float bias_b1[E];

extern const float rm_bf[E];
extern const float rs_bf[E];
extern const float scale_bf[E];
extern const float bias_bf[E];

extern const int8_t w_q_0[E * E];
extern const int8_t w_k_0[E * E];
extern const int8_t w_v_0[E * E];
extern const int32_t bias_q_0[E];
extern const int32_t bias_k_0[E];
extern const int32_t bias_v_0[E];
extern const int8_t head_0[E * E];
extern const int32_t bias_h_0[E];

extern const int8_t w_1_0[E * F];
extern const int8_t w_2_0[F * E];
extern const int32_t bias_1_0[F];
extern const int32_t bias_2_0[E];

extern const int8_t patch_kernels_0[E * IN_CHANNELS * PATCH * PATCH];
extern const int32_t patch_bias_0[E];
extern const int8_t cls_token_0[E];
extern const int8_t pos_enc_0[(T + 1) * E];

extern const int8_t clas_weights_0[E * OUT_CLS];
extern const int32_t clas_bias_0[OUT_CLS];

#endif
