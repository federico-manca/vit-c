#ifndef VIT_KERNELS_H
#define VIT_KERNELS_H

#include <math.h>
#include <stdint.h>

#include "vit_params.h"
#include "utils.h"

/* Float ops */
void softmax_f32(float *x, int M, int N);
void scale_f32(float *x, int M, int N, const float scale);
void batchnorm(
    float *A,
    const float *rm,
    const float *rs,
    const float *scale,
    const float *bias,
    int M,
    int N);

/* Quant/int ops */
void residual(int8_t *A, const int8_t *residual_tensor, int M, int N);
void activation(int8_t *A, int M, int N, int s);

void av_head_float(
    float *A,
    float *V,
    int8_t *S,
    int T,
    int E,
    int H,
    int h,
    const float scale,
    const int ZP_OUT);

void matmul_int32(
    const int8_t *matrix_a,
    const int8_t *matrix_b,
    int8_t *matrix_c,
    const int32_t *bias,
    int M,
    int K,
    int N,
    int32_t MULT,
    int32_t SHIFT,
    int32_t ZERO,
    int t);

void qk_head_int32(
    int8_t *Q,
    int8_t *K,
    int32_t *S,
    int T,
    int E,
    int H,
    int h,
    const int MULT,
    const int SHIFT,
    const int ZP_OUT);

void av_head_int32(
    int8_t *A,
    int8_t *V,
    int8_t *S,
    int T,
    int E,
    int H,
    int h,
    const int MULT,
    const int SHIFT,
    const int ZP_OUT);

void mha_int32(
    int8_t *input,
    const int8_t *w_q,
    const int8_t *w_k,
    const int8_t *w_v,
    const int32_t *bias_q,
    const int32_t *bias_k,
    const int32_t *bias_v,
    int8_t *O,
    const int8_t *head,
    const int32_t *bias_h,
    int T,
    int E,
    int H,
    float scale);

void mlp(
    int8_t *A,
    const int8_t *w_1,
    const int8_t *w_2,
    const int32_t *bias_1,
    const int32_t *bias_2,
    int8_t *O,
    int T,
    int E,
    int MULT_fc1,
    int SHIFT_fc1,
    int ZP_OUT_fc1,
    int MULT_fc2,
    int SHIFT_fc2,
    int ZP_OUT_fc2);

/* Patch embedding path */
void conv2d(
    const int8_t *input,
    int32_t *output,
    const int8_t *kernels,
    const int32_t *bias,
    int Cin,
    int Hin,
    int Win,
    int Cout,
    int Sh,
    int Sw,
    int Ph,
    int Pw,
    int Kh,
    int Kw);



void patch_embed(
    const int8_t *input,
    int8_t *output,
    const int8_t *kernels,
    const int32_t *bias,
    const int8_t *cls_token,
    const int8_t *pos_enc,
    int Cin,
    int Hin,
    int Win,
    int Cout,
    int Sh,
    int Sw,
    int Ph,
    int Pw,
    int Kh,
    int Kw);

/* Classifier: uses only CLS token, returns int32 logits */
void classifier(
    const int8_t *input,
    int32_t *output,
    const int8_t *weights,
    const int32_t *bias,
    int M,
    int N,
    int L,
    int MULT,
    int SHIFT,
    int ZERO_PT);


#endif