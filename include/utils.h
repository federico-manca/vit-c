#ifndef UTILS_H
#define UTILS_h

#include <stdint.h>

int8_t clamp_int8(int32_t x);
/* Basic quant helpers */
float dequantize_int8_to_float(int32_t q, float scale, int32_t zp);
int8_t requantize_float_to_int8(float x_real, float scale, int32_t zp_out);
int8_t requantize_int32_to_int8(int32_t acc, int32_t mult, int32_t shift, int32_t zp_out);

/* Tensor utilities */
void copy_matrix(const int8_t *A, int8_t *B, int M, int N);
void conv_out_to_tokens(const int8_t *out_conv, int8_t *tokens, int Cout, int Hout, int Wout);

void add_cls_token(const int8_t *A, const int8_t *cls, int8_t *B, int M, int N);
void add_positonal_encoding(int8_t *A, const int8_t *pos_enc, int M, int N);

/* Float/int conversion helpers */
void quantize_to_float_input(
    const float *x_real,
    int8_t *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N);

void quantize_to_float(
    const float *x_real,
    int8_t *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M);

void dequantize_to_float(
    const int8_t *x,
    float *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M);

void dequantize_to_float_32(
    const int32_t *x,
    float *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M);
    

#endif