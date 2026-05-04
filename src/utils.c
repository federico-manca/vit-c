#include "utils.h"




int8_t clamp_int8(int32_t x)
{
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

float dequantize_int8_to_float(int32_t q, float scale, int32_t zp)
{
    return scale * (float)(q - zp);
}

int8_t requantize_float_to_int8(float x_real, float scale, int32_t zp_out)
{
    // Scale division
    float qf = x_real / scale;
    int32_t q;

    //Rounding
    if (qf >= 0.0f) {
        q = (int32_t)(qf + 0.5f);
    } else {
        q = (int32_t)(qf - 0.5f);
    }

    q += zp_out;

    //Clamping
    return clamp_int8(q);
}

/*
 * Here 'shift' is the number of RIGHT-SHIFT BITS, not a divisor value.
 * Example:
 *   real_multiplier ≈ mult / 2^shift
 */

 /*
    What this code should do is take a quantized representation and convert it
    to another quantized representation, without passing through the float conversion:

    x_q = clamp(round(x_real / scale) + zero_pt)

    x_q --> x_real --> x_q1

    From x_q to x_real we have:

    x_real = scale* (x_q-zero_pt)

    Then, we convert x_real to x_q1
    -----> some passage have wrong sign but the outcome is the same basically
    x_q1 = clamp(round(x_real/scale1) - zerp_pt1) =

    x_q1 = ... (x_q * scale + zero_pt)/scale1 - zero_pt1 = (if zero_pt == 0)
         = ... (x_q * scale / scale1)

    Now, if the scales are expressed as MULT / 2^SHIFT, we have:

    x_q * (MULT / 2^SHIFT) * (2^SHIFT / MULT1) =
    = x_q * MULT / MULT1 --> mult_ratio = MULT / MULT1
    */
int8_t requantize_int32_to_int8(int32_t acc, int32_t mult_ratio, int32_t shift, int32_t zp_out)
{
    
    int64_t x = (int64_t)acc * (int64_t)mult_ratio;
    int32_t y;

    if (shift > 0) {
        int64_t rnd = (int64_t)1 << (shift - 1);
        if (x >= 0) {
            y = (int32_t)((x + rnd) >> shift);
        } else {
            y = (int32_t)((x - rnd) >> shift);
        }
    } else {
        y = (int32_t)x;
    }

    y += zp_out;
    return clamp_int8(y);
}

/* Raw accumulator helper for classifier or debug paths */
static inline int32_t dot_accum_int32(
    const int8_t *a,
    const int8_t *b,
    int K)
{
    int32_t sum = 0;
    for (int k = 0; k < K; k++) {
        sum += (int32_t)a[k] * (int32_t)b[k];
    }
    return sum;
}

/*
    MatMul kernel
*/

void copy_matrix(const int8_t *A, int8_t *B, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = A[i * N + j];
        }
    }
}






void quantize_to_float_input(
    const float *x_real,
    int8_t *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N)
{
    for (int i = 0; i < N; i++) {
        out_matrix[i] = requantize_float_to_int8(x_real[i], scale, zp_out);
    }
}

void quantize_to_float(
    const float *x_real,
    int8_t *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out_matrix[i * M + j] = requantize_float_to_int8(x_real[i * M + j], scale, zp_out);
        }
    }
}


void dequantize_to_float(
    const int8_t *x,
    float *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out_matrix[i * M + j] = dequantize_int8_to_float(x[i * M + j], scale, zp_out);
        }
    }
}

void dequantize_to_float_32(
    const int32_t *x,
    float *out_matrix,
    const float scale,
    int32_t zp_out,
    const int N,
    const int M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            out_matrix[i * M + j] = dequantize_int8_to_float(x[i * M + j], scale, zp_out);
        }
    }
}










void add_cls_token(const int8_t *A, const int8_t *cls, int8_t *B, int M, int N)
{
    for (int j = 0; j < N; j++) {
        B[j] = cls[j];
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[(i + 1) * N + j] = A[i * N + j];
        }
    }
}

void add_positonal_encoding(int8_t *A, const int8_t *pos_enc, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            A[idx] = clamp_int8((int32_t)A[idx] + (int32_t)pos_enc[idx]);
        }
    }
}

/*
Patch Embedding:
- Compute projection
- Convert conv output map -> token matrix
- Add CLS token
- Add positional encoding
*/


void conv_out_to_tokens(const int8_t *out_conv, int8_t *tokens, int Cout, int Hout, int Wout)
{
    for (int oh = 0; oh < Hout; oh++) {
        for (int ow = 0; ow < Wout; ow++) {
            int t = oh * Wout + ow;
            for (int co = 0; co < Cout; co++) {
                tokens[t * Cout + co] = out_conv[(co * Hout + oh) * Wout + ow];
            }
        }
    }
}