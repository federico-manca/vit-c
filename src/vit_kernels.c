#include "vit_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

/*
    Utils
*/



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
    int t)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = (bias != NULL) ? bias[n] : 0;

            for (int k = 0; k < K; k++) {
                sum += (int32_t)matrix_a[m * K + k] * (int32_t)matrix_b[k * N + n];
            }
            
            /*
                Here it happens the multiplication between the output of the BatchNorm,
                quantized to 8 bits with a scale bn_quant, and the quantized matrix X of the weights, 
                with scale x_quant. The outcome is basically:

                y = bn_quant*x_quant*(x_real1 * x_real2)

                So when I quantize back to 8 bits, I will do:

                x_q = x * (bn_quant*x_quant)/(out_matmul_quant)

                So here MULT = MULT_BN * MULT_X / MULT_OUT
            */
            int8_t res = requantize_int32_to_int8(sum, MULT, SHIFT, ZERO);

            if (t == 0) {
                matrix_c[m * N + n] = res;
            } else {
                matrix_c[n * M + m] = res;
            }
        }
    }
}

void softmax_f32(float *x, int M, int N)
{
    for (int m = 0; m < M; m++) {
        float row_max = x[m * N];

        for (int n = 1; n < N; n++) {
            float v = x[m * N + n];
            if (v > row_max) {
                row_max = v;
            }
        }

        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            x[m * N + n] = expf(x[m * N + n] - row_max);
            sum += x[m * N + n];
        }

        for (int n = 0; n < N; n++) {
            x[m * N + n] /= sum;
        }
    }
}

void scale_f32(float *x, int M, int N, const float scale)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            x[m * N + n] /= scale;
        }
    }
}

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
    const int ZP_OUT)
{
    const int D = E / H;
    const int offset = h * D;

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int32_t sum = 0;
            for (int d = 0; d < D; d++) {
                sum += (int32_t)Q[i * E + offset + d] * (int32_t)K[j * E + offset + d];
            }
            S[i * T + j] = sum;//requantize_int32_to_int8(sum, MULT, SHIFT, ZP_OUT);
        }
    }
}

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
    const int ZP_OUT)
{
    const int D = E / H;
    const int offset = h * D;

    for (int i = 0; i < T; i++) {
        for (int d = 0; d < D; d++) {
            int32_t sum = 0;
            for (int j = 0; j < T; j++) {
                sum += (int32_t)A[i * T + j] * (int32_t)V[j * E + offset + d];
            }
            S[i * E + offset + d] = requantize_int32_to_int8(sum, MULT, SHIFT, ZP_OUT);
        }
    }
}

void av_head_float(
    float *A,
    float *V,
    int8_t *S,
    int T,
    int E,
    int H,
    int h,
    const float scale,
    const int ZP_OUT)
{
    const int D = E / H;
    const int offset = h * D;

    for (int i = 0; i < T; i++) {
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int j = 0; j < T; j++) {
                sum += A[i * T + j] * V[j * E + offset + d];
            }
            S[i * E + offset + d] = requantize_float_to_int8(sum, scale, ZP_OUT);
        }
    }
}

void activation(int8_t *A, int M, int N, int s)
{
    if (s == 0) {
        /* ReLU */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i * N + j;
                A[idx] = (A[idx] > 0) ? A[idx] : 0;
            }
        }
    } else {
        printf("ERROR: activation not supported.\n");
    }
}

void batchnorm(
    float *A,
    const float *rm,
    const float *rs,
    const float *scale,
    const float *bias,
    int M,
    int N)
{
    /*
     * A is [M, N]
     * rm, rs, scale, bias are per-feature vectors of length N
     *
     * out = ((x - mean) * rs * scale) + bias
     * where rs is typically 1 / sqrt(var + eps)
     */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            A[idx] = (((A[idx] - rm[j]) * rs[j] * scale[j]) + bias[j]);
        }
    }
}

void residual(int8_t *A, const int8_t *residual_tensor, int M, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            int32_t s = (int32_t)A[idx] + (int32_t)residual_tensor[idx];
            A[idx] = clamp_int8(s);
        }
    }
}

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
    float scale)
{
    int8_t Q[T * E];
    int8_t K[T * E];
    int8_t V[T * E];

    float V_f[T*E];

    int8_t S[T * E];


    /*
     * NOTE:
     * Q/K/V projection
     */

    //The MULT_x will be equal to MULT_BN * MULT_X / MULT_OUT_x
    matmul_int32(input, w_q, Q, bias_q, T, E, E, MULT_q,     SHIFT_q,     ZP_OUT_q,     0);
    matmul_int32(input, w_k, K, bias_k, T, E, E, MULT_k,     SHIFT_k,     ZP_OUT_k,     0);
    matmul_int32(input, w_v, V, bias_v, T, E, E, MULT_vproj, SHIFT_vproj, ZP_OUT_vproj, 0);

    for(int k=0; k < T*E; k++){
        V_f[k] = dequantize_int8_to_float(V[k], V_OUT_SCALE, 0);
    }

    //for(int k=0; k < T*E; k++){
    //    printf("Q Matrix %f\n", (float)Q[k]*0.0361f) ;
    //}

    for (int h = 0; h < H; h++) {
        /* Each head needs its own A buffer so QK and AV operate on the same data */
        int32_t A_h[T * T];
        //int8_t A_h8[T*T];
        //float A_h8[T*T];
        float  A_float_h[T * T];

        //Here the MatMul between QK happens
        //In brevitas, this operation happens between the unquantized version
        //of the tensors: here we aproximate and perform the operations of 
        //matmul with the quantized versions, then converting to float when 
        //necessary

        //Here the qk_head will produce a int32_t Matrix with this value:
        //y = out_quant_q * out_quant_k * (x_realq * x_realk)
        //to dequantize, we need the out_quant_q * out_quant_k scale
        //So, MULT_a is unused, and SCALE_x = out_quant_q * out_quant_k
        qk_head_int32(Q, K, A_h, T, E, H, h, MULT_a, SHIFT_a, ZP_OUT_a);

        dequantize_to_float_32(A_h, A_float_h, SCALE_x, ZERO_x, T, T);

        //scale = 1 / rad(dk), precomputed
        scale_f32(A_float_h, T, T, scale);
        //softma xon float values
        softmax_f32(A_float_h, T, T);
        //In Brevitas the a_softmax @ v is performed in floating point.
        //Again, we could perform it in a quantized way here. Maybe this one
        //we will do it in floating point. To do that, I need to dequantize
        //the V matrix, which has out_v_quant as scale.
        //The output is quantized with the in_quant_head scale
        av_head_float(A_float_h, V_f, S, T, E, H, h, HEAD_IN_SCALE, ZP_OUT_v);


        //quantize_to_float(A_float_h, A_h8, SCALE_x1, ZERO_x1, T, T);

        //av_head_int32(A_h8, V, S, T, E, H, h, MULT_v, SHIFT_v, ZP_OUT_v);
    }

    //Here happens the Linear between in_quant_head and weight_quant_head
    //The requant then will happen with factor:
    // in_quant_head * weight_quant_head / out_quant = 
    // = acc_quant_head / out_quant_head
    //MULT_h = MULT_ACC_H / MULT_OUT_H
    matmul_int32(S, head, O, bias_h, T, E, E, MULT_h, SHIFT_h, ZP_OUT_h, 0);
}

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
    int ZP_OUT_fc2)
{
    const int F = 4 * E;
    int8_t FC1[T * F];

    //Here the input arrives already quantized with MLP_IN_SCALE
    //The MULT_fc1 = MULT_FC1_ACC / MULT_FC1_OUT
    matmul_int32(A, w_1, FC1, bias_1, T, E, F, MULT_fc1, SHIFT_fc1, ZP_OUT_fc1, 0);
    activation(FC1, T, F, 0);
    //Again, this module will expect a quantized tensor with MULT_FC1_OUT scale
    //The output is again MULT_fc2 = MULT_FC2_ACC / MULT_FC2_OUT
    matmul_int32(FC1, w_2, O, bias_2, T, F, E, MULT_fc2, SHIFT_fc2, ZP_OUT_fc2, 0);
}

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
    int Kw)
{
    const int Hout = (Hin + 2 * Ph - Kh) / Sh + 1;
    const int Wout = (Win + 2 * Pw - Kw) / Sw + 1;

    for (int co = 0; co < Cout; co++) {
        for (int oh = 0; oh < Hout; oh++) {
            for (int ow = 0; ow < Wout; ow++) {
                int32_t sum = (bias != NULL) ? bias[co] : 0;

                for (int ci = 0; ci < Cin; ci++) {
                    for (int kh = 0; kh < Kh; kh++) {
                        for (int kw = 0; kw < Kw; kw++) {
                            int ih = oh * Sh - Ph + kh;
                            int iw = ow * Sw - Pw + kw;

                            if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                                sum +=
                                    (int32_t)input[(ci * Hin + ih) * Win + iw] *
                                    (int32_t)kernels[((co * Cin + ci) * Kh + kh) * Kw + kw];
                            }
                        }
                    }
                }

                output[(co * Hout + oh) * Wout + ow] = sum;
            }
        }
    }
}



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
    int Kw)
{
    const int Hout = (Hin + 2 * Ph - Kh) / Sh + 1;
    const int Wout = (Win + 2 * Pw - Kw) / Sw + 1;
    const int Tlocal = Hout * Wout;

    int32_t out_conv_acc[Cout * Hout * Wout];
    int8_t out_conv_q[Cout * Hout * Wout];
    int8_t tokens[Tlocal * Cout];

   
    conv2d(input, out_conv_acc, kernels, bias, Cin, Hin, Win, Cout, Sh, Sw, Ph, Pw, Kh, Kw);


    /*
     * Requantize conv accumulators to int8.
     */

    // Patch Mult = accumulation_scale / output_scale = in_scale * w_scale / out_scale

    //1.84764*10^-5 / 0.0013743407325819135 = 
    for (int k = 0; k < Cout * Hout * Wout; k++) {
        out_conv_q[k] = requantize_int32_to_int8(out_conv_acc[k], PATCH_MULT, PATCH_SHIFT, PATCH_ZERO_PT);
    }

    

    /*
     * out_conv_q, cls_token and pos_enc must all live in the same quant domain
     * for concatenation/addition to be semantically correct.
     * That's why cls_token and pos_enc are already quantized with the output_quant
    * of the QuantConv
     */
    conv_out_to_tokens(out_conv_q, tokens, Cout, Hout, Wout);
    add_cls_token(tokens, cls_token, output, Tlocal, Cout);
    add_positonal_encoding(output, pos_enc, Tlocal + 1, Cout);

}

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
    int ZERO_PT)
{
    (void)M;
    (void)MULT;
    (void)SHIFT;
    (void)ZERO_PT;

    /*
     * Use CLS token only, i.e. the first row of shape [M, N].
     * Output stays int32 logits/accumulators.
     */
    const int8_t *cls_vec = input;
    

    for (int l = 0; l < L; l++) {
        int32_t sum = (bias != NULL) ? bias[l] : 0;
        for (int j = 0; j < N; j++) {
            sum += (int32_t)cls_vec[j] * (int32_t)weights[j * L + l];
        }
        output[l] = sum;
    }
}

