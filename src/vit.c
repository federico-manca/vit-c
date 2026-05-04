#include "vit.h"

#include <stdint.h>
#include <stdio.h>

/*
Vit Block:
    ---> NORM ---> MHA ---> + ---> NORM ---> MLP ---> + --->
   |-----------------------^ |-----------------------^
*/

void run_model(const float *input_matrix, int32_t *out_net)
{
    int8_t output_embed[(T + 1) * E] = {0};
    int8_t out_vit_0[(T + 1) * E] = {0};
    int8_t quant_matr[IN_CHANNELS * HEIGHT * WIDTH];

    /* Quantize floating-point input image to int8 */
    quantize_to_float_input(
        input_matrix,
        quant_matr,
        INPUT_SCALE,
        INPUT_ZERO_PT,
        IN_CHANNELS * HEIGHT * WIDTH);
    
    //Here I will have the input quantized to 8 bits
    //I now pass the quantized input to the embedding

    
    /* Patch embedding */
    patch_embed(
        quant_matr,
        output_embed,
        patch_kernels_0,
        patch_bias_0,
        cls_token_0,
        pos_enc_0,
        IN_CHANNELS,
        HEIGHT,
        WIDTH,
        E,
        PATCH,
        PATCH,
        0,
        0,
        PATCH,
        PATCH);

    // The output will be a (T+1)*(E) matrix, quantized to  8 bit with scale
    // equal to output_quant of the Conv
    // So, in the next module, the requant will quantize with out_quant_conv / in_quant_block
    

    /* One ViT block */
    vit_block(output_embed, out_vit_0, T + 1, E, BN_SCALE);


    /*
     * Final BN (model.norm) applied to ALL tokens before CLS extraction.
     * out_vit_0 is in res2_scale (= fc2_out_scale = HEAD_IN_SCALE) domain.
     * After BN in float, requantize back to HEAD_IN_SCALE for the classifier.
     */
    {
        float out_vit_float[(T + 1) * E];
        int8_t out_vit_bn[(T + 1) * E];

        //The tensor arrives with out_fc2_scale, so:
        //RES2_SCALE = out_fc2_scale
        for (int k = 0; k < (T + 1) * E; k++) {
            out_vit_float[k] = dequantize_int8_to_float(out_vit_0[k], RES2_SCALE, 0);
        }

        batchnorm(out_vit_float, rm_bf, rs_bf, scale_bf, bias_bf, T + 1, E);

        //quantized here with the classifier input scale
        for (int k = 0; k < (T + 1) * E; k++) {
            out_vit_bn[k] = requantize_float_to_int8(out_vit_float[k], HEAD_IN_SCALE, 0);
        }

        /* Classifier over CLS token (first row) */
        classifier(
            out_vit_bn,
            out_net,
            clas_weights_0,
            clas_bias_0,
            T + 1,
            E,
            OUT_CLS,
            HEAD_MULT,
            HEAD_SHIFT,
            HEAD_ZERO_PT);
    }
}

void vit_block(const int8_t *input, int8_t *output, int T_local, int E_local, const float IN_SCALE)
{
    int8_t x[T_local * E_local];
    int8_t attn_out[T_local * E_local];
    int8_t mlp_out[T_local * E_local];
    int8_t res1[T_local * E_local];
    int8_t res2[T_local * E_local];

    float x_batch[T_local * E_local];
    int8_t x_out_batch[T_local * E_local];

    /* Start from input */
    copy_matrix(input, x, T_local, E_local);

    /* ---------- First sublayer: Norm -> MHA -> Residual ---------- */

    copy_matrix(x, res1, T_local, E_local);

    //Here the input comes from the outside...so it will be quantized
    //with some scale. In our case, it comes from the PatchEmbedding 
    //layer, so it will be quantized with out_conv_quant scale. We need to
    //pass that value
    /* Dequantize x -> float for BN */
    for (int k = 0; k < T_local * E_local; k++) {
        x_batch[k] = dequantize_int8_to_float(x[k], IN_SCALE, 0);
    }

    /* BN in float */
    batchnorm(x_batch, rm_b0, rs_b0, scale_b0, bias_b0, T_local, E_local);

    /* Quantize BN output to the explicit MHA input domain */
    //Here, the scale will be equal to a certain quantized parameter
    //Attention because the Q, K, V, already quantized matrixes, will have
    //the necessity of being requantized
    for (int k = 0; k < T_local * E_local; k++) {
        x_out_batch[k] = requantize_float_to_int8(x_batch[k], BN0_OUT_SCALE, 0);
    }

    

    mha_int32(
        x_out_batch,
        w_q_0,
        w_k_0,
        w_v_0,
        bias_q_0,
        bias_k_0,
        bias_v_0,
        attn_out,
        head_0,
        bias_h_0,
        T_local,
        E_local,
        H,
        ATTN_SCALE);

    copy_matrix(attn_out, x, T_local, E_local);

    /*
     * Residual branch must be requantized to the same domain as the MHA output.
     * Assumption: RATIO is a fixed-point multiplier with implicit right shift by 16.
     */
    //Here the residual comes from the PatchEmbedding, quantized with out_conv_quant scale
    //The requant happens with factor:
    //RATIO = out_conv_quant / out_head_quant
    for (int k = 0; k < T_local * E_local; k++) {
        res1[k] = requantize_int32_to_int8((int32_t)res1[k], RATIO, 16, 0);
    }

    residual(x, res1, T_local, E_local);

    /* ---------- Second sublayer: Norm -> MLP -> Residual ---------- */

    copy_matrix(x, res2, T_local, E_local);

    /* Dequantize x -> float for BN */
    //The scale needs to be equal to out_head_quant
    //BN1_SCALE = OUT_HEAD_QUANT
    for (int k = 0; k < T_local * E_local; k++) {
        x_batch[k] = dequantize_int8_to_float(x[k], BN1_SCALE, 0);
    }

    /* BN in float */
    batchnorm(x_batch, rm_b1, rs_b1, scale_b1, bias_b1, T_local, E_local);

    /* Quantize BN output to MLP input domain */
    //Quantize with MLP_IN_SCALE
    for (int k = 0; k < T_local * E_local; k++) {
        x_out_batch[k] = requantize_float_to_int8(x_batch[k], MLP_SCALE, 0);
    }

    mlp(
        x_out_batch,
        w_1_0,
        w_2_0,
        bias_1_0,
        bias_2_0,
        mlp_out,
        T_local,
        E_local,
        MULT_fc1,
        SHIFT_fc1,
        ZP_OUT_fc1,
        MULT_fc2,
        SHIFT_fc2,
        ZP_OUT_fc2);

    //Matrix with MULT_FC2_OUT scale 
    copy_matrix(mlp_out, x, T_local, E_local);

    /*
     * Residual branch must be requantized to the same domain as the MLP output.
     * Assumption: RATIO1 is a fixed-point multiplier with implicit right shift by 16.
     */
    //The residual has scale of out_head_quant, so:
    //RATIO = out_head_quant / out_fc2_scale
    for (int k = 0; k < T_local * E_local; k++) {
        res2[k] = requantize_int32_to_int8((int32_t)res2[k], RATIO1, 16, 0);
    }

    residual(x, res2, T_local, E_local);

    /* Final output */
    copy_matrix(x, output, T_local, E_local);
}