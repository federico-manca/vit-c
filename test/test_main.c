#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "vit.h"

static int test_run_model_zero_input(void)
{
    float input_matrix[INPUT_DIM];
    int32_t out_net[OUT_CLS];
    int32_t out_ref[OUT_CLS];

    memset(input_matrix, 0, sizeof(input_matrix));
    memset(out_net, 0, sizeof(out_net));
    memset(out_ref, 0, sizeof(out_ref));

    run_model(input_matrix, out_net);
    run_model(input_matrix, out_ref);

    for (int i = 0; i < OUT_CLS; i++) {
        if (out_net[i] != out_ref[i]) {
            printf("FAIL: output mismatch at index %d: %d != %d\n",
                   i, out_net[i], out_ref[i]);
            return 0;
        }
    }

    printf("PASS: test_run_model_zero_input\n");
    return 1;
}

int main(void)
{
    int pass = 1;

    pass &= test_run_model_zero_input();

    if (pass) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("TESTS FAILED\n");
        return 1;
    }
}