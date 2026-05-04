#include "vit.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

static inline uint64_t diff_ns(struct timespec start, struct timespec end)
{
    return (uint64_t)(end.tv_sec - start.tv_sec) * 1000000000ull +
           (uint64_t)(end.tv_nsec - start.tv_nsec);
}

int main(void)
{
    static float input_matrix[IN_CHANNELS * HEIGHT * WIDTH];
    static int32_t out_net[OUT_CLS];

    /* Simple deterministic dummy input */
    for (int i = 0; i < IN_CHANNELS * HEIGHT * WIDTH; i++) {
        //input_matrix[i] = (float)(i % 17);
        input_matrix[i] = INPUT_MATRIX[i];
        //input_matrix[i] = 0;
    }

    

    const int warmup = 10;
    const int runs = 100;

    for (int i = 0; i < warmup; i++) {
        run_model(input_matrix, out_net);
    }

    uint64_t total_ns = 0;
    uint64_t min_ns = UINT64_MAX;
    uint64_t max_ns = 0;

    for (int i = 0; i < runs; i++) {
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        run_model(input_matrix, out_net);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        uint64_t dt = diff_ns(t0, t1);
        total_ns += dt;

        if (dt < min_ns) min_ns = dt;
        if (dt > max_ns) max_ns = dt;
    }

    double avg_ms = (double)total_ns / (double)runs / 1e6;
    double min_ms = (double)min_ns / 1e6;
    double max_ms = (double)max_ns / 1e6;

    printf("run_model timing over %d runs\n", runs);
    printf("avg: %.6f ms\n", avg_ms);
    printf("min: %.6f ms\n", min_ms);
    printf("max: %.6f ms\n", max_ms);

    printf("Sanity outputs:\n");
    for (int i = 0; i < OUT_CLS; i++) {
        printf("out_net[%d] = %d\n", i, out_net[i]);
    }

    return 0;
}