#include <stddef.h>

#include "gemm.h"

/*
 * T1：i-k-j 循环重排。
 *
 * naive 的最内层是 k，对 B 的访问步长为 N（按列跨步）；这里把 k 提到 j 外面，
 * 最内层对 j 连续访问 B 的行和 C 的行，把跨步访问变成顺序访问。
 * C 由调用方清零（beta = 0）。
 */
void gemm_ikj(const float *A, const float *B, float *C,
              int M, int N, int K, const gemm_params *p) {
    (void)p;
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    for (int i = 0; i < M; ++i) {
        const float *ai = A + (size_t)i * K;
        float *ci = C + (size_t)i * N;
        for (int k = 0; k < K; ++k) {
            const float a = ai[k];
            const float *bk = B + (size_t)k * N;
            for (int j = 0; j < N; ++j)
                ci[j] += a * bk[j];
        }
    }
}
