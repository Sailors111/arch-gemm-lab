#include "gemm.h"

/*
 * T0 基线：行优先、i-j-k 顺序、最内层对 k 归约、单线程。
 * 直接写 C（beta = 0）。
 */
void gemm_naive(const float *A, const float *B, float *C,
                int M, int N, int K, const gemm_params *p) {
    (void)p;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

/*
 * 双精度参考：i-k-j 顺序，便于在 K 较大时仍有可接受的运行时间。
 * 调用方必须先把 Cref 清零。
 */
void gemm_ref_double(const float *A, const float *B, double *Cref,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double a = (double)A[i * K + k];
            const float *brow = &B[k * N];
            double *crow = &Cref[i * N];
            for (int j = 0; j < N; ++j)
                crow[j] += a * (double)brow[j];
        }
    }
}
