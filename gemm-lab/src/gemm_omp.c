#include <stddef.h>

#include "gemm.h"

/*
 * T6：OpenMP 行块并行。
 *
 * 按行划分任务：每个线程负责若干完整的 C 行，行内做 i-k-j。
 * 不同线程写不同的 C 行，没有数据竞争，也不需要同步或归约。
 * 线程数由 gemm_params.nthreads 指定；未以 OpenMP 编译时退化为串行。
 * C 由调用方清零（beta = 0）。
 */
void gemm_omp(const float *A, const float *B, float *C,
              int M, int N, int K, const gemm_params *p) {
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    const int nt = (p && p->nthreads > 0) ? p->nthreads : 1;
#ifndef _OPENMP
    (void)nt;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
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
