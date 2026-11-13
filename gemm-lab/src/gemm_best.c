#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/*
 * 组合最优：分块 + 打包 + 8x8 AVX2 微内核 + OpenMP。
 *
 * 并行策略：按 j0 列块划分给各线程，各线程的 C 列块互不重叠；
 * 每个线程私有 B/A 打包缓冲区，避免竞争。
 *
 * 每个 j0：
 *   1. 把 B 的 K x nb 面板打包成连续内存 Bp；
 *   2. 对每个 i0，把 A 的 mb x K 面板打包成 Ap；
 *   3. 在 kb 子块上用 8x8 微内核计算，C 块常驻寄存器。
 *
 * 边界（mb/nb 不是 8 的倍数）用标量补全。C 由调用方清零（beta = 0）。
 */
void gemm_best(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p) {
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    int mb = (p && p->mb > 0) ? p->mb : 64;
    int nb = (p && p->nb > 0) ? p->nb : 64;
    int kb = (p && p->kb > 0) ? p->kb : 64;
    const int nt = (p && p->nthreads > 0) ? p->nthreads : 1;
    if (mb > M) mb = M;
    if (nb > N) nb = N;
    if (kb > K) kb = K;
#ifndef _OPENMP
    (void)nt;
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        float *Bp = malloc((size_t)K * nb * sizeof(float));
        float *Ap = malloc((size_t)mb * K * sizeof(float));
        if (Bp == NULL || Ap == NULL) {
            free(Bp);
            free(Ap);
            fprintf(stderr, "gemm_best: out of memory\n");
            exit(EXIT_FAILURE);
        }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int j0 = 0; j0 < N; j0 += nb) {
            const int nb_act = (j0 + nb <= N) ? nb : N - j0;

            /* 打包 B 面板：K x nb_act，行连续。 */
            for (int k = 0; k < K; ++k) {
                const float *src = B + (size_t)k * N + j0;
                float *dst = Bp + (size_t)k * nb_act;
                for (int j = 0; j < nb_act; ++j)
                    dst[j] = src[j];
            }

            for (int i0 = 0; i0 < M; i0 += mb) {
                const int mb_act = (i0 + mb <= M) ? mb : M - i0;

                /* 打包 A 面板：mb_act x K。 */
                for (int i = 0; i < mb_act; ++i)
                    memcpy(Ap + (size_t)i * K,
                           A + (size_t)(i0 + i) * K,
                           (size_t)K * sizeof(float));

                for (int k0 = 0; k0 < K; k0 += kb) {
                    const int kb_act = (k0 + kb <= K) ? kb : K - k0;

#if defined(__AVX2__) && defined(__FMA__)
                    for (int i = 0; i < mb_act; i += 8) {
                        for (int j = 0; j < nb_act; j += 8) {
                            if (i + 8 <= mb_act && j + 8 <= nb_act) {
                                __m256 c0 = _mm256_loadu_ps(C + (size_t)(i0 + i + 0) * N + j0 + j);
                                __m256 c1 = _mm256_loadu_ps(C + (size_t)(i0 + i + 1) * N + j0 + j);
                                __m256 c2 = _mm256_loadu_ps(C + (size_t)(i0 + i + 2) * N + j0 + j);
                                __m256 c3 = _mm256_loadu_ps(C + (size_t)(i0 + i + 3) * N + j0 + j);
                                __m256 c4 = _mm256_loadu_ps(C + (size_t)(i0 + i + 4) * N + j0 + j);
                                __m256 c5 = _mm256_loadu_ps(C + (size_t)(i0 + i + 5) * N + j0 + j);
                                __m256 c6 = _mm256_loadu_ps(C + (size_t)(i0 + i + 6) * N + j0 + j);
                                __m256 c7 = _mm256_loadu_ps(C + (size_t)(i0 + i + 7) * N + j0 + j);

                                for (int kk = 0; kk < kb_act; ++kk) {
                                    const __m256 bv = _mm256_loadu_ps(
                                        Bp + (size_t)(k0 + kk) * nb_act + j);
                                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 0) * K + k0 + kk]), bv, c0);
                                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 1) * K + k0 + kk]), bv, c1);
                                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 2) * K + k0 + kk]), bv, c2);
                                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 3) * K + k0 + kk]), bv, c3);
                                    c4 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 4) * K + k0 + kk]), bv, c4);
                                    c5 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 5) * K + k0 + kk]), bv, c5);
                                    c6 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 6) * K + k0 + kk]), bv, c6);
                                    c7 = _mm256_fmadd_ps(_mm256_set1_ps(Ap[(size_t)(i + 7) * K + k0 + kk]), bv, c7);
                                }

                                _mm256_storeu_ps(C + (size_t)(i0 + i + 0) * N + j0 + j, c0);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 1) * N + j0 + j, c1);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 2) * N + j0 + j, c2);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 3) * N + j0 + j, c3);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 4) * N + j0 + j, c4);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 5) * N + j0 + j, c5);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 6) * N + j0 + j, c6);
                                _mm256_storeu_ps(C + (size_t)(i0 + i + 7) * N + j0 + j, c7);
                            } else {
                                const int imax = (i + 8 <= mb_act) ? i + 8 : mb_act;
                                const int jmax = (j + 8 <= nb_act) ? j + 8 : nb_act;
                                for (int ii = i; ii < imax; ++ii) {
                                    for (int jj = j; jj < jmax; ++jj) {
                                        float acc = C[(size_t)(i0 + ii) * N + j0 + jj];
                                        for (int kk = 0; kk < kb_act; ++kk)
                                            acc += Ap[(size_t)ii * K + k0 + kk] *
                                                   Bp[(size_t)(k0 + kk) * nb_act + jj];
                                        C[(size_t)(i0 + ii) * N + j0 + jj] = acc;
                                    }
                                }
                            }
                        }
                    }
#else
                    /* 无 AVX2/FMA 时用打包后的标量 i-k-j。 */
                    for (int i = 0; i < mb_act; ++i) {
                        const float *ai = Ap + (size_t)i * K;
                        float *ci = C + (size_t)(i0 + i) * N + j0;
                        for (int k = k0; k < k0 + kb_act; ++k) {
                            const float a = ai[k];
                            const float *bk = Bp + (size_t)k * nb_act;
                            for (int j = 0; j < nb_act; ++j)
                                ci[j] += a * bk[j];
                        }
                    }
#endif
                }
            }
        }

        free(Bp);
        free(Ap);
    }
}
