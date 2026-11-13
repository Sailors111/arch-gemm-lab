#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"

/*
 * T4：数据打包。
 *
 * 对每个 j0 块，把 B 的 K x nb 面板打包成连续的 Bp[k*nb_act + j]；
 * 对每个 i0 块，把 A 的 mb x K 面板打包成连续的 Ap[i*K + k]；
 * 然后在打包后的面板上做 i-k-j 计算。打包消除了跨步访问，
 * 也让后续微内核可以按连续地址取数。
 *
 * 代价是额外的内存搬运，量级约为计算量的 1/nb，远小于 O(n^3)。
 * C 由调用方清零（beta = 0）。
 */
void gemm_pack(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p) {
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    int mb = (p && p->mb > 0) ? p->mb : 64;
    int nb = (p && p->nb > 0) ? p->nb : 64;
    int kb = (p && p->kb > 0) ? p->kb : 64;
    if (mb > M) mb = M;
    if (nb > N) nb = N;
    if (kb > K) kb = K;

    float *Bp = malloc((size_t)K * nb * sizeof(float));
    float *Ap = malloc((size_t)mb * K * sizeof(float));
    if (Bp == NULL || Ap == NULL) {
        free(Bp);
        free(Ap);
        /* 内存不足时退回分块实现，保证结果正确。 */
        gemm_block(A, B, C, M, N, K, p);
        return;
    }

    for (int j0 = 0; j0 < N; j0 += nb) {
        const int nb_act = (j0 + nb <= N) ? nb : N - j0;

        /* 打包 B 面板：K x nb_act。 */
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
            }
        }
    }

    free(Bp);
    free(Ap);
}
