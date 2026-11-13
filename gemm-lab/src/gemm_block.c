#include <stddef.h>

#include "gemm.h"

/*
 * T2/T3：缓存分块。
 *
 * 三层循环按 mb / nb / kb 分块，块内仍是 i-k-j：
 *   - C 的 mb x nb 块在 k 方向反复使用，尽量留在 L2；
 *   - B 的 kb x nb 块被 mb 行复用；
 *   - A 的 mb x kb 块被 nb 列复用。
 * 边界用 min 截断，支持任意 M/N/K。C 由调用方清零（beta = 0）。
 */
void gemm_block(const float *A, const float *B, float *C,
                int M, int N, int K, const gemm_params *p) {
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    int mb = (p && p->mb > 0) ? p->mb : 64;
    int nb = (p && p->nb > 0) ? p->nb : 64;
    int kb = (p && p->kb > 0) ? p->kb : 64;
    if (mb > M) mb = M;
    if (nb > N) nb = N;
    if (kb > K) kb = K;

    for (int i0 = 0; i0 < M; i0 += mb) {
        const int i1 = (i0 + mb <= M) ? i0 + mb : M;
        for (int j0 = 0; j0 < N; j0 += nb) {
            const int j1 = (j0 + nb <= N) ? j0 + nb : N;
            for (int k0 = 0; k0 < K; k0 += kb) {
                const int k1 = (k0 + kb <= K) ? k0 + kb : K;
                for (int i = i0; i < i1; ++i) {
                    const float *ai = A + (size_t)i * K;
                    float *ci = C + (size_t)i * N;
                    for (int k = k0; k < k1; ++k) {
                        const float a = ai[k];
                        const float *bk = B + (size_t)k * N;
                        for (int j = j0; j < j1; ++j)
                            ci[j] += a * bk[j];
                    }
                }
            }
        }
    }
}
