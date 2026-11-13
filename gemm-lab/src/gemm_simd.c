#include <stddef.h>

#include "gemm.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/*
 * T5：8x8 寄存器分块 + AVX2/FMA 微内核。
 *
 * 对每个 8 行 x 8 列的 C 块，用 8 个 __m256 累加器常驻寄存器：
 *   - B 的一行 8 个 float 用一次 256 位 load；
 *   - A 的 8 个元素分别广播，做 8 次 FMA；
 *   - K 维循环结束后一次写回 C。
 * 这样每个 A 元素只取一次，每个 B 元素服务 8 行，C 只在最后写回。
 *
 * M、N 不是 8 的倍数时，主块之外的边角用标量 i-k-j 补全。
 * C 由调用方清零（beta = 0）。
 */
void gemm_simd(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p) {
    (void)p;
    if (M <= 0 || N <= 0 || K <= 0)
        return;

#if defined(__AVX2__) && defined(__FMA__)
    const int i8 = M & ~7;
    const int j8 = N & ~7;

    for (int i0 = 0; i0 < i8; i0 += 8) {
        for (int j0 = 0; j0 < j8; j0 += 8) {
            __m256 c0 = _mm256_loadu_ps(C + (size_t)(i0 + 0) * N + j0);
            __m256 c1 = _mm256_loadu_ps(C + (size_t)(i0 + 1) * N + j0);
            __m256 c2 = _mm256_loadu_ps(C + (size_t)(i0 + 2) * N + j0);
            __m256 c3 = _mm256_loadu_ps(C + (size_t)(i0 + 3) * N + j0);
            __m256 c4 = _mm256_loadu_ps(C + (size_t)(i0 + 4) * N + j0);
            __m256 c5 = _mm256_loadu_ps(C + (size_t)(i0 + 5) * N + j0);
            __m256 c6 = _mm256_loadu_ps(C + (size_t)(i0 + 6) * N + j0);
            __m256 c7 = _mm256_loadu_ps(C + (size_t)(i0 + 7) * N + j0);

            for (int k = 0; k < K; ++k) {
                const __m256 bv = _mm256_loadu_ps(B + (size_t)k * N + j0);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 0) * K + k]), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 1) * K + k]), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 2) * K + k]), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 3) * K + k]), bv, c3);
                c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 4) * K + k]), bv, c4);
                c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 5) * K + k]), bv, c5);
                c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 6) * K + k]), bv, c6);
                c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)(i0 + 7) * K + k]), bv, c7);
            }

            _mm256_storeu_ps(C + (size_t)(i0 + 0) * N + j0, c0);
            _mm256_storeu_ps(C + (size_t)(i0 + 1) * N + j0, c1);
            _mm256_storeu_ps(C + (size_t)(i0 + 2) * N + j0, c2);
            _mm256_storeu_ps(C + (size_t)(i0 + 3) * N + j0, c3);
            _mm256_storeu_ps(C + (size_t)(i0 + 4) * N + j0, c4);
            _mm256_storeu_ps(C + (size_t)(i0 + 5) * N + j0, c5);
            _mm256_storeu_ps(C + (size_t)(i0 + 6) * N + j0, c6);
            _mm256_storeu_ps(C + (size_t)(i0 + 7) * N + j0, c7);
        }
    }

    /* 右边条：i 在完整 8 行块内，j 落在 j8..N-1。 */
    for (int i = 0; i < i8; ++i) {
        const float *ai = A + (size_t)i * K;
        float *ci = C + (size_t)i * N;
        for (int k = 0; k < K; ++k) {
            const float a = ai[k];
            const float *bk = B + (size_t)k * N;
            for (int j = j8; j < N; ++j)
                ci[j] += a * bk[j];
        }
    }

    /* 底边：i 落在 i8..M-1，覆盖全部 j。 */
    for (int i = i8; i < M; ++i) {
        const float *ai = A + (size_t)i * K;
        float *ci = C + (size_t)i * N;
        for (int k = 0; k < K; ++k) {
            const float a = ai[k];
            const float *bk = B + (size_t)k * N;
            for (int j = 0; j < N; ++j)
                ci[j] += a * bk[j];
        }
    }
#else
    /* 无 AVX2/FMA 时退回 i-k-j，保证功能正确。 */
    gemm_ikj(A, B, C, M, N, K, p);
#endif
}
