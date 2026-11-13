#ifndef GEMM_H
#define GEMM_H

/*
 * 统一接口与变体声明。
 *
 * 所有变体采用 beta = 0 语义：C 的内容被本次计算覆盖。
 * 累加型变体（ikj / block / pack / simd）在实现上使用 C += ...，
 * 因此调用方必须在每次调用前把 C 清零（基准程序负责，且不计入计时）。
 */

typedef struct {
    int mb;        /* M 方向分块 */
    int nb;        /* N 方向分块 */
    int kb;        /* K 方向分块 */
    int nthreads;  /* OpenMP 线程数 */
} gemm_params;

typedef void (*gemm_fn)(const float *A, const float *B, float *C,
                        int M, int N, int K, const gemm_params *p);

void gemm_naive(const float *A, const float *B, float *C,
                int M, int N, int K, const gemm_params *p);
void gemm_ikj(const float *A, const float *B, float *C,
              int M, int N, int K, const gemm_params *p);
void gemm_block(const float *A, const float *B, float *C,
                int M, int N, int K, const gemm_params *p);
void gemm_pack(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p);
void gemm_simd(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p);
void gemm_omp(const float *A, const float *B, float *C,
              int M, int N, int K, const gemm_params *p);
void gemm_best(const float *A, const float *B, float *C,
               int M, int N, int K, const gemm_params *p);

/* 双精度参考实现；Cref 需由调用方清零，大小为 M*N 个 double。 */
void gemm_ref_double(const float *A, const float *B, double *Cref,
                     int M, int N, int K);

#endif /* GEMM_H */
