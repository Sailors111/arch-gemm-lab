#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gemm.h"

#define ALIGN 64
#define MAX_N 16384

typedef struct {
    const char *name;
    gemm_fn fn;
} variant_entry;

static const variant_entry VARIANTS[] = {
    {"naive", gemm_naive},
    {"ikj",   gemm_ikj},
    {"block", gemm_block},
    {"pack",  gemm_pack},
    {"simd",  gemm_simd},
    {"omp",   gemm_omp},
    {"best",  gemm_best},
};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *alloc_aligned(size_t bytes) {
    void *p = NULL;
    int rc = posix_memalign(&p, ALIGN, bytes);
    if (rc != 0) {
        fprintf(stderr, "posix_memalign(%zu) failed: %s\n", bytes, strerror(rc));
        exit(EXIT_FAILURE);
    }
    return p;
}

/* 确定性的伪随机初始化，取值 [-1, 1)。 */
static void init_mat(float *a, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        unsigned x = (unsigned)i * 2654435761u + seed * 2246822519u;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        a[i] = (float)(x % 2001u) / 1000.0f - 1.0f;
    }
}

static void verify(const float *C, const double *Cref, size_t n,
                   double *max_abs, double *max_rel) {
    double ma = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)C[i] - Cref[i]);
        double denom = fmax(1.0, fabs(Cref[i]));
        if (d > ma) ma = d;
        if (d / denom > mr) mr = d / denom;
    }
    *max_abs = ma;
    *max_rel = mr;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void print_header(void) {
    puts("variant,technique,M,N,K,threads,block_m,block_n,block_k,"
         "rep_index,time_s,gflops,max_abs_err,rel_err,"
         "cache_misses,llc_misses,dtlb_misses,cpu_mhz,notes");
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --variant NAME   naive|ikj|block|pack|simd|omp|best (default: naive)\n"
        "  --n N            square matrix size (default: 512)\n"
        "  --reps R         fixed timed repetitions (default: auto, see below)\n"
        "  --warmup W       untimed warmup calls (default: 2)\n"
        "  --target-sec T   auto mode: aim for T seconds of timed work (default: 1.0)\n"
        "  --max-reps M     auto mode: cap on repetitions (default: 30)\n"
        "  --threads T      OpenMP threads (default: 1)\n"
        "  --mb/--nb/--kb   block sizes (default: 64)\n"
        "  --no-check       skip double-precision reference and verification\n"
        "  --header         print CSV header and exit\n"
        "  --help           show this help\n"
        "\n"
        "auto reps: after the first timed call, repeat until target-sec is\n"
        "reached, with at least 3 reps (2 if one call already takes >= 2 s),\n"
        "capped by --max-reps.  Passing --reps disables auto mode.\n", prog);
}

static const variant_entry *find_variant(const char *name) {
    for (size_t i = 0; i < sizeof(VARIANTS) / sizeof(VARIANTS[0]); ++i)
        if (strcmp(VARIANTS[i].name, name) == 0)
            return &VARIANTS[i];
    return NULL;
}

int main(int argc, char **argv) {
    const char *vname = "naive";
    int n = 512, reps = 0, threads = 1;
    int warmup = 2, max_reps = 30;
    double target_sec = 1.0;
    int mb = 64, nb = 64, kb = 64;
    int no_check = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            vname = argv[++i];
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--target-sec") == 0 && i + 1 < argc) {
            target_sec = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-reps") == 0 && i + 1 < argc) {
            max_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mb") == 0 && i + 1 < argc) {
            mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            nb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kb") == 0 && i + 1 < argc) {
            kb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-check") == 0) {
            no_check = 1;
        } else if (strcmp(argv[i], "--header") == 0) {
            print_header();
            return 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    const variant_entry *v = find_variant(vname);
    if (v == NULL) {
        fprintf(stderr, "unknown variant: %s\n", vname);
        usage(argv[0]);
        return 2;
    }
    if (n <= 0 || n > MAX_N || reps < 0 || reps > 10000 ||
        warmup < 0 || warmup > 1000 || max_reps < 2 || max_reps > 10000 ||
        target_sec <= 0.0 || threads <= 0 ||
        mb <= 0 || nb <= 0 || kb <= 0) {
        fprintf(stderr, "invalid numeric argument\n");
        return 2;
    }

    const size_t M = (size_t)n, N = (size_t)n, K = (size_t)n;
    const size_t an = M * K, bn = K * N, cn = M * N;

    float *A = alloc_aligned(an * sizeof(float));
    float *B = alloc_aligned(bn * sizeof(float));
    float *C = alloc_aligned(cn * sizeof(float));
    double *Cref = no_check ? NULL : alloc_aligned(cn * sizeof(double));

    init_mat(A, an, 1u);
    init_mat(B, bn, 2u);

    if (!no_check) {
        fprintf(stderr, "computing double reference for n=%d ...\n", n);
        double t0 = now_sec();
        memset(Cref, 0, cn * sizeof(double));
        gemm_ref_double(A, B, Cref, n, n, n);
        fprintf(stderr, "reference done in %.3f s\n", now_sec() - t0);
    }

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    gemm_params p;
    p.mb = mb;
    p.nb = nb;
    p.kb = kb;
    p.nthreads = threads;

    /* 热身（不计时，不计正确性）。 */
    for (int w = 0; w < warmup; ++w) {
        memset(C, 0, cn * sizeof(float));
        v->fn(A, B, C, n, n, n, &p);
    }

    const double flops = 2.0 * (double)n * (double)n * (double)n;
    double max_abs = -1.0, max_rel = -1.0;

    const int cap = reps > 0 ? reps : max_reps;
    double *times = malloc((size_t)cap * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "malloc for timing samples failed\n");
        return EXIT_FAILURE;
    }

    int planned = reps > 0 ? reps : -1;  /* -1 表示首轮后自适应 */
    int r = 0;
    while (planned < 0 || r < planned) {
        /* beta = 0 语义：计时区外清零。 */
        memset(C, 0, cn * sizeof(float));

        double t0 = now_sec();
        v->fn(A, B, C, n, n, n, &p);
        double dt = now_sec() - t0;

        if (r == 0 && !no_check)
            verify(C, Cref, cn, &max_abs, &max_rel);

        printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.9f,%.6f,%.9g,%.9g,,,,\n",
               v->name, v->name, n, n, n, threads, mb, nb, kb,
               r, dt, flops / dt / 1e9, max_abs, max_rel);
        fflush(stdout);

        times[r] = dt;
        ++r;

        if (r == 1 && planned < 0) {
            /* 单次 >=2 s 时 2 次即可；否则至少 3 次并补足目标时长。 */
            int floor_reps = (dt >= 2.0) ? 2 : 3;
            int want;
            if (dt < 1e-9) {
                want = max_reps;
            } else {
                double need = target_sec / dt;
                want = (need >= (double)max_reps) ? max_reps
                                                  : (int)ceil(need);
            }
            if (want < floor_reps) want = floor_reps;
            if (want > max_reps) want = max_reps;
            planned = want;
            fprintf(stderr,
                    "auto reps: %d (first call %.6f s, target %.2f s)\n",
                    planned, dt, target_sec);
        }
    }

    qsort(times, (size_t)r, sizeof(double), cmp_double);
    double med = (r % 2 == 1) ? times[r / 2]
                              : 0.5 * (times[r / 2 - 1] + times[r / 2]);
    fprintf(stderr,
            "timed reps: %d, min %.6f s, median %.6f s, max %.6f s, "
            "median %.3f GFLOPS\n",
            r, times[0], med, times[r - 1], flops / med / 1e9);
    if (!no_check) {
        double limit = (double)n * (double)FLT_EPSILON;
        fprintf(stderr, "correctness: rel_err %.3e %s K*eps %.3e\n",
                max_rel, max_rel <= limit ? "<=" : ">", limit);
        if (max_rel > limit)
            fprintf(stderr, "WARNING: rel_err exceeds K*epsilon threshold\n");
    }
    free(times);

    /* 防止死代码消除，同时给出一个可核对的结果摘要。 */
    volatile double sink = 0.0;
    for (size_t i = 0; i < cn; ++i)
        sink += (double)C[i];
    fprintf(stderr, "checksum=%.6f\n", (double)sink);

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}
