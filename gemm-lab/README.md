# gemm-lab

课程实验代码：基于体系结构技术的 GEMM 性能优化。实验方案、测量协议和结果分析见
[report/report.md](../report/report.md)。

## 构建

```bash
make                  # 默认 -O3 -march=native -fopenmp -std=gnu11
make clean
```

## 运行

```bash
./bench --variant naive --n 512 --reps 3
./bench --header > data/raw.csv
./bench --help
```

常用参数：`--variant`、`--n`、`--reps`、`--warmup`、`--target-sec`、
`--max-reps`、`--threads`、`--mb/--nb/--kb`、`--no-check`。CSV 直接写 stdout，
进度与校验和写 stderr。`data/` 下的 CSV 为本地实验数据，不随仓库上传。

不传 `--reps` 时自动决定重复次数：首次计时后补足到 `--target-sec`（默认 1 s），
最少 3 次（单次已 ≥2 s 时最少 2 次），上限 `--max-reps`（默认 30）。

## 全量扫描

```bash
./run_all.sh              # 结果写入 data/raw.csv
python3 plot.py           # 生成 figures/*.png
```

## 变体

| 名称 | 文件 | 技术 |
| --- | --- | --- |
| naive | `src/gemm_naive.c` | i-j-k 基线 |
| ikj | `src/gemm_loop.c` | 循环重排 |
| block | `src/gemm_block.c` | 缓存分块 |
| pack | `src/gemm_pack.c` | 数据打包 |
| simd | `src/gemm_simd.c` | 寄存器分块 + AVX2/FMA |
| omp | `src/gemm_omp.c` | OpenMP 并行 |
| best | `src/gemm_best.c` | 组合最优 |

所有变体均已实现：T1 i-k-j 重排、T2/T3 缓存分块、T4 数据打包、
T5 AVX2 8x8 微内核、T6 OpenMP 行并行；`best` 组合分块 + 打包 + SIMD + OpenMP。
没有 AVX2/FMA 时 T5 与 `best` 自动退回标量实现。

## 正确性约定

- 所有变体为 beta = 0 语义，调用前由 `bench` 清零 C，且不计入计时。
- 每个变体首次重复后与双精度参考比对，输出 `max_abs_err` / `rel_err`。
- 阈值见 [report/report.md](../report/report.md) 第 8 节；未通过不得作为性能结果上报。
