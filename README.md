# arch-gemm-lab

《高级计算机体系结构》课程实验：使用 C 实现并优化 FP32 方阵乘法 `C = A × B`，研究循环重排、缓存分块、数据打包、AVX2/FMA 微内核和 OpenMP 多线程的性能影响。

项目包含 T0–T6 与 `best` 组合实现，使用统一基准程序检查正确性、测量时间和 GFLOPS，覆盖 n = 128–4096 的七个矩阵规模。

详细内容见 [实验设计](report/experiment.md)、[调研与实验报告](report/report.md) 和 [PPT 汇报文稿](report/ppt.md)。

## 工程结构

以下仅列入 Git 管理、随仓库上传的内容：

```text
arch-gemm-lab/
├── .gitignore
├── README.md
├── report/
│   ├── experiment.md      # 实验设计
│   ├── report.md          # 调研与实验报告
│   ├── ppt.md             # PPT 逐页内容
│   └── picture/           # 汇报示意图
└── gemm-lab/
    ├── src/               # 基准程序与各 GEMM 实现
    ├── data/.gitkeep      # 数据目录占位文件
    ├── figures/           # 已生成的实验结果图
    ├── Makefile           # 编译配置
    ├── run_all.sh         # 基础变体与规模扫描
    ├── merge_data.py      # 合并完整实验批次
    ├── check_final.py     # 校验汇总数据
    ├── plot_final.py      # 生成报告使用的五张图
    └── plot.py            # 早期基础扫描绘图脚本
```

原始 CSV、编译产物、绘图源稿和本地工具配置由 `.gitignore` 排除，不随仓库上传。

## 运行步骤

环境：Linux 或 WSL，安装 Git、GCC（支持 OpenMP）和 GNU Make；校验与绘图另需 Python 3，绘图还需 matplotlib。以下命令除克隆步骤外均在 `gemm-lab/` 中执行。

### 1. 编译与单次测试

```bash
git clone https://github.com/Sailors111/arch-gemm-lab.git
cd arch-gemm-lab/gemm-lab
make
./bench --variant naive --n 512 --reps 3
./bench --variant best --n 1024 --threads 4
./bench --help
```

已有仓库时直接进入 `gemm-lab/`，从 `make` 开始。默认编译选项为 `-O3 -march=native -fopenmp -std=gnu11 -Wall -Wextra`。

`bench` 将逐次 CSV 样本输出到标准输出，将计时摘要和误差检查输出到标准错误。默认热身 2 次；未指定 `--reps` 时自适应重复 2–30 次，报告中位时间。首次计时后与双精度参考比对，超阈值会输出警告。

### 2. 选择变体与参数

| 实验编号 | `--variant` | 实现 |
| --- | --- | --- |
| T0 | `naive` | 朴素 i-j-k |
| T1 | `ikj` | i-k-j 循环重排 |
| T2 / T3 | `block` | i/j 分块对照 / 三维分块 |
| T4 | `pack` | 分块与面板打包 |
| T5 | `simd` | 8×8 AVX2/FMA 微内核 |
| T6 | `omp` | OpenMP 行并行 |
| best | `best` | 分块、打包、微内核与列块并行 |

`--n` 指定方阵阶数，`--threads` 指定线程数（默认 1）；`--mb`、`--nb`、`--kb` 指定块尺寸（默认均为 64）。T2 设置 `--kb` 等于矩阵阶数，T3 使用三维分块，例如：

```bash
./bench --variant block --n 1024 --mb 64 --nb 64 --kb 1024
./bench --variant block --n 1024 --mb 64 --nb 64 --kb 64
```

### 3. 基础批量扫描（可选）

```bash
./run_all.sh data/scan.csv
```

扫描 7 个默认变体与 7 个矩阵规模，覆盖指定输出文件；大规模朴素实现耗时较长。T2、额外线程数和分块参数扫描需另行运行，此脚本不会生成完整报告所需的全部批次。

### 4. 重建报告图表（需已有原始数据）

将 `merge_data.py` 中 `SOURCES` 列出的 16 个原始 CSV 放入 `gemm-lab/data/` 后，再执行以下命令；基础扫描的单个 CSV 不能直接代替这些文件。

```bash
python3 merge_data.py &&
python3 check_final.py &&
python3 plot_final.py --outdir /tmp/arch-gemm-figures
```

校验通过后，五张图输出到 `/tmp/arch-gemm-figures/`。省略 `--outdir` 会更新仓库中的 `gemm-lab/figures/`；`plot.py` 是另一套基础绘图流程，不用于重建报告图表。
