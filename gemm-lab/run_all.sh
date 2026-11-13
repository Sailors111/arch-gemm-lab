#!/usr/bin/env bash
# 运行全部变体 × 全部矩阵规模，原始结果写入 data/raw.csv。
# 用法：./run_all.sh [输出文件]
set -eu

cd "$(dirname "$0")"
out="${1:-data/raw.csv}"
sizes=(128 256 512 1024 2048 3072 4096)
variants=(naive ikj block pack simd omp best)

make

mkdir -p data
"${PWD}/bench" --header > "$out"

for v in "${variants[@]}"; do
    for n in "${sizes[@]}"; do
        # 仅对组内并行变体开启多线程，其余固定单线程，保证对比公平。
        threads=1
        if [ "$v" = "omp" ] || [ "$v" = "best" ]; then
            threads="${OMP_THREADS:-4}"
        fi
        echo "==> variant=$v n=$n threads=$threads" >&2
        # 不传 --reps：由 bench 自适应（首次计时后补足到约 1 s，
        # 慢实现最少 2 次，快实现最多 30 次）。
        ./bench --variant "$v" --n "$n" --threads "$threads" >> "$out"
    done
done

echo "wrote $out"
