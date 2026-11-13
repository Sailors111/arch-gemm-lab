CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -std=gnu11 -Wall -Wextra
# 严格 c11 需额外加：-std=c11 -D_POSIX_C_SOURCE=200809L
# 若要限定纯 AVX2：把 -march=native 换成 -march=x86-64-v3
LDFLAGS = -fopenmp
LDLIBS  = -lm

# 可选 OpenBLAS 上界对比（需先安装 libopenblas-dev）：
#   make LDLIBS="-lm -lopenblas"

SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)
BIN = bench

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

src/%.o: src/%.c src/gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
