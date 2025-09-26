# GEMM: d = alpha*a@b + beta*c
# Implement GEMM using nvmath-python
import nvmath
import cupy as cp
from cupyx.profiler import benchmark

# Define matrix dimensions m, n, k
m, n, k = 10_000_000, 40, 10

# Create random matrices
a = cp.random.rand(m, k, dtype=cp.float32)
b = cp.random.rand(k, n, dtype=cp.float32)
c = cp.random.rand(m, n, dtype=cp.float32)

alpha = 1.5
beta = 0.5

# Now benchmark with cupyx.profiler.benchmark()
print(benchmark(lambda: alpha * a @ b + beta * c, n_repeat=5, n_warmup=1))
print(benchmark(lambda: nvmath.linalg.advanced.matmul(a, b, c, alpha=alpha, beta=beta), n_repeat=5, n_warmup=1))
