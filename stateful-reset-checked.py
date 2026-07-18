import time

import nvmath
from nvmath.linalg.advanced import MatmulEpilog
import cupy as cp
from cupyx.profiler import benchmark

m, n, k = 12400, 1024, 16

repeat = 20
dtype = cp.float32

a = cp.random.rand(m, k, dtype=dtype)
b = cp.random.rand(k, n, dtype=dtype)
bias = cp.random.rand(m, dtype=dtype)


def matmul_batched_stateless(a, b, bias):
    return nvmath.linalg.advanced.matmul(a, b, epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias})


def matmul_batched_stateful_execute(mm, a, b, bias):
    mm.reset_operands(a=a, b=b, epilog_inputs={"bias": bias})
    return mm.execute()


print("Benchmarking stateless API...")
print(benchmark(lambda: matmul_batched_stateless(a, b, bias), n_repeat=repeat))
time.sleep(0.5)


with nvmath.linalg.advanced.Matmul(a, b) as mm:
    mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias})
    for i in range(1):
        print("Benchmarking stateful API execute (no autotuning)...")
        bm = benchmark(lambda: matmul_batched_stateful_execute(mm, a, b, bias), n_repeat=repeat)
        print(bm.gpu_times)
        print(bm)
        time.sleep(0.5)

with nvmath.linalg.advanced.Matmul(a, b) as mm:
    mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias})
    mm.autotune(iterations=2)
    time.sleep(0.5)
    for i in range(1):
        print("Benchmarking stateful API execute after autotuning...")
        bm = benchmark(lambda: matmul_batched_stateful_execute(mm, a, b, bias), n_repeat=repeat)
        print(bm.gpu_times)
        print(bm)
        time.sleep(0.5)


with nvmath.linalg.advanced.Matmul(a, b) as mm:
    mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias})
    print("Benchmarking autotuning timeg...")
    bm = benchmark(lambda: mm.autotune(iterations=2), n_repeat=repeat)
    print(bm.gpu_times)
    print(bm)
    time.sleep(0.5)
