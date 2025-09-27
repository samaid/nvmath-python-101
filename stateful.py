import nvmath
from nvmath.linalg.advanced import MatmulEpilog
import cupy as cp
from cupyx.profiler import benchmark

m, n, k, batch_size = 124, 1024, 1512, 1024

a = cp.random.rand(batch_size, m, k, dtype=cp.float32)
b = cp.random.rand(batch_size, k, n, dtype=cp.float32)
d = cp.empty((batch_size, m, n), dtype=cp.float32)
bias = cp.random.rand(batch_size, m, dtype=cp.float32)


def matmul_batched_stateless(d, a, b, bias):
    for i in range(batch_size):
        d[i] = nvmath.linalg.advanced.matmul(
            a[i], b[i], epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias[i]}
        )
    return d


def matmul_batched_stateful_execute(mm, d, a, b, bias):
    mm.execute()
    for i in range(1, batch_size):
        mm.reset_operands(a=a[i], b=b[i], epilog_inputs={"bias": bias[i]})
        d[i] = mm.execute()
    return d


def matmul_batched_stateful(d, a, b, bias):
    with nvmath.linalg.advanced.Matmul(a[0], b[0]) as mm:
        mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias[0]})
        matmul_batched_stateful_execute(mm, d, a, b, bias)
    return d


def matmul_batched_stateful_autotuned(d, a, b, bias):
    with nvmath.linalg.advanced.Matmul(a[0], b[0]) as mm:
        mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias[0]})
        mm.autotune(iterations=5)
        matmul_batched_stateful_execute(mm, d, a, b, bias)


print("Benchmarking stateless API...")
print(benchmark(lambda: matmul_batched_stateless(d, a, b, bias), n_repeat=5))
print("Benchmarking stateful API...")
print(benchmark(lambda: matmul_batched_stateful(d, a, b, bias), n_repeat=5))
print("Benchmarking stateful API with autotuning...")
print(benchmark(lambda: matmul_batched_stateful_autotuned(d, a, b, bias), n_repeat=5))

with nvmath.linalg.advanced.Matmul(a[0], b[0]) as mm:
    mm.plan(epilog=MatmulEpilog(MatmulEpilog.RELU_BIAS), epilog_inputs={"bias": bias[0]})
    print("Benchmarking stateful API execute before autotuning...")
    print(benchmark(lambda: matmul_batched_stateful_execute(mm, d, a, b, bias), n_repeat=5))

    mm.autotune(iterations=5)
    print("Benchmarking stateful API execute after autotuning...")
    print(benchmark(lambda: matmul_batched_stateful_execute(mm, d, a, b, bias), n_repeat=5))
