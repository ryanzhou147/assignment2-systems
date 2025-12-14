"""
Benchmark PyTorch attention implementation at different scales.

This script benchmarks attention with:
- Fixed batch size of 8
- No multihead attention (no head dimension)
- d_model (head embedding dimension) in [16, 32, 64, 128]
- Sequence length in [256, 1024, 4096, 8192, 16384]
- 100 forward passes and 100 backward passes
- Memory measurement before backward pass
"""

import torch
import itertools
import timeit
from cs336_basics.utility import scaled_dot_product_attention


def benchmark_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    warmup_steps: int = 10,
    benchmark_steps: int = 100,
) -> dict:
    """
    Benchmark attention forward and backward passes.

    Args:
        batch_size: Batch size (fixed to 8)
        seq_len: Sequence length
        d_model: Head embedding dimension
        device: CUDA device
        warmup_steps: Number of warmup iterations
        benchmark_steps: Number of benchmark iterations

    Returns:
        Dictionary with timing and memory results
    """
    results = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "forward_time": None,
        "backward_time": None,
        "memory_before_backward_gb": None,
        "peak_memory_gb": None,
        "oom": False,
        "error": None,
    }

    try:
        # Clear cache before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create random inputs Q, K, V
        # Shape: (batch_size, seq_len, d_model) - no head dimension
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = mask.unsqueeze(0)  # Add batch dimension: (1, seq_len, seq_len)

        # Warmup for forward pass
        for _ in range(warmup_steps):
            output = scaled_dot_product_attention(Q, K, V, mask)
            torch.cuda.synchronize()

        # Benchmark forward pass
        forward_times = []
        for _ in range(benchmark_steps):
            # Need fresh tensors with gradients for each iteration
            Q_iter = Q.detach().clone().requires_grad_(True)
            K_iter = K.detach().clone().requires_grad_(True)
            V_iter = V.detach().clone().requires_grad_(True)

            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            output = scaled_dot_product_attention(Q_iter, K_iter, V_iter, mask)

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            forward_times.append(end_time - start_time)

        avg_forward_time = sum(forward_times) / benchmark_steps
        results["forward_time"] = avg_forward_time

        # Measure memory before backward pass
        # First do a forward pass and measure memory
        Q_mem = Q.detach().clone().requires_grad_(True)
        K_mem = K.detach().clone().requires_grad_(True)
        V_mem = V.detach().clone().requires_grad_(True)

        torch.cuda.reset_peak_memory_stats()
        output = scaled_dot_product_attention(Q_mem, K_mem, V_mem, mask)
        torch.cuda.synchronize()

        memory_before_backward = torch.cuda.memory_allocated() / (1024 ** 3)
        results["memory_before_backward_gb"] = memory_before_backward

        # Warmup for backward pass
        for _ in range(warmup_steps):
            Q_warm = Q.detach().clone().requires_grad_(True)
            K_warm = K.detach().clone().requires_grad_(True)
            V_warm = V.detach().clone().requires_grad_(True)

            output = scaled_dot_product_attention(Q_warm, K_warm, V_warm, mask)
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()

        # Benchmark backward pass
        backward_times = []
        for _ in range(benchmark_steps):
            Q_iter = Q.detach().clone().requires_grad_(True)
            K_iter = K.detach().clone().requires_grad_(True)
            V_iter = V.detach().clone().requires_grad_(True)

            # Forward pass (not timed)
            output = scaled_dot_product_attention(Q_iter, K_iter, V_iter, mask)
            loss = output.sum()

            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            loss.backward()

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            backward_times.append(end_time - start_time)

        avg_backward_time = sum(backward_times) / benchmark_steps
        results["backward_time"] = avg_backward_time

        # Record peak memory
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results["oom"] = True
            results["error"] = "OOM"
        else:
            results["error"] = str(e)
        torch.cuda.empty_cache()

    return results


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        return

    device = torch.device("cuda")

    # Configuration
    batch_size = 8
    d_model_values = [16, 32, 64, 128]
    seq_len_values = [256, 1024, 4096, 8192, 16384]
    warmup_steps = 10
    benchmark_steps = 100

    print("=" * 100)
    print("PyTorch Attention Benchmark")
    print("=" * 100)
    print(f"Batch size: {batch_size}")
    print(f"d_model values: {d_model_values}")
    print(f"Sequence length values: {seq_len_values}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Benchmark steps: {benchmark_steps}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 100)
    print()

    # Results table header
    print(f"{'d_model':>8} | {'seq_len':>8} | {'Forward (ms)':>14} | {'Backward (ms)':>14} | {'Mem Before BW (GB)':>18} | {'Peak Mem (GB)':>14} | {'Status':>10}")
    print("-" * 100)

    all_results = []

    # Iterate through cartesian product
    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        results = benchmark_attention(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            device=device,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        )
        all_results.append(results)

        # Format output
        if results["oom"]:
            status = "OOM"
            forward_str = "N/A"
            backward_str = "N/A"
            mem_before_bw_str = "N/A"
            peak_mem_str = "N/A"
        elif results["error"]:
            status = "ERROR"
            forward_str = "N/A"
            backward_str = "N/A"
            mem_before_bw_str = "N/A"
            peak_mem_str = "N/A"
        else:
            status = "OK"
            forward_str = f"{results['forward_time'] * 1000:.4f}"
            backward_str = f"{results['backward_time'] * 1000:.4f}"
            mem_before_bw_str = f"{results['memory_before_backward_gb']:.4f}"
            peak_mem_str = f"{results['peak_memory_gb']:.4f}"

        print(f"{d_model:>8} | {seq_len:>8} | {forward_str:>14} | {backward_str:>14} | {mem_before_bw_str:>18} | {peak_mem_str:>14} | {status:>10}")

    print("=" * 100)
    print()

    # Summary of OOM configurations
    oom_configs = [r for r in all_results if r["oom"]]
    if oom_configs:
        print("Out-of-Memory Configurations:")
        for r in oom_configs:
            print(f"  - d_model={r['d_model']}, seq_len={r['seq_len']}")

    # Find the configuration just before OOM for memory analysis
    successful_configs = [r for r in all_results if not r["oom"] and not r["error"]]
    if successful_configs:
        # Sort by memory usage to find the largest successful config
        largest_config = max(successful_configs, key=lambda x: x["memory_before_backward_gb"])
        print()
        print("Largest Successful Configuration:")
        print(f"  d_model={largest_config['d_model']}, seq_len={largest_config['seq_len']}")
        print(f"  Memory before backward: {largest_config['memory_before_backward_gb']:.4f} GB")
        print(f"  Peak memory: {largest_config['peak_memory_gb']:.4f} GB")

    print()
    print("Memory scaling analysis (for successful configs with d_model=128):")
    d128_configs = [r for r in successful_configs if r["d_model"] == 128]
    if len(d128_configs) >= 2:
        for r in sorted(d128_configs, key=lambda x: x["seq_len"]):
            print(f"  seq_len={r['seq_len']:>5}: mem_before_bw={r['memory_before_backward_gb']:.4f} GB")


if __name__ == "__main__":
    main()
