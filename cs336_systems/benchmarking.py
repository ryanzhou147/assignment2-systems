import timeit
import torch
from contextlib import nullcontext
import cs336_basics.transformer.transformer as transformer
import cs336_basics.optimizer.cross_entropy as cross_entropy
import cs336_basics.optimizer.adamw as adamw
import torch.cuda.nvtx as nvtx

def benchmark_transformer(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None, batch_size: int = 4, warmup_steps: int = 10, benchmark_steps: int = 50, backward: bool = False,
                 use_bf16: bool = False, memory_snapshot: bool = False, snapshot_path: str = "memory_snapshot.pickle", compile: bool = False) -> None:
    
    print(f"  Config: batch={batch_size}, context={context_length}, layers={num_layers}")

    # Initialize the model
    model = transformer.TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, with_rope,
                 rope_theta, max_seq_len, device, dtype).to(device)
    
    if compile:
        model = torch.compile(model)

    model.train()
    # Generate a random batch of data
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    target_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    criterion = cross_entropy.CrossEntropyLoss()
    optimizer = adamw.AdamW(model.parameters())

    # choose autocast context: bf16 when requested and running on CUDA, else no-op
    if use_bf16 and torch.cuda.is_available():
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    # Warm-up steps
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            with amp_ctx:
                optimizer.zero_grad()
                outputs = model(input_data)
                if backward:
                    loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
                    loss.backward()
                    optimizer.step()
    
    # Benchmark steps
    times = []
    if torch.cuda.is_available() and hasattr(torch.cuda.memory, '_record_memory_history'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(benchmark_steps):

        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        with amp_ctx:
            with nvtx.range("zero_grad"):
                optimizer.zero_grad()
            
            with nvtx.range("forward"):
                outputs = model(input_data)
            
            if backward:
                with nvtx.range("loss"):
                    loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
                
                with nvtx.range("backward"):
                    loss.backward()
                
                with nvtx.range("optimizer_step"):
                    optimizer.step()
            
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak memory: {peak_memory_gb:.2f} GB")
    if memory_snapshot:
        if torch.cuda.is_available() and hasattr(torch.cuda.memory, '_dump_snapshot'):

            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
        else:
            print("Warning: CUDA memory snapshot API not available; no snapshot written")

    avg_time_per_step = sum(times) / benchmark_steps
    std_time_per_step = (sum((t - avg_time_per_step) ** 2 for t in times) / benchmark_steps) ** 0.5
    print(f"Average time per step ({'forward + backward' if backward else 'forward only'}): {avg_time_per_step:.6f} seconds +/- {std_time_per_step:.6f} seconds")
    return avg_time_per_step

import triton.testing.do_bench

# Write a benchmarking script using triton.testing.do_bench that compares the performance
# of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with
# a regular PyTorch implementation (i.e., not using FlashAttention).
# Specifically, you will report a table that includes latencies for forward, backward, and the endto-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly
# generate any necessary inputs before you start benchmarking, and run the benchmark on a single
# H100. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence
# lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of various powers
# of 2 from 16 up to size 128, and precisions of torch.bfloat16 and torch.float32. You will
# likely need to adjust tile sizes depending on the input sizes.
# Deliverable: A table of results comparing your implementation of FlashAttention-2 with the
# PyTorch implementation, using the settings above and reporting forward, backward, and end-toend latencies.

def triton_vs_pytorch_flash_attention_benchmark():
    import cs336_systems.flash_forward
    import cs336_systems.pytorch_attention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    head_dims = [16, 32, 64, 128]
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    precisions = [torch.float32, torch.bfloat16]

    results = []

    for d_model in head_dims:
        for seq_len in seq_lengths:
            for precision in precisions:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Initialize models
                    triton_mha = cs336_systems.flash_forward.FlashAttentionModule(d_model=d_model, num_heads=1, device=device).to(device).to(precision)
                    pytorch_mha = cs336_systems.pytorch_attention.MultiHeadSelfAttention(d_model=d_model, num_heads=1, device=device).to(device).to(precision)

                    # Generate random input
                    Q = torch.randn((batch_size, seq_len, d_model), device=device, dtype=precision, requires_grad=True)

                    # Warmup
                    for _ in range(10):
                        out_triton = triton_mha(Q)
                        out_triton.mean().backward()
                        triton_mha.zero_grad()

                        out_pytorch = pytorch_mha(Q)
                        out_pytorch.mean().backward()
                        pytorch_mha.zero_grad()
                    torch.cuda.synchronize()

                    # Time forward passes
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    for _ in range(10):
                        out_triton = triton_mha(Q)
                        torch.cuda.synchronize()
                    triton_forward_time = timeit.default_timer() - start

                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    for _ in range(10):
                        out_pytorch = pytorch_mha(Q)
                        torch.cuda.synchronize()
                    pytorch_forward_time = timeit.default_timer() - start

                    # Time backward passes
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    for _ in range(10):
                        out_triton = triton_mha(Q)
                        torch.cuda.synchronize()
                        out_triton.mean().backward()
                        torch.cuda.synchronize()
                        triton_mha.zero_grad()
                    triton_backward_time = timeit.default_timer() - start
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    for _ in range(10):
                        out_pytorch = pytorch_mha(Q)
                        torch.cuda.synchronize()
                        out_pytorch.mean().backward()
                        torch.cuda.synchronize()
                        pytorch_mha.zero_grad()
                    pytorch_backward_time = timeit.default_timer() - start
                    # Store results
                    results.append({
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'precision': str(precision).split('.')[-1],
                        'triton_forward_time': triton_forward_time / 10,
                        'pytorch_forward_time': pytorch_forward_time / 10,
                        'triton_backward_time': triton_backward_time / 10,
                        'pytorch_backward_time': pytorch_backward_time / 10,
                    })
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"OOM for d_model={d_model}, seq_len={seq_len}, precision={precision}")
                        torch.cuda.empty_cache()
    # Print results
    print(f"{'d_model':>8} {'seq_len':>8} {'precision':>10} {'triton_fwd(s)':>15} {'pytorch_fwd
    '(s)':>15} {'triton_bwd(s)':>15} {'pytorch_bwd(s)':>15}")
    for res in results:
        print(f"{res['d_model']:>8} {res['seq_len']:>8} {res['precision']:>10} {res['triton_forward_time']:>15.6f} {res['pytorch_forward_time']:>15.6f} {res['triton_backward_time']:>15.6f} {res['pytorch_backward_time']:>15.6f}")
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-snapshot', action='store_true', help='Record CUDA memory history and dump a snapshot')
    parser.add_argument('--snapshot-path', type=str, default='./memory_snapshot.pickle', help='Path to write memory snapshot')
    parser.add_argument('--bf16', action='store_true', help='Run benchmarks with BF16 autocast when available')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--sizes', type=str, nargs='*', default=['small','medium','large','xl','2.7B'],
                        help='Which model sizes to benchmark')
    parser.add_argument('--forward-only', action='store_true', help='Skip backward pass')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sizes = {
        "small": (768, 3072, 12, 12),
        "medium": (1024, 4096, 24, 16),
        "large": (1280, 5120, 36, 20),
        "xl": (1600, 6400, 48, 25),
        "2.7B": (2560, 10240, 32, 32)
    }
    vocab_size = 10000
    context_length = args.context
    batch_size = args.batch
    warmup_steps = args.warmup
    benchmark_steps = args.steps
    backward = not args.forward_only
    compile = args.compile

    for size_name in args.sizes:
        if size_name not in model_sizes:
            print(f"Unknown model size: {size_name}, skipping")
            continue
        d_model, d_ff, num_layers, num_heads = model_sizes[size_name]
        print(f"Benchmarking {size_name} model:")
        bf16_options = [False, True] if args.bf16 else [False]
        for use_bf16 in bf16_options:
            print(f"  Mixed BF16: {use_bf16}")
            try:
                benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                                      device=device, batch_size=batch_size,
                                      warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                                      backward=backward, use_bf16=use_bf16,
                                      memory_snapshot=args.memory_snapshot, snapshot_path=args.snapshot_path, compile=compile)
            except RuntimeError as e:
                print(f"  Skipping (OOM or other error): {e}")
                torch.cuda.empty_cache()