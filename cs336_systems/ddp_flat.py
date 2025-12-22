import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def broadcast_parameters(model, src=0):
    """Broadcast model parameters from src rank to all other ranks"""
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

def all_reduce_gradients_naive(model):
    """Naive: All-reduce each gradient tensor separately"""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

def all_reduce_gradients_flat(model):
    """Optimized: Flatten all gradients, single all-reduce, unflatten"""
    world_size = dist.get_world_size()
    
    # Collect all gradients
    grads = [param.grad.data for param in model.parameters() if param.grad is not None]
    
    # Flatten into single tensor
    flat_grads = _flatten_dense_tensors(grads)
    
    # Single all-reduce
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads /= world_size
    
    # Unflatten back to original shapes
    unflat_grads = _unflatten_dense_tensors(flat_grads, grads)
    
    # Copy back to parameter gradients
    for param, unflat_grad in zip(
        [p for p in model.parameters() if p.grad is not None], 
        unflat_grads
    ):
        param.grad.data.copy_(unflat_grad)


def benchmark_ddp(rank, world_size, model_config, use_flat=False, num_steps=50):
    """
    Benchmark DDP training comparing naive vs flat gradient communication.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < world_size:
        if rank == 0:
            print(f"ERROR: Requested {world_size} GPUs but only {num_gpus} available.")
        return
    
    setup(rank, world_size, backend="nccl")
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    from cs336_basics.transformer.transformer import TransformerLM
    
    vocab_size = 10000
    context_length = 512
    d_model, d_ff, num_layers, num_heads = model_config
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    ).to(device)
    
    broadcast_parameters(model, src=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    batch_size_per_gpu = 4
    input_data = torch.randint(0, vocab_size, (batch_size_per_gpu, context_length), device=device)
    target_data = torch.randint(0, vocab_size, (batch_size_per_gpu, context_length), device=device)
    
    # Select communication method
    all_reduce_fn = all_reduce_gradients_flat if use_flat else all_reduce_gradients_naive
    method_name = "FLAT" if use_flat else "NAIVE"
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
        loss.backward()
        all_reduce_fn(model)
        optimizer.step()
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    total_times = []
    comm_times = []
    
    for step in range(num_steps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
        loss.backward()
        
        torch.cuda.synchronize()
        comm_start = time.perf_counter()
        all_reduce_fn(model)
        torch.cuda.synchronize()
        comm_end = time.perf_counter()
        
        optimizer.step()
        
        torch.cuda.synchronize()
        step_end = time.perf_counter()
        
        total_times.append(step_end - step_start)
        comm_times.append(comm_end - comm_start)
    
    if rank == 0:
        avg_total = sum(total_times) / len(total_times) * 1000
        avg_comm = sum(comm_times) / len(comm_times) * 1000
        comm_percent = (avg_comm / avg_total) * 100
        
        num_params = sum(p.numel() for p in model.parameters())
        num_tensors = sum(1 for p in model.parameters())
        
        print(f"\n{'=' * 60}")
        print(f"DDP Benchmark: {method_name} gradient communication")
        print(f"{'=' * 60}")
        print(f"World size:              {world_size} GPUs")
        print(f"Model parameters:        {num_params:,} ({num_tensors} tensors)")
        print(f"{'─' * 60}")
        print(f"Avg time per step:       {avg_total:.2f} ms")
        print(f"Avg communication time:  {avg_comm:.2f} ms ({comm_percent:.1f}%)")
        print(f"Compute time:            {avg_total - avg_comm:.2f} ms ({100 - comm_percent:.1f}%)")
        print(f"{'=' * 60}")
    
    cleanup()


def run_benchmark(rank, world_size, model_config, use_flat):
    benchmark_ddp(rank, world_size, model_config, use_flat)


def compare_methods(world_size, model_config):
    """Run both methods and compare"""
    print("\n" + "=" * 60)
    print("COMPARING NAIVE vs FLAT GRADIENT COMMUNICATION")
    print("=" * 60)
    
    # Naive method
    print("\nRunning NAIVE (per-parameter all-reduce)...")
    mp.spawn(run_benchmark, args=(world_size, model_config, False), nprocs=world_size, join=True)
    
    # Flat method
    print("\nRunning FLAT (single batched all-reduce)...")
    mp.spawn(run_benchmark, args=(world_size, model_config, True), nprocs=world_size, join=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large", "xl"])
    parser.add_argument("--method", type=str, default="compare",
                        choices=["naive", "flat", "compare"])
    args = parser.parse_args()
    
    model_configs = {
        "small": (768, 3072, 12, 12),
        "medium": (1024, 4096, 24, 16),
        "large": (1280, 5120, 36, 20),
        "xl": (1600, 6400, 48, 25),
    }
    
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
        if args.world_size == 0:
            print("No GPUs available.")
            exit(1)
        print(f"Auto-detected {args.world_size} GPU(s)")
    
    config = model_configs[args.model_size]
    
    if args.method == "compare":
        compare_methods(args.world_size, config)
    else:
        use_flat = args.method == "flat"
        mp.spawn(run_benchmark, args=(args.world_size, config, use_flat), 
                 nprocs=args.world_size, join=True)