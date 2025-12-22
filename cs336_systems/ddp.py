import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_num_elements_for_mb(target_mb, dtype=torch.float32):
    """Calculate number of elements needed for target MB"""
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    target_bytes = target_mb * 1024 * 1024
    return int(target_bytes / bytes_per_element)

def benchmark_all_reduce(rank, world_size, backend, data_size_mb, num_warmup=5, num_trials=20):
    """Benchmark all-reduce for given configuration"""
    
    # Setup
    setup(rank, world_size, backend)
    
    # Determine device
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Create tensor of target size
    num_elements = get_num_elements_for_mb(data_size_mb)
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # Synchronize before timing
    dist.barrier()
    
    # Benchmark
    times = []
    for _ in range(num_trials):
        if backend == "nccl":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        
        if backend == "nccl":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    # Only rank 0 prints results
    if rank == 0:
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        bandwidth = (data_size_mb / avg_time) * 2 * (world_size - 1) / world_size  # Ring all-reduce bandwidth
        
        print(f"Backend: {backend:5s} | Procs: {world_size} | Size: {data_size_mb:4d}MB | "
              f"Time: {avg_time*1000:8.2f}ms ± {std_time*1000:.2f}ms | "
              f"Bandwidth: {bandwidth:.2f} MB/s")
    
    cleanup()

if __name__ == "__main__":

    # Configuration
    backends = ["gloo", "nccl"]  # gloo for CPU, nccl for GPU
    data_sizes_mb = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    num_processes = [2, 4, 6]
    
    results = []
    
    print("=" * 80)
    print("All-Reduce Benchmark")
    print("=" * 80)
    
    for backend in backends:
        # Skip NCCL if no GPUs available
        if backend == "nccl" and not torch.cuda.is_available():
            print(f"Skipping {backend} - no GPUs available")
            continue
        
        # Check GPU count for NCCL
        if backend == "nccl":
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
        
        for world_size in num_processes:
            # Skip if not enough GPUs for NCCL
            if backend == "nccl" and world_size > torch.cuda.device_count():
                print(f"Skipping {backend} with {world_size} procs - only {torch.cuda.device_count()} GPUs")
                continue
            
            for data_size_mb in data_sizes_mb:
                try:
                    mp.spawn(
                        fn=benchmark_all_reduce,
                        args=(world_size, backend, data_size_mb),
                        nprocs=world_size,
                        join=True
                    )
                except Exception as e:
                    print(f"Error with {backend}, {world_size} procs, {data_size_mb}MB: {e}")
        
        print("-" * 80)
