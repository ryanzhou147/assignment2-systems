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
                 use_bf16: bool = False):
    
    # Initialize the model
    model = transformer.TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, with_rope,
                 rope_theta, max_seq_len, device, dtype).to(device)

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

    avg_time_per_step = sum(times) / benchmark_steps
    std_time_per_step = (sum((t - avg_time_per_step) ** 2 for t in times) / benchmark_steps) ** 0.5
    print(f"Average time per step ({'forward + backward' if backward else 'forward only'}): {avg_time_per_step:.6f} seconds +/- {std_time_per_step:.6f} seconds")
    return avg_time_per_step

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sizes = {
        "small": (768, 3072, 12, 12),
        "medium": (1024, 4096, 24, 16),
        "large": (1280, 5120, 36, 20),
        "xl": (1600, 6400, 48, 25),
        "2.7B": (2560, 10240, 32, 32)
    }
    vocab_size = 10000
    context_length = 512
    batch_size = 4
    warmup_steps = 2
    benchmark_steps = 10

    for size_name, (d_model, d_ff, num_layers, num_heads) in model_sizes.items():
        print(f"Benchmarking {size_name} model:")
        for use_bf16 in (False, True):
            print(f"  Mixed BF16: {use_bf16}")
            try:
                benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                                      device=device, batch_size=batch_size,
                                      warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                                      backward=True, use_bf16=use_bf16)
                benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                                      device=device, batch_size=batch_size,
                                      warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                                      backward=False, use_bf16=use_bf16)
            except RuntimeError as e:
                print(f"  Skipping (OOM or other error): {e}")

# import torch.nn as nn
# if __name__ == "__main__":
#     with torch.autocast(device_type='cuda', dtype=torch.float16):
#         x = torch.randn(4, 10, device='cuda')
        
#         out = nn.Linear(10, 10).cuda()(x)
#         print(f"Linear output: {out.dtype}")        # float16
        
#         out = nn.LayerNorm(10).cuda()(x)
#         print(f"LayerNorm output: {out.dtype}")     # float32
        
#         out = nn.ReLU()(x)
#         print(f"ReLU output: {out.dtype}")          # float32 (input was fp32)

# class ToyModel(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, 10, bias=False)
#         self.ln = nn.LayerNorm(10)
#         self.fc2 = nn.Linear(10, out_features, bias=False)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.ln(x)
#         x = self.fc2(x)
#         return x