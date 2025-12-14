import timeit
import torch
import cs336_basics.transformer.transformer as transformer
import cs336_basics.optimizer.cross_entropy as cross_entropy
import cs336_basics.optimizer.adamw as adamw

def benchmark_transformer(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None, batch_size: int = 4, warmup_steps: int = 10, benchmark_steps: int = 50, backward: bool = False):
    
    # Initialize the model
    model = transformer.TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, with_rope,
                 rope_theta, max_seq_len, device, dtype).to(device)

    model.train()
    # Generate a random batch of data
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    target_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    criterion = cross_entropy.CrossEntropyLoss()
    optimizer = adamw.AdamW(model.parameters())

    # Warm-up steps
    for _ in range(warmup_steps):
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

        optimizer.zero_grad()
        outputs = model(input_data)

        if backward:
            loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        start_time = end_time

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
        benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                              device=device, batch_size=batch_size,
                              warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                              backward=True)
        benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                        device=device, batch_size=batch_size,
                        warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                        backward=False)