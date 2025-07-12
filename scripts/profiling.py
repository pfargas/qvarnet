import torch
import torch.nn as nn
import torch.profiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a trivial model
model = nn.Linear(10, 5).to(device)
inputs = torch.randn(32, 10).to(device)

# Use the profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],  # include CUDA if you’re on GPU
    record_shapes=True,
    profile_memory=True,
    with_stack=True  # optional: gets you file:line info for ops
) as prof:
    # Run the operation(s) you want to profile
    output = model(inputs)
    prof.export_chrome_trace("trace.json")

# Print the results
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))