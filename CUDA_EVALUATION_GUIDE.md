# CUDA Evaluation - Detailed Guide

## Overview
Test CUDA ray tracer with different block sizes and threads per block configurations to measure performance impact.

## Recommended Test Configurations

### Configuration Set 1: Varying Block Count (Fixed Threads/Block)
Keep threads per block constant at 16×16 (256 threads), vary grid size:

| Config | Blocks (X×Y) | Threads/Block (X×Y) | Total Blocks | Total Threads |
|--------|--------------|---------------------|--------------|---------------|
| 1      | 32×32        | 16×16               | 1,024        | 262,144       |
| 2      | 64×64        | 16×16               | 4,096        | 1,048,576     |
| 3      | 50×38        | 16×16               | 1,900        | 486,400       |

### Configuration Set 2: Varying Threads/Block (Adjusted Blocks)
Keep image coverage similar, vary thread block size:

| Config | Blocks (X×Y) | Threads/Block (X×Y) | Total Blocks | Total Threads |
|--------|--------------|---------------------|--------------|---------------|
| 4      | 100×75       | 8×8                 | 7,500        | 480,000       |
| 5      | 50×38        | 16×16               | 1,900        | 486,400       |
| 6      | 25×19        | 32×32               | 475          | 483,840       |

## Code Modifications Needed

### In your CUDA kernel launch (Jupyter notebook):

**Current code probably looks like:**
```cpp
dim3 blocks(50, 38);  // Number of blocks in X and Y
dim3 threads(16, 16); // Threads per block in X and Y

raytraceKernel<<<blocks, threads>>>(d_image, width, height, ...);
```

### For each configuration, modify these values:

**Configuration 1:**
```cpp
dim3 blocks(32, 32);
dim3 threads(16, 16);
```

**Configuration 2:**
```cpp
dim3 blocks(64, 64);
dim3 threads(16, 16);
```

**Configuration 3:**
```cpp
dim3 blocks(50, 38);
dim3 threads(16, 16);
```

**Configuration 4:**
```cpp
dim3 blocks(100, 75);
dim3 threads(8, 8);
```

**Configuration 5:**
```cpp
dim3 blocks(50, 38);
dim3 threads(16, 16);
```

**Configuration 6:**
```cpp
dim3 blocks(25, 19);
dim3 threads(32, 32);
```

## Timing Your CUDA Code

### Method 1: Using CUDA Events (Recommended)
```cpp
// Before kernel launch
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

// Your kernel launch
raytraceKernel<<<blocks, threads>>>(d_image, width, height, ...);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Kernel execution time: %.3f ms\n", milliseconds);
```

### Method 2: Using Python time module
```python
import time

start = time.time()
# Run CUDA kernel
cudaDeviceSynchronize()  # Wait for kernel to complete
end = time.time()

execution_time_ms = (end - start) * 1000
print(f"Kernel execution time: {execution_time_ms:.3f} ms")
```

## Data Collection Template

Create a table in your notebook or Excel:

```python
import pandas as pd

results = {
    'Config': [1, 2, 3, 4, 5, 6],
    'Blocks_X': [32, 64, 50, 100, 50, 25],
    'Blocks_Y': [32, 64, 38, 75, 38, 19],
    'Threads_X': [16, 16, 16, 8, 16, 32],
    'Threads_Y': [16, 16, 16, 8, 16, 32],
    'Total_Blocks': [1024, 4096, 1900, 7500, 1900, 475],
    'Total_Threads': [262144, 1048576, 486400, 480000, 486400, 483840],
    'Time_ms': [0, 0, 0, 0, 0, 0],  # Fill in your measured times
}

df = pd.DataFrame(results)
# Calculate speedup (relative to Config 1)
baseline_time = df.loc[0, 'Time_ms']
df['Speedup'] = baseline_time / df['Time_ms']
df['Efficiency'] = (df['Speedup'] / df['Total_Threads'] * df.loc[0, 'Total_Threads']) * 100

print(df)
```

## Graph Generation Code

### Graph 1: Configuration vs Execution Time
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

configs = df['Config'].values
times = df['Time_ms'].values

ax.bar(configs, times, color='#F18F01', alpha=0.8)
ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('CUDA: Configuration vs Execution Time', fontsize=14, fontweight='bold')
ax.set_xticks(configs)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (c, t) in enumerate(zip(configs, times)):
    ax.text(c, t, f'{t:.2f}ms', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('cuda_execution_time.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Graph 2: Configuration vs Speedup
```python
fig, ax = plt.subplots(figsize=(10, 6))

configs = df['Config'].values
speedup = df['Speedup'].values

ax.plot(configs, speedup, 'o-', color='#C73E1D', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax.set_title('CUDA: Configuration vs Speedup', fontsize=14, fontweight='bold')
ax.set_xticks(configs)
ax.grid(True, alpha=0.3)
ax.legend()

# Add value labels
for i, (c, s) in enumerate(zip(configs, speedup)):
    ax.annotate(f'{s:.2f}x', (c, s), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('cuda_speedup.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Screenshot Checklist

### Screenshot 1: Configuration 1 Execution
- Show the code cell with `dim3 blocks(32, 32); dim3 threads(16, 16);`
- Show the output with execution time
- Label: "CUDA Config 1: 32×32 blocks, 16×16 threads/block"

### Screenshot 2: Configuration 3 Execution  
- Show the code cell with `dim3 blocks(50, 38); dim3 threads(16, 16);`
- Show the output with execution time
- Label: "CUDA Config 3: 50×38 blocks, 16×16 threads/block"

### Screenshot 3: Configuration 6 Execution
- Show the code cell with `dim3 blocks(25, 19); dim3 threads(32, 32);`
- Show the output with execution time
- Label: "CUDA Config 6: 25×19 blocks, 32×32 threads/block"

### Screenshot 4: Performance Table
- Show your pandas DataFrame with all results
- Must include: Config, Blocks, Threads, Time, Speedup columns

### Screenshot 5: Execution Time Graph
- The bar chart showing Configuration vs Execution Time

### Screenshot 6: Speedup Graph
- The line chart showing Configuration vs Speedup

## Running in Google Colab

### Step-by-Step Process:

1. **Open your CUDA notebook** in Google Colab
   - Upload `CUDA/CUDA_Ray_Tracer.ipynb`

2. **Enable GPU Runtime**
   - Runtime → Change runtime type → Hardware accelerator: GPU → Save

3. **Create a new cell for benchmarking**:
```python
# Configuration parameters to test
configs = [
    {'name': 'Config 1', 'blocks': (32, 32), 'threads': (16, 16)},
    {'name': 'Config 2', 'blocks': (64, 64), 'threads': (16, 16)},
    {'name': 'Config 3', 'blocks': (50, 38), 'threads': (16, 16)},
    {'name': 'Config 4', 'blocks': (100, 75), 'threads': (8, 8)},
    {'name': 'Config 5', 'blocks': (50, 38), 'threads': (16, 16)},
    {'name': 'Config 6', 'blocks': (25, 19), 'threads': (32, 32)},
]

results = []

for config in configs:
    blocks_x, blocks_y = config['blocks']
    threads_x, threads_y = config['threads']
    
    # Modify your kernel launch here
    # dim3 blocks(blocks_x, blocks_y)
    # dim3 threads(threads_x, threads_y)
    # raytraceKernel<<<blocks, threads>>>(...)
    
    # Measure time and store
    # results.append({'config': config['name'], 'time': measured_time})
    
print(results)
```

4. **Generate graphs in Colab**:
   - Copy the graph generation code above
   - Modify to use your timing results
   - The graphs will display inline in the notebook

5. **Take screenshots**:
   - Screenshot each configuration's execution
   - Screenshot the final results table
   - Screenshot each graph

## Alternative: Simpler Approach

If you want to keep it simple:

1. **Test only 3 configurations**:
   - Small: 25×19 blocks, 16×16 threads
   - Medium: 50×38 blocks, 16×16 threads  
   - Large: 100×75 blocks, 16×16 threads

2. **Manually record times** in a table

3. **Create graphs in Excel/Google Sheets**:
   - Import your data
   - Create bar chart for execution time
   - Create line chart for speedup

4. **Take screenshots** of Excel charts

## Tips for Good Results

- ✅ Run each configuration 2-3 times and use the average
- ✅ Make sure GPU is warmed up (run once before timing)
- ✅ Use the same image size for all tests
- ✅ Clearly label each screenshot
- ✅ Ensure timing code is visible in screenshots
- ✅ Show actual measured values, not placeholders

## Expected Observations

You might observe:
- **Optimal thread block size**: Usually 16×16 or 32×32 performs best
- **Too many blocks**: Can cause overhead
- **Too few threads/block**: Underutilizes GPU
- **Sweet spot**: Balance between blocks and threads/block

Include these observations in your report!

---

Good luck with your CUDA evaluation! 🚀
