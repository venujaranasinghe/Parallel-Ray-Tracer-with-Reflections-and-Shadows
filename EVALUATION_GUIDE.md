# Performance Evaluation Guide

## ✅ Completed: OpenMP & MPI Evaluations

### 1. OpenMP Evaluation (6 marks) - COMPLETED ✓

#### Performance Results Generated:
- **Benchmark Data**: `OpenMP/benchmark_results/openmp_results.csv`
- **Performance Graphs**: `benchmark_results/openmp_performance_graphs.png`

#### OpenMP Performance Statistics:
```
Thread Count | Exec Time | Speedup | Efficiency
--------------------------------------------------
           1 |      0.09s |    1.00x |  100.0%
           2 |      0.03s |    3.00x |  150.0%
           4 |      0.02s |    4.50x |  112.5%
           8 |      0.01s |    9.00x |  112.5%
          16 |      0.01s |    9.00x |   56.2%
```

#### 📸 Screenshots Needed for OpenMP:

**Screenshot 1**: Terminal output showing execution with **1 thread**
```bash
cd OpenMP
export OMP_NUM_THREADS=1
./raytrace_openmp
```
Capture: Full terminal showing "Number of threads: 1" and "Rendering time: X.XXs"

**Screenshot 2**: Terminal output showing execution with **4 threads**
```bash
export OMP_NUM_THREADS=4
./raytrace_openmp
```
Capture: Full terminal showing "Number of threads: 4" and timing

**Screenshot 3**: Terminal output showing execution with **16 threads**
```bash
export OMP_NUM_THREADS=16
./raytrace_openmp
```
Capture: Full terminal showing "Number of threads: 16" and timing

**Screenshot 4**: The performance graphs
- Open: `benchmark_results/openmp_performance_graphs.png`
- Shows both graphs: Threads vs Execution Time AND Threads vs Speedup

---

### 2. MPI Evaluation (6 marks) - COMPLETED ✓

#### Performance Results Generated:
- **Benchmark Data**: `MPI/benchmark_results/mpi_results.csv`
- **Performance Graphs**: `benchmark_results/mpi_performance_graphs.png`

#### MPI Performance Statistics:
```
Process Count | Exec Time | Speedup | Efficiency
--------------------------------------------------
            1 |      0.05s |    1.00x |  100.0%
            2 |      0.05s |    1.00x |   50.0%
            4 |      0.03s |    1.67x |   41.7%
            8 |      0.02s |    2.50x |   31.2%
           16 |      0.01s |    5.00x |   31.2%
```

#### 📸 Screenshots Needed for MPI:

**Screenshot 1**: Terminal output showing execution with **1 process**
```bash
cd MPI
mpirun -np 1 ./raytrace_mpi
```
Capture: Full terminal showing "Starting MPI Ray Tracer with 1 processes" and timing

**Screenshot 2**: Terminal output showing execution with **4 processes**
```bash
mpirun -np 4 ./raytrace_mpi
```
Capture: Full terminal showing "Starting MPI Ray Tracer with 4 processes" and timing

**Screenshot 3**: Terminal output showing execution with **16 processes**
```bash
mpirun --oversubscribe -np 16 ./raytrace_mpi
```
Capture: Full terminal showing "Starting MPI Ray Tracer with 16 processes" and timing

**Screenshot 4**: The performance graphs
- Open: `benchmark_results/mpi_performance_graphs.png`
- Shows both graphs: Processes vs Execution Time AND Processes vs Speedup

---

## 📋 To-Do: CUDA Evaluation (6 marks)

### 3. CUDA Evaluation Guide

For CUDA, you need to test different configurations and measure performance.

#### Test Configurations to Try:

**Block Size Variations** (keeping threads/block constant):
- Configuration 1: 128 blocks × 128 threads
- Configuration 2: 256 blocks × 128 threads  
- Configuration 3: 512 blocks × 128 threads

**Threads Per Block Variations** (keeping total threads similar):
- Configuration 4: 256 blocks × 64 threads
- Configuration 5: 256 blocks × 256 threads
- Configuration 6: 512 blocks × 512 threads

#### Steps to Complete CUDA Evaluation:

1. **Open Google Colab** and load your CUDA notebook:
   - `CUDA/CUDA_Ray_Tracer.ipynb`

2. **Modify the kernel launch configuration** in your code:
   ```cuda
   dim3 blocks(NUM_BLOCKS_X, NUM_BLOCKS_Y);
   dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
   raytraceKernel<<<blocks, threads>>>(d_image, ...);
   ```

3. **For each configuration**, record:
   - Number of blocks (total = blocks_x × blocks_y)
   - Threads per block (total = threads_x × threads_y)
   - Execution time (from CUDA events or timing)

4. **Create a table like this**:
   ```
   Config | Blocks | Threads/Block | Total Threads | Time (ms) | Speedup
   -------|--------|---------------|---------------|-----------|--------
   1      | 128    | 128           | 16,384        | X.XX      | 1.00x
   2      | 256    | 128           | 32,768        | X.XX      | X.XXx
   3      | 512    | 128           | 65,536        | X.XX      | X.XXx
   4      | 256    | 64            | 16,384        | X.XX      | X.XXx
   5      | 256    | 256           | 65,536        | X.XX      | X.XXx
   6      | 512    | 512           | 262,144       | X.XX      | X.XXx
   ```

5. **Calculate Speedup**:
   - Use Configuration 1 as baseline
   - Speedup = Baseline_Time / Current_Time

#### 📸 Screenshots Needed for CUDA:

**Screenshot 1**: Code showing kernel launch with Configuration 1
- Show the dim3 declarations and kernel launch line
- Show the timing output

**Screenshot 2**: Code showing kernel launch with Configuration 3
- Show the modified dim3 declarations
- Show the timing output

**Screenshot 3**: Code showing kernel launch with Configuration 5 or 6
- Show the dim3 declarations with different thread counts
- Show the timing output

**Screenshot 4**: Your performance comparison table or graphs
- Can be in Excel, Python matplotlib, or even a formatted markdown table
- Must show: Configuration parameters vs Execution time
- Must show: Configuration parameters vs Speedup

#### Graphs to Create for CUDA:

You can create graphs in:
- **Python/Matplotlib** (similar to OpenMP/MPI graphs)
- **Excel/Google Sheets** (import your data and create charts)
- **Colab itself** (using matplotlib in a code cell)

**Graph 1**: Bar chart of Configuration vs Execution Time
**Graph 2**: Line chart of Total Threads vs Speedup

---

## 📁 File Locations Summary

### Generated Files:
```
Barnes-Hut-N-Body-Simulation-for-Parallel-Computing/
├── benchmark_results/
│   ├── openmp_performance_graphs.png    ← OpenMP graphs
│   └── mpi_performance_graphs.png       ← MPI graphs
├── OpenMP/
│   └── benchmark_results/
│       └── openmp_results.csv           ← OpenMP raw data
├── MPI/
│   └── benchmark_results/
│       └── mpi_results.csv              ← MPI raw data
└── EVALUATION_GUIDE.md                  ← This file
```

### Scripts Created:
- `OpenMP/benchmark_detailed.sh` - Automated OpenMP benchmark
- `MPI/benchmark_detailed.sh` - Automated MPI benchmark  
- `generate_graphs.py` - Graph generation script

---

## 🚀 Quick Commands Reference

### Re-run Benchmarks (if needed):
```bash
# OpenMP
cd OpenMP && ./benchmark_detailed.sh

# MPI
cd MPI && ./benchmark_detailed.sh

# Regenerate graphs
cd .. && python3 generate_graphs.py
```

### Take Individual Screenshots:
```bash
# OpenMP examples
cd OpenMP
export OMP_NUM_THREADS=1 && ./raytrace_openmp
export OMP_NUM_THREADS=4 && ./raytrace_openmp
export OMP_NUM_THREADS=16 && ./raytrace_openmp

# MPI examples
cd MPI
mpirun -np 1 ./raytrace_mpi
mpirun -np 4 ./raytrace_mpi
mpirun --oversubscribe -np 16 ./raytrace_mpi
```

---

## ✅ Checklist

### OpenMP (6 marks):
- [x] Benchmarks run with 1, 2, 4, 8, 16 threads
- [x] Graph: Threads vs Execution Time
- [x] Graph: Threads vs Speedup  
- [ ] Screenshot: 1 thread execution
- [ ] Screenshot: 4 threads execution
- [ ] Screenshot: 16 threads execution
- [ ] Screenshot: Performance graphs

### MPI (6 marks):
- [x] Benchmarks run with 1, 2, 4, 8, 16 processes
- [x] Graph: Processes vs Execution Time
- [x] Graph: Processes vs Speedup
- [ ] Screenshot: 1 process execution
- [ ] Screenshot: 4 processes execution
- [ ] Screenshot: 16 processes execution
- [ ] Screenshot: Performance graphs

### CUDA (6 marks):
- [ ] Benchmarks run with varying block sizes
- [ ] Benchmarks run with varying threads per block
- [ ] Graph: Configuration vs Execution Time
- [ ] Graph: Configuration vs Speedup
- [ ] Screenshot: Configuration 1 execution
- [ ] Screenshot: Configuration 3 execution
- [ ] Screenshot: Different thread count execution
- [ ] Screenshot: Performance graphs/table

---

**Total Points: 18 marks (6 + 6 + 6)**

Good luck with your evaluation! 🎯
