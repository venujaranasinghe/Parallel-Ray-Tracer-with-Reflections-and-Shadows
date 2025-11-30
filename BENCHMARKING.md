# Performance Benchmarking Guide

## Quick Start

### Run All Benchmarks and Generate Graphs
```bash
./run_all_benchmarks.sh
```

### Or Run Step by Step
```bash
./benchmark_openmp.sh       # Test OpenMP (1,2,4,8,16 threads)
./benchmark_mpi.sh          # Test MPI (1,2,4,8,16 processes)
python3 generate_graphs.py  # Generate performance graphs
```

## Requirements

```bash
pip3 install pandas matplotlib
```

## Output Files

- `openmp_performance.png` - Execution time & speedup graphs for OpenMP
- `mpi_performance.png` - Execution time & speedup graphs for MPI
- `openmp_results.csv` - Raw timing data for OpenMP
- `mpi_results.csv` - Raw timing data for MPI

## Understanding Results

### Execution Time Graph
Shows how long rendering takes with different thread/process counts. **Lower is better.**

### Speedup Graph
Shows performance improvement compared to serial execution.
- **Speedup = Serial Time / Parallel Time**
- Green line = Actual speedup
- Red dashed line = Ideal speedup (perfect scaling)

### Example Output
```
===== OpenMP Performance Summary =====
 Threads  Time(seconds)  Speedup
       1          12.34     1.00x  (baseline)
       2           6.45     1.91x  (~2x faster)
       4           3.28     3.76x  (~4x faster)
       8           1.85     6.67x  (~7x faster)
      16           1.42     8.69x  (~9x faster)

Best speedup: 8.69x with 16 threads
```

## Customization

To test different thread/process counts, edit the arrays in the scripts:

**benchmark_openmp.sh:**
```bash
THREAD_COUNTS=(1 2 4 8 16)
```

**benchmark_mpi.sh:**
```bash
PROCESS_COUNTS=(1 2 4 8 16)
```

## Troubleshooting

**Permission denied:**
```bash
chmod +x *.sh
```

**MPI not found:**
```bash
# macOS
brew install open-mpi

# Linux
sudo apt-get install libopenmpi-dev
```

**Python modules not found:**
```bash
pip3 install pandas matplotlib
```
