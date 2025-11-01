# OpenMP Ray Tracer

This is a parallelized version of the ray tracer using OpenMP.

## Key Changes from Serial Version

1. **Added OpenMP header**: `#include <omp.h>`
2. **Parallelized the main rendering loop**: Used `#pragma omp parallel for schedule(dynamic, 1)` to distribute rows among threads
3. **Dynamic scheduling**: Each thread gets one row at a time for better load balancing
4. **Timing**: Added `omp_get_wtime()` for accurate timing measurements
5. **Thread-safe progress reporting**: Only thread 0 reports progress to avoid race conditions

## Compilation and Running

### Using Makefile:
```bash
make           # Compile
make run       # Compile and run
make clean     # Clean up files
```

### Manual compilation:

**On macOS with Homebrew:**
```bash
# Install GCC if not already installed
brew install gcc

# Compile with Homebrew GCC (has OpenMP support)
/opt/homebrew/bin/gcc-15 -O3 -fopenmp -lm -o raytrace_openmp raytrace_openmp.c
./raytrace_openmp
```

**On Linux:**
```bash
gcc -O3 -fopenmp -lm -o raytrace_openmp raytrace_openmp.c
./raytrace_openmp
```

## Performance Tips

1. **Set number of threads**: 
   ```bash
   export OMP_NUM_THREADS=4  # Set to your CPU core count
   ./raytrace_openmp
   ```

2. **Thread affinity** (optional):
   ```bash
   export OMP_PROC_BIND=true
   ```

## Performance Comparison

### Quick Comparison:
```bash
./quick_compare.sh          # Compare serial vs OpenMP (default threads)
```

### Comprehensive Analysis:
```bash
./compare_performance.sh    # Test multiple thread counts and show detailed analysis
```

## Output

- Creates `raytrace_openmp.ppm` image file
- Displays rendering time and thread count
- Shows progress during rendering

## Parallelization Strategy

The code uses **embarrassingly parallel** approach where:
- Each row of pixels is computed independently
- No synchronization needed during computation
- Dynamic scheduling handles load imbalance (some rows may take longer due to reflections)
- Memory access pattern is mostly read-only (scene data) with write-only pixel data

This simple approach typically achieves good speedup proportional to the number of CPU cores.