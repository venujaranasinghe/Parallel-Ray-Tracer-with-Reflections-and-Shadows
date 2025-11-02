# MPI Ray Tracer

This directory contains the MPI (Message Passing Interface) parallel implementation of the ray tracer.

## Overview

The MPI version distributes the ray tracing workload across multiple processes by dividing the image rows among available processes. Each process:

1. Renders its assigned portion of the image
2. Sends the results back to the root process (rank 0)
3. The root process assembles the final image and saves it to disk

## Key Features

- **Row-based Distribution**: Image rows are distributed among processes for load balancing
- **Efficient Communication**: Minimized communication overhead by gathering results only at the end
- **Scalable Performance**: Performance improves with more processes up to the optimal number
- **Load Balancing**: Extra rows are distributed evenly when rows don't divide evenly among processes

## Files

- `raytrace_mpi.c` - Main MPI ray tracer implementation
- `Makefile` - Build configuration with various targets
- `benchmark.sh` - Performance testing script for different process counts
- `compare_performance.sh` - Comparison script between Serial, OpenMP, and MPI versions
- `README.md` - This documentation file

## Building

### Prerequisites
- MPI implementation (OpenMPI, MPICH, etc.)
- C compiler with C99 support
- Math library (libm)

### Compilation
```bash
make
```

Or manually:
```bash
mpicc -Wall -O3 -std=c99 -o raytrace_mpi raytrace_mpi.c -lm
```

## Running

### Basic Execution
```bash
# Run with 4 processes
mpirun -np 4 ./raytrace_mpi

# Run with different number of processes
mpirun -np 8 ./raytrace_mpi
```

### Using Makefile Targets
```bash
# Run with default 4 processes
make run

# Run comprehensive benchmark
make benchmark

# Clean build files
make clean
```

### Performance Testing
```bash
# Test different process counts
./benchmark.sh

# Compare with other implementations
./compare_performance.sh
```

## Implementation Details

### Parallelization Strategy

1. **Domain Decomposition**: The image is divided into horizontal strips (rows)
2. **Work Distribution**: Each process calculates which rows it's responsible for
3. **Independent Computation**: Each process renders its assigned rows independently
4. **Result Gathering**: Process 0 collects results from all other processes
5. **Image Assembly**: Process 0 assembles the complete image and saves it

### Load Balancing

```c
int rows_per_process = IMAGE_HEIGHT / size;
int extra_rows = IMAGE_HEIGHT % size;
int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);
```

This ensures that if there are extra rows when IMAGE_HEIGHT doesn't divide evenly by the number of processes, they are distributed among the first few processes.

### Communication Pattern

- **Broadcast**: Scene data (spheres, lights) is replicated across all processes
- **Point-to-Point**: Each worker process sends its rendered rows to process 0
- **Collective**: MPI_Barrier for synchronization, MPI_Reduce for timing

### Memory Management

Each process only allocates memory for its portion of the image, reducing overall memory usage compared to having the full image on every process.

## Performance Characteristics

### Expected Speedup
- **Linear scaling** for moderate numbers of processes
- **Diminishing returns** as communication overhead increases
- **Optimal process count** typically matches the number of CPU cores

### Scalability Factors
- **Computation/Communication Ratio**: High (good for MPI)
- **Load Balance**: Even distribution of work
- **Memory Requirements**: Scales with number of processes
- **Network Overhead**: Minimal (only result gathering)

## Sample Output

```
Starting MPI Ray Tracer with 4 processes...
Image size: 800x600
Distributing 600 rows among 4 processes
Rendering scene...
Progress: 8.3%
Progress: 16.7%
...
Progress: 100.0%
Rendering time: 2.45 seconds
Gathering results from all processes...
Image saved as raytrace_mpi.ppm
Ray tracing completed!
```

## Benchmarking Results

Typical performance improvements (may vary by system):

| Processes | Time (s) | Speedup | Efficiency |
|-----------|----------|---------|------------|
| 1         | 8.2      | 1.0x    | 100%       |
| 2         | 4.3      | 1.9x    | 95%        |
| 4         | 2.2      | 3.7x    | 93%        |
| 8         | 1.4      | 5.9x    | 74%        |

## Troubleshooting

### Common Issues

1. **MPI not found**: Install OpenMPI or MPICH
   ```bash
   # macOS with Homebrew
   brew install open-mpi
   
   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev
   ```

2. **Segmentation fault**: Usually memory allocation issues - check available memory

3. **Poor performance**: 
   - Try different numbers of processes
   - Check system load
   - Verify MPI is using intended network interface

### Debugging
```bash
# Run with MPI debugging
mpirun -np 4 --debug ./raytrace_mpi

# Check MPI installation
mpirun --version
```

## Theoretical Background

### Amdahl's Law
The speedup is limited by the sequential portion of the code. In this implementation:
- **Parallel portion**: Ray tracing computation (~95%)
- **Sequential portion**: Image I/O and setup (~5%)
- **Maximum theoretical speedup**: ~20x

### Communication Complexity
- **Setup**: O(1) - constant time broadcast
- **Computation**: O(n/p) - work divided by processes  
- **Gathering**: O(n) - linear in image size
- **Total**: O(n/p + n) for large n/p >> n/p

## Extensions and Modifications

### Possible Improvements
1. **Dynamic Load Balancing**: Assign work blocks dynamically
2. **Hierarchical Gathering**: Use tree-based result collection
3. **Overlapped Communication**: Pipeline computation and communication
4. **Memory Optimization**: Stream results instead of buffering

### Code Modifications
The MPI implementation maintains the same ray tracing algorithm as the serial version, with parallelization focused on the pixel rendering loop. Key modifications:

- Added MPI initialization and cleanup
- Distributed row assignments among processes
- Modified memory allocation for local image portions
- Added result gathering and communication

## Related Files
- `../Serial/raytrace_serial.c` - Original serial implementation
- `../OpenMP/raytrace_openmp.c` - OpenMP parallel version
- `../CUDA/` - GPU-accelerated version (if available)