# Parallel Ray Tracer with Reflections and Shadows

A high-performance ray tracing implementation with multiple parallelization strategies: Serial, OpenMP, MPI, and CUDA.

## 🎯 Project Overview

This project implements a photorealistic ray tracer that renders 3D scenes with:
- **Sphere primitives** with customizable colors
- **Reflections** for realistic surface interactions
- **Lighting effects** (ambient, diffuse, specular)
- **Multiple light sources**
- **Shadows and depth**
- **Gradient sky background**

## 🚀 Implementations

| Version | Technology | Speedup | Best For |
|---------|-----------|---------|----------|
| **Serial** | Single-threaded CPU | 1x | Baseline reference |
| **OpenMP** | Multi-threaded CPU | ~4x | Shared memory systems |
| **MPI** | Distributed computing | ~4x | Clusters, multi-node |
| **CUDA** | GPU acceleration | **~400x** | NVIDIA GPUs |

## 📁 Project Structure

```
├── Serial/                  # Baseline CPU implementation
├── OpenMP/                  # Multi-threaded shared memory version
├── MPI/                     # Distributed memory parallel version
├── CUDA/                    # GPU-accelerated version
├── benchmark_openmp.sh      # OpenMP benchmarking script
├── benchmark_mpi.sh         # MPI benchmarking script
├── run_all_benchmarks.sh    # Complete benchmarking pipeline
├── generate_graphs.py       # Generate individual performance graphs
└── compare_all_implementations.py  # Comprehensive comparison tool
```

## 🎨 Sample Output

The ray tracer renders a scene with:
- 5 colored spheres (red, green, blue, yellow, and a large gray floor)
- 2 light sources creating realistic lighting
- Reflections showing neighboring spheres
- Gradient background (blue sky effect)
- 800×600 resolution (configurable)

## 🏃 Quick Start

### Serial Version
```bash
cd Serial/
gcc -O3 -o raytrace_serial raytrace_serial.c -lm
./raytrace_serial
```

### OpenMP Version
```bash
cd OpenMP/

# macOS with Homebrew libomp
gcc -o raytrace_openmp raytrace_openmp.c -lm \
    -I$(brew --prefix libomp)/include \
    -L$(brew --prefix libomp)/lib \
    -Xpreprocessor -fopenmp -lomp -O3

# Linux
gcc -o raytrace_openmp raytrace_openmp.c -lm -fopenmp -O3

# Run with 4 threads
./raytrace_openmp 4
```

### MPI Version
```bash
cd MPI/
mpicc -o raytrace_mpi raytrace_mpi.c -lm -O3
mpirun -np 4 ./raytrace_mpi
```

### CUDA Version (Run on Google Colab)
**⚠️ The CUDA implementation is designed to run on Google Colab**

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `raytrace_cuda.cu` or copy the code
3. The file contains Jupyter magic commands (`%%writefile`) for Colab
4. Run the cells to compile and execute
5. Use `plot_results.py` to visualize performance

Alternatively, if you have a local NVIDIA GPU:
```bash
cd CUDA/
# Remove the %%writefile line from raytrace_cuda.cu first
nvcc -o raytrace_cuda raytrace_cuda.cu
./raytrace_cuda
```

## 📊 Performance Benchmarking

### Quick Benchmark - Run Everything

Run all benchmarks and generate comprehensive comparison graphs:

```bash
./run_all_benchmarks.sh
```

This automated pipeline will:
1. Test OpenMP with 1, 2, 4, 8, 16 threads
2. Test MPI with 1, 2, 4, 8, 16 processes  
3. Generate individual performance graphs
4. Create comprehensive comparison visualizations

**Outputs:**
- `openmp_results.csv` - OpenMP benchmark data
- `mpi_results.csv` - MPI benchmark data
- `cuda_results.csv` - CUDA benchmark data (if available)
- `openmp_performance.png` - OpenMP performance graphs
- `mpi_performance.png` - MPI performance graphs
- `all_implementations_comparison.csv` - Combined results
- Comprehensive comparison graphs showing all implementations

### Individual Benchmarks

**OpenMP only:**
```bash
./benchmark_openmp.sh
```

**MPI only:**
```bash
./benchmark_mpi.sh
```

**CUDA benchmarking:**
```bash
cd CUDA/
python3 plot_results.py  # After running raytrace_cuda
```

### Generate Graphs Manually

After running benchmarks, generate graphs:

```bash
# Individual implementation graphs
python3 generate_graphs.py

# Comprehensive comparison of all implementations
python3 compare_all_implementations.py
```

### Typical Results

| Implementation | Time (s) | Speedup | Efficiency |
|---------------|----------|---------|------------|
| Serial        | 8.0      | 1.0x    | 100%       |
| OpenMP (4)    | 2.0      | 4.0x    | 100%       |
| MPI (4)       | 2.0      | 4.0x    | 100%       |
| CUDA (GPU)    | 0.02     | 400x    | -          |

## 🛠️ Requirements

### Serial & OpenMP
- GCC compiler
- Math library (libm)
- OpenMP support (for OpenMP version)

### MPI
- MPI implementation (OpenMPI, MPICH)
- C compiler with C99 support

### CUDA
- **Recommended:** Google Colab (free, no local setup required)
- **Alternative:** Local NVIDIA GPU with CUDA Toolkit

### Benchmarking & Visualization
- Python 3.x
- pandas (`pip3 install pandas`)
- matplotlib (`pip3 install matplotlib`)
- numpy (`pip3 install numpy`)

## 📖 Documentation

### Benchmarking Scripts
- `run_all_benchmarks.sh` - Complete automated benchmarking pipeline
- `benchmark_openmp.sh` - OpenMP-specific benchmarks
- `benchmark_mpi.sh` - MPI-specific benchmarks
- `generate_graphs.py` - Generate individual performance graphs
- `compare_all_implementations.py` - Comprehensive comparison tool

### Implementation Guides
Each implementation directory includes:
- **Source code** with detailed comments
- **Makefile** for easy compilation (where applicable)
- Implementation-specific documentation

### CUDA Resources
- `CUDA/raytrace_cuda.cu` - CUDA implementation with Colab magic commands
- `CUDA/plot_results.py` - CUDA performance visualization
- **Note:** CUDA code is designed for Google Colab with `%%writefile` commands

## 🎓 Learning Outcomes

This project demonstrates:
- **Parallel algorithm design** for graphics applications
- **Load balancing** across different architectures
- **Communication patterns** in distributed systems
- **GPU programming** with CUDA
- **Performance optimization** techniques
- **Comparative analysis** of parallelization strategies

## 🔬 Technical Details

### Ray Tracing Algorithm
1. Generate rays from camera through each pixel
2. Test ray-sphere intersections
3. Calculate lighting at hit points
4. Compute reflections recursively
5. Return final pixel color

### Parallelization Strategies

**OpenMP**: Parallel for loop over image rows
```c
#pragma omp parallel for schedule(dynamic)
for (int y = 0; y < IMAGE_HEIGHT; y++) {
    // Render row y
}
```

**MPI**: Distribute rows among processes
```c
// Each process renders its assigned rows
for (int y = start_row; y < end_row; y++) {
    // Render row y
}
// Gather results at root process
```

**CUDA**: Each thread renders one pixel
```cuda
__global__ void raytrace_kernel(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Render pixel (x, y)
}
```

## 🌟 Features

- ✅ Multiple sphere primitives
- ✅ Recursive reflections (configurable depth)
- ✅ Phong lighting model
- ✅ Multiple light sources
- ✅ Shadows
- ✅ Anti-aliasing ready (configurable)
- ✅ PPM image output
- ✅ Performance timing
- ✅ Progress indicators

## 🔧 Customization

Key parameters in source files:

```c
#define IMAGE_WIDTH 800      // Image width in pixels
#define IMAGE_HEIGHT 600     // Image height in pixels
#define MAX_RAY_DEPTH 3      // Reflection recursion depth
#define MAX_SPHERES 5        // Number of spheres
#define MAX_LIGHTS 2         // Number of light sources
```

## 📝 Output Files

### Rendered Images
Each implementation generates:
- `raytrace_serial.ppm` (Serial version)
- `raytrace_openmp.ppm` (OpenMP version)
- `raytrace_mpi.ppm` (MPI version)
- `raytrace_cuda.ppm` (CUDA version)

Convert PPM to PNG:
```bash
convert raytrace_serial.ppm raytrace_serial.png
```

### Benchmark Results
- `openmp_results.csv` - OpenMP performance data
- `mpi_results.csv` - MPI performance data
- `cuda_results.csv` - CUDA performance data
- `all_implementations_comparison.csv` - Combined comparison
- `implementation_comparison_summary.csv` - Summary statistics

### Performance Graphs
- `openmp_performance.png` - OpenMP execution time and speedup
- `mpi_performance.png` - MPI execution time and speedup
- Various comparison graphs generated by `compare_all_implementations.py`

## 🤝 Contributing

Feel free to:
- Add more primitives (planes, triangles, etc.)
- Implement additional lighting models
- Add textures or materials
- Optimize further
- Add anti-aliasing
- Implement path tracing

## 🔍 Workflow

### 1. Run Individual Implementations
```bash
# Serial
cd Serial && gcc -O3 -o raytrace_serial raytrace_serial.c -lm && ./raytrace_serial

# OpenMP (Linux)
cd OpenMP && gcc -o raytrace_openmp raytrace_openmp.c -lm -fopenmp -O3 && ./raytrace_openmp 4

# MPI
cd MPI && mpicc -o raytrace_mpi raytrace_mpi.c -lm -O3 && mpirun -np 4 ./raytrace_mpi

# CUDA - Run on Google Colab (see Quick Start section)
```

### 2. Run Comprehensive Benchmarks
```bash
# Automated benchmarking of all implementations
./run_all_benchmarks.sh
```

### 3. Analyze Results
```bash
# Generate detailed comparison graphs
python3 compare_all_implementations.py

# View CSV results
cat all_implementations_comparison.csv
cat implementation_comparison_summary.csv
```

## 📚 References

- **Ray Tracing in One Weekend** - Peter Shirley
- **CUDA Programming Guide** - NVIDIA
- **MPI: A Message-Passing Interface Standard**
- **OpenMP Specification**

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Venuja Ranasinghe

---

**Ready to render! Choose your parallel implementation and see the speedup!** 🚀
