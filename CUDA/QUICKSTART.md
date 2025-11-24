# CUDA Ray Tracer - Quick Start

## Google Colab Setup (Simplest Method)

### Method 1: Copy-Paste Single Script

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → Select GPU → Save
4. Copy this entire code into the first cell:

```python
# Install and check GPU
!nvidia-smi

# Create the CUDA file
%%writefile raytrace_cuda.cu
# PASTE THE ENTIRE CONTENTS OF raytrace_cuda.cu HERE

# Compile
!nvcc -o raytrace_cuda raytrace_cuda.cu

# Run
!./raytrace_cuda

# Convert to PNG and display
!apt-get update -qq && apt-get install -y -qq imagemagick
!convert raytrace_cuda.ppm raytrace_cuda.png

from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

### Method 2: GitHub Clone (Recommended)

1. Push this code to your GitHub repository
2. In Google Colab, create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → Select GPU → Save
4. Run these cells:

```bash
# Clone repository
!git clone https://github.com/venujaranasinghe/Parallel-Ray-Tracer-with-Reflections-and-Shadows.git
%cd Parallel-Ray-Tracer-with-Reflections-and-Shadows/CUDA
```

```bash
# Compile and run
!nvcc -o raytrace_cuda raytrace_cuda.cu
!./raytrace_cuda
```

```bash
# Convert to viewable format
!apt-get update -qq && apt-get install -y -qq imagemagick
!convert raytrace_cuda.ppm raytrace_cuda.png
```

```python
# Display result
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

### Method 3: Upload File

1. In Google Colab, enable GPU runtime
2. Upload file:

```python
from google.colab import files
uploaded = files.upload()  # Select raytrace_cuda.cu
```

3. Compile and run:

```bash
!nvcc -o raytrace_cuda raytrace_cuda.cu
!./raytrace_cuda
```

4. View result:

```bash
!apt-get update -qq && apt-get install -y -qq imagemagick
!convert raytrace_cuda.ppm raytrace_cuda.png
```

```python
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

## Local Compilation (If you have NVIDIA GPU)

```bash
# Compile
nvcc -o raytrace_cuda raytrace_cuda.cu

# Run
./raytrace_cuda

# Convert PPM to PNG (optional)
convert raytrace_cuda.ppm raytrace_cuda.png
```

## Expected Output

```
Starting CUDA Ray Tracer...
Image size: 800x600
Using GPU: Tesla T4
Rendering scene on GPU...
Grid size: (50, 38), Block size: (16, 16)
Rendering time: 0.02 seconds
Image saved as raytrace_cuda.ppm
Ray tracing completed!
```

## Performance Comparison

| Version | Time | Speedup |
|---------|------|---------|
| Serial CPU | ~8s | 1x |
| OpenMP (4 cores) | ~2s | 4x |
| MPI (4 processes) | ~2s | 4x |
| **CUDA GPU** | **~0.02s** | **400x** |

## Troubleshooting

**Problem**: "nvcc: command not found"
- **Fix**: Enable GPU runtime in Colab

**Problem**: "No CUDA device found"
- **Fix**: Runtime → Change runtime type → GPU → Save → Restart

**Problem**: Can't view PPM file
- **Fix**: Use ImageMagick to convert to PNG (shown above)

## Key Features

✅ GPU-accelerated ray tracing
✅ 2D thread block parallelization (16x16)
✅ Reflections and lighting effects
✅ ~400x faster than CPU serial version
✅ Optimized for Google Colab
✅ Single-precision floating point for speed

## Image Output

- Resolution: 800x600
- Format: PPM (converts to PNG)
- File size: ~1.4 MB (PPM), ~150 KB (PNG)
- Scene: 5 spheres with reflections and 2 light sources
