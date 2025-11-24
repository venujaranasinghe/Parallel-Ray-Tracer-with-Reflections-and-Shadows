# CUDA Ray Tracer - Google Colab Instructions

## 🚀 How to Run on Google Colab

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### Step 2: Create a New Notebook

### Step 3: Enable GPU Runtime
1. Click on **Runtime** → **Change runtime type**
2. Select **GPU** from the Hardware accelerator dropdown
3. Click **Save**

### Step 4: Upload the CUDA File

Create a new code cell and run:

```python
# Upload the raytrace_cuda.cu file
from google.colab import files
uploaded = files.upload()
```

Then select and upload `raytrace_cuda.cu` from your local machine.

### Step 5: Compile the CUDA Code

In a new cell, run:

```bash
%%bash
nvcc -o raytrace_cuda raytrace_cuda.cu
```

### Step 6: Run the Ray Tracer

In a new cell, run:

```bash
!./raytrace_cuda
```

### Step 7: Download the Output Image

In a new cell, run:

```python
from google.colab import files
files.download('raytrace_cuda.ppm')
```

### Step 8 (Optional): Convert PPM to PNG for easier viewing

```bash
%%bash
# Install ImageMagick if needed
apt-get update && apt-get install -y imagemagick

# Convert PPM to PNG
convert raytrace_cuda.ppm raytrace_cuda.png
```

Then display the image in Colab:

```python
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

---

## 📝 Complete Colab Notebook Code

Copy and paste this entire script into a single Colab notebook:

```python
# ===== CELL 1: Check GPU =====
!nvidia-smi

# ===== CELL 2: Upload CUDA file =====
from google.colab import files
print("Upload raytrace_cuda.cu file:")
uploaded = files.upload()

# ===== CELL 3: Compile =====
%%bash
nvcc -o raytrace_cuda raytrace_cuda.cu
echo "Compilation successful!"

# ===== CELL 4: Run Ray Tracer =====
!./raytrace_cuda

# ===== CELL 5: Convert and Display =====
%%bash
apt-get update -qq && apt-get install -y -qq imagemagick
convert raytrace_cuda.ppm raytrace_cuda.png

# ===== CELL 6: Display Image =====
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))

# ===== CELL 7: Download (Optional) =====
from google.colab import files
files.download('raytrace_cuda.ppm')
files.download('raytrace_cuda.png')
```

---

## 🎯 Quick One-Command Version

If you want everything automated, use this single cell after uploading the file:

```bash
%%bash
# Compile
nvcc -o raytrace_cuda raytrace_cuda.cu

# Run
./raytrace_cuda

# Convert to PNG
apt-get update -qq && apt-get install -y -qq imagemagick
convert raytrace_cuda.ppm raytrace_cuda.png

echo "Done! Check raytrace_cuda.png"
```

Then display:

```python
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

---

## 🔧 Alternative: Clone from GitHub

If you've pushed the code to GitHub:

```bash
%%bash
# Clone your repository
git clone https://github.com/venujaranasinghe/Parallel-Ray-Tracer-with-Reflections-and-Shadows.git

# Navigate to CUDA directory
cd Parallel-Ray-Tracer-with-Reflections-and-Shadows/CUDA

# Compile
nvcc -o raytrace_cuda raytrace_cuda.cu

# Run
./raytrace_cuda
```

---

## 📊 Performance Benchmarking on Colab

To compare with CPU version:

```python
# Create a simple serial version in Colab
%%writefile raytrace_test.c
// Minimal test code to compare timings
// (You can paste a simplified version here)

# Compile and run
!gcc -O3 -o raytrace_serial raytrace_test.c -lm
!time ./raytrace_serial

# Compare with CUDA
!time ./raytrace_cuda
```

---

## 🐛 Troubleshooting

### Issue: "nvcc: command not found"
**Solution**: Make sure you've enabled GPU runtime (Step 3)

### Issue: "No CUDA-capable device found"
**Solution**: 
1. Go to Runtime → Change runtime type
2. Select GPU as hardware accelerator
3. Restart runtime

### Issue: Compilation errors
**Solution**: Make sure you uploaded the complete `.cu` file correctly

### Issue: Can't view PPM file
**Solution**: Use the ImageMagick conversion to PNG as shown in Step 8

---

## 📈 Expected Output

```
Starting CUDA Ray Tracer...
Image size: 800x600
Using GPU: Tesla T4 (or similar)
Rendering scene on GPU...
Grid size: (50, 38), Block size: (16, 16)
Rendering time: 0.02 seconds
Image saved as raytrace_cuda.ppm
Ray tracing completed!
```

---

## 🎨 What You'll See

The rendered image will show:
- **Red sphere** (center-left)
- **Green sphere** (right)
- **Blue sphere** (left)
- **Yellow sphere** (top-center)
- **Gray floor** (bottom)
- **Reflections** on sphere surfaces
- **Lighting effects** from two light sources
- **Gradient background** (blue sky effect)

---

## ⚡ Performance Notes

- **GPU**: ~0.01-0.05 seconds (depending on Colab GPU)
- **Serial CPU**: ~8-10 seconds
- **Expected Speedup**: 100-500x faster than serial!

The CUDA version uses:
- 2D thread blocks (16x16)
- Parallel processing of all pixels simultaneously
- Optimized memory transfers
- GPU-accelerated math operations

---

## 📦 File Size

- Source code: ~10 KB
- Output PPM: ~1.4 MB
- Output PNG: ~100-200 KB (after conversion)

---

## 🔄 Modifying Parameters

To change image resolution, edit these lines in `raytrace_cuda.cu`:

```c
#define IMAGE_WIDTH 800   // Change to 1920 for Full HD
#define IMAGE_HEIGHT 600  // Change to 1080 for Full HD
```

Note: Larger resolutions will take slightly longer but still be very fast on GPU!

---

## 💡 Tips for Best Results

1. **Use GPU runtime** - Essential for CUDA code
2. **Check GPU allocation** - Run `!nvidia-smi` first
3. **Monitor memory** - Large images may require more GPU memory
4. **Convert to PNG** - Easier to view than PPM format
5. **Download results** - PPM files can be used elsewhere

---

## 🎓 Educational Notes

This CUDA implementation demonstrates:
- **Parallel ray tracing** across GPU cores
- **CUDA kernel programming** with 2D thread blocks
- **Memory management** (host ↔ device transfers)
- **GPU optimization** for graphics applications
- **Iterative algorithms** on GPU (avoiding recursion limits)

Perfect for learning GPU programming and parallel computing concepts!