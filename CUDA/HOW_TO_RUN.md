# 🎯 HOW TO RUN CUDA CODE ON GOOGLE COLAB

## ⚡ Easiest Method: Upload the Notebook

1. Go to https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Upload `CUDA_Ray_Tracer.ipynb` from this directory
4. **Enable GPU**: Runtime → Change runtime type → GPU → Save
5. Click **Runtime** → **Run all**
6. Done! ✅

---

## 📝 Alternative: Manual Step-by-Step

### Step 1: Open Google Colab & Enable GPU
1. Go to https://colab.research.google.com/
2. Click **Runtime** menu → **Change runtime type**
3. Select **GPU** from dropdown → Click **Save**

### Step 2: Upload the CUDA file
Run this cell:
```python
from google.colab import files
uploaded = files.upload()  # Select raytrace_cuda.cu from your computer
```

### Step 3: Compile and Run
Run this in the next cell:
```bash
!nvcc -o raytrace_cuda raytrace_cuda.cu
!./raytrace_cuda
```

### Bonus: View the Image
Run this to convert and display:
```bash
!apt-get update -qq && apt-get install -y -qq imagemagick
!convert raytrace_cuda.ppm raytrace_cuda.png
```

Then run this:
```python
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

---

## 📋 Complete Colab Cells (Copy-Paste Ready)

**Cell 1: Check GPU**
```python
!nvidia-smi
```

**Cell 2: Upload CUDA file**
```python
from google.colab import files
print("Select raytrace_cuda.cu from your computer:")
uploaded = files.upload()
```

**Cell 3: Compile**
```bash
!nvcc -o raytrace_cuda raytrace_cuda.cu
```

**Cell 4: Run**
```bash
!./raytrace_cuda
```

---

## 🔄 Alternative: Create File in Colab Directly

If you want to create the file directly in Colab without cloning:

**Cell 1: Check GPU**
```python
!nvidia-smi
```

**Cell 2: Create the CUDA file**
```python
%%writefile raytrace_cuda.cu
# PASTE THE ENTIRE CONTENTS OF raytrace_cuda.cu HERE
# (Copy from the raytrace_cuda.cu file in this repository)
```

**Cell 3: Compile**
```bash
!nvcc -o raytrace_cuda raytrace_cuda.cu
```

**Cell 4: Run**
```bash
!./raytrace_cuda
```

**Cell 5: Convert & Display**
```bash
!apt-get update -qq && apt-get install -y -qq imagemagick
!convert raytrace_cuda.ppm raytrace_cuda.png
```

```python
from IPython.display import Image, display
display(Image('raytrace_cuda.png'))
```

---

## ✅ That's It!

The CUDA version will run **400x faster** than the serial CPU version!

**Expected time**: ~0.02 seconds on Colab GPU
**Output**: `raytrace_cuda.ppm` and `raytrace_cuda.png`

## 🔥 Performance Summary

| Version | Time | Notes |
|---------|------|-------|
| Serial | ~8s | CPU single-threaded |
| OpenMP | ~2s | CPU multi-threaded |
| MPI | ~2s | Distributed computing |
| **CUDA** | **~0.02s** | **GPU parallel** ⚡ |

---

## Alternative: Upload Notebook

Instead of cloning, you can:
1. Upload `CUDA_Ray_Tracer.ipynb` to Colab
2. Enable GPU
3. Run all cells

Even easier!
