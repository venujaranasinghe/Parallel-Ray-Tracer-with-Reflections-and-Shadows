#!/bin/bash

# Performance comparison script for Serial vs OpenMP ray tracer

echo "=== Ray Tracer Performance Comparison ==="
echo

# Compile serial version if needed
echo "Compiling serial version..."
cd ../Serial
if [ ! -f raytrace_serial ]; then
    gcc -O3 -lm -o raytrace_serial raytrace_serial.c
fi

# Run serial version
echo "Running serial version..."
time ./raytrace_serial > /dev/null
echo

# Go back to OpenMP directory
cd ../OpenMP

echo "Running OpenMP version with different thread counts..."
echo

for threads in 1 2 4 8; do
    echo "--- Testing with $threads thread(s) ---"
    export OMP_NUM_THREADS=$threads
    time ./raytrace_openmp > /dev/null
    echo
done

echo "Performance test completed!"
echo "Check the generated .ppm files to verify correctness."