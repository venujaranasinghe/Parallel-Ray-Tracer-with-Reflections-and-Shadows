#!/bin/bash

# Quick comparison script - just run both versions once
echo "Quick Performance Comparison:"
echo "============================="

# Run serial version
echo "Running Serial version..."
cd ../Serial
serial_output=$(./raytrace_serial 2>&1)
serial_time=$(echo "$serial_output" | grep "Rendering time:" | sed 's/Rendering time: \([0-9.]*\) seconds/\1/')

# Run OpenMP version with default threads
echo "Running OpenMP version..."
cd ../OpenMP
openmp_output=$(./raytrace_openmp 2>&1)
openmp_time=$(echo "$openmp_output" | grep "Rendering time:" | sed 's/Rendering time: \([0-9.]*\) seconds/\1/')
threads=$(echo "$openmp_output" | grep "Number of threads:" | sed 's/Number of threads: \([0-9]*\)/\1/')

echo
echo "Results:"
echo "--------"
echo "Serial:          ${serial_time} seconds"
echo "OpenMP ($threads threads): ${openmp_time} seconds"

# Calculate speedup
if (( $(echo "$openmp_time > 0" | bc -l) )); then
    speedup=$(echo "scale=2; $serial_time / $openmp_time" | bc)
    echo "Speedup:         ${speedup}x"
    
    efficiency=$(echo "scale=1; $speedup / $threads * 100" | bc)
    echo "Efficiency:      ${efficiency}%"
fi

echo
echo "Images saved as:"
echo "  - Serial: ../Serial/raytrace_serial.ppm"
echo "  - OpenMP: raytrace_openmp.ppm"