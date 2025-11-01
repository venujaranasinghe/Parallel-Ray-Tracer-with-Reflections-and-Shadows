#!/bin/bash

# Comprehensive performance comparison script
# Compares Serial vs OpenMP ray tracer performance

echo "=========================================="
echo "  Ray Tracer Performance Comparison"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a command and extract timing
run_and_time() {
    local output=$($1 2>&1)
    local time=$(echo "$output" | grep "Rendering time:" | sed 's/Rendering time: \([0-9.]*\) seconds/\1/')
    echo "$time"
}

# Compile both versions
echo -e "${BLUE}Compiling programs...${NC}"

# Compile serial version
cd ../Serial
if gcc -O3 -lm -o raytrace_serial raytrace_serial.c 2>/dev/null; then
    echo "✓ Serial version compiled successfully"
else
    echo "✗ Failed to compile serial version"
    exit 1
fi

# Compile OpenMP version
cd ../OpenMP
if make clean > /dev/null 2>&1 && make > /dev/null 2>&1; then
    echo "✓ OpenMP version compiled successfully"
else
    echo "✗ Failed to compile OpenMP version"
    exit 1
fi

echo

# Run serial version
echo -e "${YELLOW}Running Serial Version:${NC}"
cd ../Serial
serial_time=$(run_and_time "./raytrace_serial")
echo "Serial rendering time: ${serial_time} seconds"
echo

# Test OpenMP with different thread counts
cd ../OpenMP
echo -e "${YELLOW}Running OpenMP Version with different thread counts:${NC}"

declare -a thread_counts=(1 2 4 8)
declare -a openmp_times=()

for threads in "${thread_counts[@]}"; do
    echo -n "Testing with $threads thread(s)... "
    export OMP_NUM_THREADS=$threads
    time=$(run_and_time "./raytrace_openmp")
    openmp_times+=($time)
    echo "${time} seconds"
done

echo

# Calculate and display speedups
echo -e "${GREEN}Performance Analysis:${NC}"
echo "----------------------------------------"
printf "%-15s %-12s %-10s\n" "Configuration" "Time (s)" "Speedup"
echo "----------------------------------------"
printf "%-15s %-12s %-10s\n" "Serial" "$serial_time" "1.00x"

for i in "${!thread_counts[@]}"; do
    threads=${thread_counts[$i]}
    time=${openmp_times[$i]}
    if (( $(echo "$time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $time" | bc)
        printf "%-15s %-12s %-10s\n" "OpenMP ($threads threads)" "$time" "${speedup}x"
    fi
done

echo "----------------------------------------"

# Find best performance
best_time=${openmp_times[0]}
best_threads=1
for i in "${!openmp_times[@]}"; do
    if (( $(echo "${openmp_times[$i]} < $best_time" | bc -l) )); then
        best_time=${openmp_times[$i]}
        best_threads=${thread_counts[$i]}
    fi
done

if (( $(echo "$best_time > 0" | bc -l) )); then
    max_speedup=$(echo "scale=2; $serial_time / $best_time" | bc)
    echo -e "${GREEN}Best Performance:${NC} ${best_threads} threads with ${max_speedup}x speedup"
fi

echo
echo -e "${BLUE}Files generated:${NC}"
echo "  - ../Serial/raytrace_serial.ppm"
echo "  - raytrace_openmp.ppm"
echo
echo "You can compare the images to verify correctness!"

# Calculate efficiency for best case
if (( best_threads > 1 )) && (( $(echo "$best_time > 0" | bc -l) )); then
    efficiency=$(echo "scale=1; $max_speedup / $best_threads * 100" | bc)
    echo -e "${YELLOW}Parallel Efficiency:${NC} ${efficiency}% (${max_speedup}x speedup / ${best_threads} threads)"
fi

echo
echo "Performance comparison completed!"