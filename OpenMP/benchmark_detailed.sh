#!/bin/bash

# OpenMP Detailed Benchmark Script
# Tests with 1, 2, 4, 8, 16 threads

echo "====================================="
echo "OpenMP Ray Tracer Benchmark"
echo "====================================="
echo ""

# Create results directory
mkdir -p benchmark_results

# Output file for CSV results
RESULTS_FILE="benchmark_results/openmp_results.csv"
echo "Threads,ExecutionTime(s)" > $RESULTS_FILE

# Thread counts to test
THREAD_COUNTS=(1 2 4 8 16)

echo "Compiling OpenMP version..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1

if [ ! -f "./raytrace_openmp" ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run benchmarks
for threads in "${THREAD_COUNTS[@]}"
do
    echo "Running with $threads thread(s)..."
    export OMP_NUM_THREADS=$threads
    
    # Run the program and capture output
    output=$(./raytrace_openmp)
    
    # Extract execution time from output
    time=$(echo "$output" | grep "Rendering time:" | awk '{print $3}')
    
    echo "  Threads: $threads, Time: ${time}s"
    echo "$threads,$time" >> $RESULTS_FILE
    echo ""
done

echo "====================================="
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "====================================="
echo ""
echo "Summary:"
cat $RESULTS_FILE
