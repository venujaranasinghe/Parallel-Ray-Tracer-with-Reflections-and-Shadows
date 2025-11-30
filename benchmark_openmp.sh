#!/bin/bash

# Benchmark OpenMP Ray Tracer with different thread counts
# This script compiles and runs the OpenMP version with 1, 2, 4, 8, and 16 threads
# Results are saved to openmp_results.csv

echo "===== OpenMP Ray Tracer Benchmark ====="
echo ""

# Compile the OpenMP version
echo "Compiling OpenMP version..."
cd OpenMP

# macOS-specific OpenMP compilation with libomp
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Use Homebrew's libomp on macOS
    LIBOMP_PREFIX=$(brew --prefix libomp 2>/dev/null || echo "/opt/homebrew/opt/libomp")
    gcc -o raytrace_openmp raytrace_openmp.c -lm -I${LIBOMP_PREFIX}/include -L${LIBOMP_PREFIX}/lib -Xpreprocessor -fopenmp -lomp -O3
else
    # Standard compilation for Linux
    gcc -o raytrace_openmp raytrace_openmp.c -lm -fopenmp -O3
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"
echo ""

# Create results file
RESULTS_FILE="../openmp_results.csv"
echo "Threads,Time(seconds)" > $RESULTS_FILE

# Array of thread counts to test
THREAD_COUNTS=(1 2 4 8 16)

echo "Running benchmarks..."
echo ""

for threads in "${THREAD_COUNTS[@]}"
do
    echo "Running with $threads thread(s)..."
    
    # Run the program and extract the time
    output=$(./raytrace_openmp $threads 2>&1)
    time=$(echo "$output" | grep "Rendering time:" | awk '{print $3}')
    
    echo "  Time: $time seconds"
    echo "$threads,$time" >> $RESULTS_FILE
    echo ""
done

cd ..

echo "Benchmark complete! Results saved to openmp_results.csv"
echo ""
echo "Results summary:"
cat openmp_results.csv
