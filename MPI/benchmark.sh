#!/bin/bash

# MPI Ray Tracer Benchmark Script
# This script compares performance of the MPI version with different process counts

echo "MPI Ray Tracer Performance Benchmark"
echo "===================================="
echo ""

# Check if the executable exists
if [ ! -f "./raytrace_mpi" ]; then
    echo "Building MPI ray tracer..."
    make clean && make
    if [ $? -ne 0 ]; then
        echo "Build failed. Exiting."
        exit 1
    fi
    echo ""
fi

# Function to run benchmark with given number of processes
run_benchmark() {
    local np=$1
    echo "Running with $np process(es):"
    echo "=============================="
    
    # Remove previous output file
    rm -f raytrace_mpi.ppm
    
    # Run the benchmark
    start_time=$(date +%s.%N)
    mpirun -np $np ./raytrace_mpi
    end_time=$(date +%s.%N)
    
    # Calculate elapsed time
    elapsed=$(echo "$end_time - $start_time" | bc)
    echo "Total execution time: ${elapsed}s"
    echo ""
}

# Test with different numbers of processes
process_counts=(1 2 4 8)

for np in "${process_counts[@]}"; do
    run_benchmark $np
done

echo "Benchmark completed!"
echo "Check raytrace_mpi.ppm for the final rendered image."

# Optional: Compare with serial version if it exists
if [ -f "../Serial/raytrace_serial" ]; then
    echo ""
    echo "Comparing with serial version:"
    echo "=============================="
    cd ../Serial
    start_time=$(date +%s.%N)
    ./raytrace_serial > /dev/null
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    echo "Serial version time: ${elapsed}s"
    cd ../MPI
fi