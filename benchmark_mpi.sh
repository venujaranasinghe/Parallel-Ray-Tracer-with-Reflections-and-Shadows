#!/bin/bash

# Benchmark MPI Ray Tracer with different process counts
# This script compiles and runs the MPI version with 1, 2, 4, 8, and 16 processes
# Results are saved to mpi_results.csv

echo "===== MPI Ray Tracer Benchmark ====="
echo ""

# Compile the MPI version
echo "Compiling MPI version..."
cd MPI
mpicc -o raytrace_mpi raytrace_mpi.c -lm -O3
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"
echo ""

# Create results file
RESULTS_FILE="../mpi_results.csv"
echo "Processes,Time(seconds)" > $RESULTS_FILE

# Array of process counts to test
PROCESS_COUNTS=(1 2 4 8 16)

echo "Running benchmarks..."
echo ""

for processes in "${PROCESS_COUNTS[@]}"
do
    echo "Running with $processes process(es)..."
    
    # Run the program and extract the time
    output=$(mpirun -np $processes ./raytrace_mpi 2>&1)
    time=$(echo "$output" | grep "Total execution time:" | awk '{print $4}')
    
    echo "  Time: $time seconds"
    echo "$processes,$time" >> $RESULTS_FILE
    echo ""
done

cd ..

echo "Benchmark complete! Results saved to mpi_results.csv"
echo ""
echo "Results summary:"
cat mpi_results.csv
