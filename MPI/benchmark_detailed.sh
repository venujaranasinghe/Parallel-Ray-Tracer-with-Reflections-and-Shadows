#!/bin/bash

# MPI Detailed Benchmark Script
# Tests with 1, 2, 4, 8, 16 processes

echo "====================================="
echo "MPI Ray Tracer Benchmark"
echo "====================================="
echo ""

# Create results directory
mkdir -p benchmark_results

# Output file for CSV results
RESULTS_FILE="benchmark_results/mpi_results.csv"
echo "Processes,ExecutionTime(s)" > $RESULTS_FILE

# Process counts to test
PROCESS_COUNTS=(1 2 4 8 16)

echo "Compiling MPI version..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1

if [ ! -f "./raytrace_mpi" ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run benchmarks
for procs in "${PROCESS_COUNTS[@]}"
do
    echo "Running with $procs process(es)..."
    
    # Run the program and capture output (allow oversubscription for testing)
    output=$(mpirun --oversubscribe -np $procs ./raytrace_mpi)
    
    # Extract execution time from output
    time=$(echo "$output" | grep "Rendering time:" | awk '{print $3}')
    
    echo "  Processes: $procs, Time: ${time}s"
    echo "$procs,$time" >> $RESULTS_FILE
    echo ""
done

echo "====================================="
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "====================================="
echo ""
echo "Summary:"
cat $RESULTS_FILE
