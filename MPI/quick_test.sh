#!/bin/bash

# Quick test script for MPI ray tracer
echo "Testing MPI Ray Tracer..."
echo "========================="

# Test with 2 processes
echo "Running with 2 processes:"
mpirun -np 2 ./raytrace_mpi

# Check if output file was created
if [ -f "raytrace_mpi.ppm" ]; then
    echo "Success! Output file 'raytrace_mpi.ppm' was created."
    ls -la raytrace_mpi.ppm
else
    echo "Error: Output file was not created."
    exit 1
fi

echo ""
echo "Test completed successfully!"