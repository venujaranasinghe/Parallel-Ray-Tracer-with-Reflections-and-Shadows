#!/bin/bash

# Screenshot Helper Script
# This script runs the configurations you need to screenshot

echo "======================================================================"
echo "SCREENSHOT HELPER - OpenMP & MPI Evaluations"
echo "======================================================================"
echo ""
echo "This script will run each configuration one at a time."
echo "After each execution, take a screenshot of the terminal output."
echo "Press ENTER to continue to the next configuration."
echo ""
echo "======================================================================"
echo ""

# OpenMP Screenshots
echo "============ OPENMP SCREENSHOTS ============"
echo ""

echo "--- Screenshot 1: OpenMP with 1 thread ---"
echo "Capture: Terminal showing 'Number of threads: 1' and execution time"
read -p "Press ENTER when ready to run... "
cd OpenMP
export OMP_NUM_THREADS=1
./raytrace_openmp
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

echo "--- Screenshot 2: OpenMP with 4 threads ---"
echo "Capture: Terminal showing 'Number of threads: 4' and execution time"
read -p "Press ENTER when ready to run... "
export OMP_NUM_THREADS=4
./raytrace_openmp
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

echo "--- Screenshot 3: OpenMP with 16 threads ---"
echo "Capture: Terminal showing 'Number of threads: 16' and execution time"
read -p "Press ENTER when ready to run... "
export OMP_NUM_THREADS=16
./raytrace_openmp
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

cd ..

# MPI Screenshots
echo "============ MPI SCREENSHOTS ============"
echo ""

echo "--- Screenshot 4: MPI with 1 process ---"
echo "Capture: Terminal showing 'Starting MPI Ray Tracer with 1 processes'"
read -p "Press ENTER when ready to run... "
cd MPI
mpirun -np 1 ./raytrace_mpi
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

echo "--- Screenshot 5: MPI with 4 processes ---"
echo "Capture: Terminal showing 'Starting MPI Ray Tracer with 4 processes'"
read -p "Press ENTER when ready to run... "
mpirun -np 4 ./raytrace_mpi
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

echo "--- Screenshot 6: MPI with 16 processes ---"
echo "Capture: Terminal showing 'Starting MPI Ray Tracer with 16 processes'"
read -p "Press ENTER when ready to run... "
mpirun --oversubscribe -np 16 ./raytrace_mpi
echo ""
read -p "Did you take the screenshot? Press ENTER to continue... "
echo ""

cd ..

echo "======================================================================"
echo "✓ All terminal executions complete!"
echo ""
echo "Remaining screenshots needed:"
echo "  7. Open: benchmark_results/openmp_performance_graphs.png"
echo "  8. Open: benchmark_results/mpi_performance_graphs.png"
echo ""
echo "Then proceed to CUDA evaluation (see EVALUATION_GUIDE.md)"
echo "======================================================================"
