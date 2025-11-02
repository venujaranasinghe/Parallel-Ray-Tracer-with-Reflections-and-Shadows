#!/bin/bash

# Performance Comparison Script
# Compares MPI, OpenMP, and Serial versions

echo "Ray Tracer Performance Comparison"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to measure execution time
measure_time() {
    local cmd="$1"
    local label="$2"
    
    echo -e "${BLUE}Testing: $label${NC}"
    
    start_time=$(date +%s.%N)
    eval $cmd > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc -l)
    printf "Time: %.2f seconds\n" $elapsed
    echo $elapsed
}

# Build all versions
echo -e "${YELLOW}Building all versions...${NC}"

# Build Serial
if [ -d "../Serial" ]; then
    cd ../Serial
    if [ -f "raytrace_serial.c" ] && [ ! -f "raytrace_serial" ]; then
        gcc -O3 -o raytrace_serial raytrace_serial.c -lm
    fi
    cd ../MPI
fi

# Build OpenMP
if [ -d "../OpenMP" ]; then
    cd ../OpenMP
    if [ -f "raytrace_openmp.c" ] && [ ! -f "raytrace_openmp" ]; then
        gcc -O3 -fopenmp -o raytrace_openmp raytrace_openmp.c -lm
    fi
    cd ../MPI
fi

# Build MPI
make clean && make

echo ""
echo -e "${GREEN}Starting Performance Tests...${NC}"
echo ""

# Test Serial Version
if [ -f "../Serial/raytrace_serial" ]; then
    serial_time=$(measure_time "cd ../Serial && ./raytrace_serial" "Serial Version")
    echo ""
else
    echo -e "${RED}Serial version not found${NC}"
    serial_time=0
    echo ""
fi

# Test OpenMP Version
if [ -f "../OpenMP/raytrace_openmp" ]; then
    export OMP_NUM_THREADS=4
    openmp_time=$(measure_time "cd ../OpenMP && ./raytrace_openmp" "OpenMP Version (4 threads)")
    echo ""
else
    echo -e "${RED}OpenMP version not found${NC}"
    openmp_time=0  
    echo ""
fi

# Test MPI Versions
mpi_1_time=$(measure_time "mpirun -np 1 ./raytrace_mpi" "MPI Version (1 process)")
echo ""

mpi_2_time=$(measure_time "mpirun -np 2 ./raytrace_mpi" "MPI Version (2 processes)")
echo ""

mpi_4_time=$(measure_time "mpirun -np 4 ./raytrace_mpi" "MPI Version (4 processes)")
echo ""

mpi_8_time=$(measure_time "mpirun -np 8 ./raytrace_mpi" "MPI Version (8 processes)")
echo ""

# Summary
echo -e "${YELLOW}Performance Summary:${NC}"
echo "===================="

if (( $(echo "$serial_time > 0" | bc -l) )); then
    printf "Serial:           %.2f seconds (baseline)\n" $serial_time
    
    if (( $(echo "$openmp_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $openmp_time" | bc -l)
        printf "OpenMP (4 cores): %.2f seconds (%.2fx speedup)\n" $openmp_time $speedup
    fi
    
    if (( $(echo "$mpi_1_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $mpi_1_time" | bc -l)
        printf "MPI (1 process):  %.2f seconds (%.2fx speedup)\n" $mpi_1_time $speedup
    fi
    
    if (( $(echo "$mpi_2_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $mpi_2_time" | bc -l)
        printf "MPI (2 processes):%.2f seconds (%.2fx speedup)\n" $mpi_2_time $speedup
    fi
    
    if (( $(echo "$mpi_4_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $mpi_4_time" | bc -l)
        printf "MPI (4 processes):%.2f seconds (%.2fx speedup)\n" $mpi_4_time $speedup
    fi
    
    if (( $(echo "$mpi_8_time > 0" | bc -l) )); then
        speedup=$(echo "scale=2; $serial_time / $mpi_8_time" | bc -l)
        printf "MPI (8 processes):%.2f seconds (%.2fx speedup)\n" $mpi_8_time $speedup
    fi
else
    printf "MPI (1 process):  %.2f seconds\n" $mpi_1_time
    printf "MPI (2 processes):%.2f seconds\n" $mpi_2_time  
    printf "MPI (4 processes):%.2f seconds\n" $mpi_4_time
    printf "MPI (8 processes):%.2f seconds\n" $mpi_8_time
fi

echo ""
echo -e "${GREEN}Comparison completed!${NC}"