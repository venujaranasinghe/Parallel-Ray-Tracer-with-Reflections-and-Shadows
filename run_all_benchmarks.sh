#!/bin/bash

# Complete benchmarking pipeline - runs everything and generates graphs
# This is a convenience script that runs all benchmarks and creates graphs in one go

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Complete Ray Tracer Performance Benchmarking Pipeline     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python dependencies are installed
echo "Checking dependencies..."
if ! python3 -c "import pandas, matplotlib" 2>/dev/null; then
    echo "⚠️  Warning: Python dependencies not found"
    echo "    Please install: pip3 install pandas matplotlib"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Dependencies OK"
echo ""

# Step 1: OpenMP Benchmarks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Running OpenMP benchmarks..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./benchmark_openmp.sh
if [ $? -ne 0 ]; then
    echo "❌ OpenMP benchmark failed!"
    exit 1
fi
echo ""

# Step 2: MPI Benchmarks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Running MPI benchmarks..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./benchmark_mpi.sh
if [ $? -ne 0 ]; then
    echo "❌ MPI benchmark failed!"
    exit 1
fi
echo ""

# Step 3: Generate Graphs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Generating performance graphs..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 generate_graphs.py
if [ $? -ne 0 ]; then
    echo "❌ Graph generation failed!"
    exit 1
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ All Benchmarks Complete!                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Generated files:"
echo "   • openmp_performance.png   - OpenMP performance graphs"
echo "   • mpi_performance.png      - MPI performance graphs"
echo "   • openmp_results.csv       - OpenMP raw data"
echo "   • mpi_results.csv          - MPI raw data"
echo ""
echo "📁 View your graphs now!"
echo ""

# Optionally open the graphs (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open graphs now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open openmp_performance.png
        open mpi_performance.png
    fi
fi

echo "Done! 🎉"
