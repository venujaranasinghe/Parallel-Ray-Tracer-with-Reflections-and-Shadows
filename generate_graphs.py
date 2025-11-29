#!/usr/bin/env python3
"""
Performance Graph Generator for OpenMP and MPI Ray Tracer
Generates execution time and speedup graphs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_openmp_graphs():
    """Generate OpenMP performance graphs"""
    
    # Read OpenMP results
    results_file = 'OpenMP/benchmark_results/openmp_results.csv'
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        print("Please run benchmark_detailed.sh first")
        return
    
    df = pd.read_csv(results_file)
    threads = df['Threads'].values
    exec_time = df['ExecutionTime(s)'].values
    
    # Calculate speedup (relative to 1 thread)
    baseline_time = exec_time[0]
    speedup = baseline_time / exec_time
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Threads vs Execution Time
    ax1.plot(threads, exec_time, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Execution Time')
    ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('OpenMP: Number of Threads vs Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(threads)
    ax1.legend(fontsize=10)
    
    # Add value labels on points
    for i, (t, et) in enumerate(zip(threads, exec_time)):
        ax1.annotate(f'{et:.2f}s', (t, et), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Threads vs Speedup
    ax2.plot(threads, speedup, 's-', color='#A23B72', linewidth=2, markersize=8, label='Speedup')
    # Add ideal speedup line
    ideal_speedup = threads
    ax2.plot(threads, ideal_speedup, '--', color='gray', linewidth=1.5, alpha=0.7, label='Ideal Speedup')
    ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('OpenMP: Number of Threads vs Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(threads)
    ax2.legend(fontsize=10)
    
    # Add value labels on points
    for i, (t, s) in enumerate(zip(threads, speedup)):
        ax2.annotate(f'{s:.2f}x', (t, s), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('benchmark_results', exist_ok=True)
    output_file = 'benchmark_results/openmp_performance_graphs.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ OpenMP graphs saved to: {output_file}")
    
    # Print statistics
    print("\n--- OpenMP Performance Statistics ---")
    print(f"Baseline (1 thread): {baseline_time:.2f}s")
    print(f"\nThread Count | Exec Time | Speedup | Efficiency")
    print("-" * 50)
    for t, et, sp in zip(threads, exec_time, speedup):
        efficiency = (sp / t) * 100
        print(f"{t:12d} | {et:9.2f}s | {sp:7.2f}x | {efficiency:6.1f}%")
    
    return threads, exec_time, speedup


def generate_mpi_graphs():
    """Generate MPI performance graphs"""
    
    # Read MPI results
    results_file = 'MPI/benchmark_results/mpi_results.csv'
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        print("Please run benchmark_detailed.sh first")
        return
    
    df = pd.read_csv(results_file)
    processes = df['Processes'].values
    exec_time = df['ExecutionTime(s)'].values
    
    # Calculate speedup (relative to 1 process)
    baseline_time = exec_time[0]
    speedup = baseline_time / exec_time
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Processes vs Execution Time
    ax1.plot(processes, exec_time, 'o-', color='#F18F01', linewidth=2, markersize=8, label='Execution Time')
    ax1.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('MPI: Number of Processes vs Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(processes)
    ax1.legend(fontsize=10)
    
    # Add value labels on points
    for i, (p, et) in enumerate(zip(processes, exec_time)):
        ax1.annotate(f'{et:.2f}s', (p, et), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Processes vs Speedup
    ax2.plot(processes, speedup, 's-', color='#C73E1D', linewidth=2, markersize=8, label='Speedup')
    # Add ideal speedup line
    ideal_speedup = processes
    ax2.plot(processes, ideal_speedup, '--', color='gray', linewidth=1.5, alpha=0.7, label='Ideal Speedup')
    ax2.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('MPI: Number of Processes vs Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(processes)
    ax2.legend(fontsize=10)
    
    # Add value labels on points
    for i, (p, s) in enumerate(zip(processes, speedup)):
        ax2.annotate(f'{s:.2f}x', (p, s), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('benchmark_results', exist_ok=True)
    output_file = 'benchmark_results/mpi_performance_graphs.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ MPI graphs saved to: {output_file}")
    
    # Print statistics
    print("\n--- MPI Performance Statistics ---")
    print(f"Baseline (1 process): {baseline_time:.2f}s")
    print(f"\nProcess Count | Exec Time | Speedup | Efficiency")
    print("-" * 50)
    for p, et, sp in zip(processes, exec_time, speedup):
        efficiency = (sp / p) * 100
        print(f"{p:13d} | {et:9.2f}s | {sp:7.2f}x | {efficiency:6.1f}%")
    
    return processes, exec_time, speedup


def main():
    print("=" * 60)
    print("Performance Graph Generator")
    print("=" * 60)
    print()
    
    # Generate OpenMP graphs
    print("Generating OpenMP performance graphs...")
    openmp_data = generate_openmp_graphs()
    
    print("\n" + "=" * 60)
    print()
    
    # Generate MPI graphs
    print("Generating MPI performance graphs...")
    mpi_data = generate_mpi_graphs()
    
    print("\n" + "=" * 60)
    print("\n✓ All graphs generated successfully!")
    print("\nOutput files:")
    print("  - benchmark_results/openmp_performance_graphs.png")
    print("  - benchmark_results/mpi_performance_graphs.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
