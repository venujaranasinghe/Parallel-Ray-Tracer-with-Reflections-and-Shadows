#!/usr/bin/env python3
"""
Generate performance graphs for OpenMP and MPI Ray Tracer benchmarks
This script reads the CSV results and creates graphs showing:
- Execution time vs number of threads/processes
- Speedup vs number of threads/processes
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def read_results(filename):
    """Read benchmark results from CSV file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return None
    return pd.read_csv(filename)

def plot_openmp_results(df):
    """Generate graphs for OpenMP benchmarks"""
    # Get serial time (1 thread)
    serial_time = df[df['Threads'] == 1]['Time(seconds)'].values[0]
    
    # Calculate speedup
    df['Speedup'] = serial_time / df['Time(seconds)']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('OpenMP Ray Tracer Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time
    ax1.plot(df['Threads'], df['Time(seconds)'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Threads', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['Threads'])
    
    # Add values on points
    for i, row in df.iterrows():
        ax1.annotate(f'{row["Time(seconds)"]:.2f}s', 
                    (row['Threads'], row['Time(seconds)']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 2: Speedup
    ax2.plot(df['Threads'], df['Speedup'], 'o-', linewidth=2, markersize=8, color='green', label='Actual Speedup')
    ax2.plot(df['Threads'], df['Threads'], '--', linewidth=2, color='red', alpha=0.5, label='Ideal Speedup')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs Threads', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['Threads'])
    ax2.legend()
    
    # Add values on points
    for i, row in df.iterrows():
        ax2.annotate(f'{row["Speedup"]:.2f}x', 
                    (row['Threads'], row['Speedup']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('openmp_performance.png', dpi=300, bbox_inches='tight')
    print("OpenMP graphs saved as 'openmp_performance.png'")
    
    # Print summary
    print("\n===== OpenMP Performance Summary =====")
    print(df.to_string(index=False))
    print(f"\nBest speedup: {df['Speedup'].max():.2f}x with {df.loc[df['Speedup'].idxmax(), 'Threads']} threads")

def plot_mpi_results(df):
    """Generate graphs for MPI benchmarks"""
    # Get serial time (1 process)
    serial_time = df[df['Processes'] == 1]['Time(seconds)'].values[0]
    
    # Calculate speedup
    df['Speedup'] = serial_time / df['Time(seconds)']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MPI Ray Tracer Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time
    ax1.plot(df['Processes'], df['Time(seconds)'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Number of Processes', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Processes', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['Processes'])
    
    # Add values on points
    for i, row in df.iterrows():
        ax1.annotate(f'{row["Time(seconds)"]:.2f}s', 
                    (row['Processes'], row['Time(seconds)']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot 2: Speedup
    ax2.plot(df['Processes'], df['Speedup'], 'o-', linewidth=2, markersize=8, color='green', label='Actual Speedup')
    ax2.plot(df['Processes'], df['Processes'], '--', linewidth=2, color='red', alpha=0.5, label='Ideal Speedup')
    ax2.set_xlabel('Number of Processes', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs Processes', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['Processes'])
    ax2.legend()
    
    # Add values on points
    for i, row in df.iterrows():
        ax2.annotate(f'{row["Speedup"]:.2f}x', 
                    (row['Processes'], row['Speedup']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('mpi_performance.png', dpi=300, bbox_inches='tight')
    print("MPI graphs saved as 'mpi_performance.png'")
    
    # Print summary
    print("\n===== MPI Performance Summary =====")
    print(df.to_string(index=False))
    print(f"\nBest speedup: {df['Speedup'].max():.2f}x with {df.loc[df['Speedup'].idxmax(), 'Processes']} processes")

def main():
    print("===== Generating Performance Graphs =====\n")
    
    # Process OpenMP results
    openmp_df = read_results('openmp_results.csv')
    if openmp_df is not None:
        plot_openmp_results(openmp_df)
    else:
        print("Skipping OpenMP graphs (no data found)")
    
    print("\n")
    
    # Process MPI results
    mpi_df = read_results('mpi_results.csv')
    if mpi_df is not None:
        plot_mpi_results(mpi_df)
    else:
        print("Skipping MPI graphs (no data found)")
    
    print("\n===== Graph generation complete! =====")

if __name__ == "__main__":
    main()
