#!/usr/bin/env python3
"""
Comprehensive Comparison of Ray Tracer Implementations
Compares Serial, OpenMP, MPI, and CUDA implementations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_data():
    """Load benchmark results from all implementations"""
    
    # OpenMP results
    openmp_df = pd.read_csv('openmp_results.csv')
    openmp_df.columns = ['Threads', 'Time']
    
    # MPI results
    mpi_df = pd.read_csv('mpi_results.csv')
    mpi_df.columns = ['Processes', 'Time']
    # Remove any rows with NaN values
    mpi_df = mpi_df.dropna()
    
    # CUDA results
    cuda_df = pd.read_csv('CUDA/cuda_results.csv')
    
    # Serial baseline - use 1 thread from OpenMP as baseline
    serial_time = openmp_df[openmp_df['Threads'] == 1]['Time'].values[0]
    
    return openmp_df, mpi_df, cuda_df, serial_time

def calculate_speedup(df, time_col, serial_time):
    """Calculate speedup relative to serial baseline"""
    df['Speedup'] = serial_time / df[time_col]
    return df

def calculate_efficiency(df, parallel_col, speedup_col):
    """Calculate parallel efficiency"""
    df['Efficiency'] = (df[speedup_col] / df[parallel_col]) * 100
    return df

def create_comparison_plots(openmp_df, mpi_df, cuda_df, serial_time):
    """Create comprehensive comparison visualizations"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================================
    # Plot 1: Execution Time Comparison (Log Scale)
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot OpenMP
    ax1.plot(openmp_df['Threads'], openmp_df['Time'], 
             marker='o', linewidth=2, markersize=8, label='OpenMP')
    
    # Plot MPI
    ax1.plot(mpi_df['Processes'], mpi_df['Time'], 
             marker='s', linewidth=2, markersize=8, label='MPI')
    
    # Plot CUDA (using ThreadsPerBlock as x-axis)
    ax1.plot(cuda_df['ThreadsPerBlock'], cuda_df['ExecutionTime'], 
             marker='^', linewidth=2, markersize=8, label='CUDA')
    
    # Add serial baseline
    ax1.axhline(y=serial_time, color='r', linestyle='--', 
                linewidth=2, label=f'Serial Baseline ({serial_time:.4f}s)')
    
    ax1.set_xlabel('Number of Threads/Processes/CUDA Threads per Block', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Execution Time Comparison (Linear Scale)
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.plot(openmp_df['Threads'], openmp_df['Time'], 
             marker='o', linewidth=2, markersize=8, label='OpenMP')
    ax2.plot(mpi_df['Processes'], mpi_df['Time'], 
             marker='s', linewidth=2, markersize=8, label='MPI')
    ax2.plot(cuda_df['ThreadsPerBlock'], cuda_df['ExecutionTime'], 
             marker='^', linewidth=2, markersize=8, label='CUDA')
    ax2.axhline(y=serial_time, color='r', linestyle='--', 
                linewidth=2, label=f'Serial Baseline')
    
    ax2.set_xlabel('Number of Threads/Processes/CUDA Threads per Block', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 3: Speedup Comparison
    # ============================================================
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.plot(openmp_df['Threads'], openmp_df['Speedup'], 
             marker='o', linewidth=2, markersize=8, label='OpenMP')
    ax3.plot(mpi_df['Processes'], mpi_df['Speedup'], 
             marker='s', linewidth=2, markersize=8, label='MPI')
    ax3.plot(cuda_df['ThreadsPerBlock'], cuda_df['Speedup'], 
             marker='^', linewidth=2, markersize=8, label='CUDA')
    
    # Add ideal speedup line
    max_threads = max(openmp_df['Threads'].max(), 
                      mpi_df['Processes'].max())
    ideal_x = np.array([1, 2, 4, 8, 16])
    ax3.plot(ideal_x, ideal_x, 'k--', linewidth=2, label='Ideal Speedup', alpha=0.5)
    
    ax3.set_xlabel('Number of Threads/Processes/CUDA Threads per Block', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax3.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 4: Parallel Efficiency
    # ============================================================
    ax4 = plt.subplot(2, 3, 4)
    
    ax4.plot(openmp_df['Threads'], openmp_df['Efficiency'], 
             marker='o', linewidth=2, markersize=8, label='OpenMP')
    ax4.plot(mpi_df['Processes'], mpi_df['Efficiency'], 
             marker='s', linewidth=2, markersize=8, label='MPI')
    
    # CUDA efficiency (relative to threads per block)
    cuda_efficiency = (cuda_df['Speedup'] / cuda_df['ThreadsPerBlock']) * 100
    ax4.plot(cuda_df['ThreadsPerBlock'], cuda_efficiency, 
             marker='^', linewidth=2, markersize=8, label='CUDA')
    
    ax4.axhline(y=100, color='k', linestyle='--', linewidth=2, 
                label='100% Efficiency', alpha=0.5)
    
    ax4.set_xlabel('Number of Threads/Processes/CUDA Threads per Block', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Parallel Efficiency Comparison', fontsize=14, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 5: Bar Chart - Best Performance Comparison
    # ============================================================
    ax5 = plt.subplot(2, 3, 5)
    
    implementations = ['Serial', 'OpenMP\n(16 threads)', 'MPI\n(8 processes)', 'CUDA\n(32 TPB)']
    times = [
        serial_time,
        openmp_df[openmp_df['Threads'] == 16]['Time'].values[0],
        mpi_df[mpi_df['Processes'] == 8]['Time'].values[0],
        cuda_df[cuda_df['ThreadsPerBlock'] == 32]['ExecutionTime'].values[0]
    ]
    
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    bars = ax5.bar(implementations, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.4f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax5.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Best Performance Comparison', fontsize=14, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)
    
    # ============================================================
    # Plot 6: Bar Chart - Maximum Speedup Comparison
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)
    
    speedups = [
        1.0,
        openmp_df['Speedup'].max(),
        mpi_df['Speedup'].max(),
        cuda_df['Speedup'].max()
    ]
    
    bars = ax6.bar(implementations, speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax6.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax6.set_title('Maximum Speedup Comparison', fontsize=14, fontweight='bold')
    ax6.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("Comprehensive comparison plot saved as 'comprehensive_comparison.png'")
    plt.show()

def create_summary_table(openmp_df, mpi_df, cuda_df, serial_time):
    """Create a summary table of all implementations"""
    
    summary_data = {
        'Implementation': ['Serial', 'OpenMP (Best)', 'MPI (Best)', 'CUDA (Best)'],
        'Configuration': [
            'Single Thread',
            f"{openmp_df.loc[openmp_df['Speedup'].idxmax(), 'Threads']:.0f} threads",
            f"{mpi_df.loc[mpi_df['Speedup'].idxmax(), 'Processes']:.0f} processes",
            f"{cuda_df.loc[cuda_df['Speedup'].idxmax(), 'ThreadsPerBlock']:.0f} threads/block ({cuda_df.loc[cuda_df['Speedup'].idxmax(), 'BlockSizeX']:.0f}x{cuda_df.loc[cuda_df['Speedup'].idxmax(), 'BlockSizeY']:.0f})"
        ],
        'Execution Time (s)': [
            serial_time,
            openmp_df['Time'].min(),
            mpi_df['Time'].min(),
            cuda_df['ExecutionTime'].min()
        ],
        'Speedup': [
            1.0,
            openmp_df['Speedup'].max(),
            mpi_df['Speedup'].max(),
            cuda_df['Speedup'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Efficiency (%)'] = (summary_df['Speedup'] / 
                                      [1, 16, 8, 32]) * 100
    
    return summary_df

def print_detailed_statistics(openmp_df, mpi_df, cuda_df, serial_time):
    """Print detailed statistics for all implementations"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON OF RAY TRACER IMPLEMENTATIONS")
    print("="*80)
    
    print(f"\n📊 SERIAL BASELINE")
    print(f"{'='*80}")
    print(f"Execution Time: {serial_time:.4f} seconds")
    
    print(f"\n🔧 OPENMP IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"Best Configuration: {openmp_df.loc[openmp_df['Speedup'].idxmax(), 'Threads']:.0f} threads")
    print(f"Best Time: {openmp_df['Time'].min():.4f} seconds")
    print(f"Best Speedup: {openmp_df['Speedup'].max():.2f}x")
    print(f"Best Efficiency: {openmp_df['Efficiency'].max():.2f}%")
    print(f"Worst Time: {openmp_df['Time'].max():.4f} seconds")
    print(f"Average Time: {openmp_df['Time'].mean():.4f} seconds")
    
    print(f"\n🌐 MPI IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"Best Configuration: {mpi_df.loc[mpi_df['Speedup'].idxmax(), 'Processes']:.0f} processes")
    print(f"Best Time: {mpi_df['Time'].min():.4f} seconds")
    print(f"Best Speedup: {mpi_df['Speedup'].max():.2f}x")
    print(f"Best Efficiency: {mpi_df['Efficiency'].max():.2f}%")
    print(f"Worst Time: {mpi_df['Time'].max():.4f} seconds")
    print(f"Average Time: {mpi_df['Time'].mean():.4f} seconds")
    
    print(f"\n🎮 CUDA IMPLEMENTATION")
    print(f"{'='*80}")
    best_cuda_idx = cuda_df['Speedup'].idxmax()
    print(f"Best Configuration: {cuda_df.loc[best_cuda_idx, 'BlockSizeX']:.0f}x{cuda_df.loc[best_cuda_idx, 'BlockSizeY']:.0f} blocks ({cuda_df.loc[best_cuda_idx, 'ThreadsPerBlock']:.0f} threads/block)")
    print(f"Best Time: {cuda_df['ExecutionTime'].min():.4f} seconds")
    print(f"Best Speedup: {cuda_df['Speedup'].max():.2f}x")
    cuda_best_eff = (cuda_df['Speedup'].max() / cuda_df.loc[best_cuda_idx, 'ThreadsPerBlock']) * 100
    print(f"Best Efficiency: {cuda_best_eff:.2f}%")
    print(f"Worst Time: {cuda_df['ExecutionTime'].max():.4f} seconds")
    print(f"Average Time: {cuda_df['ExecutionTime'].mean():.4f} seconds")
    
    print(f"\n🏆 OVERALL COMPARISON")
    print(f"{'='*80}")
    
    all_best_times = {
        'Serial': serial_time,
        'OpenMP': openmp_df['Time'].min(),
        'MPI': mpi_df['Time'].min(),
        'CUDA': cuda_df['ExecutionTime'].min()
    }
    
    winner = min(all_best_times, key=all_best_times.get)
    print(f"Fastest Implementation: {winner} ({all_best_times[winner]:.4f}s)")
    print(f"Overall Best Speedup: {max(openmp_df['Speedup'].max(), mpi_df['Speedup'].max(), cuda_df['Speedup'].max()):.2f}x (CUDA)")
    
    improvement_serial_to_best = ((serial_time - all_best_times[winner]) / serial_time) * 100
    print(f"Performance Improvement over Serial: {improvement_serial_to_best:.1f}%")
    
    print(f"\n📈 SCALABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"OpenMP Scalability (1→16 threads): {openmp_df[openmp_df['Threads']==16]['Speedup'].values[0]:.2f}x speedup")
    print(f"MPI Scalability (1→8 processes): {mpi_df[mpi_df['Processes']==8]['Speedup'].values[0]:.2f}x speedup")
    print(f"CUDA Scalability (1→32 TPB): {cuda_df[cuda_df['ThreadsPerBlock']==32]['Speedup'].values[0]:.2f}x speedup")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main function to run comprehensive comparison"""
    
    print("Loading benchmark data...")
    openmp_df, mpi_df, cuda_df, serial_time = load_data()
    
    print("Calculating speedup and efficiency metrics...")
    openmp_df = calculate_speedup(openmp_df, 'Time', serial_time)
    openmp_df = calculate_efficiency(openmp_df, 'Threads', 'Speedup')
    
    mpi_df = calculate_speedup(mpi_df, 'Time', serial_time)
    mpi_df = calculate_efficiency(mpi_df, 'Processes', 'Speedup')
    
    print("Creating comprehensive comparison visualizations...")
    create_comparison_plots(openmp_df, mpi_df, cuda_df, serial_time)
    
    print("Generating summary table...")
    summary_df = create_summary_table(openmp_df, mpi_df, cuda_df, serial_time)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Save summary to CSV
    summary_df.to_csv('implementation_comparison_summary.csv', index=False)
    print("\nSummary table saved to 'implementation_comparison_summary.csv'")
    
    # Print detailed statistics
    print_detailed_statistics(openmp_df, mpi_df, cuda_df, serial_time)
    
    # Save detailed results
    print("Saving detailed comparison data...")
    
    # Combine all results
    openmp_detailed = openmp_df.copy()
    openmp_detailed['Implementation'] = 'OpenMP'
    openmp_detailed.rename(columns={'Threads': 'ParallelUnits'}, inplace=True)
    
    mpi_detailed = mpi_df.copy()
    mpi_detailed['Implementation'] = 'MPI'
    mpi_detailed.rename(columns={'Processes': 'ParallelUnits'}, inplace=True)
    
    cuda_detailed = cuda_df.copy()
    cuda_detailed['Implementation'] = 'CUDA'
    cuda_detailed['ParallelUnits'] = cuda_detailed['ThreadsPerBlock']
    cuda_detailed.rename(columns={'ExecutionTime': 'Time'}, inplace=True)
    cuda_detailed = cuda_detailed[['Implementation', 'ParallelUnits', 'Time', 'Speedup']]
    
    all_results = pd.concat([openmp_detailed, mpi_detailed, cuda_detailed], ignore_index=True)
    all_results.to_csv('all_implementations_comparison.csv', index=False)
    print("Detailed comparison saved to 'all_implementations_comparison.csv'")
    
    print("\n✅ Comprehensive comparison complete!")

if __name__ == "__main__":
    main()
