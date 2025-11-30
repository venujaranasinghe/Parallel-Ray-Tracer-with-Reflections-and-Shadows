%%writefile plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def plot_benchmark_results():
    # Read results from CSV
    try:
        df = pd.read_csv('benchmark_results.csv')
    except FileNotFoundError:
        print("Error: benchmark_results.csv not found. Run the CUDA benchmark first.")
        return
    
    print("Benchmark Results:")
    print(df.to_string(index=False))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Threads per Block vs Execution Time
    ax1.plot(df['ThreadsPerBlock'], df['ExecutionTime'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Threads per Block')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Threads per Block vs Execution Time')
    ax1.grid(True, alpha=0.3)
    
    # Annotate the best point
    best_time_idx = df['ExecutionTime'].idxmin()
    ax1.annotate(f"Best: {df.loc[best_time_idx, 'ThreadsPerBlock']} threads\n{df.loc[best_time_idx, 'ExecutionTime']:.4f}s",
                xy=(df.loc[best_time_idx, 'ThreadsPerBlock'], df.loc[best_time_idx, 'ExecutionTime']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Threads per Block vs Speedup
    ax2.plot(df['ThreadsPerBlock'], df['Speedup'], 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Threads per Block')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Threads per Block vs Speedup')
    ax2.grid(True, alpha=0.3)
    
    # Annotate the best point
    best_speedup_idx = df['Speedup'].idxmax()
    ax2.annotate(f"Best: {df.loc[best_speedup_idx, 'ThreadsPerBlock']} threads\n{df.loc[best_speedup_idx, 'Speedup']:.2f}x",
                xy=(df.loc[best_speedup_idx, 'ThreadsPerBlock'], df.loc[best_speedup_idx, 'Speedup']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Plot 3: Block Configuration vs Execution Time (Heatmap style)
    # Create a matrix for block configurations
    config_labels = [f"{x}x{y}" for x, y in zip(df['BlockSizeX'], df['BlockSizeY'])]
    x_pos = np.arange(len(config_labels))
    
    bars = ax3.bar(x_pos, df['ExecutionTime'], color='skyblue', alpha=0.7)
    ax3.set_xlabel('Block Configuration (X x Y)')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Block Configuration vs Execution Time')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(config_labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, df['ExecutionTime']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{time:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Block Configuration vs Speedup
    bars = ax4.bar(x_pos, df['Speedup'], color='lightcoral', alpha=0.7)
    ax4.set_xlabel('Block Configuration (X x Y)')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('Block Configuration vs Speedup')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(config_labels, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, df['Speedup']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Best Execution Time: {df['ExecutionTime'].min():.4f} seconds")
    print(f"Worst Execution Time: {df['ExecutionTime'].max():.4f} seconds")
    print(f"Average Execution Time: {df['ExecutionTime'].mean():.4f} seconds")
    print(f"Best Speedup: {df['Speedup'].max():.2f}x")
    print(f"Optimal Configuration: {config_labels[best_speedup_idx]} "
          f"({df.loc[best_speedup_idx, 'ThreadsPerBlock']} threads/block)")
    
    # Performance improvement analysis
    worst_time = df['ExecutionTime'].max()
    best_time = df['ExecutionTime'].min()
    improvement = ((worst_time - best_time) / worst_time) * 100
    print(f"Performance Improvement: {improvement:.1f}% (from worst to best configuration)")

if __name__ == "__main__":
    plot_benchmark_results()