import matplotlib.pyplot as plt

# Data from your input
matrix_sizes = [64, 128, 256, 512, 1024]
cpu_runtimes = [0.00313, 0.01419, 0.09297, 0.99927, 8.95927]
gpu_runtimes = [0.00568, 0.01265, 0.04628, 0.20169, 0.87069]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot CPU optimizations
plt.plot(matrix_sizes, cpu_runtimes, 'o-', color='red', markersize=10, label='CPU Optimization')

# Plot GPU optimizations
plt.plot(matrix_sizes, gpu_runtimes, 'o-', color='blue', markersize=10, label='GPU Optimization')

# Add labels, title, and legend
plt.xlabel('Matrix Size', fontsize=12)
plt.ylabel('Runtime (s)', fontsize=12)
plt.title('Matrix Multiplication runtime Comparison: CPU vs GPU', fontsize=14)
plt.legend(fontsize=12)

# Save and display the plot
plt.savefig('./assets/optimization_plot.png')
plt.show()