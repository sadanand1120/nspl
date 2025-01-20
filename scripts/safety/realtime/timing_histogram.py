import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mode
import os


def plot_execution_times(file_path):
    # Load execution times from the text file
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Convert data to a list of floats
    execution_times = [float(line.strip()) for line in data]

    # Convert to NumPy array for statistical analysis
    execution_times_array = np.array(execution_times)

    # Calculate statistics
    mean_time = np.mean(execution_times_array)
    median_time = np.median(execution_times_array)
    # mode_time = mode(execution_times_array).mode[0]

    # Plotting
    plt.figure()

    # Histogram with KDE
    sns.histplot(execution_times, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}')
    plt.axvline(median_time, color='green', linestyle='--', label=f'Median: {median_time:.2f}')
    # plt.axvline(mode_time, color='purple', linestyle='--', label=f'Mode: {mode_time:.2f}')
    plt.title(f'Histogram of Execution Times accross {len(execution_times)} samples')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot image in the same directory as the text file
    plot_filename = os.path.splitext(file_path)[0] + '_plot.png'
    plt.savefig(plot_filename)
    plt.close()


def process_all_files_in_directory(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            plot_execution_times(file_path)


# Directory containing the text files
directory_path = '/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/timings'

# Process all text files in the specified directory
process_all_files_in_directory(directory_path)
