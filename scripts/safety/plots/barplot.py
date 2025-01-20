import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

methods = [
    'Synapse',
    'SF-RGB-b0',
    'SF-RGB-b5',
    'SF-RGBD-b0',
    'SF-RGBD-b5',
    'GPT4V',
    'GPT4V+',
    'VisProg',
    'VisProg+'
]

iou_data = {
    'train': [77.64-1.41, 70.49, 74.59, 72.17, 76.48, 26.56, 28.73, 45.62, 38.94],
    'in-test': [76.29-1.06, 62.75, 70.48, 67.83, 67.81, 29.71, 29.86, 45.63, 39.21],
    'out-test': [74.07-1.89, 57.42, 56.00, 54.75, 56.11, 30.45, 33.92, 44.98, 41.83]
}

error_bounds = {
    'train':    [1.41, 0.0, 0.0, 0.0, 0.0, 4.22, 3.23, 3.12, 3.56],
    'in-test':  [1.06, 0.0, 0.0, 0.0, 0.0, 3.76, 4.34, 2.92, 1.65],
    'out-test': [1.89, 0.0, 0.0, 0.0, 0.0, 5.61, 4.98, 2.87, 2.53]
}


# Define colors for each method
colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

# Start plotting
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
# Define the bar width and positions for each group
bar_width = 0.15
half_num_bars = 4
indices = np.arange(start=0, stop=2 * len(iou_data), step=2)

# Create bars for each method
for i, method in enumerate(methods):
    bar_positions = indices - bar_width * half_num_bars + i * bar_width
    print(bar_positions)
    ax.bar(bar_positions[0], iou_data['train'][i], width=bar_width, color=colors[i], label=method)
    ax.bar(bar_positions[1], iou_data['in-test'][i], width=bar_width, color=colors[i])
    ax.bar(bar_positions[2], iou_data['out-test'][i], width=bar_width, color=colors[i])

    # Error bars
    err_capsize = 2  # Size of the cap on error bars
    ax.errorbar(bar_positions[0], iou_data['train'][i], yerr=error_bounds['train'][i], fmt='none', ecolor='darkred', capsize=err_capsize)
    ax.errorbar(bar_positions[1], iou_data['in-test'][i], yerr=error_bounds['in-test'][i], fmt='none', ecolor='darkred', capsize=err_capsize)
    ax.errorbar(bar_positions[2], iou_data['out-test'][i], yerr=error_bounds['out-test'][i], fmt='none', ecolor='darkred', capsize=err_capsize)


# # Add lines connecting the tops of the bars for each method
# for i, color in enumerate(colors):
#     ax.plot(indices, [iou_data['train'][i], iou_data['in-test'][i], iou_data['out-test'][i]], color=color, marker='o')

# Add x-tick labels
ax.set_xticks(indices)
ax.set_xticklabels(['train', 'in-test', 'out-test'])

# Setting the y-axis label
ax.set_ylabel('miou (%)')
ax.set_ylim(0, 80)

# Add legend
ax.legend(loc='upper right', fontsize='x-small')

# Add a title and labels to the axes
# plt.title('Comparison of IOU_safe scores across different methods')

# Show plot
plt.tight_layout()
# plt.show()

# save
fig.savefig('/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/figs/barplot_methods.png', dpi=300)
