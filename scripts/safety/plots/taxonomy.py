import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configure plot settings
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

# Sample data_points dictionary
data_points = {
    'NSPL (ours)': [3, 10, 1, 3, -0.15, 3.0],
    'NMN': [1.25, 100000, 1, 0, 0.02, 0.1],
    '[2]': [2.5, 100, 1, 0, 0.02, 0.1],  # JCL
    'NS-VQA': [1.5, 4000, 0, 0, 0.02, 0.1],
    'NSCL': [2, 5000, 1, 0, 0.02, 0.1],
    '[5]': [1.25, 5, 1, 0, 0.02, 0.1],  # VDP
    'PUMICE': [1.2, 50, 1, 2, -0.07, 0.1],
    'LDIPS': [1, 60, 1, 2, -0.02, 0.1],
    'VCML': [2, 100000, 1, 0, 0.02, 0.1],
    'CBM': [1, 30000, 1, 1, 0.02, 0.1],
    'LRRL-RE': [1.1, 7500, 1, 1, 0.02, 0.1],
    'DCL': [2, 6000, 1, 0, -0.09, 0.8],
    'FALCON': [2, 10, 1, 0, 0.02, 0.1],
    'VisProg': [1.1, 1, 0, 0, 0.02, 0.1],
    'CaP': [1, 1, 0, 2, 0.02, 0.1],
    'NSRMP': [1.7, 5000, 1, 0, 0.02, 0.1],
    'RCL': [1.3, 50, 1, 2, 0.02, 0.1],
    'DRPL': [2.8, 100000, 1, 1, 0.02, 0.1],
    'BPNS': [1.7, 1000, 1, 0, 0.02, 0.1],
    'NS3D': [2, 500, 1, 0, 0.02, 0.1],
    '[21]': [2.5, 5, 0, 2, 0.02, 0.1],  # LLM-GROP
    'LLM+P': [1, 5, 0, 2, 0.02, 0.1],
    'ViperGPT': [1.1, 1, 0, 0, 0.02, -0.3],
    'Voyager': [1, 100, 1, 2, 0.02, 0.1],
    'CodeBotler': [1, 1, 0, 2, -0.06, 0.4],
    'LEFT': [2, 2000, 1, 0, 0.02, 0.1],
    'PROGPORT': [1.5, 500, 0, 0, 0.02, 0.1],
}

# Define marker styles
marker_styles = {0: 'o', 1: 'o', 2: 'o', 3: '*'}

# Define color styles
color_styles = {0: 'lightgray', 1: 'lightgray'}

# Create the plot with specified figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Plot each data point
for label, (x, y, color_key, marker_key, x_off, y_off) in data_points.items():
    color = color_styles[color_key]
    marker = marker_styles[marker_key]
    if label == "NSPL (ours)":
        ax.scatter(x, y, label=label, color='green', marker='*', s=40)
        ax.text(x + x_off, y + y_off, label, color='green', fontsize=12)
    if label == "[2]" or label == "[5]" or label == "[21]":
        ax.scatter(x, y, label=label, color='black', marker='o', s=20)
        ax.text(x + x_off, y + y_off, label, color='black', fontsize=10)
    # else:
    #     ax.scatter(x, y, label=label, color=color, marker=marker, s=10)
    #     ax.text(x + x_off, y + y_off, label, color=color, fontsize=6)

# Define custom markers for the legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ns', markerfacecolor='black', markersize=6),
    Line2D([0], [0], marker='s', color='w', label='pure-nn', markerfacecolor='black', markersize=6),
    Line2D([0], [0], marker='x', color='w', label='pure-symb', markerfacecolor='black', markeredgecolor='black', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='no-concept', markerfacecolor='lightgray', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='concept', markerfacecolor='black', markersize=6)
]

# Set x-axis labels
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Factual', 'Abstract', 'Preferential'])
ax.set_xlabel('Concept Spectrum (ordered complexity)')
ax.set_xlim(0.9, 3.1)

# Set y-axis label
yticks = [10, 1000, 100000]
yticklabels = ['0.01k', '1k', '100k']

# Set the y-ticks
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Data Efficiency (# training samples)')
ax.set_ylim(0.56, 177827)

# Hide x and y ticks
ax.tick_params(axis='both', which='both', length=0)
# ax.set_yticklabels([])

# plt.show()
plt.savefig(os.path.join(nspl_root_dir, "scripts/safety/plots/figs/taxonomy.png"))
