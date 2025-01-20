import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Patch

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Times New Roman'


# method: [miou, error, color, solid]
COLOR1 = '#4E79A7'
COLOR2 = '#CCCCCC'
COLOR3 = '#BBBBBB'
COLOR4 = '#DDDDDD'
DATA = {
    'Synapse': [76.11, 1.32, COLOR1, 'solid'],
    'Synapse-SynthDirect': [60.74, 3.12, COLOR2, 'dashed'],
    'Synapse-SynthDirect+': [68.86, 0.92, COLOR2, 'dashed'],
    'Synapse-SynthCaP': [64.11, 3.98, COLOR2, 'dashed'],
    'Synapse-CodeLLama': [69.88, 2.95, COLOR2, 'dashed'],
    'Synapse-StarCoder': [63.62, 4.12, COLOR2, 'dashed'],
    'Synapse-PaLM2': [71.62, 2.29, COLOR2, 'dashed'],
    'Synapse-OWLViTSAM': [70.17, 1.44, COLOR2, 'dashed'],
    'Synapse-GroupViT': [73.41, 1.29, COLOR2, 'dashed'],
    'SF-RGB-b0(29)': [44.58, 0.0, COLOR4, 'dashed'],
    'SF-RGB-b5(29)': [46.30, 0.0, COLOR4, 'dashed'],
    'SF-RGBD-b0(29)': [45.84, 0.0, COLOR4, 'dashed'],
    'SF-RGBD-b5(29)': [53.81, 0.0, COLOR4, 'dashed'],
    'GPT4V': [29.00, 4.0, COLOR3, 'solid'],
    'GPT4V+': [32.00, 4.0, COLOR3, 'solid'],
    'VisProg': [45.00, 3.0, COLOR3, 'solid'],
    'VisProg+': [42.00, 3.0, COLOR3, 'solid'],
    'SF-RGB-b0': [62.75, 0.0, COLOR4, 'solid'],
    'SF-RGB-b5': [64.81, 0.0, COLOR4, 'solid'],
    'SF-RGBD-b0': [65.84, 0.0, COLOR4, 'solid'],
    'SF-RGBD-b5': [67.00, 0.0, COLOR4, 'solid']
}

# Sort data by mIOU
sorted_methods = sorted(DATA.keys(), key=lambda x: DATA[x][0], reverse=True)
sorted_miou = [DATA[method][0] for method in sorted_methods]
sorted_error = [DATA[method][1] for method in sorted_methods]
sorted_colors = [DATA[method][2] for method in sorted_methods]
sorted_hatches = ["/" if DATA[method][3] == "dashed" else None for method in sorted_methods]

# # Define a better color palette
# colors = list(TABLEAU_COLORS.values())
# for i in range(len(sorted_colors)):
#     sorted_colors[i] = colors[i % len(colors)]

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 3))
bars = ax.bar(sorted_methods, sorted_miou, yerr=sorted_error, color=sorted_colors, capsize=2, hatch=sorted_hatches,
              ecolor='darkred')

# Customize the plot
ax.set_ylabel('mIOU (%)', fontsize=12, family='Times New Roman', weight='bold')
ax.set_ylim(20, 80)
ax.set_yticks(np.arange(20, 81, 10))
plt.xticks(rotation=30, ha='right', fontsize=10, family='Times New Roman')
plt.yticks(fontsize=12, family='Times New Roman')
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add gridlines and improve the appearance
# ax.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Create custom legend patches
legend_patches = [
    Patch(facecolor=COLOR1, label='ours', hatch=None),
    Patch(facecolor=COLOR2, label='baselines', hatch=None),
    Patch(facecolor=COLOR2, label='ablations', hatch='//')
]

# Add legend to the plot
ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1, 1), ncol=3, frameon=False)

# Show plot
plt.tight_layout()
plt.show()

# save
# fig.savefig('/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/overview_plot.png', dpi=300)
