import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# Methods and mIOU values
methods = [
    'Synapse',
    'Synapse-SynthDirect',
    'Synapse-SynthDirect+',
    'Synapse-SynthCap',
    'Synapse-CodeLLama',
    'Synapse-StarCoder',
    'Synapse-PaLM2',
    'SF-RGB-b0',
    'SF-RGB-b5',
    'SF-RGBD-b0',
    'SF-RGBD-b5'
]

mious = [
    76.11,
    60.74,
    68.86,
    64.11,
    69.88,
    63.62,
    71.62,
    54.58,
    56.30,
    55.84,
    63.81
]

error_bounds = [1.32, 3.12, 0.92, 3.98, 2.95, 4.12, 2.29, 0.0, 0.0, 0.0, 0.0]

# Define colors for each method
colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

# Start plotting
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

# Positions for each method
y_pos = np.arange(len(methods))

# Create horizontal bars
ax.barh(y_pos, mious, height=1, color=colors)

# Error bars for horizontal plot
err_capsize = 2  # Size of the cap on error bars
ax.errorbar(mious, y_pos, xerr=error_bounds, fmt='none', ecolor='darkred', capsize=err_capsize)

# Reverse the y-axis to have the highest value on top
ax.invert_yaxis()

# Add y-tick labels
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)

# Setting the x-axis label
ax.set_xlabel('miou (%)')
ax.set_xlim(0, 80)  # Assuming mIOU percentage is between 0 and 100

# Remove legend for horizontal bar plot (typically not used)
# ax.legend(loc='upper right', fontsize='x-small')  # Comment this out if no legend is required

# Show plot
plt.tight_layout()
# plt.show()

# save
fig.savefig('/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/figs/barplot_ablations.png', dpi=300)
