import os
import sys
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.interpolate import make_interp_spline
from math import factorial
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def smooth_line(x, y):
    # spl = make_interp_spline(x, y, k=5)
    # x_smooth = np.linspace(x.min(), x.max(), 800)
    # y_smooth = spl(x_smooth)
    # yhat = savitzky_golay(np.asarray(y), 27, 9)  # window size 51, polynomial order 3
    # return x, list(yhat.squeeze())
    return x, y


train_binpath = os.path.join(nspl_root_dir, "scripts/safety/lifelong_seqn_permuts/train/utcustom/0000.bin")
test_binpath = os.path.join(nspl_root_dir, "scripts/safety/lifelong_seqn_permuts/test/0000.bin")
eval_binpath = os.path.join(nspl_root_dir, "scripts/safety/lifelong_seqn_permuts/eval/0000.bin")

# cols: iou_safe, iou_unsafe, miou
train_np = np.fromfile(train_binpath, dtype=np.float32).reshape((29, 3))
test_np = np.fromfile(test_binpath, dtype=np.float32).reshape((29, 3))
eval_np = np.fromfile(eval_binpath, dtype=np.float32).reshape((29, 3))

train_miou = [0.0] + list(train_np[:, 2])
test_miou = [0.0] + list(test_np[:, 2])
eval_miou = [0.0] + list(eval_np[:, 2])

# Create x-axis values
x = np.arange(0, len(train_miou))

x_train_miou, y_train_miou = smooth_line(x, train_miou)
x_test_miou, y_test_miou = smooth_line(x, test_miou)
x_eval_miou, y_eval_miou = smooth_line(x, eval_miou)

# Plotting
# Train - darkred
# Test - darkblue
# Eval - darkgreen
plt.figure(figsize=(8, 6))
plt.plot(x_train_miou,
         y_train_miou,
         linestyle='-',
         color='darkred')
plt.plot(x_test_miou,
         y_test_miou,
         linestyle='-',
         color='darkblue')
plt.plot(x_eval_miou,
         y_eval_miou,
         linestyle='-',
         color='darkgreen')

# legend
darkred_dot = mlines.Line2D([], [], color='darkred', marker='o', linestyle='None', markersize=7, label='train')
darkblue_dot = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None', markersize=7, label='in-test')
darkgreen_dot = mlines.Line2D([], [], color='darkgreen', marker='o', linestyle='None', markersize=7, label='out-test')
# synapse = mlines.Line2D([], [], color='black', linestyle='-', label='synapse', linewidth=2)
baseline = mlines.Line2D([], [], color='black', linestyle='--', label='best baseline', linewidth=2, alpha=0.2)
plt.legend(handles=[darkred_dot, darkblue_dot, darkgreen_dot, baseline], loc='upper right', fontsize=9)

# Add annotations
special_points = [0, 3, 4, 10, 26]
annotations = ['is_on, is_far_away_from', 'is_in_the_way', 'is_in_front_of', 'is_close_to', 'is_too_inclined']
for point, annotation in zip(special_points, annotations):
    plt.axvline(x=point, color='lightgray', linestyle='--')
    if point == 0:
        plt.text(point, 58.0, annotation, horizontalalignment='center', rotation=80, color='gray')
    else:
        plt.text(point, 15.0, annotation, horizontalalignment='center', rotation=80, color='gray')

# train iou safe baseline
y_value_for_line = 76.48
plt.axhline(y=y_value_for_line, color='red', linestyle='--', alpha=0.2)
# plt.text(18, y_value_for_line * 1.01, 'baseline', horizontalalignment='right', verticalalignment='bottom')

# test iou safe baseline
y_value_for_line = 70.48
plt.axhline(y=y_value_for_line, color='blue', linestyle='--', alpha=0.2)

# eval iou safe baseline
y_value_for_line = 57.42
plt.axhline(y=y_value_for_line, color='green', linestyle='--', alpha=0.2)

plt.xlabel('# demonstration', fontweight='bold')
plt.ylabel('miou (%)', fontweight='bold')
plt.xlim(-1, len(train_miou) - 1)
plt.ylim(0, 100)
plt.xticks(range(0, len(train_miou), 2))
# plt.show()
plt.savefig("/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/figs/lifelong_single.png", dpi=300)
