import os
import sys
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.interpolate import make_interp_spline
from math import factorial
plt.rcParams['font.size'] = 15
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
    spl = make_interp_spline(x, y, k=5)
    x_smooth = np.linspace(x.min(), x.max(), 2000)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth


def plot_distn(root_dirname, bin_col_idx=0, linestyle='-', linecolor='red'):
    bins_rootdir = os.path.join(nspl_root_dir, f"scripts/safety/lifelong_seqn_permuts/{root_dirname}")
    all_binpaths = [os.path.join(bins_rootdir, f) for f in sorted(os.listdir(bins_rootdir)) if f.endswith(".bin")]
    seqns = []
    for binpath in all_binpaths:
        bin_np = np.fromfile(binpath, dtype=np.float32).reshape((29, 3))
        bin_col = [0.0] + list(bin_np[:, bin_col_idx])
        seqns.append(bin_col)

    seqns_np = np.array(seqns)
    y_data = list(seqns_np.T)

    # Create x-axis values
    x = np.arange(0, len(bin_col))

    mean_values = [np.mean(data) for data in y_data]
    percentile_low = [np.percentile(data, 0 + i / 1.5) for i, data in enumerate(y_data)]
    percentile_high = [np.percentile(data, 100 - i / 1.5) for i, data in enumerate(y_data)]

    x1, mean_values = smooth_line(x, mean_values)
    x2, percentile_low = smooth_line(x, percentile_low)
    x3, percentile_high = smooth_line(x, percentile_high)

    # Plotting
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x1,
             mean_values,
             linestyle=linestyle,
             color=f"dark{linecolor}",
             label='centerline',
             linewidth=3.5)

    plt.fill_between(x1, percentile_low, percentile_high, color=f'lightgray', alpha=0.5, label='spread')
    plt.xlabel('# demonstration', fontweight='bold', fontsize=18)
    plt.ylabel('miou (%)', fontweight='bold', fontsize=18)
    plt.xlim(0, len(bin_col) - 1)
    plt.ylim(0, 100)
    plt.xticks(range(0, len(bin_col), 4))
    # plt.legend()
    plt.grid(True, color='lightgray')
    # plt.show()
    plt.savefig(os.path.join(nspl_root_dir, "scripts/safety/plots/figs", f"lifelong_learning_distn_{dataset_name.split('/')[0]}_miou.png"))


if __name__ == "__main__":
    for d in ["train/utcustom", "test", "eval"]:
        dataset_name = d
        datadicts = {"train/utcustom": "red", "test": "blue", "eval": "green"}
        plot_distn(root_dirname=dataset_name,
                   bin_col_idx=2,
                   linestyle="-",
                   linecolor=datadicts[dataset_name])
