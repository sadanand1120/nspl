import json
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter


def reader(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    return content


def writer(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)


def append_writer(filepath, content):
    with open(filepath, 'a') as f:
        f.write(content)


def json_reader(filepath):
    with open(filepath, 'r') as f:
        dict_content = json.load(f)
    return dict_content


def json_writer(filepath, dict_content):
    with open(filepath, 'w') as f:
        json.dump(dict_content, f, indent=4)


def smoothing_filter(segmentation, window_size=11):
    def calculate_mode(values):
        counts = np.bincount(values.astype(int))
        return np.argmax(counts)
    filtered_seg = ndimage.generic_filter(segmentation, function=calculate_mode, size=window_size, mode='nearest')
    return np.asarray(filtered_seg).astype(np.asarray(segmentation).dtype).reshape(segmentation.shape)


def compute_confidence_array(binary_array, window_size=21):
    conf_score = uniform_filter(binary_array.astype(float), size=window_size, mode='constant', cval=0.0)
    return conf_score


if __name__ == "__main__":
    # random 0 1 segmentation
    seg = np.random.randint(0, 2, (100, 100))
    seg[0:60, 0:60] = 1
    seg[0:40, 0:40] = 0
    smoothed_seg = smoothing_filter(seg)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(seg, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(compute_confidence_array(smoothing_filter(seg)), cmap='gray')
    plt.title("Smoothed")
    plt.show()
