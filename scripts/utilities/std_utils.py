import json
import numpy as np
from scipy import ndimage


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
