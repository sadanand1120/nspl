import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from utilities.std_utils import json_writer, json_reader
from ldips_datagen import LDIPSdatagenSingleEx
import numpy as np
from synthesis.synthesize import LDIPS_synthesize


def read_data(nums_list):
    data_table_dir = os.path.join(nspl_root_dir, "demonstrations/SSR_notraj")
    all_json_paths = [os.path.join(data_table_dir, file) for file in sorted(os.listdir(data_table_dir))]
    prev_example_data = []  # list of tuples (label, ldips_features)
    for i in nums_list:
        json_path = all_json_paths[i - 1]
        data = json_reader(json_path)
        prev_example_data.append((data['label'], data['ldips_features']))
    return prev_example_data


if __name__ == "__main__":
    seqn_filled_lfps_sketches = {}
    seqn_lfps_sketches = json_reader(os.path.join(nspl_root_dir, "scripts/llm/seqn_lfps_sketches.json"))
    for i in range(1, 30):
        print(f"Processing {i} of 29")
        nums_list = np.arange(1, i + 1)
        examples_data = read_data(nums_list)
        lfps_sketch = seqn_lfps_sketches[str(i)]
        lsps_sketch = LDIPSdatagenSingleEx.convert_LFPS_to_LSPS(lfps_sketch)
        params = LDIPS_synthesize(examples_data, lsps_sketch)
        filled_lfps_sketch = LDIPSdatagenSingleEx.fillparams_in_LFPS(lfps_sketch, params)
        seqn_filled_lfps_sketches[str(i)] = filled_lfps_sketch
    json_writer(os.path.join(nspl_root_dir, "scripts/llm/ablation_notraj/seqn_filled_lfps_sketches.json"), seqn_filled_lfps_sketches)
