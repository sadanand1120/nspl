import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from utilities.std_utils import json_writer, json_reader
from ldips_datagen import LDIPSdatagenSingleEx
import numpy as np
from synthesis.synthesize import LDIPS_synthesize, read_data

if __name__ == "__main__":
    seqn_filled_lfps_sketches = {}
    seqn_lfps_sketches = json_reader(os.path.join(nspl_root_dir, "scripts/llm/ablation_directplus/seqn_lfps_sketches.json"))
    for i in range(1, 30):
        print(f"Processing {i} of 29")
        nums_list = np.arange(1, i + 1)
        examples_data = read_data(nums_list)
        lfps_sketch = seqn_lfps_sketches[str(i)]
        if i == 29:
            lsps_sketch = LDIPSdatagenSingleEx.convert_LFPS_to_LSPS(lfps_sketch)
            params = LDIPS_synthesize(examples_data, lsps_sketch)
            filled_lfps_sketch = LDIPSdatagenSingleEx.fillparams_in_LFPS(lfps_sketch, params)
            seqn_filled_lfps_sketches[str(i)] = filled_lfps_sketch
        else:
            seqn_filled_lfps_sketches[str(i)] = lfps_sketch
    json_writer(os.path.join(nspl_root_dir, "scripts/llm/ablation_directplus/seqn_filled_lfps_sketches.json"), seqn_filled_lfps_sketches)
