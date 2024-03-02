import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from datasets import load_dataset
import numpy as np

if __name__ == "__main__":
    hfdi = "sam1120/parking-terrain_marks"
    rootdir = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/gt_terrainmarks"
    os.makedirs(rootdir, exist_ok=True)
    ds_dict = load_dataset(hfdi)
    ds = ds_dict['train']
    for i in range(len(ds)):
        print(f"Processing {i+1}/{len(ds)}")
        pil_img = ds[i]['pixel_values']
        W, H = pil_img.size
        label_img = ds[i]['labels']
        img_name = ds[i]['name']
        noext_img_name = os.path.splitext(img_name)[0]
        label_np = np.array(label_img).reshape((H, W))
        flat_label_np = label_np.reshape(-1).astype(np.uint8)
        flat_label_np.tofile(os.path.join(rootdir, f"{noext_img_name}.bin"))
