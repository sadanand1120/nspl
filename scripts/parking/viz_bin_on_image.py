import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
import evaluate
import numpy as np
from terrainseg.inference import TerrainSegFormer
import cv2
from pprint import pprint
from simple_colors import red, green


def iou_viz_individual(image_dirpath, bin_dirpath, save_dirpath, H=540, W=960):
    id2label = {
        0: "unlabeled",
        1: "parking",
        2: "unparking",
    }
    all_image_paths = [os.path.join(image_dirpath, f) for f in sorted(os.listdir(image_dirpath)) if f.endswith(".png")]
    all_bin_paths = [os.path.join(bin_dirpath, f) for f in sorted(os.listdir(bin_dirpath)) if f.endswith(".bin")]
    all_images = []
    for image_path in all_image_paths:
        image = cv2.imread(image_path)
        all_images.append(image)

    for i in range(len(all_bin_paths)):
        try:
            bin = np.fromfile(all_bin_paths[i], dtype=np.uint8).reshape((H, W))
        except:
            continue
        # print(f"Processing idx {i}/{len(all_bin_paths)}")
        noext_binname = os.path.splitext(os.path.basename(all_bin_paths[i]))[0]
        for j in range(len(all_image_paths)):
            noext_imgname = os.path.splitext(os.path.basename(all_image_paths[j]))[0]
            if noext_imgname in noext_binname:
                break
        cv2_img = all_images[j]
        bin_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, bin, alpha=0.24)
        # cv2.imshow(f"{noext_binname}.png", bin_overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        os.makedirs(save_dirpath, exist_ok=True)
        cv2.imwrite(f"{save_dirpath}/{noext_binname}.png", bin_overlay)


if __name__ == "__main__":
    iou_viz_individual(image_dirpath="/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/eval/images",
                       bin_dirpath="/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/eval/methods_preds/1",
                       save_dirpath="/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/temp_preds/eval")
