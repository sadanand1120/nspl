import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
import evaluate
import numpy as np
from terrainseg.inference import TerrainSegFormer
import cv2
from pprint import pprint
from simple_colors import red, green


"""
eval-000035_morning_mode1_11062023_000828
eval-000036_morning_mode1_2_11062023_000054
eval-000041_morning_mode1_2_11062023_000129
"""
METHOD_NUMS = [1, 2, 3, 7, 9, 11]


def overlay_classification_on_image(image_noext_name, root_dirpath, save_dirpath):
    global METHOD_NUMS
    cur_save_dirpath = os.path.join(save_dirpath, image_noext_name)
    os.makedirs(cur_save_dirpath, exist_ok=True)
    image_path = os.path.join(root_dirpath, "images", f"{image_noext_name}.png")
    image = cv2.imread(image_path)
    H, W, _ = image.shape

    # Ensure the image is in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for method_num in METHOD_NUMS:
        bin_path = os.path.join(root_dirpath, "methods_preds", str(method_num), f"{image_noext_name}.bin")
        pred_bin = np.fromfile(bin_path, dtype=np.uint8).reshape((H, W))
        pred_bin[pred_bin == 2] = 0

        gt_bin_path = os.path.join(root_dirpath, "gt_preds", f"{image_noext_name}.bin")
        gt_bin = np.fromfile(gt_bin_path, dtype=np.uint8).reshape((H, W))
        gt_bin[gt_bin == 2] = 0

        # Calculate TP, FP, FN, TN
        TP = (pred_bin == 1) & (gt_bin == 1)
        FP = (pred_bin == 1) & (gt_bin == 0)
        FN = (pred_bin == 0) & (gt_bin == 1)
        # TN is implicitly represented by not being TP, FP, or FN

        # Initialize the overlay as a copy of the original image to keep TN regions unchanged
        overlay = image_rgb.copy()

        # Apply color codes: TP=Green, FP=Red, FN=Blue. TN remains as in the original image
        overlay[TP] = [255, 255, 0]  # Yellow for TP
        overlay[FP] = [255, 0, 0]  # Red for FP
        overlay[FN] = [0, 255, 0]  # Green for FN

        # Blend the overlay with the original image
        overlay_image = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)

        # Convert back to BGR for saving with OpenCV
        overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

        # Save the resultant image
        save_path = os.path.join(cur_save_dirpath, f"overlay_classification_{method_num}.png")
        cv2.imwrite(save_path, overlay_image_bgr)


if __name__ == "__main__":
    root_dirpath = os.path.join(nspl_root_dir, "evals_data_safety/utcustom/eval")
    save_dirpath = os.path.join(nspl_root_dir, "paper_images")
    for image_noext_name in ["000035_morning_mode1_11062023_000828", "000036_morning_mode1_2_11062023_000054", "000041_morning_mode1_2_11062023_000129"]:
        overlay_classification_on_image(image_noext_name, root_dirpath, save_dirpath)
