import os
import sys
import cv2
import rosbag
from PIL import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from terrainseg.inference import TerrainSegFormer
import matplotlib.pyplot as plt
from safety.realtime.fast_utils import FastModels
from simple_colors import red, green
from tqdm.auto import tqdm


def save_image_overlays(image_path: str, gtbin_path: str, save_rootdir_path: str, fm: FastModels):
    """
    Saves the raw image, gt, pred, and NN module overlays.
    """
    save_rootdir_path = os.path.join(save_rootdir_path, os.path.basename(image_path).split(".")[0])
    os.makedirs(save_rootdir_path, exist_ok=True)
    cv2_img_np = cv2.imread(image_path)
    pil_img = Image.open(image_path)
    full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(pil_img)

    # terrain
    pred_img_terrain = terrain_results[0]
    pred_img_terrain = cv2.cvtColor(np.asarray(pred_img_terrain), cv2.COLOR_RGB2BGR)

    # gsam
    gsam_ann_img = gsam_results[0]
    cv2_gsam_pred = cv2.cvtColor(np.asarray(gsam_ann_img), cv2.COLOR_RGB2BGR)

    # depth
    pred_metric_depth = depth_results[0]
    normalized_image = cv2.normalize(pred_metric_depth, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    colormap = plt.get_cmap('inferno')
    colored_image = colormap(normalized_image)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # pred
    cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_img_np, full_pred_seg, alpha=0.24)

    # gt
    gt_bin = np.fromfile(gtbin_path, dtype=np.uint8).reshape((cv2_img_np.shape[0], cv2_img_np.shape[1]))
    gt_bin[gt_bin == 2] = 0
    gt_overlay = TerrainSegFormer.get_seg_overlay(cv2_img_np, gt_bin, alpha=0.24)

    # saves
    cv2.imwrite(os.path.join(save_rootdir_path, "raw.png"), cv2_img_np)
    cv2.imwrite(os.path.join(save_rootdir_path, "gt.png"), gt_overlay)
    cv2.imwrite(os.path.join(save_rootdir_path, "pred.png"), cv2_pred_overlay)
    cv2.imwrite(os.path.join(save_rootdir_path, "terrain.png"), pred_img_terrain)
    cv2.imwrite(os.path.join(save_rootdir_path, "gsam.png"), cv2_gsam_pred)
    cv2.imwrite(os.path.join(save_rootdir_path, "depth.png"), colored_image)
    print(green("Overlays saved successfully."))


if __name__ == "__main__":
    image_path = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_safety/utcustom/eval/images/000007_morning_mode1_11062023_000211.png"
    gtbin_path = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_safety/utcustom/eval/gt_preds/000007_morning_mode1_11062023_000211.bin"
    save_rootdir_path = "/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/overlays/safety"
    fm = FastModels()
    save_image_overlays(image_path, gtbin_path, save_rootdir_path, fm)
