import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
import numpy as np
import cv2
from third_party.jackal_calib import JackalLidarCamCalibration

args = argparse.ArgumentParser()
args.add_argument("--rootdir", type=str, required=True)
args = args.parse_args()

lcc = JackalLidarCamCalibration(ros_flag=False)
root_dir = args.rootdir
depth_root_dir = os.path.join(root_dir, "depth")
os.makedirs(depth_root_dir, exist_ok=True)
img_root_dir = os.path.join(root_dir, "images")
pc_root_dir = os.path.join(root_dir, "pcs")

all_image_names = sorted(os.listdir(img_root_dir))
all_noext_filenames = [os.path.splitext(img_name)[0] for img_name in all_image_names]

for i in range(len(all_image_names)):
    print(f"Processing {i+1}/{len(all_image_names)}")
    img_path = os.path.join(img_root_dir, f"{all_noext_filenames[i]}.png")
    pc_path = os.path.join(pc_root_dir, f"{all_noext_filenames[i]}.bin")
    depth_path = os.path.join(depth_root_dir, f"{all_noext_filenames[i]}.png")
    cv2_img = cv2.imread(img_path)
    cur_pc_xyz = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))[:, :3]
    *_, ret_imgs = lcc.projectPCtoImageFull(cur_pc_xyz, cv2_img, ret_imgs=True, resize=False)
    full_ccs_img = ret_imgs[2]
    cv2.imwrite(depth_path, full_ccs_img)
