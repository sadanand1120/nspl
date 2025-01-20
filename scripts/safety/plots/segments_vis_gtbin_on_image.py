import os
import sys
nspl_root_dir = os.environ.get("NSPL_REPO")
from segments import SegmentsClient
import shutil
import requests
import cv2
import numpy as np
from terrainseg.inference import TerrainSegFormer

img_dir = "/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/userstudy/figs"
gt_dir = "/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/userstudy/gts"
overlay_dir = "/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/userstudy/overlays"


def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file in binary write mode
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def download_imgs_gts(dataset_identifier):
    # Set up Segments.ai client
    api_key = os.environ.get("SEGMENTSAI_API_KEY")
    client = SegmentsClient(api_key)

    all_samples = client.get_samples(dataset_identifier,
                                     labelset="ground-truth",
                                     sort="name",
                                     direction="asc")

    for sample in all_samples:
        filename = sample.name
        print(f"Processing {filename}...")
        url = sample.attributes.image.url
        print(f"Downloading image from {url}...")
        os.makedirs(img_dir, exist_ok=True)
        download_file(url, f"{img_dir}/{filename}")
        sample_uuid = sample.uuid
        label = client.get_label(sample_uuid, labelset="ground-truth")
        print("Downloading segmentation bitmap...")
        url = label.attributes.segmentation_bitmap.url
        os.makedirs(gt_dir, exist_ok=True)
        download_file(url, f"{gt_dir}/{filename}")


def read_img_bin(img_path):
    img = cv2.imread(img_path)
    img_np = np.array(img)
    return img_np[:, :, 2].squeeze()


dataset_identifier = "smodak/parking-utcustom-train"
# download_imgs_gts(dataset_identifier)

all_gt_paths = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if f.endswith(".png")]

for gt_path in all_gt_paths:
    gt = read_img_bin(gt_path)
    gt[gt == 2] = 0
    noext_gtname = os.path.splitext(os.path.basename(gt_path))[0]
    cv2_img = cv2.imread(f"{img_dir}/{noext_gtname}.png")
    bin_overlay = TerrainSegFormer.get_seg_overlay(cv2_img, gt, alpha=0.24)
    cv2.imshow(f"{noext_gtname}.png", bin_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(f"{overlay_dir}/{noext_gtname}.png", bin_overlay)
