import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
from datasets import load_dataset
import numpy as np
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import reader, json_reader, writer, json_writer
import cv2
from PIL import Image
from dropoff.faster_ns_inference import FasterImageInference, FasterImageInferenceCaP
from dropoff._visprog_inference import infer_visprog
from llm._vlm import get_vlm_response
from segments import SegmentsClient
import re
from pprint import pprint
from simple_colors import red, green
from terrainseg.inference import TerrainSegFormer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0", help='CUDA gpus allowed to be used (comma-separated)')
    parser.add_argument("--eval_hfdi", type=str, required=True)
    parser.add_argument("--hfmi", type=str, required=True)
    args = parser.parse_args()

    s = TerrainSegFormer(cuda=args.cuda,
                         hf_dataset_name=args.eval_hfdi,
                         hf_model_name=args.hfmi)
    s.load_model_inference()
    s.prepare_dataset()

    pred_ds = s.ds
    print(green(f"LOGINFO: Predicting metrics...", "bold"))
    metrics = s.predict_ds_metrics_wrapper(pred_ds)
    iou_dict = {}
    iou_dict["mIOU"] = round(metrics["mean_iou"] * 100, 2)
    iou_dict["IOU_dropoff"] = round(metrics["iou_dropoff"] * 100, 2)
    iou_dict["IOU_undropoff"] = round(metrics["iou_undropoff"] * 100, 2)
    pprint(green(iou_dict, "bold"))
