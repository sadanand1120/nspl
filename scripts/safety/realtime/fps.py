import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import argparse
from datasets import load_dataset
import numpy as np
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import reader, json_reader
import cv2
from PIL import Image
from safety.faster_ns_inference import FasterImageInference, FasterImageInferenceCaP
from safety._visprog_inference import infer_visprog
from llm._vlm import get_vlm_response
from segments import SegmentsClient
import re
from simple_colors import red, green
from tqdm.auto import tqdm
from timeit import default_timer as timer
import torch
from zeroshot_objdet.sam_dino import GroundedSAM
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_hq_model_registry, SamPredictor
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from third_party.lightHQSAM.setup_light_hqsam import setup_model
from safety.realtime.fast_utils import FastGSAM
from safety.realtime.fast_utils import print_num_params
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True


def estimate_fps_terrain(model: TerrainSegFormer):
    test_dir = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_safety/utcustom/eval/images"
    all_image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")]

    print("Estimating FPS for Terrain Segmentation")
    start = timer()
    for i in tqdm(range(10), desc='Outer Loop'):
        for path in tqdm(all_image_paths, desc='Inner Loop', leave=False):
            img = Image.open(path)
            _ = model.predict_new(img)
    end = timer()
    fps = 10 * len(all_image_paths) / (end - start)
    print(f"FPS: {round(fps, 2)}")


def estimate_fps_groundedsam(model: FastGSAM):
    test_dir = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_safety/utcustom/eval/images"
    all_image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")]
    prompts = ["person", "bush", "car", "pole", "board", "entrance", "staircase"]

    print("Estimating FPS for Grounded SAM")
    start = timer()
    for i in tqdm(range(10), desc='Outer Loop'):
        for path in tqdm(all_image_paths, desc='Inner Loop', leave=False):
            img = cv2.imread(path)
            _ = model.predict_and_segment_on_image(img, prompts)
    end = timer()
    fps = 10 * len(all_image_paths) / (end - start)
    print(f"FPS: {round(fps, 2)}")


if __name__ == "__main__":
    with torch.inference_mode():
        modelname = "sam1120/safety-utcustom-terrain-b0-optim"
        s = TerrainSegFormer(hf_dataset_name="sam1120/safety-utcustom-terrain-jackal-full-391", hf_model_name=modelname)
        s.load_model_inference()
        s.prepare_dataset()
        print_num_params(s.model, "TerrainSegFormer")
        estimate_fps_terrain(s)

        g = FastGSAM(box_threshold=0.25, text_threshold=0.25, nms_threshold=0.25)
        estimate_fps_groundedsam(g)
