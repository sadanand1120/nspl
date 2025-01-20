import os
import PIL.Image
nspl_root_dir = os.environ.get("NSPL_REPO")
import cv2
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy
import os
from sensor_msgs.msg import PointCloud2, CompressedImage
import matplotlib.pyplot as plt
import rospy
import time
from cv_bridge import CvBridge
from copy import deepcopy
import yaml
from scipy.interpolate import griddata
from copy import deepcopy
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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import PIL
from PIL import Image
import requests
from llm.preprompts.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, NSLABELS_TRAVERSABLE_TERRAINS, NSLABELS_NON_TRAVERSABLE_TERRAINS
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import json_reader
from ldips_inference import NSInferObjDet, NSInferTerrainSeg
import numpy as np
import cv2
from copy import deepcopy
import cv2
from scipy.spatial import cKDTree
from copy import deepcopy
from PIL import Image
from sklearn.decomposition import PCA
from llm.preprompts.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, NSLABELS_TRAVERSABLE_TERRAINS, NSLABELS_NON_TRAVERSABLE_TERRAINS
from terrainseg.inference import TerrainSegFormer
from utilities.local_to_map_frame import LocalToMapFrame
from zeroshot_objdet.sam_dino import GroundedSAM
from third_party.jackal_calib import JackalLidarCamCalibration, JackalCameraCalibration
from utilities.std_utils import smoothing_filter
import open3d as o3d
import argparse
import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from third_party.Depth_Anything.metric_depth.zoedepth.models.builder import build_model
from third_party.Depth_Anything.metric_depth.zoedepth.utils.config import get_config
from third_party.jackal_calib.cam_calib import JackalCameraCalibration
import time
import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from safety.realtime.fast_utils import FastModels
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True


def estimate_full_fps(model_fm: FastModels):
    test_dir = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_safety/utcustom/eval/images"
    all_image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")]

    print("Estimating Full FPS")
    start = timer()
    for i in tqdm(range(2), desc='Outer Loop'):
        for path in tqdm(all_image_paths, desc='Inner Loop', leave=False):
            pil_img = Image.open(path)
            _ = model_fm.predict_new(pil_img)
    end = timer()
    fps = 2 * len(all_image_paths) / (end - start)
    print(f"FPS: {round(fps, 2)}")


if __name__ == "__main__":
    with torch.inference_mode():
        fm = FastModels()
        estimate_full_fps(fm)
