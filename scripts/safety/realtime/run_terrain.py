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


class FasterSynapse:
    def __init__(self):
        self.latest_cv2_img_np = None
        self.fm = FastModels()
        self.cv_bridge = CvBridge()
        rospy.Subscriber("/zed2i/zed_node/left/image_rect_color/compressed", CompressedImage, self.image_callback, queue_size=1)
        # self.terrainseg_pub = rospy.Publisher("/terrainseg/compressed", CompressedImage, queue_size=1)
        # rospy.Timer(rospy.Duration(1 / 11), lambda event: self.terrainseg(self.latest_cv2_img_np))
        self.full_pub = rospy.Publisher("/full/compressed", CompressedImage, queue_size=1)
        rospy.Timer(rospy.Duration(1), lambda event: self.full(self.latest_cv2_img_np))
        rospy.loginfo("Faster synapse module initialized")

    def image_callback(self, msg):
        self.latest_cv2_img_np = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # def terrainseg(self, cv2_img_np, event=None):
    #     if cv2_img_np is None:
    #         return
    #     pil_img = Image.fromarray(cv2.cvtColor(cv2_img_np, cv2.COLOR_BGR2RGB))
    #     pred_img, _ = self.fm.terrain_pred(pil_img)
    #     cv2_pred_img = cv2.cvtColor(np.asarray(pred_img), cv2.COLOR_RGB2BGR)
    #     msg = CompressedImage()
    #     msg.header.stamp = rospy.Time.now()
    #     msg.format = "jpeg"
    #     msg.data = np.asarray(cv2.imencode('.jpg', cv2_pred_img)[1]).tobytes()
    #     self.terrainseg_pub.publish(msg)

    def full(self, cv2_img_np, event=None):
        if cv2_img_np is None:
            return
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img_np, cv2.COLOR_BGR2RGB))
        pred_seg, *_ = self.fm.predict_new(pil_img)
        cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_img_np, pred_seg)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.asarray(cv2.imencode('.jpg', cv2_pred_overlay)[1]).tobytes()
        self.full_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node('faster_synapse', anonymous=True)
    e = FasterSynapse()
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down faster synapse module")
