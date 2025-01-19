import torchvision
from torchvision import transforms
from segdinov2.models.dino2 import DINO2SEG
from segdinov2.utils.mscoco import COCOSegmentation
from segdinov2.utils.segmentationMetric import *
from segdinov2.utils.vis import decode_segmap
import numpy as np
import torch
import os
from PIL import Image, ImageOps
from simple_colors import green
from tqdm import tqdm

from copy import deepcopy
import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2

from terrainseg.inference import TerrainSegFormer
from dropoff.faster_ns_inference import FasterImageInference


class NSCLpredicates:
    @torch.inference_mode()
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.padding = (0, 210, 0, 210)  # left, top, right, bottom
        self.model_rootpath = "segdinov2/mymodels/nscl"
        self.PREDICATES = os.listdir(self.model_rootpath)
        self.PREDICATE_MODELS = {}
        for p in self.PREDICATES:
            self.PREDICATE_MODELS[p] = DINO2SEG(3, big=False).to(self.device)
            model_path = f"{self.model_rootpath}/{p}/dinov2_mscoco_best_model.pth"
            ckpt = torch.load(model_path, map_location=self.device)
            self.PREDICATE_MODELS[p].load_state_dict(ckpt)
            self.PREDICATE_MODELS[p].eval()

    @torch.inference_mode()
    def predict_predicate(self, img: Image.Image, predicate: str):
        img = img.convert('RGB')
        padding = self.padding
        img = ImageOps.expand(img, padding, fill=0)
        img = img.resize((540, 540), Image.BILINEAR)
        w, h = img.size
        short_size = 448
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        img = np.array(img)
        img = self.input_transform(img).unsqueeze(0).to(self.device)
        out = self.PREDICATE_MODELS[predicate](img)
        upsampled_logits = torch.nn.functional.interpolate(
            out,
            size=(960, 960),
            mode='bilinear',
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        reduced_predseg = pred_seg[210:750, :].cpu().numpy().astype(np.uint8).reshape((540, 960))
        return reduced_predseg


class NSCLFasterImageInference(FasterImageInference):
    def __init__(self, domain):
        super().__init__(domain)
        self.nscl_predicates = NSCLpredicates()

    def _terrain(self, pixel_loc):
        """
        Returns the terrain idx at the given pixel location
        """
        if self.cur_main_terrain_output is None:
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)),
                                                             predicate="terrain")
            # mapping correct ones to 0, rest to -1
            ret_out[ret_out != 1] = -1
            ret_out[ret_out == 1] = 0
            self.cur_main_terrain_output = ret_out
        return int(self.cur_main_terrain_output[pixel_loc[1], pixel_loc[0]])

    def _in_the_way(self, pixel_loc):
        """
        Returns whether the given pixel location is in the way
        """
        if self.cur_main_in_the_way_output is None:
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)),
                                                             predicate="in_the_way")
            ret_out[ret_out == 2] = 0
            self.cur_main_in_the_way_output = ret_out
        return bool(self.cur_main_in_the_way_output[pixel_loc[1], pixel_loc[0]])

    def _distance_to(self, pixel_loc, obj_name):
        """
        Returns the distance to the given object at the given pixel location
        """
        if self.cur_distance_to_output is None:
            self.cur_distance_to_output = {}
        if obj_name not in self.cur_distance_to_output.keys():
            orig_out = self.distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
            obj_mask = (orig_out < 0.05).astype(np.uint8)
            overlay_img = deepcopy(self.cur_img_bgr)
            overlayed_img = TerrainSegFormer.get_seg_overlay(overlay_img, obj_mask, alpha=0.24)
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)),
                                                             predicate="distance_to")
            ret_out = ret_out.astype(np.float32)
            ret_out[ret_out != 1.0] = 0.0
            ret_out[ret_out == 1.0] = 100.0
            self.cur_distance_to_output[obj_name] = ret_out
        return float(self.cur_distance_to_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def _frontal_distance(self, pixel_loc, obj_name):
        """
        Returns the frontal distance to the given object at the given pixel location
        """
        if self.cur_frontal_distance_output is None:
            self.cur_frontal_distance_output = {}
        if obj_name not in self.cur_frontal_distance_output.keys():
            orig_out = self.frontal_distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
            obj_mask = (orig_out < 0.05).astype(np.uint8)
            overlay_img = deepcopy(self.cur_img_bgr)
            overlayed_img = TerrainSegFormer.get_seg_overlay(overlay_img, obj_mask, alpha=0.24)
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)),
                                                             predicate="frontal_distance_to")
            ret_out = ret_out.astype(np.float32)
            ret_out[ret_out != 1.0] = 0.0
            ret_out[ret_out == 1.0] = 100.0
            self.cur_frontal_distance_output[obj_name] = ret_out
        return float(self.cur_frontal_distance_output[obj_name][pixel_loc[1], pixel_loc[0]])


class ParkingNSCLFasterImageInference(FasterImageInference):
    def __init__(self, domain):
        super().__init__(domain)
        self.nscl_predicates = NSCLpredicates()

    def _terrain(self, pixel_loc):
        """
        Returns the terrain idx at the given pixel location
        """
        if self.cur_main_terrain_output is None:
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)),
                                                             predicate="pterrain")
            # mapping correct ones to 0, rest to -1
            ret_out[ret_out != 1] = -1
            self.cur_main_terrain_output = ret_out
        return int(self.cur_main_terrain_output[pixel_loc[1], pixel_loc[0]])

    def _terrainmarks(self, pixel_loc):
        """
        Returns the terrainmark idx at the given pixel location
        """
        if self.cur_main_terrainmarks_output is None:
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)),
                                                             predicate="pterrainmarks")
            # mapping correct ones to 0, rest to -1
            ret_out[ret_out != 1] = -1
            ret_out[ret_out == 1] = 0
            self.cur_main_terrainmarks_output = ret_out
        return int(self.cur_main_terrainmarks_output[pixel_loc[1], pixel_loc[0]])

    def _distance_to(self, pixel_loc, obj_name):
        """
        Returns the distance to the given object at the given pixel location
        """
        if self.cur_distance_to_output is None:
            self.cur_distance_to_output = {}
        if obj_name not in self.cur_distance_to_output.keys():
            orig_out = self.distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
            obj_mask = (orig_out < 0.05).astype(np.uint8)
            overlay_img = deepcopy(self.cur_img_bgr)
            overlayed_img = TerrainSegFormer.get_seg_overlay(overlay_img, obj_mask, alpha=0.24)
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)),
                                                             predicate="pdistance_to")
            ret_out = ret_out.astype(np.float32)
            ret_out[ret_out != 1.0] = 0.0
            ret_out[ret_out == 1.0] = 100.0
            self.cur_distance_to_output[obj_name] = ret_out
        return float(self.cur_distance_to_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def _within(self, pixel_loc, obj_name):
        """
        Returns the within value
        """
        if self.cur_within_output is None:
            self.cur_within_output = {}
        if obj_name not in self.cur_within_output.keys():
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)),
                                                             predicate="pwithin")
            ret_out[ret_out != 1] = -1
            self.cur_within_output[obj_name] = ret_out
        return bool(self.cur_within_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def _next_to(self, pixel_loc, obj_name):
        if self.cur_next_to_output is None:
            self.cur_next_to_output = {}
        if obj_name not in self.cur_next_to_output.keys():
            orig_out = self.distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
            obj_mask = (orig_out < 0.05).astype(np.uint8)
            overlay_img = deepcopy(self.cur_img_bgr)
            overlayed_img = TerrainSegFormer.get_seg_overlay(overlay_img, obj_mask, alpha=0.24)
            ret_out = self.nscl_predicates.predict_predicate(img=Image.fromarray(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)),
                                                             predicate="pnext_to")
            ret_out = ret_out.astype(np.uint8)
            ret_out[ret_out != 1] = -1
            self.cur_next_to_output[obj_name] = ret_out
        return int(self.cur_next_to_output[obj_name][pixel_loc[1], pixel_loc[0]])
