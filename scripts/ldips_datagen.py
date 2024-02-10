"""
Reads examples one at a time (to simulate lifelong online learning) and appends them to a data table.
Data table basically is a folder of jsons. Each json is symbolic representation of an example.
All ldips features are with respect to the ego location.
Json format:
{
    "human_nl": ??,  ## string for human's NL demonstration
    "label": "??",  ## represents the label for the example (SAFE/UNSAFE)
    "ldips_features": {
        "terrain": ??,  ## represents the terrain feature value
        "in_the_way": ??,  ## represents the in_the_way feature value
        "slope": ??,  ## represents the slope feature value (basically max z-diff)
        "distance_to_<object_class>": ??,  ## represents the distance to <object_class> feature value
        "frontal_distance_<object_class>": ??,  ## represents the frontal distance to <object_class> feature value
    },
    "ldips_synth_program_sketch": "??",  ## represents the ldips program sketch to be used as is (i.e., (pixel_loc) removed, etc)
    "hitl_llm_state": "??",  ## represents the hitl llm module state, according to which the data table is being generated
}
"""
import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2
import re
import yaml
import cv2
import yaml
from scipy.spatial import cKDTree
from typing import Optional

from llm.preprompts.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, NSLABELS_TRAVERSABLE_TERRAINS, NSLABELS_NON_TRAVERSABLE_TERRAINS
from llm.hitl_llm import HITLllm
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import json_reader, json_writer
from utilities.local_to_map_frame import LocalToMapFrame
from zeroshot_objdet.sam_dino import GroundedSAM
from third_party.jackal_calib import JackalLidarCamCalibration
from ldips_inference import NSInferObjDet, NSInferTerrainSeg


def extract_all_labels(nl_feedbacks, state_path):
    all_labels = []
    for i, nl_feedback in enumerate(nl_feedbacks):
        print(f"Processing nl_feedback {i+1}/{len(nl_feedbacks)}")
        temp = HITLllm(human_nl_demo=nl_feedback.strip(),
                       state_path=state_path,
                       is_first_demo=False)
        all_labels.append(temp.extract_label().upper())
    return all_labels


class NSTrainObjDet:
    def __init__(self, images_bgr: list, pcs_xyz: list, localizations: list, ego_ldict: dict, MODELS: dict, kDebug=False):
        self.images_bgr = images_bgr
        self.pcs_xyz = pcs_xyz
        self.localizations = localizations
        self.ego_ldict = ego_ldict
        self.MODELS = MODELS
        self.kDebug = kDebug
        self.ns_infer_objdet = NSInferObjDet(MODELS=self.MODELS)

    def main_distance_to(self, seq_idx, obj_name: str, box_threshold=0.5, text_threshold=0.5, nms_threshold=0.6):
        """
        Runs SAM on one image and returns the distance to the closest object of obj_name. If no object is detected, returns None
        """
        img_bgr = self.images_bgr[seq_idx]
        pc_np = self.pcs_xyz[seq_idx]
        ldict = self.localizations[seq_idx]
        ego_maploc = np.array([self.ego_ldict["x"], self.ego_ldict["y"], 0]).squeeze()
        ann_img, detections, per_class_mask = self.ns_infer_objdet.sam_dino_model.predict_and_segment_on_image(img_bgr, [obj_name],
                                                                                                               box_threshold=box_threshold,
                                                                                                               text_threshold=text_threshold,
                                                                                                               nms_threshold=nms_threshold)
        per_class_mask = per_class_mask[0].squeeze()  # since we only have one class
        _, all_pixel_locs, all_vlp_coords, _ = self.ns_infer_objdet.lidar_cam_calib.projectPCtoImage(pc_np, img_bgr)  # Project vlp onto image, but does not do any interpolation
        all_mask_values = per_class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
        class_pixel_locs = all_pixel_locs[all_mask_values]
        class_vlp_coords = all_vlp_coords[all_mask_values]
        if class_vlp_coords is None or class_vlp_coords.shape[0] == 0:
            if self.kDebug:
                print("main_distance_to: no vlp points on object")
            return None
        class_map_coords = self.ns_infer_objdet.projectVLPtoMap(class_vlp_coords, ldict)
        tree = cKDTree(class_map_coords)
        dist, pidx = tree.query(ego_maploc)
        if self.kDebug:
            print("main_distance_to: dist", dist)
            class_pt = class_map_coords[pidx].squeeze()
            d_wcs_coords = self.ns_infer_objdet.projectMaptoWCS(np.vstack((ego_maploc, class_pt)), ldict)
            d_pcs_coords, *_ = self.ns_infer_objdet.lidar_cam_calib.jackal_cam_calib.projectWCStoPCS(d_wcs_coords, mode="none")
            if d_pcs_coords.shape[0] == 2:
                # egoloc in green color dot, closest point in red color dot, line in blue
                # Ideally, d_pcs_coords[1] should be the same as class_pixel_locs[pidx]
                cv2.line(ann_img, (d_pcs_coords[0][0], d_pcs_coords[0][1]), (d_pcs_coords[1][0], d_pcs_coords[1][1]), (255, 0, 0), 2)  # blue
                cv2.circle(ann_img, (d_pcs_coords[0][0], d_pcs_coords[0][1]), 4, (0, 255, 0), -1)  # green
                cv2.circle(ann_img, (class_pixel_locs[pidx][0], class_pixel_locs[pidx][1]), 4, (0, 0, 255), -1)  # red
            cv2.imshow("main_distance_to: ann_img", ann_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return dist

    def main_frontal_distance(self, seq_idx, obj_name: str, box_threshold=0.5, text_threshold=0.5, nms_threshold=0.6):
        """
        Run SAM on one image and returns the frontal distance to the closest object of obj_name. If no object is detected, returns None
        """
        img_bgr = self.images_bgr[seq_idx]
        pc_np = self.pcs_xyz[seq_idx]
        ldict = self.localizations[seq_idx]
        ego_maploc_xy = np.array([self.ego_ldict["x"], self.ego_ldict["y"]]).reshape((1, 2))
        ann_img, detections, per_class_mask = self.ns_infer_objdet.sam_dino_model.predict_and_segment_on_image(img_bgr, [obj_name],
                                                                                                               box_threshold=box_threshold,
                                                                                                               text_threshold=text_threshold,
                                                                                                               nms_threshold=nms_threshold)
        per_class_mask = per_class_mask[0].squeeze()  # since we only have one class
        _, all_pixel_locs, all_vlp_coords, _ = self.ns_infer_objdet.lidar_cam_calib.projectPCtoImage(pc_np, img_bgr)  # Project vlp onto image, but does not do any interpolation
        all_mask_values = per_class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
        class_pixel_locs = all_pixel_locs[all_mask_values]
        class_vlp_coords = all_vlp_coords[all_mask_values]
        if class_vlp_coords is None or class_vlp_coords.shape[0] == 0:
            if self.kDebug:
                print("main_distance_to: no vlp points on object")
            return None
        class_map_coords = self.ns_infer_objdet.projectVLPtoMap(class_vlp_coords, ldict)
        class_map_coords_xy = class_map_coords[:, :2].reshape((-1, 2))
        front_distances, *_ = NSInferObjDet.frontal_distances_and_sweepability(class_map_coords_xy, ego_maploc_xy)
        return front_distances[0][0]


class NSTrainTerrainSeg:
    def __init__(self, images_bgr: list, pcs_xyz: list, localizations: list, ego_ldict: dict, MODELS: dict, kDebug=False):
        self.images_bgr = images_bgr
        self.pcs_xyz = pcs_xyz
        self.localizations = localizations
        self.ego_ldict = ego_ldict
        self.MODELS = MODELS
        self.kDebug = kDebug
        self.ns_infer_terrainseg = NSInferTerrainSeg(MODELS=self.MODELS)

    def main_in_the_way(self, seq_idx, domain_terrains, param=1.5):
        """
        Runs terrain segmentation on one image and predicts in the way
        Returns: the predicted in the way bool at ego location or None
        """
        target_map_coords = np.array([self.ego_ldict["x"], self.ego_ldict["y"], 0]).reshape((1, 3))
        target_pixel_coords, *_ = self.ns_infer_terrainseg.projectMapToPCS(target_map_coords, self.localizations[seq_idx])
        if target_pixel_coords is None or target_pixel_coords.shape[0] == 0:
            if self.kDebug:
                print("main_in_the_way: ego not visible")
            return None
        target_pixel_coords = target_pixel_coords.squeeze()
        img_bgr = self.images_bgr[seq_idx]
        pc_xyz = self.pcs_xyz[seq_idx]
        final_mask = self.ns_infer_terrainseg.main_in_the_way(img_bgr, pc_xyz, domain_terrains, param=param)
        return final_mask[target_pixel_coords[1], target_pixel_coords[0]]

    def main_terrain(self, seq_idx, domain_terrains):
        """
        Runs terrain segmentation on one image
        Returns: the predicted terrain string at ego location + pred_seg + new_terrains
        """
        pred_seg, new_terrains = self.ns_infer_terrainseg.main_terrain(self.images_bgr[seq_idx], domain_terrains)
        target_map_coords = np.array([self.ego_ldict["x"], self.ego_ldict["y"], 0]).reshape((1, 3))
        target_pixel_coords, *_ = self.ns_infer_terrainseg.projectMapToPCS(target_map_coords, self.localizations[seq_idx])
        if target_pixel_coords is None or target_pixel_coords.shape[0] == 0:
            if self.kDebug:
                print("main_terrain: ego not visible")
            return "Not visible", pred_seg, new_terrains
        target_pixel_coords = target_pixel_coords.squeeze()
        pred_terrain_idx = pred_seg[target_pixel_coords[1], target_pixel_coords[0]]
        pred_terrain = new_terrains[pred_terrain_idx]
        if self.kDebug:
            print("main_terrain: pred_terrain", pred_terrain)
            overlay = TerrainSegFormer.get_seg_overlay(self.images_bgr[seq_idx], pred_seg, alpha=0.24)
            cv2.circle(overlay, (target_pixel_coords[0], target_pixel_coords[1]), 4, (0, 0, 255), -1)
            cv2.imshow("main_terrain: overlay", overlay)
            cv2.waitKey(0)
        return pred_terrain, pred_seg, new_terrains

    def main_slope(self, seq_idx, scale=5e2):
        """
        Runs slope estimation. Uses only those points which are on a terrain, ie, not NAT
        Returns: the slope at ego location, or None
        """
        target_map_coords = np.array([self.ego_ldict["x"], self.ego_ldict["y"], 0]).reshape((1, 3))
        target_pixel_coords, *_ = self.ns_infer_terrainseg.projectMapToPCS(target_map_coords, self.localizations[seq_idx])
        if target_pixel_coords is None or target_pixel_coords.shape[0] == 0:
            if self.kDebug:
                print("main_slope: ego not visible")
            return None
        target_pixel_coords = target_pixel_coords.squeeze()
        grad_zs_mag = self.ns_infer_terrainseg.main_slope(self.images_bgr[seq_idx], self.pcs_xyz[seq_idx], scale=scale)
        return grad_zs_mag[target_pixel_coords[1], target_pixel_coords[0]]


class LDIPSdatagenSingleEx:
    WAY_PARAM = 1.8
    SLOPE_SCALE = 5e2
    OBJECT_THRESHOLDS = {  # 3-tuple of (box_threshold, text_threshold, nms_threshold)
        "barricade": (0.5, 0.5, 0.3),
        "board": (0.25, 0.25, 0.5),
        "bush": (0.35, 0.35, 0.4),
        "car": (0.37, 0.37, 0.3),
        "entrance": (0.35, 0.35, 0.2),
        "person": (0.47, 0.47, 0.6),
        "pole": (0.35, 0.35, 0.5),
        "staircase": (0.45, 0.45, 0.4),
        "tree": (0.4, 0.4, 0.45),
        "wall": (0.5, 0.5, 0.4)
    }

    def __init__(self, examples_root_path: str, example_num: int, hitl_llm_state_path: str, example_label: str, nl_feedback: str, MODELS: dict, kDebug=False, do_only: Optional[str] = None, data_table_dirname: str = "SSR"):
        self.examples_root_path = examples_root_path
        self.example_num = example_num
        self.hitl_llm_state_path = hitl_llm_state_path
        self.example_label = example_label
        self.nl_feedback = nl_feedback
        self.kDebug = kDebug
        self.do_only = do_only
        self.MODELS = MODELS
        self.data_rootpath = os.path.join(self.examples_root_path, "syncdata", str(self.example_num))
        self.data_table_path = os.path.join(self.examples_root_path, data_table_dirname)
        os.makedirs(self.data_table_path, exist_ok=True)
        self.jsonf = {}  # final json to be written to the data table
        self.images_bgr = []
        self.pcs_xyz = []
        self.localizations = []
        self.ego_ldict = None
        self.read_data_and_initialize()

        self.terrain_module = NSTrainTerrainSeg(images_bgr=self.images_bgr,
                                                pcs_xyz=self.pcs_xyz,
                                                localizations=self.localizations,
                                                ego_ldict=self.ego_ldict,
                                                MODELS=self.MODELS,
                                                kDebug=False)

        self.objdet_module = NSTrainObjDet(images_bgr=self.images_bgr,
                                           pcs_xyz=self.pcs_xyz,
                                           localizations=self.localizations,
                                           ego_ldict=self.ego_ldict,
                                           MODELS=self.MODELS,
                                           kDebug=False)

        self.load_reqd()
        self.store_json()

    def load_reqd(self):
        if self.do_only is None:
            hitl_llm_state = json_reader(self.hitl_llm_state_path)
            self.jsonf["hitl_llm_state"] = hitl_llm_state
            self.jsonf["ldips_synth_program_sketch"] = self.convert_LFPS_to_LSPS(hitl_llm_state["ldips_func_program_sketch"])
            self.jsonf["label"] = self.example_label
            self.jsonf["human_nl"] = self.nl_feedback
            if self.kDebug:
                print("self.jsonf['label']", self.jsonf["label"])

            # parsing ldips sketch to get features needed to be computed
            self.jsonf["ldips_features"] = {}
            all_pred_names = self.extract_ldips_state_vars(hitl_llm_state["ldips_func_program_sketch"])
            if "terrain" in all_pred_names:
                self.jsonf["ldips_features"]["terrain"] = self.get_terrain(hitl_llm_state["domain"]["terrains"])
            if "in_the_way" in all_pred_names:
                self.jsonf["ldips_features"]["in_the_way"] = self.get_in_the_way(hitl_llm_state["domain"]["terrains"])
            if "slope" in all_pred_names:
                self.jsonf["ldips_features"]["slope"] = self.get_slope(hitl_llm_state["domain"]["terrains"])

            # parse to get distance_to objnames and frontal_distance_to objnames
            distance_to_objects = [name.split('distance_to_')[1] for name in all_pred_names if 'distance_to_' in name]
            frontal_objects = [name.split('frontal_distance_')[1] for name in all_pred_names if 'frontal_distance_' in name]
            distances_objs = self.get_distances_to_objects(distance_to_objects)  # returns a dict
            frontal_distances_objs = self.get_frontal_distances_to_objects(frontal_objects)  # returns a dict

            for obj in distance_to_objects:
                self.jsonf["ldips_features"][f"distance_to_{obj}"] = distances_objs[obj]

            for obj in frontal_objects:
                self.jsonf["ldips_features"][f"frontal_distance_{obj}"] = frontal_distances_objs[obj]
        elif self.do_only == "REPAIR":
            # retain ldips_features, modify else part
            json_path = os.path.join(self.data_table_path, f"{self.example_num:03}.json")
            self.jsonf = json_reader(json_path)
            hitl_llm_state = json_reader(self.hitl_llm_state_path)
            self.jsonf["hitl_llm_state"] = hitl_llm_state
            self.jsonf["ldips_synth_program_sketch"] = self.convert_LFPS_to_LSPS(hitl_llm_state["ldips_func_program_sketch"])
            self.jsonf["label"] = self.example_label
            self.jsonf["human_nl"] = self.nl_feedback
        else:
            json_path = os.path.join(self.data_table_path, f"{self.example_num:03}.json")
            self.jsonf = json_reader(json_path)
            if self.do_only == "terrain":
                self.jsonf["ldips_features"]["terrain"] = self.get_terrain(self.jsonf["hitl_llm_state"]["domain"]["terrains"])
            elif self.do_only == "in_the_way":
                self.jsonf["ldips_features"]["in_the_way"] = self.get_in_the_way(self.jsonf["hitl_llm_state"]["domain"]["terrains"])
            elif self.do_only == "slope":
                self.jsonf["ldips_features"]["slope"] = self.get_slope(self.jsonf["hitl_llm_state"]["domain"]["terrains"])
            elif self.do_only == "distance_to":
                distance_to_objects = [name.split('distance_to_')[1] for name in self.jsonf["ldips_features"].keys() if 'distance_to_' in name]
                distances_objs = self.get_distances_to_objects(distance_to_objects)
                for obj in distance_to_objects:
                    self.jsonf["ldips_features"][f"distance_to_{obj}"] = distances_objs[obj]
            elif self.do_only == "frontal_distance":
                frontal_objects = [name.split('frontal_distance_')[1] for name in self.jsonf["ldips_features"].keys() if 'frontal_distance_' in name]
                frontal_distances_objs = self.get_frontal_distances_to_objects(frontal_objects)
                for obj in frontal_objects:
                    self.jsonf["ldips_features"][f"frontal_distance_{obj}"] = frontal_distances_objs[obj]
            else:
                raise ValueError(f"Invalid value for self.do_only: {self.do_only}")

    def get_terrain(self, domain_terrains: list):
        print("=========================================================> get_terrain")
        DEFAULT_VAL = 1111
        terrain_voting = {"Not visible": 0}
        for i in range(len(self.images_bgr)):
            print(f"Processing image {i+1}/{len(self.images_bgr)}")
            pred_terrain, pred_seg, new_terrains = self.terrain_module.main_terrain(i, domain_terrains)
            terrain_voting[pred_terrain] = terrain_voting.get(pred_terrain, 0) + 1
        if self.kDebug:
            print("terrain_voting", terrain_voting)
        terrain_voting.pop("Not visible", None)
        terrain_voting.pop("dunno", None)
        if len(terrain_voting) == 0:
            if self.kDebug:
                print("Terrain unidentifiable in any image")
            return DEFAULT_VAL
        max_terrain = max(terrain_voting, key=terrain_voting.get)
        try:
            terrain_idx = NSLABELS_TWOWAY_NSINT[max_terrain]
            return terrain_idx
        except KeyError:
            if self.kDebug:
                print(f"*******ERROR: Terrain {max_terrain} not in PREDEFINED_TERRAINS")
            return DEFAULT_VAL

    def get_in_the_way(self, domain_terrains: list):
        print("=========================================================> get_in_the_way")
        DEFAULT_VAL = 0
        in_the_way_voting = {None: 0}
        for i in range(len(self.images_bgr)):
            print(f"Processing image {i+1}/{len(self.images_bgr)}")
            in_the_way = self.terrain_module.main_in_the_way(i, domain_terrains, param=self.WAY_PARAM)
            if in_the_way is not None:
                in_the_way = int(in_the_way)  # convert to 0 or 1
            in_the_way_voting[in_the_way] = in_the_way_voting.get(in_the_way, 0) + 1
        if self.kDebug:
            print("in_the_way_voting", in_the_way_voting)
        in_the_way_voting.pop(None, None)
        if len(in_the_way_voting) == 0:
            if self.kDebug:
                print("in_the_way unidentifiable in any image as ego not visible")
            return DEFAULT_VAL
        max_in_the_way = max(in_the_way_voting, key=in_the_way_voting.get)
        return max_in_the_way

    def get_slope(self, domain_terrains: list):
        print("=========================================================> get_slope")
        sum_slope = 0.0  # DEFAULT_VAL
        tot_count = 0
        for i in range(len(self.images_bgr)):
            print(f"Processing image {i+1}/{len(self.images_bgr)}")
            max_grad = self.terrain_module.main_slope(i, scale=self.SLOPE_SCALE)
            if max_grad is not None:
                sum_slope += max_grad
                tot_count += 1
            if self.kDebug:
                print("max_zdiff", max_grad)
        if tot_count == 0:
            if self.kDebug:
                print("Slope unidentifiable in any image as ego not visible")
            return sum_slope
        return sum_slope / tot_count

    def get_distances_to_objects(self, object_names: list):
        print("=========================================================> get_distances_to_objects")
        DEFAULT_VAL = 1111.0
        min_distances = {}
        for obj in object_names:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing object {obj}")
            min_distances[obj] = DEFAULT_VAL
            for i in range(len(self.images_bgr)):
                print(f"Processing image {i+1}/{len(self.images_bgr)}")
                dist = self.objdet_module.main_distance_to(i, obj,
                                                           box_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[0],
                                                           text_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[1],
                                                           nms_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[2])
                if self.kDebug:
                    print("dist", dist)
                if dist is not None:
                    min_distances[obj] = min(min_distances[obj], dist)
        return min_distances

    def get_frontal_distances_to_objects(self, object_names: list):
        print("=========================================================> get_frontal_distances_to_objects")
        DEFAULT_VAL = 1111.0
        min_distances = {}
        for obj in object_names:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing object {obj}")
            min_distances[obj] = DEFAULT_VAL
            for i in range(len(self.images_bgr)):
                print(f"Processing image {i+1}/{len(self.images_bgr)}")
                dist = self.objdet_module.main_frontal_distance(i, obj,
                                                                box_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[0],
                                                                text_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[1],
                                                                nms_threshold=self.OBJECT_THRESHOLDS.get(obj, [None])[2])
                if self.kDebug:
                    print("dist", dist)
                if dist is not None and not np.isinf(dist):
                    min_distances[obj] = min(min_distances[obj], dist)
        return min_distances

    @staticmethod
    def extract_ldips_state_vars(function_str):
        # Regular expression to find all occurrences of function names before "(pixel_loc)"
        function_names = re.findall(r'(\w+)\(pixel_loc\)', function_str)
        # Remove 'is_safe' and duplicates, and sort the list
        unique_function_names = sorted(set(function_names) - {'is_safe'})
        return unique_function_names

    def read_data_and_initialize(self):
        """
        reads and sorts bgr images, pcs_xyz, and localization info
        """
        # Read images
        image_dir = os.path.join(self.data_rootpath, "images")
        image_names = sorted(os.listdir(image_dir))
        image_fullpaths = [os.path.join(image_dir, image_name) for image_name in image_names]
        image_fullpaths.sort()
        self.images_bgr = [cv2.imread(image_fullpath) for image_fullpath in image_fullpaths]

        # Read pcs_xyz
        pcs_dir = os.path.join(self.data_rootpath, "pcs")
        pcs_names = sorted(os.listdir(pcs_dir))
        pcs_fullpaths = [os.path.join(pcs_dir, pcs_name) for pcs_name in pcs_names]
        pcs_fullpaths.sort()
        for pcf in pcs_fullpaths:
            pc_np = np.fromfile(pcf, dtype=np.float32).reshape((-1, 4))
            pc_np = pc_np[:, :3]
            self.pcs_xyz.append(pc_np)

        # Read localization info
        loc_dir = os.path.join(self.data_rootpath, "locs")
        loc_names = sorted(os.listdir(loc_dir))
        loc_fullpaths = [os.path.join(loc_dir, loc_name) for loc_name in loc_names]
        loc_fullpaths.sort()
        for lcf in loc_fullpaths:
            with open(lcf, 'r') as f:
                ldict = yaml.safe_load(f)
                self.localizations.append(ldict)  # x y theta

        self.ego_ldict = self.localizations[-1]

    @staticmethod
    def replace_question_marks(s):
        def replacer(match):
            replacer.counter += 1
            return f"pX{replacer.counter:03}"

        replacer.counter = 0
        return re.sub(r'\?\?', replacer, s)

    @staticmethod
    def convert_LFPS_to_LSPS(sketch):
        if sketch == "":
            return sketch
        sketch = sketch.replace("(pixel_loc)", "")
        sketch = sketch.replace("is_safe", "is_safe(pixel_loc)")
        sketch = LDIPSdatagenSingleEx.replace_question_marks(sketch)
        sketch = sketch.strip()
        return sketch

    @staticmethod
    def fillparams_in_LFPS(sketch, params):
        """
        sketch: string representing the lfps form of sketch
        params: list of tuples of params to be filled in the sketch
        """
        if sketch == "":
            return sketch
        num_sketch = LDIPSdatagenSingleEx.replace_question_marks(sketch)
        for k, v in params:
            num_sketch = num_sketch.replace(k, str(v))
        return num_sketch

    @staticmethod
    def fillparams_in_LSPS(sketch, params):
        """
        sketch: string representing the lsps form of sketch
        params: list of tuples of params to be filled in the sketch
        """
        if sketch == "":
            return sketch
        for k, v in params:
            sketch = sketch.replace(k, str(v))
        return sketch

    def store_json(self):
        json_path = os.path.join(self.data_table_path, f"{self.example_num:03}.json")
        json_writer(filepath=json_path,
                    dict_content=self.jsonf)


if __name__ == "__main__":
    nl_feedbacks_path = os.path.join(nspl_root_dir, "demonstrations/nl_feedback.txt")
    state_path = os.path.join(nspl_root_dir, "scripts/llm/state.json")
    with open(nl_feedbacks_path, "r") as f:
        nl_feedbacks = f.readlines()
    all_labels = extract_all_labels(nl_feedbacks, state_path)
    print("*********************************ALL LABELS extracted**************************************")
    lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)
    local_to_map_frame = LocalToMapFrame()
    terrain_model = TerrainSegFormer(hf_model_ver=None)
    terrain_model.load_model_inference()
    terrain_model.prepare_dataset()
    sam_dino_model = GroundedSAM(box_threshold=0.5, text_threshold=0.5, nms_threshold=0.6)
    MODELS = {"lidar_cam_calib": lidar_cam_calib,
              "local_to_map_frame": local_to_map_frame,
              "terrain_model": terrain_model,
              "sam_dino_model": sam_dino_model}
    print("*********************************ALL MODELS loaded**************************************")
    START = 1
    END = 29
    for i in range(START, END + 1):
        print(f"----------------------------------------------------------------------------------------------------------Processing example {i}")
        datagen = LDIPSdatagenSingleEx(examples_root_path=os.path.join(nspl_root_dir, "demonstrations"),
                                       example_num=i,
                                       hitl_llm_state_path=state_path,
                                       example_label=all_labels[i - 1],
                                       nl_feedback=nl_feedbacks[i - 1].strip(),
                                       MODELS=MODELS,
                                       kDebug=False,
                                       do_only=None)
        print(f"----------------------------------------------------------------------------------------------------------DONE example {i}")
