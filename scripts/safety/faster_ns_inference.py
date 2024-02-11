import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2

from terrainseg.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, NSLABELS_TRAVERSABLE_TERRAINS, NSLABELS_NON_TRAVERSABLE_TERRAINS
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import json_reader
from safety.ldips_inference import NSInferObjDet, NSInferTerrainSeg


class FasterImageInference:
    WAY_PARAM = 1.7
    SLOPE_SCALE = 0.0  # HACK
    OBJECT_THRESHOLDS = {  # 3-tuple of (box_threshold, text_threshold, nms_threshold)
        "barricade": (0.5, 0.5, 0.3),
        "board": (0.3, 0.3, 0.5),
        "bush": (0.4, 0.4, 0.4),
        "car": (0.3, 0.3, 0.3),
        "entrance": (0.3, 0.3, 0.2),
        "person": (0.25, 0.25, 0.6),
        "pole": (0.4, 0.4, 0.5),
        "staircase": (0.25, 0.25, 0.4),
        "tree": (0.4, 0.4, 0.45),
        "wall": (0.5, 0.5, 0.4)
    }

    def __init__(self, domain):
        print("Initializing FasterImageInference...")
        self.ns_infer_objdet = NSInferObjDet()
        self.ns_infer_terrainseg = NSInferTerrainSeg()
        self.domain = domain
        self.predefined_terrains = NSLABELS_TWOWAY_NSINT
        self.predefined_terrains["dunno"] = 1111
        self.predefined_terrains[1111] = "dunno"
        self.fi_data_dir = None
        self.cur_noext_name = None
        self.cur_img_bgr = None
        self.cur_pc_xyz = None
        self.cur_main_terrain_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs

    def set_state(self, fi_data_dir, noext_name, img_bgr, pc_xyz):
        self.fi_data_dir = fi_data_dir
        self.noext_name = noext_name
        self.cur_img_bgr = img_bgr
        self.cur_pc_xyz = pc_xyz
        self.cur_main_terrain_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs

    def __getattr__(self, name):
        if name.startswith("_distance_to_"):
            def dynamic_method(arg):
                target = name[len("_distance_to_"):]
                return self._distance_to(arg, target)
            return dynamic_method
        elif name.startswith("_frontal_distance_"):
            def dynamic_method(arg):
                target = name[len("_frontal_distance_"):]
                return self._frontal_distance(arg, target)
            return dynamic_method
        raise AttributeError(f"{name} not found")

    def _terrain(self, pixel_loc):
        """
        Returns the terrain idx at the given pixel location
        """
        if self.cur_main_terrain_output is None:
            self.cur_main_terrain_output = self.terrain(self.cur_img_bgr, self.cur_pc_xyz)
        return int(self.cur_main_terrain_output[pixel_loc[1], pixel_loc[0]])

    def _in_the_way(self, pixel_loc):
        """
        Returns whether the given pixel location is in the way
        """
        if self.cur_main_in_the_way_output is None:
            self.cur_main_in_the_way_output = self.in_the_way(self.cur_img_bgr, self.cur_pc_xyz)
        return bool(self.cur_main_in_the_way_output[pixel_loc[1], pixel_loc[0]])

    def _slope(self, pixel_loc):
        """
        Returns the slope at the given pixel location
        """
        if self.cur_main_slope_output is None:
            self.cur_main_slope_output = self.slope(self.cur_img_bgr, self.cur_pc_xyz)
        return float(self.cur_main_slope_output[pixel_loc[1], pixel_loc[0]])

    def _distance_to(self, pixel_loc, obj_name):
        """
        Returns the distance to the given object at the given pixel location
        """
        if self.cur_distance_to_output is None:
            self.cur_distance_to_output = {}
        if obj_name not in self.cur_distance_to_output.keys():
            self.cur_distance_to_output[obj_name] = self.distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return float(self.cur_distance_to_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def _frontal_distance(self, pixel_loc, obj_name):
        """
        Returns the frontal distance to the given object at the given pixel location
        """
        if self.cur_frontal_distance_output is None:
            self.cur_frontal_distance_output = {}
        if obj_name not in self.cur_frontal_distance_output.keys():
            self.cur_frontal_distance_output[obj_name] = self.frontal_distance_to_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return float(self.cur_frontal_distance_output[obj_name][pixel_loc[1], pixel_loc[0]])

    def terrain(self, img_bgr, pc_xyz):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_terrain.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_main_terrain_output = self.ns_infer_terrainseg.main_terrain(img_bgr, self.domain["terrains"])
        pred_seg, new_terrains = cur_main_terrain_output
        new_terrains_array = np.array(new_terrains)
        result = new_terrains_array[pred_seg]
        map_to_ldips = np.vectorize(lambda x: self.predefined_terrains.get(x, 1111))
        ret_out = map_to_ldips(result).squeeze().reshape((img_bgr.shape[0], img_bgr.shape[1]))
        flat_ret_out = ret_out.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def in_the_way(self, img_bgr, pc_xyz):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_in_the_way.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_main_in_the_way_output = self.ns_infer_terrainseg.main_in_the_way(img_bgr, pc_xyz, self.domain["terrains"], param=self.WAY_PARAM)
        ret_out = cur_main_in_the_way_output.reshape((img_bgr.shape[0], img_bgr.shape[1]))
        flat_ret_out = ret_out.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def slope(self, img_bgr, pc_xyz):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_slope.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.float32).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_main_slope_output = self.ns_infer_terrainseg.main_slope(img_bgr, pc_xyz, scale=self.SLOPE_SCALE)
        ret_out = cur_main_slope_output.reshape((img_bgr.shape[0], img_bgr.shape[1]))
        flat_ret_out = ret_out.reshape(-1).astype(np.float32)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_distance_to_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.float32).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_distance_to_obj_output = self.ns_infer_objdet.main_distance_to(img_bgr, pc_xyz, obj_name,
                                                                           box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                           text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[1],
                                                                           nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[2])
        dist_arr, found = cur_distance_to_obj_output
        ret_out = dist_arr.reshape((img_bgr.shape[0], img_bgr.shape[1])).astype(np.float32)
        flat_ret_out = ret_out.reshape(-1).astype(np.float32)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def frontal_distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_frontal_distance_to_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.float32).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_frontal_distance_to_obj_output = self.ns_infer_objdet.main_frontal_distance(img_bgr, pc_xyz, obj_name,
                                                                                        box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                                        text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[1],
                                                                                        nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[2])
        dist_arr, found = cur_frontal_distance_to_obj_output
        ret_out = dist_arr.reshape((img_bgr.shape[0], img_bgr.shape[1])).astype(np.float32)
        flat_ret_out = ret_out.reshape(-1).astype(np.float32)
        flat_ret_out.tofile(fullpath)
        return ret_out


class FasterImageInferenceCaP(FasterImageInference):
    def __init__(self, domain):
        super().__init__(domain)

    def frontal_distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"cap_{self.noext_name}_frontal_distance_to_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.float32).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        ret_out = self.distance_to_obj(img_bgr, pc_xyz, obj_name)
        flat_ret_out = ret_out.reshape(-1).astype(np.float32)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def in_the_way(self, img_bgr, pc_xyz):
        fullpath = os.path.join(self.fi_data_dir, f"cap_{self.noext_name}_in_the_way.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        ret_out_terrain = self.terrain(img_bgr, pc_xyz)
        traversable_indices = [NSLABELS_TWOWAY_NSINT[label] for label in NSLABELS_TRAVERSABLE_TERRAINS]
        traversable_mask = np.isin(ret_out_terrain, traversable_indices)
        not_terrain_or_not_traversable_mask = ~traversable_mask
        ret_out = not_terrain_or_not_traversable_mask.reshape((img_bgr.shape[0], img_bgr.shape[1]))
        flat_ret_out = ret_out.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return ret_out


if __name__ == "__main__":
    hitl_llm_state = json_reader(os.path.join(nspl_root_dir, "scripts/llm/state.json"))

    CUR_IMG_BGR = cv2.imread(os.path.join(nspl_root_dir, "evals_data_safety/utcustom/train/utcustom/images/000102_morning_random_3_11042023_000200.png"))
    CUR_PC_XYZ = np.fromfile(os.path.join(nspl_root_dir, "evals_data_safety/utcustom/train/utcustom/pcs/000102_morning_random_3_11042023_000200.bin"), dtype=np.float32).reshape((-1, 4))[:, :3]
    DOMAIN = hitl_llm_state["domain"]
    fi = FasterImageInference(DOMAIN)
    res = fi.exec_program(CUR_IMG_BGR, CUR_PC_XYZ).astype(np.uint8)
    overlay = TerrainSegFormer.get_seg_overlay(CUR_IMG_BGR, res, alpha=0.24)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
