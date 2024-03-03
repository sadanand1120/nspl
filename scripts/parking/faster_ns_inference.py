import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2

from terrainseg.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, TERRAINMARKS_NSLABELS_TWOWAY_NSINT, TERRAINMARKS_DATASETINTstr_TO_DATASETLABELS, TERRAINMARKS_DATASETLABELS_TO_NSLABELS
from terrainseg.inference import TerrainSegFormer
from utilities.std_utils import json_reader
from parking.ldips_inference import NSInferObjDet, NSInferTerrainSeg


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
        terrainmarks_model = TerrainSegFormer(hf_dataset_name="sam1120/parking-terrain_marks", hf_model_name="sam1120/parking-terrain_marks")
        terrainmarks_model.load_model_inference()
        terrainmarks_model.prepare_dataset()
        self.ns_infer_terrainmarksseg = NSInferTerrainSeg(MODELS={"terrain_model": terrainmarks_model})
        self.domain = domain
        self.predefined_terrains = NSLABELS_TWOWAY_NSINT
        self.predefined_terrains["dunno"] = 1111
        self.predefined_terrains[1111] = "dunno"
        self.predefined_terrainmarks = TERRAINMARKS_NSLABELS_TWOWAY_NSINT
        self.predefined_terrainmarks["dunno"] = 1111
        self.predefined_terrainmarks[1111] = "dunno"
        self.fi_data_dir = None
        self.cur_noext_name = None
        self.cur_img_bgr = None
        self.cur_pc_xyz = None
        self.cur_main_terrain_output = None
        self.cur_main_terrainmarks_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs
        self.cur_available_output = None  # dict of objects to outputs
        self.cur_within_output = None  # dict of objects to outputs
        self.USE_GT_TERRAIN = True
        self.USE_GT_TERRAINMARKS = True

    def set_state(self, fi_data_dir, noext_name, img_bgr, pc_xyz):
        self.fi_data_dir = fi_data_dir
        self.noext_name = noext_name
        self.cur_img_bgr = img_bgr
        self.cur_pc_xyz = pc_xyz
        self.cur_main_terrain_output = None
        self.cur_main_terrainmarks_output = None
        self.cur_main_in_the_way_output = None
        self.cur_main_slope_output = None
        self.cur_distance_to_output = None  # dict of objects to outputs
        self.cur_frontal_distance_output = None  # dict of objects to outputs
        self.cur_available_output = None  # dict of objects to outputs
        self.cur_within_output = None  # dict of objects to outputs

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
        elif name.startswith("_available_"):
            def dynamic_method(arg):
                target = name[len("_available_"):]
                return self._available(arg, target)
            return dynamic_method
        elif name.startswith("_within_"):
            def dynamic_method(arg):
                target = name[len("_within_"):]
                return self._within(arg, target)
            return dynamic_method
        raise AttributeError(f"{name} not found")

    def _terrain(self, pixel_loc):
        """
        Returns the terrain idx at the given pixel location
        """
        if self.cur_main_terrain_output is None:
            self.cur_main_terrain_output = self.terrain(self.cur_img_bgr, self.cur_pc_xyz)
        return int(self.cur_main_terrain_output[pixel_loc[1], pixel_loc[0]])

    def _terrainmarks(self, pixel_loc):
        """
        Returns the terrainmark idx at the given pixel location
        """
        if self.cur_main_terrainmarks_output is None:
            self.cur_main_terrainmarks_output = self.terrainmarks(self.cur_img_bgr, self.cur_pc_xyz)
        return int(self.cur_main_terrainmarks_output[pixel_loc[1], pixel_loc[0]])

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

    def _available(self, pixel_loc, obj_name):
        """
        Returns the available value
        """
        if self.cur_available_output is None:
            self.cur_available_output = {}
        if obj_name not in self.cur_available_output.keys():
            self.cur_available_output[obj_name] = self.available_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return bool(self.cur_available_output[obj_name])

    def _within(self, pixel_loc, obj_name):
        """
        Returns the within value
        """
        if self.cur_within_output is None:
            self.cur_within_output = {}
        if obj_name not in self.cur_within_output.keys():
            self.cur_within_output[obj_name] = self.within_obj(self.cur_img_bgr, self.cur_pc_xyz, obj_name)
        return bool(self.cur_within_output[obj_name][pixel_loc[1], pixel_loc[0]])

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
        if self.USE_GT_TERRAIN:
            gt_terrain_path = os.path.join(nspl_root_dir, "evals_data_parking/utcustom/gt_terrains", f"{self.noext_name}.bin")
            gt_seg = np.fromfile(gt_terrain_path, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        else:
            gt_seg = None
        cur_main_terrain_output = self.ns_infer_terrainseg.main_terrain(img_bgr, self.domain["terrains"], gt_seg=gt_seg)
        pred_seg, new_terrains = cur_main_terrain_output
        new_terrains_array = np.array(new_terrains)
        result = new_terrains_array[pred_seg]
        map_to_ldips = np.vectorize(lambda x: self.predefined_terrains.get(x, 1111))
        ret_out = map_to_ldips(result).squeeze().reshape((img_bgr.shape[0], img_bgr.shape[1]))
        flat_ret_out = ret_out.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def terrainmarks(self, img_bgr, pc_xyz):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_terrainmarks.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        if self.USE_GT_TERRAINMARKS:
            gt_terrainmarks_path = os.path.join(nspl_root_dir, "evals_data_parking/utcustom/gt_terrainmarks", f"{self.noext_name}.bin")
            gt_seg = np.fromfile(gt_terrainmarks_path, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        else:
            gt_seg = None
        cur_main_terrainmarks_output = self.ns_infer_terrainmarksseg.main_terrain(img_bgr, self.domain["terrainmarks"], gt_seg=gt_seg)
        pred_seg, new_terrainmarks = cur_main_terrainmarks_output
        new_terrainmarks_array = np.array(new_terrainmarks)
        result = new_terrainmarks_array[pred_seg]
        map_to_ldips = np.vectorize(lambda x: self.predefined_terrainmarks.get(x, 1111))
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
        if obj_name in self.domain["terrains"]:
            ego_terrain_idx = self.predefined_terrains[obj_name]
            terrain_out = self.terrain(img_bgr, pc_xyz)
            per_class_mask = (terrain_out == ego_terrain_idx).squeeze()
        elif obj_name in self.domain["terrainmarks"]:
            ego_terrainmark_idx = self.predefined_terrainmarks[obj_name]
            terrainmarks_out = self.terrainmarks(img_bgr, pc_xyz)
            per_class_mask = (terrainmarks_out == ego_terrainmark_idx).squeeze()
        else:
            per_class_mask = None
        cur_distance_to_obj_output = self.ns_infer_objdet.main_distance_to(img_bgr, pc_xyz, obj_name,
                                                                           box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                           text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                           nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                           per_class_mask=per_class_mask)
        dist_arr, found = cur_distance_to_obj_output
        ret_out = dist_arr.reshape((img_bgr.shape[0], img_bgr.shape[1])).astype(np.float32)
        flat_ret_out = ret_out.reshape(-1).astype(np.float32)
        flat_ret_out.tofile(fullpath)
        return ret_out

    def available_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_available_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))[0][0]
        ego_terrainmark_idx = self.predefined_terrainmarks[obj_name]
        terrainmarks_out = self.terrainmarks(img_bgr, pc_xyz)
        per_class_mask = (terrainmarks_out == ego_terrainmark_idx).astype(np.uint8).squeeze()
        tot = np.sum(per_class_mask)
        b = tot > 100  # more than 100 pixels
        bool_arr = np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * b
        flat_ret_out = bool_arr.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return b

    def within_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_within_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.uint8).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        if not self.available_obj(img_bgr, pc_xyz, obj_name):
            return np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)

        dist_to_obj = self.distance_to_obj(img_bgr, pc_xyz, obj_name)

        # ego_terrainmark_idx = self.predefined_terrainmarks[obj_name]
        # terrainmarks_out = self.terrainmarks(img_bgr, pc_xyz)
        # per_class_mask_obj = (terrainmarks_out == ego_terrainmark_idx).astype(np.uint8).squeeze()

        # # distance to obj -> obj_name, sidewalk, ELSE
        # dist_to_obj = self.distance_to_obj(img_bgr, pc_xyz, obj_name)
        # dist_to_sidewalk = self.distance_to_obj(img_bgr, pc_xyz, "sidewalk")
        # dist_to_else = self.distance_to_obj(img_bgr, pc_xyz, "ELSE")
        # dist_min_sidewalk_else = np.minimum(dist_to_sidewalk, dist_to_else)

        bool_arr = (0.0 < dist_to_obj) & (dist_to_obj < 2.0)
        flat_ret_out = bool_arr.reshape(-1).astype(np.uint8)
        flat_ret_out.tofile(fullpath)
        return bool_arr

    def frontal_distance_to_obj(self, img_bgr, pc_xyz, obj_name):
        fullpath = os.path.join(self.fi_data_dir, f"{self.noext_name}_frontal_distance_to_{obj_name}.bin")
        if os.path.exists(fullpath):
            return np.fromfile(fullpath, dtype=np.float32).reshape((img_bgr.shape[0], img_bgr.shape[1]))
        cur_frontal_distance_to_obj_output = self.ns_infer_objdet.main_frontal_distance(img_bgr, pc_xyz, obj_name,
                                                                                        box_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                                        text_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0],
                                                                                        nms_threshold=self.OBJECT_THRESHOLDS.get(obj_name, [None])[0])
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
