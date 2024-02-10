import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2
import yaml
import cv2
import yaml
from typing import Optional

from terrainseg.inference import TerrainSegFormer
from utilities.local_to_map_frame import LocalToMapFrame
from zeroshot_objdet.sam_dino import GroundedSAM
from third_party.jackal_calib import JackalLidarCamCalibration
from ldips_datagen import LDIPSdatagenSingleEx, extract_all_labels, NSTrainTerrainSeg, NSTrainObjDet


class LDIPSdatagenSingleEx_notraj(LDIPSdatagenSingleEx):
    def __init__(self, examples_root_path: str, example_num: int, hitl_llm_state_path: str, example_label: str, nl_feedback: str, MODELS: dict, kDebug=False, do_only: Optional[str] = None, data_table_dirname: str = "SSR_notraj"):
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
        self.truncate_to_first()

    def truncate_to_first(self):
        """
        Only keeps the first sample for image, pc and localizations. This is for the notraj ablation study.
        """
        self.images_bgr = self.images_bgr[0:1]
        self.pcs_xyz = self.pcs_xyz[0:1]
        self.localizations = self.localizations[0:1]


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
        datagen = LDIPSdatagenSingleEx_notraj(examples_root_path=os.path.join(nspl_root_dir, "demonstrations"),
                                              example_num=i,
                                              hitl_llm_state_path=state_path,
                                              example_label=all_labels[i - 1],
                                              nl_feedback=nl_feedbacks[i - 1].strip(),
                                              MODELS=MODELS,
                                              kDebug=False,
                                              do_only=None)
        print(f"----------------------------------------------------------------------------------------------------------DONE example {i}")
