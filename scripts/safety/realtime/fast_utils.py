import os
import PIL.Image
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
from terrainseg.inference import TerrainSegFormer
import cv2
from PIL import Image
from simple_colors import green
import torch
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from third_party.lightHQSAM.setup_light_hqsam import setup_model
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import PIL
from PIL import Image
from terrainseg.inference import TerrainSegFormer
import numpy as np
import cv2
import cv2
from pykdtree.kdtree import KDTree
from PIL import Image
from terrainseg.inference import TerrainSegFormer
from third_party.jackal_calib import JackalLidarCamCalibration
import open3d as o3d
import argparse
import os
import torch
import cv2
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
import matplotlib.pyplot as plt
from third_party.Depth_Anything.metric_depth.zoedepth.models.builder import build_model
from third_party.Depth_Anything.metric_depth.zoedepth.utils.config import get_config
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
# grid = make_grid(images, nrow=4, padding=20, pad_value=1.0)

import os
import time


class CodeBlockTimer:
    def __init__(self, file_name):
        # Construct the full file path
        self.file_path = os.path.join(
            "/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/timings",
            file_name
        )

    def __enter__(self):
        # Record the start time when entering the context
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        # Record the end time when exiting the context
        end_time = time.time()
        # Calculate execution time in milliseconds
        execution_time = (end_time - self.start_time) * 1e3
        # Append the execution time to the file
        with open(self.file_path, 'a') as file:
            file.write(f"{execution_time:.3f}\n")  # Log time in milliseconds


def log_execution_time(file_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1e3
            file_path = os.path.join("/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/realtime/timings", file_name)
            with open(file_path, 'a') as file:
                file.write(f"{execution_time:.3f}\n")  # milliseconds
            return result
        return wrapper
    return decorator


def print_num_params(model: torch.nn.Module, modelname: str):
    tot_params = sum(p.numel() for p in model.parameters())
    print(green(f"Total parameters for {modelname}: {round(tot_params/1e6)}M", 'bold'))


class FastGSAM:
    def __init__(self, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6, ann_thickness=2, ann_text_scale=0.3, ann_text_thickness=1, ann_text_padding=5):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GROUNDING_DINO_CONFIG_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth")
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
        print_num_params(self.grounding_dino_model.model, "GroundingDINO")

        self.HQSAM_CHECKPOINT_PATH = os.path.join(nspl_root_dir, "third_party/Grounded-Segment-Anything/weights/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(self.HQSAM_CHECKPOINT_PATH)
        self.light_hqsam = setup_model()
        self.light_hqsam.load_state_dict(checkpoint, strict=True)
        self.light_hqsam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.light_hqsam)
        print_num_params(self.light_hqsam, "SAM-tiny")

        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.box_annotator = sv.BoxAnnotator(
            thickness=ann_thickness,
            text_scale=ann_text_scale,
            text_thickness=ann_text_thickness,
            text_padding=ann_text_padding
        )
        self.mask_annotator = sv.MaskAnnotator()

    @log_execution_time("fastgsam_dino.txt")
    def predict_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO on image.
        img: A x B x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels
            detections:
                - xyxy: (N, 4) boxes (float pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
        """
        if box_threshold is None:
            box_threshold = self.BOX_THRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.NMS_THRESHOLD
        detections = self.grounding_dino_model.predict_with_classes(
            image=img,
            classes=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # print(f"Detected {len(detections.xyxy)} boxes")
        if do_nms:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            # print(f"After NMS: {len(detections.xyxy)} boxes")
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        return annotated_frame, detections

    @log_execution_time("fastgsam_sam.txt")
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    @staticmethod
    def get_per_class_mask(img, masks, class_ids, num_classes):
        """
        Create a per-class segmentation mask.
        Parameters:
            masks: N x H x W array, where N is the number of masks
            class_ids: (N,) array of corresponding class ids
            num_classes: Total number of classes, C
        Returns:
            per_class_mask: C x H x W array
        """
        H, W = img.shape[0], img.shape[1]
        per_class_mask = np.zeros((num_classes, H, W), dtype=bool)
        if len(masks) == 0:
            return per_class_mask
        for i in range(num_classes):
            class_idx = np.where(class_ids == i)[0]
            if class_idx.size > 0:
                per_class_mask[i] = np.any(masks[class_idx], axis=0)
        return per_class_mask

    def predict_and_segment_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO and segmentation using HQ-SAM on image.
        img: H x W x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels and segment masks
            detections: If there are N detections,
                - xyxy: (N, 4) boxes (int pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
                - mask: (N, H, W) boolean segmentation masks, i.e., True at locations belonging to corresponding class
        """
        _, detections = self.predict_on_image(img, text_prompts, do_nms=do_nms, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = self.mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        detections.xyxy = detections.xyxy.astype(np.int32).reshape((-1, 4))
        return annotated_image, detections, FastGSAM.get_per_class_mask(img, detections.mask, detections.class_id, len(text_prompts))


class FastModels:
    @log_execution_time("fastmodels_init.txt")
    def __init__(self):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                # Terrain model
                # self.terrain_modelname = "sam1120/safety-utcustom-terrain-b0-optim"
                self.terrain_modelname = "sam1120/safety-utcustom-terrain"
                self.terrain_model = TerrainSegFormer(hf_dataset_name="sam1120/safety-utcustom-terrain-jackal-full-391", hf_model_name=self.terrain_modelname)
                self.terrain_model.load_model_inference()
                self.terrain_model.prepare_dataset()
                print_num_params(self.terrain_model.model, "TerrainSegFormer")

                # Grounded SAM object detection model
                self.gsam = FastGSAM(box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6)
                self.gsam_object_dict = {
                    "person": 0,
                    "bush": 1,
                    "car": 2,
                    "pole": 3,
                    "entrance": 4,
                    "staircase": 5
                }

                # Lidar to camera calibration
                self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)

                # Relative Depth model
                self.depth_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
                self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

                # Metric Depth model
                self.metric_depth_config = get_config("zoedepth", "eval", 'nyu')
                self.metric_depth_config.pretrained_resource = "local::/home/dynamo/AMRL_Research/repos/nspl/third_party/Depth_Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt"
                self.metric_depth_model = build_model(self.metric_depth_config).to('cuda' if torch.cuda.is_available() else 'cpu')
                self.metric_depth_model.eval()
                print_num_params(self.metric_depth_model, "MetricDepth")

    @log_execution_time("fastmodels_terrain_nn.txt")
    def terrain_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                pred_img, pred_seg = self.terrain_model.predict_new(image)
            return pred_img, pred_seg.squeeze().cpu().numpy()

    def depth_relative_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                inputs = self.depth_image_processor(images=image, return_tensors="pt")
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                return prediction.squeeze().cpu().numpy()

    @log_execution_time("fastmodels_metric_depth_nn.txt")
    def depth_metric_pred(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                color_image = image.convert('RGB')
                image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
                pred = self.metric_depth_model(image_tensor, dataset='nyu')
                if isinstance(pred, dict):
                    pred = pred.get('metric_depth', pred.get('out'))
                elif isinstance(pred, (list, tuple)):
                    pred = pred[-1]
                pred = pred.squeeze().detach().cpu().numpy()
                resized_pred = Image.fromarray(pred).resize((self.lidar_cam_calib.img_width, self.lidar_cam_calib.img_height), Image.NEAREST)
                return np.asarray(resized_pred)

    @log_execution_time("fastmodels_pc_metric_depth_nn.txt")
    def get_pc_from_depth(self, image: PIL.Image.Image):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                z = self.depth_metric_pred(image)
                image = image.convert('RGB')
                FX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 0]
                FY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 1]
                CX = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][0, 2]
                CY = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix'][1, 2]
                K = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['camera_matrix']
                d = self.lidar_cam_calib.jackal_cam_calib.intrinsics_dict['dist_coeffs']
                R = np.eye(3)
                x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
                # undistort pixel coordinates
                pcs_coords = np.stack((x.flatten(), y.flatten()), axis=-1).astype(np.float64)
                undistorted_pcs_coords = cv2.undistortPoints(pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
                undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
                x, y = np.split(undistorted_pcs_coords, 2, axis=1)
                x = x.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
                y = y.reshape(self.lidar_cam_calib.img_height, self.lidar_cam_calib.img_width)
                # back project (along the camera ray) the pixel coordinates to 3D using the depth
                """
                For understanding, ignore distortions. According to pinhole model, pinhole is at (0,0,0) in CCS and image plane at -f in z axis, and equivalent image plane at +f in z axis which is what we actually use for calculations.
                So similarity of triangles gives that, for (X, Y, Z) in CCS, we get the point mapped to (fxX/Z, fyY/Z, f) on image plane.
                Dropping z=f, we get pixel coordinates upon shifting origin to top-left corner, (cx + fxX/Z, cy + fyY/Z).
                So, HERE below, what we are basically doing is the inverse process of this, i.e., given the pixel coords x y z and the CCS depth Z:
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                """
                x = (x - CX) / FX
                y = (y - CY) / FY
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = np.asarray(image).reshape(-1, 3) / 255.0
                # convert to open3d point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                # o3d.visualization.draw_geometries([pcd])  # visualize the point cloud
                # pcd = o3d.io.read_point_cloud("pointcloud.ply")  # read the point cloud
                # o3d.io.write_point_cloud("pointcloud.ply", pcd)  # save the point cloud
                return z, pcd

    @log_execution_time("fastmodels_dino_sam_nn.txt")
    def gsam_pred(self, image: cv2.imread):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                ann_img, dets, per_class_mask = self.gsam.predict_and_segment_on_image(image, list(self.gsam_object_dict.keys()))
                return ann_img, dets, per_class_mask

    def depth_true(self, pc_xyz: np.ndarray):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            vlp_points = np.asarray(pc_xyz).astype(np.float64)
            wcs_coords = self.lidar_cam_calib.projectVLPtoWCS(vlp_points)
            ccs_coords = self.lidar_cam_calib.jackal_cam_calib.projectWCStoCCS(wcs_coords)
            corresponding_pcs_coords, mask = self.lidar_cam_calib.jackal_cam_calib.projectCCStoPCS(ccs_coords, mode='skip')
            ccs_coords = ccs_coords[mask]
            corresponding_ccs_depths = ccs_coords[:, 2].reshape((-1, 1))
            all_ys, all_xs = np.meshgrid(np.arange(self.lidar_cam_calib.img_height), np.arange(self.lidar_cam_calib.img_width))
            all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)
            all_ccs_depths, _ = self.lidar_cam_calib.double_interp(a=corresponding_pcs_coords, b=corresponding_ccs_depths, x=all_pixel_locs, do_nearest=True, firstmethod="linear")
            return all_pixel_locs, all_ccs_depths

    @log_execution_time("fastmodels_symbolic_execution.txt")
    def process(self, image: PIL.Image.Image, terrain_results, depth_results, gsam_results):
        with CodeBlockTimer("predicate_terrain.txt"):
            terrain_pred_seg = terrain_results[1].squeeze().copy()
            terrain_pred_seg[terrain_pred_seg == 0] = 20
            terrain_pred_seg[(terrain_pred_seg == 2) | (terrain_pred_seg == 6) | (terrain_pred_seg == 9)] = 0
            terrain_pred_seg[terrain_pred_seg != 0] = 1
            terrain_pred_seg_bool = terrain_pred_seg.astype(bool)
            inv_terrain_pred_seg_bool = ~terrain_pred_seg_bool

        with CodeBlockTimer("predicate_distance_to.txt"):
            per_class_mask = gsam_results[2]
            pcd = depth_results[1]
            all_pc_coords = np.asarray(pcd.points).reshape((-1, 3))
            x, y = np.meshgrid(np.arange(self.lidar_cam_calib.img_width), np.arange(self.lidar_cam_calib.img_height))
            all_pixel_locs = np.stack((x.flatten(), y.flatten()), axis=-1)
            terrain_pixel_locs = all_pixel_locs[inv_terrain_pred_seg_bool.flatten()]
            terrain_pc_coords = all_pc_coords[inv_terrain_pred_seg_bool.flatten()]

            distance_to_arrays = {}
            for cidx in range(per_class_mask.shape[0]):
                dist_arr = np.ones((image.height, image.width)) * (-1.0)
                class_mask = per_class_mask[cidx].squeeze()
                all_mask_values = class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
                class_pixel_locs = all_pixel_locs[all_mask_values]
                class_pc_coords = all_pc_coords[all_mask_values]
                if class_pc_coords is None or class_pc_coords.shape[0] == 0:
                    dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = np.inf
                else:
                    kdtree = KDTree(class_pc_coords)
                    distances, _ = kdtree.query(terrain_pc_coords)
                    dist_arr[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = distances
                distance_to_arrays[cidx] = dist_arr

        with CodeBlockTimer("predicate_in_the_way.txt"):
            in_the_way_mask = np.ones((image.height, image.width), dtype=np.uint8) * (-1)
            tpred_seg = terrain_results[1].squeeze()
            non_traversable_seg = ((tpred_seg == 0) | (tpred_seg == 1) | (tpred_seg == 3) | (tpred_seg == 13)).astype(bool)
            linearized_non_traversable_seg = non_traversable_seg.flatten()
            non_traversable_pc_coords = all_pc_coords[linearized_non_traversable_seg]
            # terrain_pc_xy = terrain_pc_coords[:, :2]
            # non_traversable_pc_xy = non_traversable_pc_coords[:, :2]
            tree_nontraversable = KDTree(non_traversable_pc_coords)
            distances, _ = tree_nontraversable.query(terrain_pc_coords)
            too_far_mask = (distances > 2.5).astype(np.uint8)
            in_the_way_mask[terrain_pixel_locs[:, 1], terrain_pixel_locs[:, 0]] = too_far_mask
        return terrain_pred_seg, distance_to_arrays, in_the_way_mask

    def _terrain(self, image: PIL.Image.Image, accumulated_results):
        return accumulated_results[0]

    def _in_the_way(self, image: PIL.Image.Image, accumulated_results):
        return accumulated_results[2]

    def _distance_to(self, image: PIL.Image.Image, class_name: str, accumulated_results):
        return accumulated_results[1][self.gsam_object_dict[class_name]]

    def _frontal_distance(self, image: PIL.Image.Image, class_name: str, accumulated_results):
        return self._distance_to(image, class_name, accumulated_results)

    @log_execution_time("fastmodels_predict_new.txt")
    def predict_new(self, image: PIL.Image.Image, do_car=True):
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            with torch.inference_mode():
                terrain_results = self.terrain_pred(image)
                depth_results = self.get_pc_from_depth(image)
                gsam_results = self.gsam_pred(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                accumulated_results = self.process(image, terrain_results, depth_results, gsam_results)
                if do_car:
                    return ((self._terrain(image, accumulated_results) == 0) &
                            (self._distance_to(image, "person", accumulated_results) > 2.0) &
                            (self._distance_to(image, "pole", accumulated_results) > 0.0) &
                            (self._in_the_way(image, accumulated_results) == 0) &
                            (self._frontal_distance(image, "entrance", accumulated_results) > 3.5) &
                            (self._distance_to(image, "bush", accumulated_results) > 0.0) &
                            (self._frontal_distance(image, "staircase", accumulated_results) > 3.5) &
                            (self._distance_to(image, "car", accumulated_results) > 8.0)
                            ).astype(np.uint8), accumulated_results, terrain_results, depth_results, gsam_results
                return ((self._terrain(image, accumulated_results) == 0) &
                        (self._distance_to(image, "person", accumulated_results) > 2.0) &
                        (self._distance_to(image, "pole", accumulated_results) > 0.0) &
                        (self._in_the_way(image, accumulated_results) == 0)
                        # (self._frontal_distance(image, "entrance", accumulated_results) > 3.5) &
                        # (self._distance_to(image, "bush", accumulated_results) > 0.0) &
                        # (self._frontal_distance(image, "staircase", accumulated_results) > 3.5) &
                        # (self._distance_to(image, "car", accumulated_results) > 8.0)
                        ).astype(np.uint8), accumulated_results, terrain_results, depth_results, gsam_results


# if __name__ == "__main__":
#     test_sample_image_path = "/home/dynamo/Music/jackal_bags/backup_unified_dataset/images/morning_mode1_2_11062023_000008.png"
#     with torch.inference_mode():
#         fm = FastModels()
#         # test_path = "/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/scripts/results/saved/compares/a_new_images/008.png"
#         # pred, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(Image.open(test_path))
#         pil_img = Image.open(test_sample_image_path)
#         pred, *_ = fm.predict_new(pil_img)
#         plt.imshow(pred * 255)
#         plt.show()
#         # counts of 0 and 1
#         print(np.unique(pred, return_counts=True))
#         print("Done")

if __name__ == "__main__":
    from tqdm.auto import tqdm
    images_dirpath = "/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/scripts/results/saved/compares/a_new_images"
    all_images = os.listdir(images_dirpath)
    all_images_paths = [os.path.join(images_dirpath, img) for img in all_images if img.endswith(".png")]
    all_images_paths.sort()
    pil_preds_np = []
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    with torch.inference_mode():
        fm = FastModels()
        for idx, img_path in enumerate(tqdm(all_images_paths)):
            pil_img = Image.open(img_path)
            cv2_img_np = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            pred, *_ = fm.predict_new(pil_img)
            fm.gsam.BOX_THRESHOLD = 0.4
            fm.gsam.TEXT_THRESHOLD = 0.4
            cv2_overlay_np = TerrainSegFormer.get_seg_overlay(cv2_img_np, pred)
            pil_overlay = Image.fromarray(cv2.cvtColor(cv2_overlay_np, cv2.COLOR_BGR2RGB))
            pil_preds_np.append(transform(pil_overlay))
    grid = make_grid(pil_preds_np, nrow=4, padding=30, pad_value=0.80)
    plt.figure(figsize=(40, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig("/home/dynamo/AMRL_Research/repos/lifelong_concept_learner/scripts/results/saved/compares/a_new_images/results/results4.png")
