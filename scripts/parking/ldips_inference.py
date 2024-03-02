import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
import cv2
from copy import deepcopy
import cv2
from scipy.spatial import cKDTree
from copy import deepcopy
from PIL import Image
from sklearn.decomposition import PCA

from terrainseg.ALL_TERRAINS_MAPS import NSLABELS_TWOWAY_NSINT, DATASETINTstr_TO_DATASETLABELS, DATASETLABELS_TO_NSLABELS, TERRAINMARKS_NSLABELS_TWOWAY_NSINT, TERRAINMARKS_DATASETINTstr_TO_DATASETLABELS, TERRAINMARKS_DATASETLABELS_TO_NSLABELS
from terrainseg.inference import TerrainSegFormer
from utilities.local_to_map_frame import LocalToMapFrame
from zeroshot_objdet.sam_dino import GroundedSAM
from third_party.jackal_calib import JackalLidarCamCalibration, JackalCameraCalibration
from utilities.std_utils import smoothing_filter


class NSInferObjDet:
    def __init__(self, box_threshold=0.5, text_threshold=0.5, nms_threshold=0.6, MODELS: dict = None):
        if MODELS is None:
            self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)
            self.local_to_map_frame = LocalToMapFrame()
            self.sam_dino_model = GroundedSAM(box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        else:
            self.lidar_cam_calib = MODELS["lidar_cam_calib"]
            self.local_to_map_frame = MODELS["local_to_map_frame"]
            self.sam_dino_model = MODELS["sam_dino_model"]

    def projectVLPtoMap(self, vlp_coords, ldict):
        """
        Projects VLP points to map frame
        vlp_coords: N x 3 np array of xyz points in lidar frame
        ldict: dict of localization info (x, y, theta) in map frame
        Returns: N x 3 np array of xyz points in map frame
        """
        all_mats = self.local_to_map_frame.get_all_M_exts(ldict)
        wcs_coords = self.local_to_map_frame.lidar_cam_calib.projectVLPtoWCS(vlp_coords)
        # wcs_coords[:, 2] = 0  # Make z = 0, since we only care about distance in x, y plane
        return JackalCameraCalibration.general_project_A_to_B(wcs_coords, all_mats["wcs_to_map"])

    def projectMaptoWCS(self, map_coords, ldict):
        """
        Projects map points to WCS frame
        map_coords: N x 3 np array of xyz points in map frame
        ldict: dict of localization info (x, y, theta) in map frame
        Returns: N x 3 np array of xyz points in wcs frame
        """
        all_mats = self.local_to_map_frame.get_all_M_exts(ldict)
        return JackalCameraCalibration.general_project_A_to_B(map_coords, all_mats["map_to_wcs"])

    def projectMapToPCS(self, map_coords, ldict):
        """
        Projects map_coords to the pixel coordinates
        map_coords: N x 3 np array of xyz map coordinates
        ldict: dict of localization info (x, y, theta) in map for current image
        Returns: N x 2 np array of xy points in pcs + the mask
        """
        wcs_coords = self.projectMaptoWCS(map_coords, ldict)
        return self.lidar_cam_calib.jackal_cam_calib.projectWCStoPCS(wcs_coords)

    def main_distance_to(self, img_bgr, pc_np, obj_name: str, box_threshold=None, text_threshold=None, nms_threshold=None, per_class_mask=None):
        """
        Runs SAM on the image for the given object and returns the corresponding distance array for each pixel in the image
        Returns: dist_array H x W + bool to indicate if detections were found
        """
        if per_class_mask is None:
            ann_img, detections, per_class_mask = self.sam_dino_model.predict_and_segment_on_image(img_bgr, [obj_name], box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
            per_class_mask = per_class_mask[0].squeeze()  # since we only have one class
        all_pixel_locs, all_vlp_coords, *_ = self.lidar_cam_calib.projectPCtoImageFull(pc_np, img_bgr)
        all_mask_values = per_class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
        class_pixel_locs = all_pixel_locs[all_mask_values]
        class_vlp_coords = all_vlp_coords[all_mask_values]
        if class_vlp_coords is None or class_vlp_coords.shape[0] == 0:
            # print("main_distance_to: No detections found for object: ", obj_name)
            dist_arr = np.ones((all_pixel_locs.shape[0], 1)) * np.inf
            return self.linear_to_img_values(dist_arr, img_bgr, all_pixel_locs), False
        kdtree = cKDTree(class_vlp_coords)
        distances, _ = kdtree.query(all_vlp_coords)
        dist_arr = distances.reshape((-1, 1))
        return self.linear_to_img_values(dist_arr, img_bgr, all_pixel_locs), True

    def main_frontal_distance(self, img_bgr, pc_np, obj_name: str, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Runs SAM on the image for the given object and returns the corresponding frontal distance array for each pixel in the image
        Returns: frontal_dist_array H x W + bool to indicate if detections were found
        NOTE: different instances of the object matter here TODO
        """
        ann_img, detections, _ = self.sam_dino_model.predict_and_segment_on_image(img_bgr, [obj_name], box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        num_detected_instances = detections.mask.shape[0]
        all_pixel_locs, all_vlp_coords, *_ = self.lidar_cam_calib.projectPCtoImageFull(pc_np, img_bgr)
        dist_arr = np.ones((all_pixel_locs.shape[0], 1)) * np.inf
        for i in range(num_detected_instances):
            per_class_mask = detections.mask[i].squeeze()
            all_mask_values = per_class_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
            class_pixel_locs = all_pixel_locs[all_mask_values]
            class_vlp_coords = all_vlp_coords[all_mask_values]
            class_vlp_coords_xy = class_vlp_coords[:, :2].reshape((-1, 2))
            all_vlp_coords_xy = all_vlp_coords[:, :2].reshape((-1, 2))
            front_distances, line_segment = NSInferObjDet.frontal_distances_and_sweepability(class_vlp_coords_xy, all_vlp_coords_xy)
            dist_arr = np.minimum(dist_arr, front_distances)
            # line_segment = np.hstack((line_segment, np.zeros((line_segment.shape[0], 1))))
            # pcs_line_segment, *_ = self.lidar_cam_calib.projectVLPtoPCS(line_segment, mode="none")
            # cv2.line(img_bgr, tuple(pcs_line_segment[0]), tuple(pcs_line_segment[1]), (0, 0, 255), 4)
            # cv2.imshow("img_bgr", img_bgr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return self.linear_to_img_values(dist_arr, img_bgr, all_pixel_locs), detections.mask.shape[0] > 0

    @staticmethod
    def linear_to_img_values(linear_corresponding_values, img_bgr, all_pixel_locs):
        """
        Converts linear values to img values
        Returns: img_values
        """
        img_values = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float64)
        x_indices = all_pixel_locs[:, 1].astype(int)
        y_indices = all_pixel_locs[:, 0].astype(int)
        img_values[x_indices, y_indices] = linear_corresponding_values.squeeze()
        return img_values

    @staticmethod
    def estimate_line_segment(points):
        pca = PCA(n_components=1)
        pca.fit(points)
        line_direction = pca.components_[0]
        projected_points = pca.transform(points)
        min_point, max_point = projected_points.min(axis=0), projected_points.max(axis=0)
        line_point1 = pca.inverse_transform(min_point)
        line_point2 = pca.inverse_transform(max_point)
        return np.array([line_point1, line_point2])

    @staticmethod
    def project_point_on_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        return line_start + t * line_vec

    @staticmethod
    def check_sweepability(point, line_segment):
        projected_point = NSInferObjDet.project_point_on_line(point, line_segment[0], line_segment[1])
        if np.all(np.isclose(projected_point, line_segment[0])) or np.all(np.isclose(projected_point, line_segment[1])):
            return True
        if np.linalg.norm(projected_point - line_segment[0]) < np.linalg.norm(line_segment[1] - line_segment[0]) and np.linalg.norm(projected_point - line_segment[1]) < np.linalg.norm(line_segment[1] - line_segment[0]):
            return True
        return False

    @staticmethod
    def frontal_distances_and_sweepability(points, ego_points):
        """
        Given class points and ego points, returns the frontal distance to the class for each ego point.
        points: N x 2 np array of class points (x, y in map frame)
        ego_points: M x 2 np array of ego points (x, y in map frame)
        Returns: M x 1 np array of frontal distances to the class for each ego point + line_segment (2 x 2 np array)
        """
        line_segment = NSInferObjDet.estimate_line_segment(points)
        distances = np.zeros(len(ego_points))
        for i, ego_point in enumerate(ego_points):
            if NSInferObjDet.check_sweepability(ego_point, line_segment):
                projected_point = NSInferObjDet.project_point_on_line(ego_point, line_segment[0], line_segment[1])
                distances[i] = np.linalg.norm(ego_point - projected_point)
            else:
                distances[i] = np.inf
        return distances.reshape((-1, 1)), line_segment


class NSInferTerrainSeg:
    def __init__(self, MODELS: dict = None):
        if MODELS is None:
            self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)
            self.local_to_map_frame = LocalToMapFrame()
            self.terrain_model = TerrainSegFormer(hf_model_ver=None)
            self.terrain_model.load_model_inference()
            self.terrain_model.prepare_dataset()
        else:
            # Check individual models
            self.lidar_cam_calib = MODELS["lidar_cam_calib"] if "lidar_cam_calib" in MODELS.keys() else JackalLidarCamCalibration(ros_flag=False)
            self.local_to_map_frame = MODELS["local_to_map_frame"] if "local_to_map_frame" in MODELS.keys() else LocalToMapFrame()
            if "terrain_model" in MODELS.keys():
                self.terrain_model = MODELS["terrain_model"]
            else:
                self.terrain_model = TerrainSegFormer(hf_model_ver=None)
                self.terrain_model.load_model_inference()
                self.terrain_model.prepare_dataset()

    def projectMapToPCS(self, map_coords, ldict):
        """
        Projects map_coords to the pixel coordinates
        map_coords: N x 3 np array of xyz map coordinates
        ldict: dict of localization info (x, y, theta) in map for current image
        Returns: N x 2 np array of xy points in pcs + the mask
        """
        all_mats = self.local_to_map_frame.get_all_M_exts(ldict)
        wcs_coords = JackalCameraCalibration.general_project_A_to_B(map_coords, all_mats["map_to_wcs"])
        return self.lidar_cam_calib.jackal_cam_calib.projectWCStoPCS(wcs_coords)

    def projectVLPtoMap(self, vlp_coords, ldict):
        """
        Projects VLP points to map frame
        vlp_coords: N x 3 np array of xyz points in lidar frame
        ldict: dict of localization info (x, y, theta) in map frame
        Returns: N x 3 np array of xyz points in map frame
        """
        all_mats = self.local_to_map_frame.get_all_M_exts(ldict)
        wcs_coords = self.local_to_map_frame.lidar_cam_calib.projectVLPtoWCS(vlp_coords)
        # wcs_coords[:, 2] = 0  # Make z = 0, since we only care about distance in x, y plane
        return JackalCameraCalibration.general_project_A_to_B(wcs_coords, all_mats["wcs_to_map"])

    @staticmethod
    def convert_label(label, id2label, SOFT_TERRAIN_MAPPING, new_terrains):
        initial_name = id2label[label]
        new_name = SOFT_TERRAIN_MAPPING.get(initial_name, initial_name)
        final_index = new_terrains.index(new_name) if new_name in new_terrains else new_terrains.index('dunno')
        return final_index

    def main_terrain(self, img_bgr, domain_terrains, gt_seg=None):
        """
        Runs terrain segmentation on one image
        Returns: pred_seg H x W + new_terrains
        """
        pil_img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img_np)
        new_terrains = ["dunno"] + domain_terrains
        if gt_seg is None:
            _, pred_seg = self.terrain_model.predict_new(pil_img)
            pred_seg = smoothing_filter(pred_seg)
            pred_seg = np.asarray(pred_seg).squeeze()
        else:
            pred_seg = gt_seg
        vec_convert_label = np.vectorize(self.convert_label, excluded=[1, 2, 3])
        pred_seg = np.asarray(vec_convert_label(pred_seg, self.terrain_model.id2label, DATASETLABELS_TO_NSLABELS, new_terrains)).squeeze()
        return pred_seg.squeeze(), new_terrains

    @staticmethod
    def linear_to_img_mask(linear_corresponding_mask, img_bgr, all_pixel_locs):
        """
        Converts linear mask to img mask
        Returns: img_mask
        """
        img_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.int64)
        x_indices = all_pixel_locs[:, 1].astype(int)
        y_indices = all_pixel_locs[:, 0].astype(int)
        img_mask[x_indices, y_indices] = linear_corresponding_mask.astype(int)
        return img_mask

    def main_in_the_way(self, img_bgr, pc_xyz, domain_terrains, param=1.5):
        """
        Runs in the way on one image
        Returns: bool_mask H x W
        # acc to image points, non_trav is near ego: dist to non_trav is less than param
            # NOT in the way
        # acc to image points, no non_trav is near ego: dist to non_trav is greater than param
            # maybe there is non_trav but occluded: not in the way
            # maybe there is no non_trav: in the way
        """
        _pred_seg, _new_terrains = self.main_terrain(img_bgr, domain_terrains)
        all_pixel_locs, all_vlp_coords, *_ = self.lidar_cam_calib.projectPCtoImageFull(pc_xyz, img_bgr)
        _new_terrains_dict = {i: terrain for i, terrain in enumerate(_new_terrains)}
        terrain_labels = np.vectorize(_new_terrains_dict.get)(_pred_seg)
        traversable_mask = np.isin(terrain_labels, NSLABELS_TRAVERSABLE_TERRAINS)
        expanded_traversable_mask = traversable_mask[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
        expanded_non_traversable_mask = ~expanded_traversable_mask
        traversable_vlp_coords = all_vlp_coords[expanded_traversable_mask]
        nontraversable_vlp_coords = all_vlp_coords[expanded_non_traversable_mask]
        traversable_xy = traversable_vlp_coords[:, :2]
        nontraversable_xy = nontraversable_vlp_coords[:, :2]
        tree_nontraversable = cKDTree(nontraversable_xy)
        distances, _ = tree_nontraversable.query(traversable_xy)
        far_enough_mask = distances > param
        in_the_way_mask = deepcopy(expanded_traversable_mask)
        in_the_way_mask[expanded_traversable_mask] = far_enough_mask
        final_mask = self.linear_to_img_mask(in_the_way_mask, img_bgr, all_pixel_locs)
        return final_mask.squeeze()

    @staticmethod
    def calculate_max_zdiff(terrain_pixel_locs, terrain_pixel_zs, K):
        # Create a cKDTree from the pixel locations
        ckdtree = cKDTree(terrain_pixel_locs)
        # Query the K+1 nearest neighbors for all points at once
        distances, indices = ckdtree.query(terrain_pixel_locs, k=K + 1)
        # Retrieve the Z values of these neighbors
        neighbor_zs = terrain_pixel_zs[indices]
        # Calculate the maximum Z difference
        max_zdiff = np.max(np.abs(neighbor_zs - terrain_pixel_zs[:, None]), axis=1)
        return max_zdiff

    def main_slope(self, img_bgr, pc_xyz, scale=5e2):
        """
        Runs slope estimation on one image
        Returns: grad_mask H x W
        """
        pil_img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img_np)
        _, _pred_seg = self.terrain_model.predict_new(pil_img)  # unlabeled (0) and NAT (1)
        _pred_seg = smoothing_filter(_pred_seg)
        _, corresponding_pcs_coords, corresponding_vlp_coords, _ = self.lidar_cam_calib.projectPCtoImage(pc_xyz, img_bgr)
        corresponding_vlp_zs = corresponding_vlp_coords[:, 2].reshape((-1, 1))
        all_pixel_locs, all_vlp_coords, _, all_vlp_zs, _ = self.lidar_cam_calib.projectPCtoImageFull(pc_xyz, img_bgr)
        labels = _pred_seg[all_pixel_locs[:, 1], all_pixel_locs[:, 0]]
        is_terrain_mask = np.array((labels != 0) & (labels != 1)).squeeze()
        is_terrain_mask_img = self.linear_to_img_mask(is_terrain_mask, img_bgr, all_pixel_locs).astype(bool)
        corresponding_x_indices = corresponding_pcs_coords[:, 1].astype(int)
        corresponding_y_indices = corresponding_pcs_coords[:, 0].astype(int)
        corresponding_is_terrain_mask = is_terrain_mask_img[corresponding_x_indices, corresponding_y_indices]
        corresponding_pcs_coords = corresponding_pcs_coords[corresponding_is_terrain_mask]
        corresponding_vlp_zs = corresponding_vlp_zs[corresponding_is_terrain_mask]
        smoothed_all_vlp_zs, *_ = self.lidar_cam_calib.double_interp(a=corresponding_pcs_coords,
                                                                     b=corresponding_vlp_zs,
                                                                     x=all_pixel_locs)

        terrain_pixel_locs = all_pixel_locs[is_terrain_mask]
        terrain_pixel_zs = smoothed_all_vlp_zs[is_terrain_mask]
        # smoothed_all_vlp_zs, *_ = self.lidar_cam_calib.double_interp(a=terrain_pixel_locs,
        #                                                              b=terrain_pixel_zs,
        #                                                              x=all_pixel_locs,
        #                                                              do_nearest=False,
        #                                                              firstmethod="nearest")
        # vlp_zs_img = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float32)
        # x_indices = all_pixel_locs[:, 1].astype(int)
        # y_indices = all_pixel_locs[:, 0].astype(int)
        # vlp_zs_img[x_indices, y_indices] = smoothed_all_vlp_zs.squeeze()
        # grad_zs = np.gradient(vlp_zs_img)
        # grad_zs_mag = np.sqrt(grad_zs[0] ** 2 + grad_zs[1] ** 2)
        # grad_zs_mag = grad_zs_mag * self.linear_to_img_mask(is_terrain_mask, img_bgr, all_pixel_locs) * scale
        # return grad_zs_mag.squeeze()
        max_zdiff = self.calculate_max_zdiff(terrain_pixel_locs, terrain_pixel_zs, K=8)
        max_zdiff_img = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float32)
        x_indices = terrain_pixel_locs[:, 1].astype(int)
        y_indices = terrain_pixel_locs[:, 0].astype(int)
        max_zdiff_img[x_indices, y_indices] = max_zdiff.squeeze()
        max_zdiff_img = max_zdiff_img * scale
        return max_zdiff_img.squeeze()
