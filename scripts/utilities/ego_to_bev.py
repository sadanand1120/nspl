import numpy as np
import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from third_party.jackal_calib import JackalLidarCamCalibration
from PIL import Image
import matplotlib.pyplot as plt
import cv2


class EgoToBEV:
    def __init__(self):
        self.lidar_cam = JackalLidarCamCalibration(ros_flag=False)
        self.bev_lidar_cam = JackalLidarCamCalibration(ros_flag=False, cam_extrinsics_filepath=os.path.join(nspl_root_dir, "scripts/utilities/bev_params/baselink_to_zed_left_extrinsics.yaml"))

    def get_2d_bev(self, pil_img, inpaint=True):
        """
        Assumes all pixel points on Z=0 plane and calculates BEV image. It implicitly only does it for points below horizon actually.
        pil_img: A x B PIL Image object (RGB)
        Returns: A x B Numpy array of BEV image (RGB)
        """
        np_pil_img = np.array(pil_img)
        np_cv2_img = cv2.cvtColor(np_pil_img, cv2.COLOR_RGB2BGR)
        all_ys, all_xs = np.meshgrid(np.arange(pil_img.height), np.arange(pil_img.width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        all_wcs_coords, horizon_mask = self.lidar_cam.jackal_cam_calib.projectPCStoWCSground(all_pixel_locs)
        # img_mask = np.zeros((pil_img.height, pil_img.width), dtype=np.int64)
        # x_indices = all_pixel_locs[:, 1].astype(int)
        # y_indices = all_pixel_locs[:, 0].astype(int)
        # img_mask[x_indices, y_indices] = horizon_mask.astype(int)  # 0 1 mask, 1 means below horizon so we want to keep it
        # overlayed = TerrainSegFormer.get_seg_overlay(np_pil_img, img_mask, alpha=0.2)
        # plt.imshow(overlayed)
        # plt.show()
        all_pixel_locs = all_pixel_locs[horizon_mask]
        bev_np_pil_img = np.zeros_like(np_pil_img)
        bev_pixel_locs, bev_mask = self.bev_lidar_cam.jackal_cam_calib.projectWCStoPCS(all_wcs_coords)
        all_pixel_locs = all_pixel_locs[bev_mask]
        rows_bev, cols_bev = bev_pixel_locs[:, 1], bev_pixel_locs[:, 0]
        rows_all, cols_all = all_pixel_locs[:, 1], all_pixel_locs[:, 0]
        bev_np_pil_img[rows_bev, cols_bev] = np_pil_img[rows_all, cols_all]
        if inpaint:
            inpaint_mask = np.all(bev_np_pil_img == [0, 0, 0], axis=-1).astype(np.uint8)
            polygon_mask = np.zeros((bev_np_pil_img.shape[0], bev_np_pil_img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(polygon_mask, bev_pixel_locs, 1)
            combined_mask = cv2.bitwise_and(inpaint_mask, polygon_mask)
            bev_np_pil_img = cv2.inpaint(bev_np_pil_img, combined_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        return bev_np_pil_img

    def get_3d_bev(self, pil_img, pc_np, using="pc", inpaint=True):
        """
        Uses the lidar pointcloud to calculate BEV image.
        pil_img: A x B PIL Image object (RGB)
        pc_np: N x ? numpy array of point cloud (x, y, z, whatever)
        using: "pc" or "z" to use the pointcloud OR Z values of the pointcloud (+ camera homography) respectively
        Returns: A x B Numpy array of BEV image (RGB)        
        """
        pc_np = pc_np[:, :3]
        np_pil_img = np.array(pil_img)
        np_cv2_img = cv2.cvtColor(np_pil_img, cv2.COLOR_RGB2BGR)
        all_ys, all_xs = np.meshgrid(np.arange(pil_img.height), np.arange(pil_img.width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        if using == "z":
            _, _, _, all_vlp_zs, _ = self.lidar_cam.projectPCtoImageFull(pc_np, np_cv2_img)
            all_wcs_coords, mask = self.lidar_cam.projectPCStoWCSusingZ(all_pixel_locs, all_vlp_zs)
            all_pixel_locs = all_pixel_locs[mask]
            bev_np_pil_img = np.zeros_like(np_pil_img)
            bev_pixel_locs, bev_mask = self.bev_lidar_cam.jackal_cam_calib.projectWCStoPCS(all_wcs_coords)
        elif using == "pc":
            _, all_vlp_coords, _, _, interp_mask = self.lidar_cam.projectPCtoImageFull(pc_np, np_cv2_img, do_nearest=False)
            all_pixel_locs = all_pixel_locs[interp_mask]
            bev_np_pil_img = np.zeros_like(np_pil_img)
            bev_pixel_locs, bev_mask, _ = self.bev_lidar_cam.projectVLPtoPCS(all_vlp_coords)
        else:
            raise ValueError("using must be 'pc' or 'z'")
        all_pixel_locs = all_pixel_locs[bev_mask]
        rows_bev, cols_bev = bev_pixel_locs[:, 1], bev_pixel_locs[:, 0]
        rows_all, cols_all = all_pixel_locs[:, 1], all_pixel_locs[:, 0]
        bev_np_pil_img[rows_bev, cols_bev] = np_pil_img[rows_all, cols_all]
        if inpaint:
            inpaint_mask = np.all(bev_np_pil_img == [0, 0, 0], axis=-1).astype(np.uint8)
            polygon_mask = np.zeros((bev_np_pil_img.shape[0], bev_np_pil_img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(polygon_mask, bev_pixel_locs, 1)
            combined_mask = cv2.bitwise_and(inpaint_mask, polygon_mask)
            bev_np_pil_img = cv2.inpaint(bev_np_pil_img, combined_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        return bev_np_pil_img


if __name__ == "__main__":
    raw_pil_img = Image.open(os.path.join(nspl_root_dir, "examples/syncdata/1/images/000000.png"))
    pc_np = np.fromfile(os.path.join(nspl_root_dir, "examples/syncdata/1/pcs/000000.bin"), dtype=np.float32).reshape((-1, 4))
    ego_to_bev = EgoToBEV()

    f, axs = plt.subplots(1, 2)
    f.set_figheight(30)
    f.set_figwidth(50)
    axs[0].set_title("Raw", {'fontsize': 40})
    axs[0].imshow(raw_pil_img)
    axs[1].set_title("BEV", {'fontsize': 40})
    axs[1].imshow(ego_to_bev.get_3d_bev(raw_pil_img, pc_np, using="pc"))
    plt.show()
