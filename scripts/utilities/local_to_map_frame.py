import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import numpy as np
from third_party.jackal_calib import JackalLidarCamCalibration, JackalCameraCalibration


class LocalToMapFrame:
    def __init__(self):
        self.lidar_cam_calib = JackalLidarCamCalibration(ros_flag=False)
        self.wcs_to_vlp = self.lidar_cam_calib.get_M_ext()
        self.vlp_to_wcs = np.linalg.inv(self.wcs_to_vlp)

    def get_M_ext_from_localization(self, localization):
        """
        Returns the extrinsic matrix (4 x 4) that transforms from Map frame to WCS
        localization: dict with keys x, y, theta
        NOTE: assumes that z translation b/w local wcs frame and map frame is 0
        """
        x = localization['x']  # m
        y = localization['y']  # m
        theta = localization['theta']  # rad

        T1 = JackalCameraCalibration.get_std_trans(cx=x, cy=y, cz=0)
        T2 = JackalCameraCalibration.get_std_rot(axis="Z", alpha=theta)
        return T2 @ T1

    def get_all_M_exts(self, localization):
        map_to_wcs = self.get_M_ext_from_localization(localization)
        wcs_to_map = np.linalg.inv(map_to_wcs)
        map_to_vlp = self.wcs_to_vlp @ map_to_wcs
        vlp_to_map = np.linalg.inv(map_to_vlp)
        mats = {
            "map_to_wcs": map_to_wcs,
            "wcs_to_map": wcs_to_map,
            "map_to_vlp": map_to_vlp,
            "vlp_to_map": vlp_to_map,
        }
        return mats
