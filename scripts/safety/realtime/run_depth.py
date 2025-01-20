import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import cv2
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
from sensor_msgs.msg import PointCloud2, CompressedImage, PointField
import matplotlib.pyplot as plt
import rospy
from cv_bridge import CvBridge
from PIL import Image
import torch
import time
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from safety.realtime.fast_utils import FastModels
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True


class FasterSynapse:
    def __init__(self):
        self.latest_cv2_img_np = None
        self.fm = FastModels()
        self.cv_bridge = CvBridge()
        rospy.Subscriber("/zed2i/zed_node/left/image_rect_color/compressed", CompressedImage, self.image_callback, queue_size=1)
        self.depth_pub = rospy.Publisher("/depth/compressed", CompressedImage, queue_size=1)
        self.pc_pub = rospy.Publisher('/depth_point_cloud', PointCloud2, queue_size=1)
        rospy.Timer(rospy.Duration(1 / 11), lambda event: self.depth(self.latest_cv2_img_np))
        rospy.loginfo("Faster synapse module initialized")

    def image_callback(self, msg):
        self.latest_cv2_img_np = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def depth(self, cv2_img_np, event=None):
        if cv2_img_np is None:
            return
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img_np, cv2.COLOR_BGR2RGB))
        pred_metric_depth, pcd = self.fm.get_pc_from_depth(pil_img)
        normalized_image = cv2.normalize(pred_metric_depth, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        colormap = plt.get_cmap('inferno')
        colored_image = colormap(normalized_image)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        colored_image = cv2.resize(colored_image, None, fx=0.75, fy=0.75)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.asarray(cv2.imencode('.jpg', colored_image)[1]).tobytes()
        self.depth_pub.publish(msg)
        ros_pcd = self.convert_open3d_pcd_to_ros_pcd(pcd)
        self.pc_pub.publish(ros_pcd)

    @staticmethod
    def convert_open3d_pcd_to_ros_pcd(opcd):
        points = np.asarray(opcd.points)
        colors = np.asarray(opcd.colors)
        points = np.hstack((points, colors))
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1),
        ]
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        return pc2.create_cloud(header, fields, points)


if __name__ == "__main__":
    rospy.init_node('faster_synapse', anonymous=True)
    e = FasterSynapse()
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down faster synapse module")
