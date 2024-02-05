import os
nspl_root_dir = os.environ.get("NSPL_REPO")
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import rosbag
import yaml
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy  # Used in sensor_msgs.msg apt-get install ros-noetic-ros-numpy
from third_party.jackal_calib import JackalLidarCamCalibration
from itertools import tee


def save_image_to_png(image_msg, filename, bridge, images_out_dir):
    img_bgr = bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    filepath = os.path.join(images_out_dir, filename)
    cv2.imwrite(filepath, img_bgr)


def save_pointcloud_to_bin(pointcloud_msg, filename, lcc, pointclouds_out_dir, raw=False):
    pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg).reshape((1, -1))
    pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], 4), dtype=np.float32)
    pc_np[..., 0] = pc_cloud['x']
    pc_np[..., 1] = pc_cloud['y']
    pc_np[..., 2] = pc_cloud['z']
    pc_np[..., 3] = pc_cloud['intensity']
    pc_np = pc_np.reshape((-1, 4))
    if not raw:
        pc_np = lcc._correct_pc(pc_np)
    filepath = os.path.join(pointclouds_out_dir, filename)
    flat_pc = pc_np.reshape(-1).astype(np.float32)
    flat_pc.tofile(filepath)


def save_localization_to_yaml(localization_msg, filename, localizations_out_dir):
    x = localization_msg.pose.x
    y = localization_msg.pose.y
    theta = localization_msg.pose.theta
    data = {
        "x": x,
        "y": y,
        "theta": theta
    }
    filepath = os.path.join(localizations_out_dir, filename)
    with open(filepath, 'w') as outfile:
        yaml.dump(data, outfile)


def generate3(bag_path, root_dir, image_topic, pointcloud_topic, localization_topic, syncimg_pc_min_lag=0.025, syncimg_pc_max_lag=0.035, syncimg_abs_loc_threshold=0.01, filename_prefix="", user=True):
    """
    Image and pc synced according to msg header timestamps
    Image and loc synced according to rosbag timestamps
    """
    if os.path.exists(root_dir):
        print("Skipping...")
        return
    syncimg_lag_pc_threshold = syncimg_pc_max_lag - syncimg_pc_min_lag
    images_out_dir = os.path.join(root_dir, "images")
    pointclouds_out_dir = os.path.join(root_dir, "pcs")
    localizations_out_dir = os.path.join(root_dir, "locs")
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    lcc = JackalLidarCamCalibration(ros_flag=False)

    # Create generators for each topic
    image_messages_gen = ((msg.header.stamp.to_sec(), t.to_sec(), msg) for topic, msg, t in bag.read_messages() if topic == image_topic)
    pointcloud_messages_gen = ((msg.header.stamp.to_sec() + syncimg_pc_min_lag, t.to_sec(), msg) for topic, msg, t in bag.read_messages() if topic == pointcloud_topic)
    localization_messages_gen = ((msg.header.stamp.to_sec(), t.to_sec(), msg) for topic, msg, t in bag.read_messages() if topic == localization_topic)

    # Tee the iterators to enable advancing them independently
    image_iter, image_iter_copy = tee(image_messages_gen)
    pc_iter, pc_iter_copy = tee(pointcloud_messages_gen)
    loc_iter, loc_iter_copy = tee(localization_messages_gen)

    # Function to advance the iterator
    def advance_iterator(it):
        try:
            return next(it)
        except StopIteration:
            return None

    # Initialize the iterators
    image_item = advance_iterator(image_iter_copy)
    pc_item = advance_iterator(pc_iter_copy)
    loc_item = advance_iterator(loc_iter_copy)

    sync_data = []

    # Iterate and sync data
    while image_item and pc_item and loc_item:
        img_ht, img_t, image_msg = image_item
        pc_ht, pc_t, pc_msg = pc_item
        loc_ht, loc_t, loc_msg = loc_item

        # Check if all timestamps are within range
        if abs(img_t - loc_t) <= syncimg_abs_loc_threshold and 0 <= (img_ht - pc_ht) <= syncimg_lag_pc_threshold:
            sync_data.append((image_msg, pc_msg, loc_msg))
            image_item = advance_iterator(image_iter_copy)
            pc_item = advance_iterator(pc_iter_copy)
            loc_item = advance_iterator(loc_iter_copy)
        else:
            # advance individual iterators
            if pc_ht > img_ht or loc_t > img_t + syncimg_abs_loc_threshold:
                image_item = advance_iterator(image_iter_copy)
            elif loc_t < img_t - syncimg_abs_loc_threshold:
                loc_item = advance_iterator(loc_iter_copy)
            elif pc_ht < img_ht - syncimg_lag_pc_threshold:
                pc_item = advance_iterator(pc_iter_copy)
            else:
                print("************Something went wrong*************")
                raise ValueError

    print(f"Found {len(sync_data)} synced data points.")
    if user:
        user_input = input(f"Do you want to continue? (y/n)\n")
        if user_input == "n":
            return

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(pointclouds_out_dir, exist_ok=True)
    os.makedirs(localizations_out_dir, exist_ok=True)

    # STORE
    for i, data in enumerate(sync_data):
        img_msg, pc_msg, loc_msg = data
        name_wo_ext = f'{filename_prefix}{i:06}'
        save_image_to_png(img_msg, f'{name_wo_ext}.png', bridge, images_out_dir)
        save_pointcloud_to_bin(pc_msg, f'{name_wo_ext}.bin', lcc, pointclouds_out_dir)
        save_localization_to_yaml(loc_msg, f'{name_wo_ext}.yaml', localizations_out_dir)
        print(f"Saved {i+1} of {len(sync_data)}")


def generate2(bag_path, root_dir, image_topic, pointcloud_topic, syncimg_pc_min_lag=0.025, syncimg_pc_max_lag=0.035, filename_prefix="", user=True):
    """
    Image and pc synced according to msg header timestamps
    """
    syncimg_lag_pc_threshold = syncimg_pc_max_lag - syncimg_pc_min_lag
    images_out_dir = os.path.join(root_dir, "images")
    pointclouds_out_dir = os.path.join(root_dir, "pcs")
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    lcc = JackalLidarCamCalibration(ros_flag=False)

    # Create generators for each topic
    image_messages_gen = ((msg.header.stamp.to_sec(), t.to_sec(), msg) for topic, msg, t in bag.read_messages() if topic == image_topic)
    pointcloud_messages_gen = ((msg.header.stamp.to_sec() + syncimg_pc_min_lag, t.to_sec(), msg) for topic, msg, t in bag.read_messages() if topic == pointcloud_topic)

    # Tee the iterators to enable advancing them independently
    image_iter, image_iter_copy = tee(image_messages_gen)
    pc_iter, pc_iter_copy = tee(pointcloud_messages_gen)

    # Function to advance the iterator
    def advance_iterator(it):
        try:
            return next(it)
        except StopIteration:
            return None

    # Initialize the iterators
    image_item = advance_iterator(image_iter_copy)
    pc_item = advance_iterator(pc_iter_copy)

    sync_data = []

    # Iterate and sync data
    while image_item and pc_item:
        img_ht, img_t, image_msg = image_item
        pc_ht, pc_t, pc_msg = pc_item

        # Check if all timestamps are within range
        if 0 <= (img_ht - pc_ht) <= syncimg_lag_pc_threshold:
            sync_data.append((image_msg, pc_msg))
            image_item = advance_iterator(image_iter_copy)
            pc_item = advance_iterator(pc_iter_copy)
        else:
            # advance individual iterators
            if pc_ht > img_ht:
                image_item = advance_iterator(image_iter_copy)
            elif pc_ht < img_ht - syncimg_lag_pc_threshold:
                pc_item = advance_iterator(pc_iter_copy)
            else:
                print("************Something went wrong*************")
                raise ValueError

    print(f"Found {len(sync_data)} synced data points.")
    if user:
        user_input = input(f"Do you want to continue? (y/n)\n")
        if user_input == "n":
            return

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(pointclouds_out_dir, exist_ok=True)

    # STORE
    for i, data in enumerate(sync_data):
        img_msg, pc_msg = data
        name_wo_ext = f'{filename_prefix}{i:06}'
        save_image_to_png(img_msg, f'{name_wo_ext}.png', bridge, images_out_dir)
        save_pointcloud_to_bin(pc_msg, f'{name_wo_ext}.bin', lcc, pointclouds_out_dir)
        print(f"Saved {i+1} of {len(sync_data)}")


if __name__ == "__main__":
    # bagnumber = 4
    for bagnumber in range(3, 31):
        print(f"Generating data for bag {bagnumber}")
        bag_path = os.path.join(nspl_root_dir, f"demonstrations/{bagnumber}.bag")
        bagname = os.path.basename(bag_path).split('.')[0]
        out_dir = os.path.join(nspl_root_dir, f"demonstrations/syncdata/{bagname}")
        IMAGE_TOPIC = "/zed2i/zed_node/left/image_rect_color/compressed"  # sensor_msgs/CompressedImage
        POINTCLOUD_TOPIC = "/ouster/points"  # sensor_msgs/PointCloud2
        LOCALIZATION_TOPIC = "/localization"  # amrl_msgs/Localization2DMsg
        prefix = ""
        generate3(bag_path, out_dir, IMAGE_TOPIC, POINTCLOUD_TOPIC, LOCALIZATION_TOPIC,
                  syncimg_pc_min_lag=0.025, syncimg_pc_max_lag=0.035, syncimg_abs_loc_threshold=0.01, user=True, filename_prefix=prefix)

    # bag_path = f"/home/dynamo/Music/jackal_bags/mode1/morning_mode1_2_11062023_4.bag"
    # bagname = os.path.basename(bag_path).split('.')[0]
    # out_dir = f"/home/dynamo/Music/jackal_bags/full_dataset/{bagname}"
    # IMAGE_TOPIC = "/zed2i/zed_node/left/image_rect_color/compressed"  # sensor_msgs/CompressedImage
    # POINTCLOUD_TOPIC = "/ouster/points"  # sensor_msgs/PointCloud2
    # SYNC_THRESHOLD = 0.01  # seconds
    # prefix = f"{bagname}_"
    # generate2(bag_path, out_dir, IMAGE_TOPIC, POINTCLOUD_TOPIC, SYNC_THRESHOLD, prefix)

    # mode1_bagdir = "/robodata/smodak/LCL/jackal_modes_bags/mode1"
    # mode2_bagdir = "/robodata/smodak/LCL/jackal_modes_bags/mode2"
    # OUT_ROOT_DIR = "/robodata/smodak/LCL/full_eval_dataset"
    # IMAGE_TOPIC = "/zed2i/zed_node/left/image_rect_color/compressed"  # sensor_msgs/CompressedImage
    # POINTCLOUD_TOPIC = "/ouster/points"  # sensor_msgs/PointCloud2
    # all_bags1_full_paths = [os.path.join(mode1_bagdir, bag) for bag in sorted(os.listdir(mode1_bagdir))]
    # all_bags2_full_paths = [os.path.join(mode2_bagdir, bag) for bag in sorted(os.listdir(mode2_bagdir))]
    # all_bags_full_paths = all_bags1_full_paths + all_bags2_full_paths
    # print(f"Found {len(all_bags_full_paths)} bags.")
    # for bag_path in all_bags_full_paths:
    #     print(f"************************************************ Generating data for bag {os.path.basename(bag_path)}")
    #     bagname = os.path.basename(bag_path).split('.')[0]
    #     out_dir = os.path.join(OUT_ROOT_DIR, bagname)
    #     prefix = f"{bagname}_"
    #     generate2(bag_path, out_dir, IMAGE_TOPIC, POINTCLOUD_TOPIC,
    #               syncimg_pc_min_lag=0.025, syncimg_pc_max_lag=0.035, user=False, filename_prefix=prefix)
