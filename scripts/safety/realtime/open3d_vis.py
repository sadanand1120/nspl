import open3d as o3d
from safety.realtime.fast_utils import FastModels
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
import torch
import torch
import numpy as np
import PIL
from PIL import Image
from terrainseg.inference import TerrainSegFormer
import numpy as np
import cv2
from PIL import Image
from terrainseg.inference import TerrainSegFormer
import open3d as o3d
import os
import torch
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from tqdm.auto import tqdm
from simple_colors import green
import shutil
torch.set_default_device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True


def pil_img_to_pointcloud(image: PIL.Image.Image):
    full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(image)
    cv2_image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_image_np, full_pred_seg)
    pilnp_pred_overlay = cv2.cvtColor(cv2_pred_overlay, cv2.COLOR_BGR2RGB)
    pcd = depth_results[1]
    colors = np.asarray(pilnp_pred_overlay).reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def read_images_from_bag(bag_file, topic):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file)
    images = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        # Using cv_bridge to convert compressed image message to cv2 format
        cv2_frame_np = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        images.append(cv2_frame_np)
    bag.close()
    return images


def images_to_video(dir_path, video_path, skip_rate=1):
    images = sorted([img for img in os.listdir(dir_path) if img.endswith(".png")])
    if not images:
        print("No images found in the directory.")
        return
    first_image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (width, height), True)
    for i, image in enumerate(tqdm(images, desc="Writing Video", total=len(images))):
        img_path = os.path.join(dir_path, image)
        frame = cv2.imread(img_path)
        if i % skip_rate != 0:
            out.write(save_frame)
        else:
            save_frame = frame
            out.write(save_frame)
    out.release()
    print(green(f"Video saved at {video_path}", ["bold"]))


def main(bag_file, image_topic, camera_params_json, output_video, tempdir="/home/dynamo/Music/jackal_bags/VIDEOS/tempdir", skip_rate=1):
    os.makedirs(tempdir, exist_ok=True)

    # Read images from the bag file
    print(green(f"Reading images from the bag file: {bag_file}", ["bold"]))
    images = read_images_from_bag(bag_file, image_topic)
    print(green(f"Number of images read: {len(images)}", ["bold"]))

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Read camera parameters
    param = o3d.io.read_pinhole_camera_parameters(camera_params_json)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

    # Set up video writer
    frame_width = 960  # Set based on your requirement
    frame_height = 540  # Set based on your requirement

    for i, image in enumerate(tqdm(images, desc="Saving Images", total=len(images))):
        # Convert image to point cloud
        pcd = pil_img_to_pointcloud(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        vis.clear_geometries()
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()

        # Capture screen image
        vis.capture_screen_image(f"{tempdir}/{i:06}.png")

    vis.destroy_window()

    # Convert images to video
    images_to_video(tempdir, output_video, skip_rate=skip_rate)

    # Clean up
    shutil.rmtree(tempdir)


if __name__ == "__main__":
    fm = FastModels()
    # test_sample_image_path = "/home/dynamo/Music/jackal_bags/backup_unified_dataset/images/morning_mode1_2_11062023_000008.png"
    # test_sample_image = Image.open(test_sample_image_path)
    # full_pred_seg, accumulated_results, terrain_results, depth_results, gsam_results = fm.predict_new(test_sample_image)
    # cv2_image_np = cv2.cvtColor(np.asarray(test_sample_image), cv2.COLOR_RGB2BGR)
    # cv2_pred_overlay = TerrainSegFormer.get_seg_overlay(cv2_image_np, full_pred_seg)
    # pilnp_pred_overlay = cv2.cvtColor(cv2_pred_overlay, cv2.COLOR_BGR2RGB)
    # pcd = depth_results[1]
    # colors = np.asarray(pilnp_pred_overlay).reshape(-1, 3) / 255.0
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # vis = o3d.visualization.Visualizer()
    # param = o3d.io.read_pinhole_camera_parameters("/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json")
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.add_geometry(pcd)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.run()
    # vis.destroy_window()

    # o3d.visualization.draw_geometries([pcd])
    # print(green("Done!"))

    # main(bag_file="/home/dynamo/Music/jackal_bags/new1.bag",
    #      image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
    #      camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
    #      output_video="/home/dynamo/Music/jackal_bags/VIDEOS/1/pcd.mp4",
    #      skip_rate=1)
    # main(bag_file="/home/dynamo/Music/jackal_bags/new1.bag",
    #      image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
    #      camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
    #      output_video="/home/dynamo/Music/jackal_bags/VIDEOS/1/pcd_realtime.mp4",
    #      skip_rate=10)

    # main(bag_file="/home/dynamo/Music/jackal_bags/new2.bag",
    #      image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
    #      camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
    #      output_video="/home/dynamo/Music/jackal_bags/VIDEOS/2/pcd.mp4",
    #      skip_rate=1)
    # main(bag_file="/home/dynamo/Music/jackal_bags/new2.bag",
    #      image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
    #      camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
    #      output_video="/home/dynamo/Music/jackal_bags/VIDEOS/2/pcd_realtime.mp4",
    #      skip_rate=10)

    main(bag_file="/home/dynamo/Music/jackal_bags/new3.bag",
         image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
         camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
         output_video="/home/dynamo/Music/jackal_bags/VIDEOS/3/pcd.mp4",
         skip_rate=1)
    main(bag_file="/home/dynamo/Music/jackal_bags/new3.bag",
         image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
         camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
         output_video="/home/dynamo/Music/jackal_bags/VIDEOS/3/pcd_realtime.mp4",
         skip_rate=10)

    main(bag_file="/home/dynamo/Music/jackal_bags/new4.bag",
         image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
         camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
         output_video="/home/dynamo/Music/jackal_bags/VIDEOS/4/pcd.mp4",
         skip_rate=1)
    main(bag_file="/home/dynamo/Music/jackal_bags/new4.bag",
         image_topic="/zed2i/zed_node/left/image_rect_color/compressed",
         camera_params_json="/home/dynamo/AMRL_Research/repos/nspl/ScreenCamera_2024-05-29-01-43-10.json",
         output_video="/home/dynamo/Music/jackal_bags/VIDEOS/4/pcd_realtime.mp4",
         skip_rate=10)
