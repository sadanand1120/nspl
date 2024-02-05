import numpy as np
import open3d as o3d


def visualize_point_cloud(pc_np):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    orange_color = [1, 0.647, 0]  # RGB for orange
    pcd.colors = o3d.utility.Vector3dVector(np.tile(orange_color, (len(xyz), 1)))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 0.75  # Smaller point size
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    vis.run()
    vis.destroy_window()
