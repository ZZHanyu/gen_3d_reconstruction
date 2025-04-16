import open3d as o3d
import numpy as np


def check_pcd_scale(gaussian):
    """
    Check the boundary box range of the input point cloud.

    Args:
        gaussian: Open3D PointCloud object containing the point cloud data.
    """
    # Extract point cloud positions
    positions = np.asarray(gaussian.points)

    # Calculate the boundary box
    min_bounds = positions.min(axis=0)
    max_bounds = positions.max(axis=0)

    # Print the boundary box range
    print(f"Point Cloud Boundary Box:")
    print(f"Min Bounds: {min_bounds}")
    print(f"Max Bounds: {max_bounds}")



if __name__ == "__main__":
    # "/home/zhy01/gaussian-splatting/output/c77ba6be-e/point_cloud/iteration_30000/point_cloud.ply"
    # Load the point cloud


    # point_cloud_path = "/home/zhy01/gaussian-splatting/output/c77ba6be-e/point_cloud/iteration_30000/point_cloud.ply"
    # '''
    #     Scale of the Gaussian (per axis): [2.08046826 1.40174136 1.28585104]
    #     Point Cloud Boundary Box:
    #     Min Bounds: [-5.76914644 -0.96589994 -0.42373353]
    #     Max Bounds: [5.03678989 6.64617777 6.77788782]
    # '''

    point_cloud_path = "/home/zhy01/桌面/feicuiwan_all/fused.ply"
    '''
    colmap output:
        Scale of the Gaussian (per axis): [1.64111482 1.11758483 1.01862492]
        Point Cloud Boundary Box:
        Min Bounds: [-4.76835251 -0.77058887  0.57503533]
        Max Bounds: [4.68534565 5.71688795 7.44791031]
    '''

    pcd = o3d.io.read_point_cloud(point_cloud_path)

    # Get the points as a numpy array
    points = np.asarray(pcd.points)

    # Calculate the scale (e.g., standard deviation of the points)
    scale = np.std(points, axis=0)

    print("Scale of the Gaussian (per axis):", scale)


    check_pcd_scale(pcd)