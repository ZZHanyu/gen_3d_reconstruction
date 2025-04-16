import open3d as o3d
import numpy as np
import trimesh
import copy 

pcd = o3d.io.read_point_cloud("/home/zhy01/桌面/feicuiwan_all/fused.ply")

#转换的mash存在孔洞
# 法线估计
radius1 = 0.1   # 搜索半径
max_nn = 100     # 邻域内用于估算法线的最大点数
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius1, max_nn)
)     # 执行法线估计

# 滚球半径的估计
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
)

print(mesh.get_surface_area())
o3d.visualization.draw_geometries(
    [mesh], 
    window_name='Open3D downSample', 
    width=800, 
    height=600, 
    left=50,
    top=50, 
    point_show_normal=True, 
    mesh_show_wireframe=True, 
    mesh_show_back_face=True,
)