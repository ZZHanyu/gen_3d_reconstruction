import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import pyopencl as cl
import numpy as np
from PIL import Image
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
import cv2
import time
import open3d as o3d
import random
from unidepth.models import UniDepthV1, UniDepthV2

from distillanydepth.modeling.archs.dam.dam import DepthAnything
# from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
# from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
# from torchvision.transforms import Compose
from app import process_image
import math

from transformers import pipeline
import requests

if torch.cuda.is_available():
    print(f"当前使用的GPU设备：{torch.cuda.current_device()}")
    print(f"GPU设备名称：{torch.cuda.get_device_name(0)}")
else:
    print("没有可用的cuda设备!")

import sys
sys.path.append("/home/zhy01/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def readCamerasTimeFromTransforms(fovx, frames, height, width, white_background=None):
    c2w = np.array(frames["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    # c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    # image_path = cam_name
    # image_name = Path(cam_name).stem
    # image = Image.open(image_path)

    # im_data = np.array(image.convert("RGBA"))

    # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    # norm_data = im_data / 255.0
    # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    # alpha_mask = norm_data[:, :, 3]
    # alpha_mask = Image.fromarray(np.array(alpha_mask*255.0, dtype=np.byte), "L")
    # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
    # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

    # normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
    # normal_image_path = os.path.join(path, normal_cam_name)
    # if os.path.exists(normal_image_path):
    #     normal_image = Image.open(normal_image_path)
        
    #     normal_im_data = np.array(normal_image.convert("RGBA"))
    #     normal_bg_mask = (normal_im_data==128).sum(-1)==3
    #     normal_norm_data = normal_im_data / 255.0
    #     normal_arr = normal_norm_data[:,:,:3] * normal_norm_data[:, :, 3:4] + bg * (1 - normal_norm_data[:, :, 3:4])
    #     normal_arr[normal_bg_mask] = 0
    #     normal_image = Image.fromarray(np.array(normal_arr*255.0, dtype=np.byte), "RGB")
    # else:
    #     normal_image = None
    # norm_data = im_data / 255.0
    # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

    fovy = focal2fov(fov2focal(fovx, width), height)
    FovY = fovy 
    FovX = fovx

    K = compute_intrinsic_matrix(width, height, FovX, FovY)
            
    return R, T, FovY, FovX, w2c, K


def compute_intrinsic_matrix(W, H, fov_x, fov_y, fov_unit='rad'):
    """
    计算内参数矩阵 K。
    
    参数：
        W (int): 图像宽度（像素）
        H (int): 图像高度（像素）
        fov_x (float): 水平视场角
        fov_y (float): 垂直视场角
        fov_unit (str): 视场角的单位，'deg'（度）或 'rad'（弧度），默认为 'deg'
    
    返回：
        K (np.ndarray): 内参数矩阵，3x3
    """
    # 如果视场角以度为单位，转换为弧度
    if fov_unit == 'deg':
        fov_x_rad = np.deg2rad(fov_x)
        fov_y_rad = np.deg2rad(fov_y)
    elif fov_unit == 'rad':
        fov_x_rad = fov_x
        fov_y_rad = fov_y
    else:
        raise ValueError("fov_unit must be 'deg' or 'rad'")

    # 计算焦距（以像素为单位）
    f_x = W / (2 * np.tan(fov_x_rad / 2))
    f_y = H / (2 * np.tan(fov_y_rad / 2))

    # 主点（假设在图像中心）
    c_x = W / 2
    c_y = H / 2

    # 构建内参数矩阵 K
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return K



def depth_to_pointcloud(depth_map, intrinsics, extrinsic):
    # 输入数据（根据图片尺寸）
    # height, width = depth_map.shape[:2]

    # point_cloud = np.zeros((height * width, 3), dtype=np.float32)

    # # 创建 OpenCL 缓冲区
    # depth_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=depth_map.flatten())
    # pc_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, point_cloud.nbytes)

    # # 运行 Kernel
    # kernel = program.depth_to_pointcloud
    # fx = intrinsics[0][0]
    # fy = intrinsics[1][1]
    # cx = intrinsics[0][3]
    # cy = intrinsics[1][3]
    # kernel.set_args(depth_buf, pc_buf, np.int32(width), np.int32(height), np.float32(fx), np.float32(fy), np.float32(cx), np.float32(cy))

    # # 并行计算
    # global_size = (width, height)
    # cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)

    # # 读取结果
    # cl.enqueue_copy(queue, point_cloud, pc_buf)
    # queue.finish()

    # rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(depth_map, intrinsics, extrinsic)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # 保存点云为 PLY 文件
    o3d.io.write_point_cloud("/home/zhy01/Distill-Any-Depth/output.ply", pcd)

    # # 保存点云为 PLY 文件
    # ply_header = f"""ply
    #     format ascii 1.0
    #     element vertex {height * width}
    #     property float x
    #     property float y
    #     property float z
    #     end_header
    #     """
    # ply_data = ply_header + "\n".join(f"{point[0]} {point[1]} {point[2]}" for point in point_cloud)
    # with open("/home/zhy01/Distill-Any-Depth/output.ply", "w+") as f:
    #     try:
    #         f.write(ply_data)
    #     except Exception as e:
    #         print(f"[ERROR] {e}!")



def distill_pipeline():
     # config model
    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    # # 设置 OpenCL 环境
    # platforms = cl.get_platforms()
    # gpu_devices = platforms[0].get_devices(cl.device_type.GPU)
    # context = cl.Context(gpu_devices)
    # queue = cl.CommandQueue(context)

    # # OpenCL Kernel 代码
    # kernel_code = """
    # __kernel void depth_to_pointcloud(
    #     __global const float* depth_map,
    #     __global float3* point_cloud,
    #     const int width, 
    #     const int height,
    #     const float fx, const float fy,
    #     const float cx, const float cy) {
        
    #     int x = get_global_id(0);
    #     int y = get_global_id(1);
    #     if (x >= width || y >= height) return;

    #     int idx = y * width + x;
    #     float depth = depth_map[idx];

    #     if (depth > 0.0f) {
    #         float X = (x - cx) * depth / fx;
    #         float Y = (y - cy) * depth / fy;
    #         float Z = depth;
    #         point_cloud[idx] = (float3)(X, Y, Z);
    #     } else {
    #         point_cloud[idx] = (float3)(0.0f, 0.0f, 0.0f);
    #     }
    # }
    # """

    # # 编译 OpenCL 程序
    # program = cl.Program(context, kernel_code).build()


    # select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device selected = {device}!")

    # Distill-anything mmodel
    model = DepthAnything(**model_kwargs['vitl']).to(device)
        # if use hf_hub_download, you can use the following code
    checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")
        # if use local path, you can use the following code
    # checkpoint_path = "path/to/your/model.safetensors"    

    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model = model.to(device)

    # check whether has a model
    if model is None:
        print("[ERROR] you have to load a model...")
        raise ValueError

    # Read image from path
    image_path = '/home/zhy01/data/s1/train'
    image_list =  os.listdir(image_path)
    random.shuffle(image_list)
    for single_image in tqdm(image_list, leave=True, desc="[info] Now loading2depth2pcd pipeline..."):
        # load image        
        image = Image.open(os.path.join(image_path, single_image))
        image = np.array(image)
        # load intrinsics of this image
        with open('/home/zhy01/data/s1/transforms_train.json', 'r') as f:
            data = json.load(f)
            target_path = f"./train/{single_image.split('.')[0]}"
            # build a dict
            frame_dict = {frame["file_path"]: frame for frame in data["frames"]}
            angel_x = data["camera_angle_x"]
            selected_frame = frame_dict.get(target_path, None)
            if selected_frame:
                height, width = image.shape[:2]
                R, T, FovY, FovX, w2c, K = readCamerasTimeFromTransforms(angel_x, selected_frame, height, width)
                # transform_matrix = selected_frame["transform_matrix"]
                
                # Process image and return output
                depth_image, raw_depth = process_image(image, model, device)
                depth_image = raw_depth
                depth_image = np.array(depth_image, dtype=np.float32)
                # print(f"[Value]: type = {type(depth_image)}, value: {depth_image}")
                # # 如果是 uint16，转换到 0-255 范围以便可视化
                # if depth_image.dtype == np.uint16:
                #     depth_vis = (depth_image / depth_image.max() * 255).astype(np.uint8)
                # else:
                #     depth_vis = depth_image  # 可能已经是 uint8
                
                # # 归一化深度范围（根据你的数据集设定）
                # depth_image = np.clip(depth_image, np.min(depth_image), np.max(depth_image))

                # print(f"[Value]: after normlization: {depth_image}")

                # Convert image and depth_image to Open3D RGBDImage
                color_image = o3d.geometry.Image(image)
                # depth_image = Image.fromarray(depth_image)
                depth_image = o3d.geometry.Image(depth_image)
                # o3d.io.write_image("/home/zhy01/Distill-Any-Depth/depth_image.png", depth_image)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
                o3d.io.write_image("/home/zhy01/Distill-Any-Depth/rgbd_image.png", rgbd_image.color)
                o3d.io.write_image("/home/zhy01/Distill-Any-Depth/depth_image.png", rgbd_image.depth)

                # Convert intrinsics to Open3D PinholeCameraIntrinsic
                # # using default intrinsic setting
                # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
                # intrinsic_matrix = pinhole_camera_intrinsic.intrinsic_matrix
                # fx, fy = pinhole_camera_intrinsic.get_focal_length()
                # cx, cy = pinhole_camera_intrinsic.get_principal_point()
                # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                #     width, height, fx, fy, cx, cy
                # )

                intrinsic_matrix = K

                # Convert extrinsics to numpy array
                # extrinsic = np.linalg.inv(transform_matrix)
                extrinsic = w2c

                # depth2pcd
                depth_to_pointcloud(rgbd_image, pinhole_camera_intrinsic,extrinsic)
            else:
                # print("not found!")
                continue




def unidepth_pipeline(
        image_path,
        camera=None, 
        intrinsics=None
):
    mode_v1 = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone
    model_v2 = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14")

    # Move to CUDA, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     try:
    #         mode_v1 = torch.nn.DataParallel(mode_v1, device_ids=[0, 1])
    #         print(f"[INFO]: CUDA is avaliable, enable parallel...")
    #     except Exception as e:
    #         print(f"[ERROR]: {e}, NOW using single GPU mode...")
    
    model_v1 = mode_v1.to(device)
    model_v2 = model_v2.to(device)

    # Load the RGB image and the normalization will be taken care of by the model
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1) # C, H, W

    if camera is None:
        predictions = model_v2.infer(rgb)
    else:
        predictions = model_v2.infer(rgb, intrinsics)
    
    print(f"[INFO]: predictions has {predictions.keys()} attributes!")

    # Metric Depth Estimation
    depth = predictions["depth"]
    print(f"[INFO]: shape of depth = {depth.shape}")

    # Point Cloud in Camera Coordinate
    xyz = predictions["points"]
    print(f"[INFO]: shape of xyz = {xyz.shape}")

    # Intrinsics Prediction
    intrinsics = predictions["intrinsics"]
    print(f"[INFO]: shape of intrinsics = {intrinsics.shape}")

    '''
        [INFO]: predictions has dict_keys(['intrinsics', 'points', 'depth']) attributes!
        [INFO]: shape of depth = torch.Size([1, 1, 512, 640])
        [INFO]: shape of xyz = torch.Size([1, 3, 512, 640])
        [INFO]: shape of intrinsics = torch.Size([1, 3, 3])
    '''

    return depth, xyz, intrinsics



def depth_anything_v2(
    image,
    camera=None, 
    intrinsics=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] depth anything v2 using device = {device}")

    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load('/home/zhy01/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth', map_location=device))
    model.to(device).eval()

    # if image.device != device:
    #     image = image.to(device)

    pth = model.infer_image(image) # HxW raw depth map
    depth_image = Image.fromarray((pth * 255 / np.max(pth)).astype(np.uint8))
    
    try:
        depth_image.save("/home/zhy01/Distill-Any-Depth/depth_image_depth_anything_v2.png")
    except Exception as e:
        print(f"[ERROR] saving depth error!")
    
    return depth_image



def trellis_pipeline(
        image_path,
        camera,
        intrinsics
):
    pass



def single_image(
        image_folder=None
):
    ''''
        no intrinsics 
        input is diffusion genrated
    '''
    if image_folder is None:
        raise FileExistsError
    
    for single_image in os.listdir(image_folder):
        image_path = os.path.join(image_folder, single_image)
        image = Image.open(image_path)
        image.load()
        
        image_cv = cv2.imread(image_path)
        image_np = np.asarray(image, dtype=np.float32)
        print(f"[INFO]: single image name is {single_image}!")

        start_time = time.time()
        # depth, xyz, intrinsics = unidepth_pipeline(image_path)
        depth = depth_anything_v2(image_cv)
        end_time = time.time()

        print(f"[INFO] total time costing on single view reconstruction is {end_time - start_time}s!")

        if isinstance(depth, Image):
            depth.save("/home/zhy01/Distill-Any-Depth/depth_image_unidepth.png")
        else:

            # Convert depth to numpy array and save as PNG
            depth_np = depth.squeeze().cpu().numpy()
            depth_image = Image.fromarray((depth_np * 255 / np.max(depth_np)).astype(np.uint8))
            depth_image.save("/home/zhy01/Distill-Any-Depth/depth_image_unidepth.png")
        
            # Convert xyz to numpy array and reshape
            xyz_np = xyz.squeeze().cpu().numpy().reshape(3, -1).T

            # Create Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # ori-pcd is reversed we need transfer it back
            pcd.points = o3d.utility.Vector3dVector(xyz_np)

            # Save point cloud to PLY file
            o3d.io.write_point_cloud("/home/zhy01/Distill-Any-Depth/output_SD_unidepth.ply", pcd)





if __name__ == "__main__":
    # # Read image from path
    # image_base_path = '/home/zhy01/data/s1/train'
    # image_list =  os.listdir(image_base_path)
    # random.shuffle(image_list)
    
    # for single_image in tqdm(image_list, leave=True, desc="[info] Now loading2depth2pcd pipeline..."):
    #     image_path = os.path.join(image_base_path, single_image)
    #     image = Image.open(image_path)

    #     print(f"[INFO] single image name is {single_image}!")

    #     with open('/home/zhy01/data/s1/transforms_train.json', 'r') as f:
    #         data = json.load(f)
    #         target_path = f"./train/{single_image.split('.')[0]}"
    #         # build a dict
    #         frame_dict = {frame["file_path"]: frame for frame in data["frames"]}
    #         angel_x = data["camera_angle_x"]
    #         selected_frame = frame_dict.get(target_path, None)
    #         if selected_frame:
    #             height, width = image.size
    #             R, T, FovY, FovX, w2c, K = readCamerasTimeFromTransforms(angel_x, selected_frame, height, width)
    #             intrinsics = torch.from_numpy(K) # 3 x 3
    #             from unidepth.utils.camera import Pinhole, Fisheye624
    #             camera = Pinhole(K=intrinsics) # pinhole 
    #         else:
    #             camera = None
        
    #     start_time = time.time()
    #     depth, xyz, intrinsics = unidepth_pipeline(image_path, camera, intrinsics)
    #     end_time = time.time()

    #     print(f"[INFO] total time costing on single view reconstruction is {end_time - start_time}s!")
        
    #     # Convert depth to numpy array and save as PNG
    #     depth_np = depth.squeeze().cpu().numpy()
    #     depth_image = Image.fromarray((depth_np * 255 / np.max(depth_np)).astype(np.uint8))
    #     depth_image.save("/home/zhy01/Distill-Any-Depth/depth_image_unidepth.png")
        
    #     # Convert xyz to numpy array and reshape
    #     xyz_np = xyz.squeeze().cpu().numpy().reshape(3, -1).T

    #     # Create Open3D PointCloud object
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # ori-pcd is reversed we need transfer it back
    #     pcd.points = o3d.utility.Vector3dVector(xyz_np)

    #     # Save point cloud to PLY file
    #     o3d.io.write_point_cloud("/home/zhy01/Distill-Any-Depth/output_unidepth.ply", pcd)



    single_image("/home/zhy01/Distill-Any-Depth/distroyed_city")