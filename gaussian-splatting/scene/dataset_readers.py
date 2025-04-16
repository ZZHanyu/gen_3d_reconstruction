#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud, GaussianPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


class GaussianSceneInfo(NamedTuple):
    point_cloud: GaussianPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras_new(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):

    print(f"[INFO] image folder is = {images_folder}")
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        inter_name = image_name.split('/')[-1].split('_')[1] + '.png'
        image_all = os.listdir(images_folder)

        if inter_name not in image_all:
            print(f"[WRONG] image {inter_name} is not found in path!")
            continue

        image_path = os.path.join(images_folder, inter_name)
        image_name = inter_name
        

        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
            image_path=image_path, image_name=image_name, depth_path=depth_path,
            width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


# def fetchPly(path):
#     plydata = PlyData.read(path)
#     vertices = plydata['vertex']
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)



def judge_whether_gaussian(ply_path):
    
    ply_data = PlyData.read(ply_path)
    vertex_element = ply_data['vertex']

    # actually elements we had
    available_props = set(vertex_element.data.dtype.names)

    # gaussian requried data type
    required_properties = ['f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)]
    missing_props = [prop for prop in required_properties if prop not in available_props]


    def fetchPly(path):
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals)


    def fetchGaussian(ply_path):
        """
        读取 PLY 文件并提取高斯点信息。
        
        参数：
            ply_path (str): PLY 文件路径。
        
        返回：
            dict: 包含位置、法向量、颜色、其他属性的 NumPy 数组。
        """
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex']
        
        # 解析 PLY 数据
        positions = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
        
        # 将高斯颜色属性 f_dc 转化为 RGB
        f_dc = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T
        colors = SH2RGB(f_dc)
        
        # 解析高阶特征 f_rest
        # f_rest_keys = [f'f_rest_{i}' for i in range(45)]
        # f_rest = np.vstack([vertex_data[key] for key in f_rest_keys]).T
        
        # # 解析其他属性
        # opacity = np.array(vertex_data['opacity'])
        # scale = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T
        # rotation = np.vstack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T
        
        return BasicPointCloud(points=positions, normals=normals, colors=colors)

        # {
        #     'positions': positions,  # (N, 3)
        #     'normals': normals,      # (N, 3)
        #     'colors': colors,        # (N, 3)
        #     'f_rest': f_rest,        # (N, 45)
        #     'opacity': opacity,      # (N,)
        #     'scale': scale,          # (N, 3)
        #     'rotation': rotation     # (N, 4)
        # }

    if missing_props:
        # its normal pcd
        pcd = fetchPly(ply_path)
        return pcd
    else:
        # its gaussian
        gaussian = fetchGaussian(ply_path)
        return gaussian


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos


def readCamerasFromTransforms_update(
        path, 
        transformsfile, 
        depths_folder, 
        white_background, 
        is_test, 
        extension='.png', 
        image_path=None, 
        all_frame=None,
        train_angle_camera=None,
    ):

    init_image_path = image_path
    cam_infos = []

    if transformsfile is None and all_frame is None:
        raise ValueError("Either transformsfile or all_frame must be provided.")
    elif all_frame is not None:
        if train_angle_camera is None:
            raise ValueError("train_angle_camera must be provided if all_frame is provided.")
        fovx = train_angle_camera
        frames = all_frame
    elif image_path is None:
        raise ValueError("[ERROR] The new image path cannot be None!")
    else:
        with open(os.path.join(path, transformsfile['frames']['file_path'])) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]
            frames = contents["frames"]

    # for idx, frame in enumerate(frames):
    #     image_path = '/home/zhy01/Distill-Any-Depth/depth'
    #     if isinstance(frame["file_path"], list):
    #         continue
    #     cam_name = frame["file_path"].split('_')[1] + extension

    #     # NeRF 'transform_matrix' is a camera-to-world transform
    #     c2w = np.array(frame["transform_matrix"])

    #     # get the world-to-camera transform and set R, T
    #     w2c = np.linalg.inv(c2w)
    #     R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #     T = w2c[:3, 3]

    #     image_path = os.path.join(image_path, cam_name)
    #     image_name = Path(cam_name).stem

    #     try:
    #         image = Image.open(image_path)
    #     except FileNotFoundError:
    #         print(f"[WARNING] {image_path} not found!")
    #         continue

    #     im_data = np.array(image.convert("RGBA"))

    #     bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

    #     norm_data = im_data / 255.0
    #     arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    #     image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

    #     fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
    #     FovY = fovy 
    #     FovX = fovx

    #     depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

    #     cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
    #                     image_path=image_path, image_name=image_name,
    #                     width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
    for idx, frame in enumerate(frames):

        image_path = init_image_path
        cam_name = os.path.join(path, frame["file_path"] + extension)

        inter_name = frame["file_path"].split('/')[-1].split('_')[1] + extension

        new_cam_path = os.path.join(image_path, inter_name)
        search_area = os.listdir(image_path)

        if inter_name not in search_area:
            print(f"[WRONG] did not found diffusion image {inter_name}! continue..")
            continue
            # raise FileNotFoundError("[ERROR] file did not found in new image folder...")

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image_path = new_cam_path
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                        image_path=image_path, image_name=inter_name,
                        width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
        
    return cam_infos



def readNerfSyntheticInfo(path, whitebackground, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = judge_whether_gaussian(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


# def readDiffusionInfo(
#     gaussian_path,
#     white_background,
#     depths, 
#     eval, 
#     extension=".png",
#     no_split_data=True,
#     new_image_path=None,
# ):  
#     gaussian_path = "/home/zhy01/gaussian-splatting/output/c77ba6be-e/point_cloud/iteration_30000/point_cloud.ply"
#     blender_transformer_path = "/home/zhy01/data/s1/"
#     # new_image_path = "/home/zhy01/Distill-Any-Depth/depth"
#     if new_image_path is None:
#         raise FileNotFoundError("No new image dataset found!")
#     else:
#         print(f"[INFO] Fetching new dataset from {new_image_path}!")
    
#     depths_folder=os.path.join(path, depths) if depths != "" else ""


#     if not eval and no_split_data is True:
#         with open(os.path.join(blender_transformer_path, "transforms_train.json")) as json_file_train, open(os.path.join(blender_transformer_path, "transforms_test.json")) as json_file_test:
#             train_data = json.load(json_file_train)
#             test_data = json.load(json_file_test)

#             # Merge the "frames" from both train and test JSON files
#             merged_data = train_data.copy()
#             merged_data["frames"].extend(test_data["frames"])

#             all_frames = train_data["frames"] + test_data["frames"]
#             train_angle_camera = train_data["camera_angle_x"]
#             test_angle_camera = test_data["camera_angle_x"]
            
#             all_camera_info = readCamerasFromTransforms_update(
#                 blender_transformer_path,
#                 merged_data,
#                 depths_folder,
#                 white_background,
#                 False,
#                 extension,
#                 new_image_path,
#                 all_frame=all_frames, 
#                 train_angle_camera=train_angle_camera
#             )

#             train_cam_infos = all_camera_info
#             test_cam_infos = all_camera_info

#     elif no_split_data is False:
#         print("Reading Training Transforms")
#         train_cam_infos = readCamerasFromTransforms_update(blender_transformer_path, "transforms_train.json", depths_folder, white_background, False, extension, new_image_path)
#         print("Reading Test Transforms")
#         test_cam_infos = readCamerasFromTransforms_update(blender_transformer_path, "transforms_test.json", depths_folder, white_background, True, extension, new_image_path)

#         if not eval:
#             train_cam_infos.extend(test_cam_infos)
#             test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = gaussian_path
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")
        
#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     else:
#         pcd = judge_whether_gaussian(ply_path)

#     scene_info = SceneInfo(
#         point_cloud=pcd,
#         train_cameras=train_cam_infos,
#         test_cameras=test_cam_infos,
#         nerf_normalization=nerf_normalization,
#         ply_path=ply_path,
#         is_nerf_synthetic=True
#     )

    
    
#     # try:
#     #     pcd = judge_whether_gaussian(ply_path)
#     # except:
#     #     pcd = None
#     #     raise ValueError("The point cloud is not a Gaussian point cloud.")

#     # if isinstance(pcd, BasicPointCloud):
#     #     scene_info = SceneInfo(
#     #         point_cloud=pcd,
#     #         train_cameras=train_cam_infos,
#     #         test_cameras=test_cam_infos,
#     #         nerf_normalization=nerf_normalization,
#     #         ply_path=ply_path,
#     #         is_nerf_synthetic=True
#     #     )
#     # elif isinstance(pcd, GaussianPointCloud):
#     #     scene_info = GaussianSceneInfo(
#     #         point_cloud=pcd,
#     #         train_cameras=train_cam_infos,
#     #         test_cameras=test_cam_infos,
#     #         nerf_normalization=nerf_normalization,
#     #         ply_path=ply_path,
#     #         is_nerf_synthetic=True
#     #     )
    
#     return scene_info


def readDiffusionInfo(
    path, depths, eval, train_test_exp, llffhold=8, new_image_path=None
):

    # new_image_path = "/home/zhy01/Distill-Any-Depth/depth"
    if new_image_path is None:
        raise FileNotFoundError("No new image dataset found!")
    else:
        print(f"[INFO] Fetching new dataset from {new_image_path}!")
    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    # reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_new(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=new_image_path, 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join("/home/zhy01/gen_3d_recon/gaussian-splatting/output/c77ba6be-e/point_cloud/iteration_30000/point_cloud.ply")

    # ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = judge_whether_gaussian(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info
    

    

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Diffusion" : readDiffusionInfo,
}