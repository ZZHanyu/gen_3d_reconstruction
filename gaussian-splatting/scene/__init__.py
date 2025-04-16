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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene: 
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], loaded_gaussian_path=None, new_image_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        if loaded_gaussian_path is None and new_image_path is None:
            raise ValueError("Either loaded_gaussian_path or new_image_path must be provided.")

        self.model_path = args.model_path
        self.loaded_iter = load_iteration
        self.loaded_gaussian_path = loaded_gaussian_path

        print(f"[INFO] loading scene from {self.model_path}!")
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.load_iteration = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.load_iteration = load_iteration
            print("Loading trained model at iteration {}".format(self.load_iteration))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print(f"[INFO] Found sparse folder, assuming COLMAP data set!")
            # scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
            scene_info = sceneLoadTypeCallbacks["Diffusion"](
                args.source_path, 
                args.depths, 
                args.eval, 
                args.train_test_exp,
                new_image_path=new_image_path,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("[INFO] Found transforms_train.json file, assuming Diffusion data set!")
            # scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
            # blender version
            scene_info = sceneLoadTypeCallbacks["Diffusion"](
                args.source_path, 
                args.white_background, 
                args.depths, 
                args.eval, 
                new_image_path=new_image_path,
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_gaussian_path:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            # /home/zhy01/output/ade59c79-6/point_cloud/iteration_30000/point_cloud.ply
            # self.gaussians.load_ply(os.path.join(
            #     self.model_path,
            #     "point_cloud",
            #     "iteration_" + str(self.loaded_iter),
            #     "point_cloud.ply"
            # ), args.train_test_exp)
            # self.loaded_iter = loaded_iter
            try:
                self.gaussians.load_ply(os.path.join(
                    self.loaded_gaussian_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                ), args.train_test_exp)
            except:
                print(f"[WRONG] No iteration {self.loaded_iter} founded!")
                try:
                    self.loaded_iter = 7000
                    self.gaussians.load_ply(os.path.join(
                        self.loaded_gaussian_path,
                        "point_cloud",
                        f"iteration_{self.loaded_iter}",
                        "point_cloud.ply",
                    ), args.train_test_exp)
                except Exception as e:
                    print(f"[WRONG] No iteration {self.loaded_iter} founded!")
                    print(e)
                    raise FileNotFoundError
        else: 
            print(f"[INFO] no trained gaussian founded! now init new gaussian from pcd...")
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.exposure_mapping is not None:
            exposure_dict = {
                image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.gaussians.exposure_mapping
            }
            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def update_dataset(self, dataset):
        print(f"Updating dataset with new COLMAP images folder: {self.model_path}")
        if os.path.exists(os.path.join(self.model_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.model_path, None, None, False, None)
        else:
            raise ValueError("COLMAP sparse folder not found in the specified path.")

        self.train_cameras.clear()
        self.test_cameras.clear()

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Reloading Training Cameras")
        self.train_cameras[1.0] = cameraList_from_camInfos(scene_info.train_cameras, 1.0, None, scene_info.is_nerf_synthetic, False)
        print("Reloading Test Cameras")
        self.test_cameras[1.0] = cameraList_from_camInfos(scene_info.test_cameras, 1.0, None, scene_info.is_nerf_synthetic, True)

        print("Dataset update complete.")
        pass


    def update_dataset_new(self, transforms_json_path, image_folder):
        """
        更新数据集中的相机视图和位姿，支持基于 transforms_train.json 和图片文件夹的格式。

        Args:
            transforms_json_path (str): 包含相机外参的 transforms_train.json 文件路径。
            image_folder (str): 包含监督图片的文件夹路径。
        """
        import json
        from utils.camera_utils import CameraInfo

        # 加载 transforms_train.json
        with open(transforms_json_path, 'r') as f:
            transforms_data = json.load(f)

        # 清空现有的相机数据
        self.train_cameras = {}
        self.test_cameras = {}

        # 解析 transforms_train.json
        train_cameras = []
        for frame in transforms_data["frames"]:
            image_path = os.path.join(image_folder, os.path.basename(frame["file_path"]))
            if not os.path.exists(image_path):
                print(f"[WARNING] Image {image_path} not found, skipping.")
                continue

            # 创建 CameraInfo 对象
            camera_info = CameraInfo(
                image_path=image_path,
                transform_matrix=frame["transform_matrix"],
                focal_length=transforms_data["fl_x"],  # 假设 focal length 在 JSON 中
                image_width=transforms_data["w"],
                image_height=transforms_data["h"]
            )
            train_cameras.append(camera_info)

        # 更新训练相机
        print("Updating Training Cameras")
        self.train_cameras[1.0] = cameraList_from_camInfos(train_cameras, 1.0, None)

        # 更新相机范围（假设可以从 transforms_train.json 中计算）
        self.cameras_extent = transforms_data.get("scene_extent", 1.0)  # 默认范围为 1.0

        # 打印更新完成信息
        print(f"Dataset updated with {len(train_cameras)} training cameras.")