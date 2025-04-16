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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import wandb
wandb_status = True
try:
    wandb.login()
    print("[INFO] Wandb init successfully.")
    run = wandb.init(
        project = "Gaussian-Splatting-Surface-Reconstruction",
    )
except:
    wandb_status = False
    print("[ERROR] WandB is not installed or login failed. Skipping WandB integration.")
    

def get_projection_image_check(
        gaussians, 
        camera, 
        iteration, 
        model_path='/home/zhy01/gaussian-splatting/inter_output'
    ):
    """
    Project the Gaussian point cloud onto the camera planes and visualize with a bounding box.

    Args:
        gaussians: GaussianModel object containing the point cloud data.
        camera: Camera object with RT matrices.
        iteration: Current training iteration.
        model_path: Path to save the projection images.
    """
    if os.path.exists(model_path) is False:
        os.mkdir(model_path)

    # Extract Gaussian positions
    positions = gaussians.get_xyz.cpu().detach().numpy()

    # Compute camera intrinsic matrix using simple pinhole model
    fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
    fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
    cx = camera.image_width / 2
    cy = camera.image_height / 2
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

    # Use RT matrix as extrinsic
    extrinsic = np.hstack((camera.R, camera.T.reshape(-1, 1)))

    # Project Gaussian positions to the camera plane
    homogeneous_positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
    camera_coords = extrinsic @ homogeneous_positions.T
    image_coords = intrinsic @ camera_coords[:3, :]
    image_coords /= image_coords[2, :]  # Normalize by depth

    # Filter points within the image bounds
    valid_mask = (image_coords[0, :] >= 0) & (image_coords[0, :] < camera.image_width) & \
                 (image_coords[1, :] >= 0) & (image_coords[1, :] < camera.image_height)
    valid_coords = image_coords[:, valid_mask]
    invalid_coords = image_coords[:, ~valid_mask]

    # Create an empty image and plot the projected points
    projection_image = np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)

    # Draw valid points in green
    for x, y in valid_coords[:2, :].T:
        projection_image[int(y), int(x)] = [0, 255, 0]  # Green points

    # Draw invalid points in red
    for x, y in invalid_coords[:2, :].T:
        if 0 <= int(y) < camera.image_height and 0 <= int(x) < camera.image_width:
            projection_image[int(y), int(x)] = [255, 0, 0]  # Red points

    # Draw the camera frame as a bounding box
    plt.figure(figsize=(10, 8))
    plt.imshow(projection_image)
    plt.gca().add_patch(plt.Rectangle((0, 0), camera.image_width, camera.image_height,
                                       edgecolor='blue', fill=False, linewidth=2, label='Camera Frame'))
    plt.legend()
    plt.title(f'Projection at Iteration {iteration}')
    plt.axis('off')

    # Save the projection image
    output_path = os.path.join(model_path, f'projection_camera_iter_{iteration}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Projection image saved to {output_path}")


# def visualize_gaussians_and_cameras(
#         gaussians, 
#         camera, 
#         iteration, 
#         model_path='/home/zhy01/gaussian-splatting/inter_output'
#     ):
#     """
#     Visualize the Gaussian point cloud and camera frustums.

#     Args:
#         gaussians: GaussianModel object containing the point cloud data.
#         cameras: List of camera objects to visualize.
#         iteration: Current training iteration.
#         model_path: Path to save the visualization output.
#     """
#     if not os.path.exists(model_path):
#         os.mkdir(model_path)
    
#     # Extract Gaussian positions
#     positions = gaussians.get_xyz.cpu().detach().numpy()

#     # Downsample Gaussian positions to 1%
#     downsampled_indices = np.random.choice(positions.shape[0], size=max(1, positions.shape[0] // 100), replace=False)
#     downsampled_positions = positions[downsampled_indices]

#     # Extract camera positions and frustums
#     camera_positions = []
#     frustum_lines = []
#     center = camera.camera_center.cpu().numpy()
#     camera_positions.append(center)

#     # Compute frustum corners in world space
#     fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
#     fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
#     intrinsic = np.array([[fx, 0, camera.image_width / 2],
#                             [0, fy, camera.image_height / 2],
#                             [0,  0,  1]])
#     extrinsic = np.hstack((camera.R, camera.T.reshape(-1, 1)))

#     # Add a small regularization term to ensure numerical stability
#     extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
#     # Compute the corners of the imaging plane in camera space
#     z = 1  # Assume the imaging plane is at z = 1 in camera space
#     fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
#     fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
#     cx = camera.image_width / 2
#     cy = camera.image_height / 2

#     corners = np.array(
#         [
#             [(-cx) / fx, (-cy) / fy, z],  # Top-left corner
#             [(camera.image_width - cx) / fx, (-cy) / fy, z],  # Top-right corner
#             [(camera.image_width - cx) / fx, (camera.image_height - cy) / fy, z],  # Bottom-right corner
#             [(-cx) / fx, (camera.image_height - cy) / fy, z]  # Bottom-left corner
#         ]
#     )
#     corners = np.linalg.inv(intrinsic) @ corners.T
#     corners = np.vstack((corners, np.ones((1, corners.shape[1]))))
#     corners_world = np.linalg.inv(extrinsic) @ corners
#     corners_world /= corners_world[3, :]
#     corners_world = corners_world[:3, :].T

#     # Add lines from camera center to frustum corners
#     for corner in corners_world:
#         frustum_lines.append((center, corner))

#     camera_positions = np.array(camera_positions)

#     # Create a 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot downsampled Gaussian positions
#     ax.scatter(downsampled_positions[:, 0], downsampled_positions[:, 1], downsampled_positions[:, 2], c='blue', s=10, label='Gaussians (1%)')  # Ensure 's' is a valid scalar

#     # Plot camera positions
#     ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='red', s=20, label='Cameras')  # Adjust 's' for better visibility

#     # Plot frustum lines
#     for line in frustum_lines:
#         start, end = line
#         ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c='green', linewidth=1)  # Adjust linewidth for better visibility

#     # Add labels and legend
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')  # Ensure the plot is a 3D plot (Axes3D object)
#     ax.legend()

#     # Set title
#     ax.set_title(f'Visualization at Iteration {iteration}')

#     # Save the plot
#     output_path = os.path.join(model_path, f'visualization_iter_{iteration}.png')
#     plt.savefig(output_path)
#     plt.close(fig)

#     print(f"Visualization saved to {output_path}")

def visualize_gaussians_and_cameras(
    gaussians, 
    camera, 
    iteration, 
    model_path='/home/zhy01/gaussian-splatting/inter_output'
):
    """
    Visualize the Gaussian point cloud and camera frustum.

    Args:
        gaussians: GaussianModel object containing the point cloud data.
        camera: A single camera object with attributes like camera_center, R, T, FoVx, FoVy, image_width, image_height.
        iteration: Current training iteration.
        model_path: Path to save the visualization output.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # === 1. Downsample Gaussians ===
    positions = gaussians.get_xyz.cpu().detach().numpy()
    num_points = positions.shape[0]
    downsampled_indices = np.random.choice(num_points, size=max(1, num_points // 100), replace=False)
    downsampled_positions = positions[downsampled_indices]

    # === 2. Get camera center ===
    center = camera.camera_center.cpu().numpy().reshape(3,)
    camera_positions = np.array([center])
    frustum_lines = []

    # === 3. Compute frustum corners in world space ===
    image_width = camera.image_width
    image_height = camera.image_height
    FoVx = camera.FoVx
    FoVy = camera.FoVy

    fx = image_width / (2 * np.tan(FoVx / 2))
    fy = image_height / (2 * np.tan(FoVy / 2))
    cx = image_width / 2
    cy = image_height / 2

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Image plane assumed at z=1 in camera space
    image_corners = np.array([
        [0, 0, 1],  # Top-left
        [image_width, 0, 1],  # Top-right
        [image_width, image_height, 1],  # Bottom-right
        [0, image_height, 1]   # Bottom-left
    ]).T  # Shape (3, 4)

    # Normalize camera space coords
    normalized_corners = np.linalg.inv(intrinsic) @ image_corners
    normalized_corners = np.vstack((normalized_corners, np.ones((1, 4))))  # Homogeneous

    # Camera-to-world transformation
    extrinsic = np.vstack((np.hstack((camera.R, camera.T.reshape(-1, 1))), [0, 0, 0, 1]))
    corners_world = (extrinsic @ normalized_corners)[:3].T  # shape (4, 3)

    # === 4. Create frustum lines ===
    for corner in corners_world:
        frustum_lines.append((center, corner))  # From camera center to corner

    # Add image plane rectangle lines (for visual clarity)
    for i in range(4):
        start = corners_world[i]
        end = corners_world[(i + 1) % 4]
        frustum_lines.append((start, end))  # Between adjacent corners

    # === 5. Create 3D plot ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Gaussians
    ax.scatter(downsampled_positions[:, 0], downsampled_positions[:, 1], downsampled_positions[:, 2],
               c='blue', s=1, label='Gaussians (1%)')

    # Plot camera center
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='red', s=20, label='Camera')

    # Plot frustum lines
    for start, end in frustum_lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                c='green', linewidth=1)

    # Labels, legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Gaussian + Camera Frustum at Iter {iteration}')

    # Save
    output_path = os.path.join(model_path, f'visualization_iter_{iteration}.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"✅ Visualization saved to {output_path}")


def check_gaussian_attributes(gaussian):
    """
    Check the boundary box range of the input Gaussian.

    Args:
        gaussian: GaussianModel object containing the point cloud data.
    """
    # Extract Gaussian positions
    positions = gaussian.get_xyz.cpu().detach().numpy()

    # Calculate the boundary box
    min_bounds = positions.min(axis=0)
    max_bounds = positions.max(axis=0)

    # Print the boundary box range
    print(f"Gaussian Boundary Box:")
    print(f"Min Bounds: {min_bounds}")
    print(f"Max Bounds: {max_bounds}")

    pass


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

        # Visualization of Gaussian point cloud and camera frustums
        visualize_gaussians_and_cameras(gaussians, scene.getTrainCameras(), iteration, args.model_path)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visibility_filter = visibility_filter.to(gaussians.max_radii2D.device)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def train_from_trained_model(
    dataset, 
    opt, 
    pipe, 
    testing_iterations, 
    saving_iterations, 
    checkpoint_iterations, 
    debug_from,
    new_image_path=None,    # new supervising image folder
    load_iter=None # ori gaussian splatting model folder 
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    elif load_iter is None:
        raise ValueError("[ERROR] the ori gaussian path is None....")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    scene = Scene(
        dataset, 
        gaussians, 
        loaded_gaussian_path=load_iter,
        new_image_path=new_image_path     
    )
    # loaded_iter="/home/zhy01/gaussian-splatting/output/c77ba6be-e/"
    gaussians.training_setup(opt)
    

    # if not isinstance(checkpoint, str):
    #     raise ValueError("[ERROR] the checkpoint is not a path!")


    first_iter = 0  # Reset iteration count for further training

    # Update dataset with new COLMAP images folder
    # print(f"Updating dataset with new COLMAP images folder: {dataset.source_path}")
    # scene.update_dataset(dataset)
    # 更新数据集中的相机视图和位姿
    # scene.update_dataset_new(dataset)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

        # Visualization of Gaussian point cloud and camera frustums
        # visualize_gaussians_and_cameras(gaussians, viewpoint_cam, iteration)

        # visualization of Gaussian point cloud projected into camera plane
        # get_projection_image_check(gaussians, viewpoint_cam, iteration)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        wandb.log({"Ll1": Ll1.item()})
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        wandb.log({"ssim_value": ssim_value.item()})

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        wandb.log({"total_loss": loss.item()})

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth: int | float | bool | os.Any = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                visibility_filter = visibility_filter.to(gaussians.max_radii2D.device)
                # gaussians.max_radii2D = gaussians.max_radii2D.to("cuda")
                # Keep track of max radii in image-space for pruning, 是所有视角下，每个高斯点在图像空间内出现过的最大半径。它是在多视角训练中持续更新的
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    pass


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--ori_gaussian_path", type=str, default="/home/zhy01/gen_3d_recon/gaussian-splatting/output/c77ba6be-e/", help="The ori trained gaussian path folder, which trained from this suitations")
    parser.add_argument("--new_image_path", type=str, default=None, required=True)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
        # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    
    train_from_trained_model(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        # args.start_checkpoint, 
        args.debug_from,
        args.new_image_path,
        args.ori_gaussian_path
    )

    # All done
    print("\nTraining complete.")
