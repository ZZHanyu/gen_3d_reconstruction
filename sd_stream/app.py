import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

try:
    checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")
except Exception as e:
    raise ValueError

# Image processing function
def process_image(image, model, device):
    if model is None:
        # return None
        print("[ERROR]: the model is None...")
        raise FileExistsError
    
    # Preprocess the image
    image_np = np.array(image)[..., ::-1] / 255
    
    transform = Compose([
        Resize(512, 512, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    image_tensor = transform({'image': image_np})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
        pred_disp, _ = model(image_tensor)
    torch.cuda.empty_cache()

    # Ensure the depth map is in the correct shape before colorization
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions
    
    h, w = image_np.shape[:2]
    raw_depth = cv2.resize(pred_disp_np,  (w, h), cv2.INTER_LINEAR)
    
    # Save the raw depth map as a PNG file
    raw_depth_image = Image.fromarray((raw_depth * 255).astype(np.uint8))
    raw_depth_image.save('/home/zhy01/Distill-Any-Depth/depth_map.png')
    
    # Normalize depth map
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

    # Colorize depth map
    cmap = "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()  # Ensure correct dimension
    
    # Convert to uint8 for image display
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert to HWC format (height, width, channels)
    depth_colored_hwc = chw2hwc(depth_colored)
    
    # Resize to match the original image dimensions (height, width)
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    
    # Convert to a PIL image
    depth_image = Image.fromarray(depth_colored_hwc)
    raw_depth = raw_depth.astype(np.uint16)
    
    # Save the depth image as a PNG file
    depth_image.save('/home/zhy01/Distill-Any-Depth/depth_colored.png')
    return depth_image, raw_depth

# Gradio interface function
def gradio_interface(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Load model
    model = DepthAnything(**model_kwargs['vitl']).to(device)
    # if use hf_hub_download, you can use the following code
    checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

    # if use local path, you can use the following code
    # checkpoint_path = "path/to/your/model.safetensors"

    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model = model.to(device)
    
    if model is None:
        return None
    
    # Process image and return output
    depth_image, raw_depth = process_image(image, model, device)
    return depth_image

# # Create Gradio interface
# iface = gr.Interface(
#     fn=gradio_interface,
#     inputs=gr.Image(type="pil"),  # Only image input, no mode selection
#     outputs=gr.Image(type="pil"),  # Only depth image output, no debug info
#     title="Depth Estimation Demo",
#     description="Upload an image to see the depth estimation results. Our model is running on GPU for faster processing."
# )

# # Launch the Gradio interface
# iface.launch(share=True)