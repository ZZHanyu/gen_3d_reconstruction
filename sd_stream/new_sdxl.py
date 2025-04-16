import torch
from torch import nn
# from transformers import AutoModelForImageGeneration
from diffusers import StableDiffusionPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import os
from diffusers import AutoencoderKL
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import time
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn
from diffusers import StableDiffusionPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import os
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline


# 控制模型类
class ControlNetModelWrapper(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        self.model = ControlNetModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

    def forward(self, input_tensor):
        return self.model(input_tensor)

# VAE 解码器类
class VAE(nn.Module):
    def __init__(self, vae_checkpoint, device):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            vae_checkpoint,
            torch_dtype=torch.float16
        ).to(device)

    def forward(self, latent_input):
        return self.vae.decode(latent_input)

# 图片预处理类
class ImagePreProcessor:
    def __init__(self, model_name="Intel/dpt-hybrid-midas", device="cuda"):
        self.device = device
        self.feature_extractor = DPTImageProcessor.from_pretrained(model_name)
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(model_name).to(device)

    def get_depth_map(self, image):
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image_tensor).predicted_depth

        # 调整深度图大小
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = torch.cat([depth_map] * 3, dim=1)
        depth_map = depth_map.permute(0, 2, 3, 1).cpu().numpy()[0]
        return Image.fromarray((depth_map * 255.0).clip(0, 255).astype(np.uint8))

# Stable Diffusion 生成模型类
class StableDiffusionGenerator:
    def __init__(self, base_model_checkpoint, device, controlnet_model, vae_model, ip_adapter_model):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_checkpoint,
            controlnet=controlnet_model,
            vae=vae_model,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)

        # IP-Adapter模型加载
        self.ip_adapter_model = ip_adapter_model
        self.pipe.load_ip_adapter(pretrained_model_name_or_path_or_dict="h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", device=device)
        self.pipe.set_ip_adapter_scale(0.8)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, init_image, control_image, ip_adapter_image, strength=0.99):
        # 调用管道生成图像
        result = self.pipe(
            prompt,
            init_image=init_image,
            control_image=control_image,
            ip_adapter_image=ip_adapter_image,
            strength=strength,
            num_inference_steps=50,
            controlnet_conditioning_scale=0.5
        ).images[0]
        return result

# 主程序类
class ControlNetSDXL:
    def __init__(self, device="cuda", task="canny", prompt="A post-apocalyptic cityscape, destroyed and collapsed buildings..."):
        self.device = device
        self.task = task
        self.prompt = prompt
        self.neg_prompt = "blurry, low-resolution, cartoon, unrealistic"
        
        # 模型加载路径
        self.config = {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "checkpoint": [
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusers/controlnet-zoe-depth-sdxl-1.0"
            ],
            "vae_checkpoint": "madebyollin/sdxl-vae-fp16-fix",
            "ip_adapter_model": "h94/IP-Adapter"
        }

        # 加载模型
        self.controlnet_model = ControlNetModelWrapper(self.config["checkpoint"][0], self.device)  # 根据任务选择模型
        self.vae_model = VAE(self.config["vae_checkpoint"], self.device)
        self.ip_adapter_model = self.load_ip_adapter(self.config["ip_adapter_model"])

        # 初始化生成器
        self.generator = StableDiffusionGenerator(
            self.config["base_model"], 
            self.device,
            self.controlnet_model,
            self.vae_model,
            self.ip_adapter_model
        )

        # 输出文件夹路径
        self.output_path = "/home/zhy01/gen_3d_recon/sd_stream/outputs"
        os.makedirs(self.output_path, exist_ok=True)

    # 修改加载 IP Adapter 模型的代码
    def load_ip_adapter(self, model_name_or_path):
        try:
            # 加载 IP Adapter 模型
            ip_adapter = self.pipe.load_ip_adapter(
                pretrained_model_name_or_path_or_dict=model_name_or_path,
                subfolder="sdxl_models",  # 如果模型有子文件夹，可以在这里指定
                weight_name="ip-adapter_sdxl.bin",  # 如果模型权重文件有具体名称，可以指定
                device=self.device
            )
            return ip_adapter
        except Exception as e:
            raise FileNotFoundError(f"[ERROR] The IP Adapter model file '{model_name_or_path}' could not be loaded. Error: {str(e)}")


    def forward(self, raw_image, refer_image, single_image_name):
        # 处理深度图
        image_processor = ImagePreProcessor(device=self.device)
        depth_image = image_processor.get_depth_map(raw_image)

        # 使用生成器进行图像生成
        result = self.generator.generate(
            prompt=self.prompt,
            init_image=raw_image,
            control_image=depth_image,
            ip_adapter_image=refer_image,
            strength=0.99
        )

        # 保存生成的图像
        result.save(os.path.join(self.output_path, single_image_name + ".png"))
        print(f"[INFO] Image saved to {self.output_path}/{single_image_name}.png")

if __name__ == "__main__":
    parser = ArgumentParser(description="SD plus ControlNet")
    parser.add_argument("--base_path", type=str, default="/home/zhy01/data/s1/", help="base path of image")
    parser.add_argument("--image_path", type=str, default="/home/zhy01/data/s1/train", help="base path of image")
    parser.add_argument("--image_path_test", type=str, default="/home/zhy01/data/s1/test", help="base path of image")
    parser.add_argument("--prompt", type=str, default="convert into destroyed and collapse buildings", help="prompt for image generation")
    parser.add_argument("--out_dir", type=str, default="/home/zhy01/Distill-Any-Depth", help="output directory for generated images")
    parser.add_argument("--SD_ver", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="version of stable diffusion")
    parser.add_argument("--task", type=str, default="lineart", help="task name for controlnet")

    args = parser.parse_args()


    base_path = args.base_path
    image_path = args.image_path
    image_path_test = args.image_path_test

    image_queue_train = sorted(
        [f for f in os.listdir(image_path) if f.endswith('.png') and len(f.split('_')) > 1],
        key=lambda x: int(x.split('_')[-2])
    )
    len_train = len(image_queue_train)

    image_queue_test = sorted(
        [f for f in os.listdir(image_path_test) if f.endswith('.png') and len(f.split('_')) > 1],
        key=lambda x: int(x.split('_')[-2])
    )
    len_test = len(image_queue_test)


    image_queue = image_queue_test + image_queue_train   
    image_queue = sorted(image_queue) 
    # print(image_queue)

    prompt = args.prompt
    outdir = args.out_dir

    past_image=None


    pipeline = ControlNetSDXL(
        task="canny",
        prompt=prompt,
        device="cuda",
    )

    # Method1: 统一风格（一个完备prompt）
    for single_image in tqdm(image_queue, leave=True, desc="M1: all transfer using same prompt..."):
        if single_image in image_queue_test:
            mid_str = "test"
        elif single_image in image_queue_train:
            mid_str = "train"
        else:
            raise ValueError(f"[ERROR] image {single_image} is not in either train or test!")
        
        init_image_path = os.path.join(base_path, mid_str, single_image)
        
        pli_image = Image.open(init_image_path)

        # # Upscale image
        # print(f"[INFO] Doing upscaling...")
        # upscaled_image = pipe(prompt="A high-resolution detailed image", image=pli_image).images[0]   
        # upscaled_image.save(os.path.join(outdir, "superResoulution.png"))
        # pli_image = upscaled_image
        # print(f"[INFO] Upscaling is done successfully!")

        start_time = time.time()

        # for sdxl
        pipeline.forward(
            raw_image=pli_image,
            refer_image=Image.open("/home/zhy01/gen_3d_recon/sd_stream/refer_image.jpg").convert("RGB"),
            single_image_name=single_image.split('_')[1],
        )
        
        end_time = time.time()
        print(f"[INFO] Time cost: {end_time - start_time}s!")