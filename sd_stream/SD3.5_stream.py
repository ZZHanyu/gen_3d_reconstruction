from ast import List
from genericpath import isdir
import os
from threading import local
from turtle import st

from kiwisolver import strength
from sympy import lowergamma
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from diffusers import AutoencoderKL
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from flask import config
import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from torchvision import transforms
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionUpscalePipeline
from transformers import pipeline
# from ip_adapter import IPAdapter  
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.loading_utils import load_image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, BertTokenizer, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import huggingface_hub
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import time
from PIL import Image
import cv2
import numpy as np
from controlnet_aux import LineartDetector
from argparse import ArgumentParser, Namespace
from ip_adapter import IPAdapterXL
import PIL
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLControlNetImg2ImgPipeline

import sys
sys.path.append("/home/zhy01") 

# import customized packages
sys.path.append("/home/zhy01/ControlNet_Plus_Plus")
from ControlNet_Plus_Plus.eval.utils import get_reward_model

sys.path.append("/home/zhy01/stablediffusion")
from stablediffusion.scripts.img2img import SD2_outer_interface

sys.path.append("/home/zhy01/sd3_5")
from sd3_5.sd3_infer import SD3_large_controlnet

sys.path.append("/home/zhy01/ControlNet_v1_1_nightly")
from ControlNet_v1_1_nightly.cldm.model import create_model, load_state_dict
from ControlNet_v1_1_nightly.cldm.ddim_hacked import DDIMSampler
# from ControlNet_v1_1_nightly.annotator.lineart import LineartDetector
from ControlNet_v1_1_nightly.annotator.util import resize_image, HWC3
from diffusers import EulerAncestralDiscreteScheduler

from ControlNetPlus.models.controlnet_union import ControlNetModel_Union
from ControlNetPlus.pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline


if torch.cuda.is_available():
    print(f"[INFO] 当前使用的GPU设备 {torch.cuda.current_device()},")
    print(f"GPU设备名称 {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] 没有可用的cuda设备!")

# huggingface_hub.login(
#     token="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"
# )

'''
Startup:
    python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_large.safetensors

    python sd3_infer.py --model models/sd3.5_large.safetensors --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors --controlnet_cond_image inputs/canny.png --prompt "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms"
'''

def SD_controlnet():
    hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors", local_dir="models")

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
    pipe = torch.nn.DataParallel(pipe, device_ids=[0, 1])
    pipe = pipe.to("cuda")

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]
    image.save("/home/zhy01/Distill-Any-Depth/SD_test.png")


def SD3_5_controlnet(
    out_dir="/home/zhy01/Distill-Any-Depth",
    image_path=None,
    prompt=None,
    model="models/sd3.5_large.safetensors",
):
    hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors", local_dir="models")

    if image_path is None or prompt is None or model is None:
        raise FileNotFoundError
    init_image = load_image(image_path).convert("RGB")
    width, height = init_image.size
    SD3_large_controlnet(
        out_dir=out_dir,
        controlnet_cond_image=image_path,
        controlnet_ckpt="models/sd3.5_large_controlnet_canny.safetensors",
        model=model,
        prompt=prompt,
        model_folder="/home/zhy01/sd3_5/models",
        # width=width,
        # height=height
    )


def SDXL_refiner(prompt, path):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    # pipe = torch.nn.DataParallel(pipe, device_ids=[0, 1])
    pipe = pipe.to("cuda")
    # url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

    init_image = load_image(path).convert("RGB")
    
    image = pipe(prompt, image=init_image).images
    image[0].save("/home/zhy01/Distill-Any-Depth/SDXL_refiner1.0.png")


{# TODO: transfer every module into class
# class controlnet_plus_plus_main:
#     def __init__(self):
#         from diffusers import (
#             T2IAdapter, StableDiffusionAdapterPipeline,
#             StableDiffusionControlNetPipeline, ControlNetModel,
#             UniPCMultistepScheduler, DDIMScheduler,
#             StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
#             StableDiffusionXLControlNetPipeline, AutoencoderKL
#         )

#         self.config = {
#             "model": "controlnet",
#             "model_path": 'lllyasviel/control_v11p_sd15_seg',
#             "task_name": "lineart",
#             "cache_dir": None,
#             "sd_path": 'runwayml/stable-diffusion-v1-5',
#             "outdir": "/home/zhy01/Distill-Any-Depth",
#             "image_path": "/home/zhy01/data/s1/train",
#             "prompt": "convert into destroyed and collapse buildings",
#             "num_inference_steps": 20,
#             "guidance_scale": 7.5,
#             "batch_size": 4,
#             "sd_path": "runwayml/stable-diffusion-v1-5",
#         }

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# class controlnet_sdxl:
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
    
    
#     def __init__(
#         self,
#         task="canny",
#         prompt="A post-apocalyptic cityscape, destroyed and collapsed buildings, scattered rubble and debris, broken structures, dark atmosphere, realistic style with Ultra high definition resolution, best quality",
#         neg_prompt="blurry, low-resolution, cartoon, unrealistic, distorted, poorly drawn, low quality",
#     ):
        
#         self.config= {
#             "task": task,
#             "image_path": "/home/zhy01/data/s1/train",
#             "outdir": "/home/zhy01/gen_3d_recon/sd_stream",
#             "checkpoint": [
#                 "diffusers/controlnet-canny-sdxl-1.0", # 0
#                 "diffusers/controlnet-depth-sdxl-1.0",  # 1
#                 "diffusers/controlnet-zoe-depth-sdxl-1.0", # 2
#             ],
#             "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
#             "out_path": "/home/zhy01/gen_3d_recon/sd_stream/outputs",
#             "combine_list":["canny", "depth", "zoe_depth"],
#             "prompt":prompt,
#             "neg_prompt": neg_prompt
#         }
#         # print the config
#         for key, value in self.config.items():
#             print("[INFO] {}: {}".format(key, value))

#         # task init
#         self.task = self.config["task"]

#         # init_all controlnet compoments
#         self.control_canny = self.instantiation_controlnet(0)
#         self.control_depth = self.instantiation_controlnet(1)
#         self.control_zoe_depth = self.instantiation_controlnet(2)
        
#         if self.config["task"] == "combine":
#             # combine two of them(depth and canny)
#             controlnet = [
#                 self.control_canny,
#                 self.control_depth
#             ]                
            
#         elif self.config['task'] == "canny":
#             controlnet = self.control_canny
#         elif self.config['task'] == "depth":
#             controlnet = self.control_depth
#         elif self.config['task'] == "zoe_depth":
#             controlnet = self.control_zoe_depth
#         else:
#             raise ValueError(f"[ERROR] the task name is wrong! now is {self.config['task']}!")

        
#         self.pipe = self.init_pipeline(controlnet)

#         self.judge_save_folder(task)

#         print(f"[INFO] controlnet sdxl pipeline init done!")


#     def instantiation_controlnet(self, idx=None):
#         if idx is None:
#             print(f"[ERROR] init controlnet module error...")
#             raise NotImplementedError
#         try:
#             checkpoint = self.config["checkpoint"][idx]
#             controlnet = ControlNetModel.from_pretrained(
#                 checkpoint, 
#                 torch_dtype=torch.float16,
#                 # local_files_only=True,
#             )
#         except Exception as e:
#             print(f"[ERROR] {e} ")
#             raise ValueError
#         return controlnet
        

#     def init_pipeline(self, controlnet):
#         if isinstance(controlnet, list):
#             print("[INFO] combine module has:")
#             for idx, single_controlnet in enumerate(controlnet):
#                 print(f"\b layer idx {idx} - {single_controlnet}\n")

#             pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#                 pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0",
#                 controlnet=[x for x in controlnet],
#                 torch_dtype=torch.float16
#             ).to(controlnet_sdxl.device)

#             pipe.load_ip_adapter(
#                 pretrained_model_name_or_path_or_dict="h94/IP-Adapter", 
#                 subfolder="sdxl_models",
#                 weight_name="ip-adapter_sdxl.bin",    
#                 device=controlnet_sdxl.device,
#             )
#             pipe.set_ip_adapter_scale(0.8)

#             # # choice 2 for loading ip adapter...
#             # ip_adapter = IPAdapterXL(
#             #     pipe, 
#             #     "h94/IP-Adapter", 
#             #     "h94/IP-Adapter/ip-adapter_sdxl.bin",
#             #     controlnet_sdxl.device
#             # )

#         elif isinstance(controlnet, ControlNetModel):
#             print(f"[INFO] single control net module...")
#             pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#                 pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", 
#                 controlnet=controlnet, 
#                 torch_dtype=torch.float16,
#                 add_watermarker=False,
#             ).to(controlnet_sdxl.device)

#             pipe.load_ip_adapter(
#                 pretrained_model_name_or_path_or_dict="h94/IP-Adapter", 
#                 subfolder="sdxl_models",
#                 weight_name="ip-adapter_sdxl.bin",  
#                 # image_encoder=image_encoder,  # Explicitly pass the image encoder
#                 device=controlnet_sdxl.device
#             )
#             pipe.set_ip_adapter_scale(0.8)

#             # # choice 2 for loading ip adapter...
#             # # init ip_model
#             # base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
#             # image_encoder_path = "sdxl_models/image_encoder"
#             # ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
#             # device = "cuda"

#             # ip_model = IPAdapterXL(
#             #     pipe, 
#             #     image_encoder_path, 
#             #     ip_ckpt, 
#             #     device
#             # )
#         else:
#             print(f"[ERROR] CanNot reginzed this controlnet type...")
#             raise NotImplementedError

#         # select a sampler/scheduler
#         pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#         # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#         pipe.enable_model_cpu_offload()

#         return pipe


#     def canny_forward(self, raw_image, refer_image):
#         pipe = self.pipe

#         prompt= self.config["prompt"]
#         neg_prompt= self.config["neg_prompt"]

#         generator = torch.manual_seed(0)
#         image = pipe(
#             prompt=prompt, 
#             negative_prompt=neg_prompt,
#             num_inference_steps=30, 
#             generator=generator, 
#             image=raw_image,    # extract control line
#             control_image=raw_image,
#             ip_adapter_image=refer_image   # referred style
#         ).images[0]

#         if self.config["out_path"] is None:
#             self.judge_save_folder(task_name=self.config["task"])
#         image_name = os.path.basename(single_image).split('_')[1] + ".png"
#         curr_save_path = os.path.join(self.config["outdir"], image_name)
#         if not os.path.exists(curr_save_path):
#             os.mkdir(curr_save_path)
#         image.save(os.path.join(curr_save_path, image_name))
#         # image.save(self.config["out_path"])


#     def depth_forward(self, raw_image, refer_image):
#         pipe = self.pipe

#         prompt=self.config["prompt"]
#         neg_prompt=self.config["neg_prompt"]

#         generator = torch.manual_seed(0)
#         # result = pipe(
#         #     prompt=prompt, 
#         #     negative_prompt=neg_prompt,
#         #     num_inference_steps=30, 
#         #     generator=generator, 
#         #     control_image=raw_image,
#         #     image=raw_image,    # extract control line
#         #     ip_adapter_image=refer_image   # referred style
#         # )
#         # 1. 先编码文本 prompt 成 embeddings（必须做）
#         prompt_embeds, negative_prompt_embeds  = pipe.encode_prompt(
#             prompt=prompt,
#             negative_prompt=neg_prompt,
#             device=pipe.device,
#             do_classifier_free_guidance=True,
#             num_images_per_prompt=1,
#         )

#         # 2. IP-Adapter 编码风格图为 image_embeds
#         ip_adapter_embeds = pipe.prepare_ip_adapter_image_embeds(
#             refer_image,
#             ip_adapter_image_embeds=None,
#             device=pipe.device,
#             num_images_per_prompt=1,
#             do_classifier_free_guidance=True,
#         )

#         # 3. 执行推理（关键参数都传了）
#         result = pipe(
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             image=raw_image,                   # img2img 的基础图
#             control_image=raw_image,          # ControlNet 输入
#             ip_adapter_image_embeds=ip_adapter_embeds,  # IP-Adapter style embedding
#             num_inference_steps=30,
#             guidance_scale=7.5,
#             generator=torch.manual_seed(0),
#         )
#         image = result.images[0]  # Access the first image from the result

#         image_name = os.path.basename(single_image).split('_')[1] + ".png"
#         if self.config["out_path"] is None:
#             raise FileNotFoundError("[ERROR] the out path is not defined!")
#         image.save(os.path.join(self.config["out_path"], image_name))


#     def combine_forward_Parallel(self, raw_image, refer_image):
#         # Initialize the pipeline with the combined controlnets
#         pipe = self.pipe

#         # Define prompts and negative prompts
#         prompt = self.config["prompt"]
#         neg_prompt = self.config["neg_prompt"]

#         # Generate the image using the pipeline
#         generator = torch.manual_seed(0)
#         result = pipe(
#             prompt=[prompt] * 2,  # Two prompts for the two controlnets
#             negative_prompt=[neg_prompt] * 2,
#             num_inference_steps=30,
#             generator=generator,
#             image=[raw_image, raw_image],  # Input image for both controlnets
#             control_image=[raw_image, raw_image],
#             ip_adapter_image=refer_image,
#             control_guidance_start=[0.0, 0.5],  # 第一阶段只用 ControlNet1
#             control_guidance_end=[0.5, 1.0]     # 第二阶段只用 ControlNet2
#         )

#         # Save the generated image
#         output_image = result.images[0]
#         image_name = os.path.basename(single_image).split('_')[1] + ".png"
#         curr_save_path = os.path.join(self.config["out_path"], "combine_parallel")
#         if not os.path.exists(curr_save_path):
#             os.mkdir(curr_save_path)
#         output_image.save(os.path.join(curr_save_path, image_name))
    

    
#     def output_save(self, task_name: str, image, image_name, outdir):
#         if task_name not in ["canny", "depth", "combine"]:
#             print(f"[ERROR] task name {task_name} is not supported!")
#             raise NotImplementedError
        
#         folder_suffix = 0
#         while True:
#             # curr_time = time.strftime("%m-%d %H:%M", time.localtime())
#             # save_name = task_name + curr_time
#             # save_path = os.path.join(outdir, save_name)
#             save_path = outdir
#             save_path = os.path.join(save_path, task_name + str(folder_suffix))
           
#             if os.path.exists(save_path):
#                 folder_suffix += 1
#                 continue
#             else:
#                 os.mkdir(save_path)
#                 break
#         image.save(os.path.join(save_path, image_name))

    
#     def judge_save_folder(self, task_name: str,):
#         if task_name not in ["canny", "depth", "combine"]:
#             print(f"[ERROR] task name {task_name} is not supported!")
#             raise NotImplementedError
        
#         folder_suffix = 0
#         while True:
#             save_folder_name = task_name + str(folder_suffix)
#             save_path = os.path.join(self.config["outdir"], save_folder_name)
#             if os.path.exists(save_path):
#                 folder_suffix += 1
#                 continue
#             else:
#                 os.mkdir(save_path)
#                 break
        
#         self.config["out_path"] = save_path
        
#         print(f"[INFO] the save path is {save_path}")
}

        




class controlnet_sdxl:
    def __init__(
        self,
        task="canny",
        prompt="A post-apocalyptic cityscape, destroyed and collapsed buildings, scattered rubble and debris, broken structures, dark atmosphere, realistic style with Ultra high definition resolution, best quality",
        neg_prompt="blurry, low-resolution, cartoon, unrealistic, distorted, poorly drawn, low quality",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = {
            "task": task,
            "image_path": "/home/zhy01/data/s1/train",
            "outdir": "/home/zhy01/gen_3d_recon/sd_stream/outputs",
            "checkpoint": [
                "diffusers/controlnet-canny-sdxl-1.0",
                "diffusers/controlnet-depth-sdxl-1.0",
                "diffusers/controlnet-zoe-depth-sdxl-1.0",
            ],
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "out_path": "/home/zhy01/gen_3d_recon/sd_stream/outputs",
            "combine_list": ["canny", "depth", "zoe_depth"],
            "prompt": prompt,
            "neg_prompt": neg_prompt
        }

        for key, value in self.config.items():
            print(f"[INFO] {key}: {value}")

        self.task = self.config["task"]


        if task == "combine":
            controlnet = [
                self.instantiation_controlnet(0),
                self.instantiation_controlnet(3)
            ]
        elif task == "canny":
            self.control_canny = self.instantiation_controlnet(0)
            controlnet = self.control_canny
        elif task == "depth":
            self.control_depth = self.instantiation_controlnet(1)
            controlnet = self.control_depth
        elif task == "zoe_depth":
            self.control_zoe_depth = self.instantiation_controlnet(2)
            controlnet = self.control_zoe_depth
        else:
            raise ValueError(f"[ERROR] Invalid task name: {task}")

        
        self.pipe = self.init_pipeline(controlnet)
        os.makedirs(self.config["out_path"], exist_ok=True)
        print("[INFO] ControlNet SDXL pipeline initialized successfully!")

        self.judge_save_folder()
        print(f"[INFO] saving folder is {self.config['out_path']}")


    def judge_save_folder(self):
        task_name = self.config['task']
        if task_name not in ["canny", "depth", "combine", "lineart"]:
            print(f"[ERROR] task name {task_name} is not supported!")
            raise NotImplementedError
        
        folder_suffix = 0
        while True:
            save_folder_name = task_name + str(folder_suffix)
            save_path = os.path.join(self.config["outdir"], save_folder_name)
            if os.path.exists(save_path):
                folder_suffix += 1
                continue
            else:
                os.mkdir(save_path)
                break
        
        self.config["out_path"] = save_path
        
        print(f"[INFO] the save path is {save_path}")


    def instantiation_controlnet(self, idx):
        checkpoint = self.config["checkpoint"][idx]
        controlnet = ControlNetModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        return controlnet


    def init_pipeline(self, controlnet):
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=self.config["base_model"],
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            torch_dtype=torch.float16,
            # use_safetensors=True,
        ).to(self.device)
        if self.config["task"] == "lineart":
            pipe.controlnet.set_control_mode("lineart", mode="standard")

        pipe.load_ip_adapter(
            pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
            device=self.device
        )
        pipe.set_ip_adapter_scale(0.8)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        return pipe


    def get_depth_map(self, image):
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image
        

    def forward(self, raw_image, refer_image, single_image_name):
        pipe = self.pipe
        prompt = self.config["prompt"]
        neg_prompt = self.config["neg_prompt"]
        generator = torch.manual_seed(420)

        # depth_image = self.get_depth_map(raw_image)

        # Ensure all images have the same dimensions
        raw_image = raw_image.resize((512, 512))
        # depth_image = depth_image.resize((512, 512))
        refer_image = refer_image.resize((512, 512))

        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=raw_image,
            control_image=raw_image,
            ip_adapter_image=refer_image,
            strength=0.9, # 控制原图保留多少，默认0.8，越小越接近原图
            height=1024,
            width=1024,
            num_inference_steps=50, # sampler推理次数，越高生成越好
            generator=generator,
            guidance_scale=10.0, # CFG指引强度，越大越贴合prompt，越小越自由生成，范围5-12
            controlnet_conditioning_scale=1.4, # 控制controlnet强度，默认1.0
        ).images[0]
        

        image = result
        save_path = os.path.join(self.config["out_path"], single_image_name + ".png")
        image.save(save_path)
        print(f"[INFO] Image saved to {save_path}")




class sdxl_controlnet_plus(controlnet_sdxl):
    def __init__(self, task, prompt, neg_prompt):
        super().__init__()
        self.congfig["checkpoint"] = "xinsir/controlnet-union-sdxl-1.0"
        self.config["task"] = task
        self.config["prompt"] = prompt
        self.config["neg_prompt"] = neg_prompt

        for key, value in self.config.items():
            print(f"[INFO] {key}: {value}")
        self.task = self.config["task"]

        self.vae = self.custmize_vae()
        self.controlnet_model = self.instantiation_controlnet()
        self.pipe = self.init_pipeline(self.controlnet_model)
        os.makedirs(self.config["out_path"], exist_ok=True)
        print("[INFO] ControlNet SDXL pipeline initialized successfully!")

        self.judge_save_folder()
        print(f"[INFO] saving folder is {self.config['out_path']}")


    def custmize_vae(self):
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


    def instantiation_controlnet(self):
        return ControlNetModel_Union.from_pretrained(self.config["checkpoint"] , torch_dtype=torch.float16, use_safetensors=True)
    

    def init_pipeline(self, controlnet):
        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

        pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.config["base_model"], controlnet=self.controlnet_model, 
            vae=self.vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        return pipe

    @torch.no_grad()
    def forward(self, raw_image, refer_image, single_image_name):
        generator = torch.manual_seed(423)
        result = self.pipe(
            prompt=self.config["prompt"],
            negative_prompt=self.config["neg_prompt"],
            image=raw_image,
            control_image=raw_image,
            ip_adapter_image=refer_image,
            strength=0.7, # 控制原图保留多少，默认0.8，越小越接近原图
            height=1024,
            width=1024,
            num_inference_steps=30, # sampler推理次数，越高生成越好
            generator=generator,
            guidance_scale=10.0, # CFG指引强度，越大越贴合prompt，越小越自由生成，范围5-12
            controlnet_conditioning_scale=1.4, # 控制controlnet强度，默认1.0
        ).images[0]

        image = result
        save_path = os.path.join(self.config["out_path"], single_image_name + ".png")
        image.save(save_path)
        print(f"[INFO] Image saved to {save_path}")




class controlnet_v1_1:
    def __init__(
        self,
        task="combine",
        prompt="a city with destroyed and collapse buildings, accompanied by the smoke and fires of war",
        neg_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = {
            "task": task,
            "image_path": "/home/zhy01/data/s1/train",
            "outdir": "/home/zhy01/gen_3d_recon/sd_stream",
            "checkpoint": [
                "lllyasviel/control_v11e_sd15_ip2p",
                "lllyasviel/control_v11p_sd15_canny",
                "ControlNet-1-1-preview/control_v11p_sd15_lineart",
                "lllyasviel/control_v11f1p_sd15_depth",
            ],
            "prompt": prompt,
            "neg_prompt": neg_prompt,
        }

        for key, value in self.config.items():
            print(f"[INFO] {key}: {value}")

        self.task = self.config["task"]

        if task == "combine":
            self.control_ip2p = self.instantiation_controlnet(0)
            self.control_lineart = self.instantiation_controlnet(2)
            controlnet = [self.control_ip2p, self.control_lineart]
        elif task == "ip2p":
            controlnet = self.control_ip2p
        elif task == "canny":
            self.control_canny = self.instantiation_controlnet(1)
            controlnet = self.control_canny
        elif task == "lineart":
            self.control_lineart = self.instantiation_controlnet(2)
            controlnet = self.control_lineart
        elif task == "depth":
            self.control_depth = self.instantiation_controlnet(3)
            controlnet = self.control_depth
        else:
            raise ValueError(f"[ERROR] Invalid task name: {task}")

        self.pipe = self.init_pipeline(controlnet)
        os.makedirs(self.config["outdir"], exist_ok=True)
        print("[INFO] ControlNet v1.1 pipeline initialized successfully!")

        self.judge_save_folder()
        print(f"[INFO] saving folder is {self.config['out_path']}")


    def judge_save_folder(self):
        task_name = self.config['task']
        if task_name not in ["canny", "depth", "combine", "lineart"]:
            print(f"[ERROR] task name {task_name} is not supported!")
            raise NotImplementedError
        
        folder_suffix = 0
        while True:
            save_folder_name = task_name + str(folder_suffix)
            save_path = os.path.join(self.config["outdir"], save_folder_name)
            if os.path.exists(save_path):
                folder_suffix += 1
                continue
            else:
                os.mkdir(save_path)
                break
        
        self.config["out_path"] = save_path
        
        print(f"[INFO] the save path is {save_path}")


    def instantiation_controlnet(self, idx):
        try:
            checkpoint = self.config["checkpoint"][idx]
            controlnet = ControlNetModel.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16
            )
            return controlnet
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load ControlNet checkpoint: {e}")


    def init_pipeline(self, controlnet):
        if isinstance(controlnet, list):
            print("[INFO] Combining multiple ControlNet modules...")
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            pipe.load_ip_adapter(
                pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                device=self.device
            )
        elif isinstance(controlnet, ControlNetModel):
            print("[INFO] Initializing single ControlNet module...")
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to(self.device)
            # if self.config["task"] == "lineart":
                # pipe.controlnet.set_control_mode("lineart", mode="standard")
            pipe.load_ip_adapter(
                pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                device=self.device
            )
        else:
            raise ValueError("[ERROR] Invalid ControlNet type!")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        return pipe

    def forward(self, raw_image, refer_image, single_image_name):
        pipe = self.pipe
        prompt = self.config["prompt"]
        neg_prompt = self.config["neg_prompt"]
        generator = torch.manual_seed(0)

        # depth_image = self.get_depth_map(raw_image)


        # Ensure all images have the same dimensions
        raw_image = raw_image.resize((512, 512))
        # depth_image = depth_image.resize((512, 512))
        refer_image = refer_image.resize((512, 512))

        if self.config['task'] != 'combine':
            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=raw_image,
                control_image=raw_image,
                ip_adapter_image=refer_image,
                # strength=0.6, # 控制原图保留多少，默认0.8，越小越接近原图
                height=1024,
                width=1024,
                num_inference_steps=50, # sampler推理次数，越高生成越好
                generator=generator,
                guidance_scale=10.0, # CFG指引强度，越大越贴合prompt，越小越自由生成，范围5-12
                controlnet_conditioning_scale=1.2, # 控制controlnet强度，默认1.0
            ).images[0]
        else:
            result = pipe(
                prompt=[prompt] * 2,  # Two prompts for the two controlnets
                negative_prompt=[neg_prompt] * 2,
                image=raw_image,
                control_image=[raw_image] * 2,
                ip_adapter_image=refer_image,
                # strength=0.6, # 控制原图保留多少，默认0.8，越小越接近原图
                height=1024,
                width=1024,
                num_inference_steps=50, # sampler推理次数，越高生成越好
                generator=generator,
                guidance_scale=10.0, # CFG指引强度，越大越贴合prompt，越小越自由生成，范围5-12
                controlnet_conditioning_scale=1.2, # 控制controlnet强度，默认1.0
            ).images[0]

        image = result
        save_path = os.path.join(self.config["out_path"], single_image_name + ".png")
        image.save(save_path)
        print(f"[INFO] Image saved to {save_path}")


def controlnet_v11():
    config = {
        "task": "combine",
        "image_path":"/home/zhy01/data/s1/train",
        "outdir":"/home/zhy01/Distill-Any-Depth",
        "checkpoint_1" : "lllyasviel/control_v11e_sd15_ip2p",
        "checkpoint_2" : "lllyasviel/control_v11p_sd15_canny",
        "checkpoint_3" : "ControlNet-1-1-preview/control_v11p_sd15_lineart",
        "checkpoint_4" : "lllyasviel/control_v11p_sd15_depth"
    }


    if config["task"] == "canny":
        checkpoint = "lllyasviel/control_v11p_sd15_canny"

        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        
    elif config["task"] == "ip2p":
        checkpoint = "lllyasviel/control_v11e_sd15_ip2p"
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
    elif config["task"] == "lineart":
        checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    elif config["task"] == "combine":
        checkpoint_1 = "lllyasviel/control_v11e_sd15_ip2p"
        checkpoint_2 = "lllyasviel/control_v11p_sd15_canny"
        # TODO：待完成
        checkpoint_3 = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
        checkpoint_4 = "lllyasviel/control_v11p_sd15_depth"

        control_ip2p = ControlNetModel.from_pretrained(checkpoint_1, torch_dtype=torch.float16)
        control_canny = ControlNetModel.from_pretrained(checkpoint_2, torch_dtype=torch.float16)
        control_lineart = ControlNetModel.from_pretrained(checkpoint_3, torch_dtype=torch.float16)
        control_depth = ControlNetModel.from_pretrained(checkpoint_4, torch_dtype=torch.float16)
        

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", 
            controlnet=[control_ip2p, control_canny, control_lineart, control_depth], # TIPs: set multiple ControlNets as a list, the outputs from each ControlNet are added together to create one combined additional conditioning.
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

    image_queue = os.listdir(config["image_path"])

    for single_image in tqdm(image_queue, leave=True, desc=f"Controlnet v1_1 task-{config['task']}-!"):
        image = Image.open(os.path.join(config["image_path"], single_image))
        image = image.resize((512, 512))
        prompt = "convert into destroyed and collapse buildings"

        if config["task"] == "lineart":
            processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
            control_image = processor(image)
            control_image.save(os.path.join(config["outdir"], "lineart_control.png"))
            generator = torch.manual_seed(0)
            image = pipe(
                prompt, 
                num_inference_steps=30, 
                generator=generator, 
                image=control_image
            ).images[0]
            image_name = os.path.basename(single_image).split('_')[1] + ".png"
            curr_save_path = os.path.join(config["outdir"], "lineart")
            if not os.path.exists(curr_save_path):
                os.mkdir(curr_save_path)
            image.save(os.path.join(curr_save_path, image_name))
        elif config['task'] == 'ip2p':
            generator = torch.manual_seed(0)
            image = pipe(
                prompt, 
                num_inference_steps=30, 
                generator=generator, 
                image=single_image
            ).images[0]
            image.save(os.path.join(config["outdir"], 'ip2p_image_out.png'))
        elif config['task'] == 'canny':
            image = np.array(single_image)
            low_threshold = 100
            high_threshold = 200
            image = image.astype(np.uint8)
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)

            control_image.save(config["outdir"], "canny_control.png")

            generator = torch.manual_seed(33)
            image = pipe(
                prompt=prompt, 
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=20, 
                generator=generator, 
                image=control_image
            ).images[0]

            image.save(config["outdir"], 'cannny_image_out.png')
        
        elif config["task"] == 'combine':
            generator = torch.manual_seed(int(time.time()))
            pass
            


        else:
            print("[ERROR] CanNOT Found matched task name...")
            raise NotImplementedError


def controlnet_plus_plus():
    from diffusers import (
        T2IAdapter, StableDiffusionAdapterPipeline,
        StableDiffusionControlNetPipeline, ControlNetModel,
        UniPCMultistepScheduler, DDIMScheduler,
        StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
        StableDiffusionXLControlNetPipeline, AutoencoderKL
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] controlnet++ device selected is {device}")

    config={
        "model":"T2IAdapter-sdxl",
        "model_path":'lllyasviel/control_v11p_sd15_seg',
        "task_name":"lineart",
        "cache_dir":None,
        "sd_path":'runwayml/stable-diffusion-v1-5',
        "outdir":"/home/zhy01/Distill-Any-Depth",
        "image_path":"/home/zhy01/data/s1/train",
        "prompt":"convert into destroyed and collapse buildings",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "batch_size": 4,
        "sd_path": "runwayml/stable-diffusion-v1-5",
    }

    # config["prompts"] = [config["prompt"]] * config["batch_size"]
    image_queue = os.listdir(config["image_path"])

    # base model load
    if config["model"] == "controlnet":
        controlnet = ControlNetModel.from_pretrained(config["model_path"], torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=config["sd_path"],
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )
    elif config["model"] == "T2IAdapter-sdxl":
        # T2IAdapter-sdxl
        adapter = T2IAdapter.from_pretrained(config["model_path"], torch_dtype=torch.float16, varient="fp16")
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(config["sd_path"], subfolder="scheduler")
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        text_encoder_2 = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16)

        tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

        image_encoder = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            pretrained_model_name_or_path=config["sd_path"],
            vae=vae, 
            # tokenizer=tokenizer, 
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            image_encoder=image_encoder,
            adapter=adapter, 
            scheduler=euler_a, 
            torch_dtype=torch.float16, 
            variant="fp16",
        )
    elif config["model"] == "t2i-adapter":
        # StableDiffusionAdapterPipeline
        adapter = T2IAdapter.from_pretrained(config["model_path"], torch_dtype=torch.float16)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            config["sd_path"], adapter=adapter, safety_checker=None, torch_dtype=torch.float16, variant="fp16"
        )
    else:
        assert 1==2, print(f"[ERROR] no base model specfic!!!")

    
    if config["task_name"] == "lineart":
        model = get_reward_model(task='lineart', model_path='https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth')
        model.eval()
        model.to(device)
    else:
        print("[ERROR]: NO speicfic task selected!")
        raise NotImplementedError
    

    start_time = time.time()
    count = 0

    for single_image in tqdm(image_queue, leave=True, desc="M1: all transfer using same prompt..."):
        if not single_image.endswith('.png'):
            continue
        init_image_path = os.path.join(image_path, single_image)
        
        cond_image = Image.open(init_image_path)
        raw_input_cond_image = cond_image.copy()
        width, height = cond_image.size
        cond_image = F.pil_to_tensor(cond_image.resize((width, height))).unsqueeze(0) / 255.0 # keep ori size of image
        cond_image = cond_image.to(device)
        with torch.no_grad():
            condition = model(cond_image, device=device) # get control net condition embbedings
        condition = 1 - condition
        condition = condition.reshape(width, height)
        condition = F.to_pil_image(condition, 'L').convert('RGB')

        prompts, conditions = [config["prompt"]] * config["batch_size"], [condition] * config["batch_size"]

        # print(pipe)  # 查看模型结构
        # print(pipe.unet)  # 检查 U-Net 是否正确加载
        # print(pipe.text_encoder)  # 检查文本编码器
        # print(pipe.vae)  # 检查 VAE
        print(pipe)


        # get condition image latent
        if config["model"] == "T2IAdapter-sdxl":
            images = pipe(
                prompt=prompts,
                image=conditions,
                num_inference_steps=config["num_inference_steps"],
                guidance_scale=config["guidance_scale"],
                adapter_conditioning_scale=0.5,
                negative_prompt=['worst quality, low quality'] *  config["batch_size"]
            ).images[0]
        elif config["model"] == "controlnet" or config["model"] == "t2i-adapter":
            images = pipe(
                prompt=prompts,
                image=conditions,
                num_inference_steps=config["num_inference_steps"],
                guidance_scale=config["guidance_scale"],
                negative_prompt=['worst quality, low quality'] *  config["batch_size"]
            ).images[0]
        else:
            # generate image
            generator = torch.manual_seed(0)
            image = pipe(
                prompt="convert into destroyed and collapse buildings", 
                num_inference_steps=20, 
                generator=generator, 
                image=raw_input_cond_image,
            ).images[0]



        if config["task_name"] == "lineart":
            lineart = [F.pil_to_tensor(img)/255.0 for img in images]

            for single_element in lineart:
                single_element = single_element.to(device)

            with torch.no_grad():
                lineart = model(torch.stack(lineart), device=device)
            lineart = torch.chunk(lineart, config["batch_size"], dim=0)
            lineart = [x.reshape(1, width, height) for x in lineart]
            lineart = [F.to_pil_image(x).convert('RGB') for x in lineart]
            for i, img in enumerate(lineart):
                if not os.path.exists(f"{config['outdir']}/group_{i}/"):
                    os.mkdir(f"{config['outdir']}/group_{i}/")
                img.save(f"{config['outdir']}/group_{i}/lineart_{count}.png")

        # using lineart condition to generate the final image
        condition = 255 - F.pil_to_tensor(condition)
        condition = F.to_pil_image(condition)

        cond_image = torch.squeeze(cond_image)
        cond_image_np = np.asarray(cond_image.cpu(), dtype=np.uint8)
        cond_image_np = np.transpose(cond_image_np, (1, 2 ,0))
        cond_image_pli = Image.fromarray(cond_image_np)

        # # todo find out the meaning of sum of these things
        # single_image = Image.open(init_image_path)
        # single_image = np.array(single_image)
        # single_image = np.transpose(single_image, (1, 2, 0))
        # single_image = Image.fromarray(single_image)
        

        images = [cond_image_pli] + images + [condition] + lineart
        images = [img.convert('RGB') for img in single_image] if config["model"] == 't2i-adapter' else images
        
        images = [F.pil_to_tensor(x) for x in images]
        # images = [F.pil_to_tensor(img) if isinstance(img, Image.Image) else img for img in images]
        if images[0].shape != images[1].shape:
            images[0] = np.transpose(images[0], (0, 2 ,1))
        images = make_grid(images, nrow=len(images)//2)
        print(f"[INFO] single image path is {init_image_path}!")
        try:
            save_path = "/home/zhy01/Distill-Any-Depth/cpp_result.png"
            F.to_pil_image(images).save(save_path)
        except Exception as e:
            print(f"[INFO] ERROR in saving image... {e}")
            pass

        count += 1

    end_time = time.time()
    print(f"[INFO] controlnet++ cost {end_time-start_time} s!")



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

    pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler").to("cuda")


    pipeline_config = {
        "SD_ver": args.SD_ver,
        "task": args.task,
    }
    for key, value in pipeline_config.items():
        print(f"[INFO] {key} is {value}")

    # pipeline = controlnet_v1_1(
    #     task="lineart",
    #     # prompt="A post-apocalyptic cityscape, destroyed and collapsed buildings, scattered rubble and debris, broken structures, dark atmosphere, realistic style with Ultra high definition resolution, best quality",
    #     prompt="a futuristic cyberpunk cityscape at night, neon lights, towering skyscrapers, flying cars, rainy streets, glowing signs, high-tech urban environment, vibrant colors, detailed architecture, foggy atmosphere",
    #     # neg_prompt="modern buildings, skyscrapers, neon lights, crowded street, traffic, industrial structures, blurry, low quality, distorted, text, logo, watermark, deformed, extra limbs, bad anatomy"
    #     neg_prompt="blurry, low-resolution, cartoon, unrealistic, distorted, poorly drawn, low quality, dark"
    # )
    
    # pipeline = controlnet_sdxl(
    #     task="canny",
    #     # prompt="a picturesque European town, cobblestone streets, colorful medieval houses, rustic architecture, charming small town, old brick buildings, flower-covered balconies, vintage street lamps, warm soft lighting, sunny afternoon, scenic view, highly detailed, cinematic atmosphere",
    #     prompt="a cyberpunk cityscape at night, neon lights, high-tech urban environment, vibrant colors, detailed architecture, foggy atmosphere, 4K, super-resoltion",
    #     # neg_prompt="modern buildings, skyscrapers, neon lights, crowded street, traffic, industrial structures, blurry, low quality, distorted, text, logo, watermark, deformed, extra limbs, bad anatomy"
    #     neg_prompt = "blurry, low-resolution, cartoon, unrealistic, distorted, poorly drawn, low quality",
    # )

    pipeline = sdxl_controlnet_plus(
        task="ctrl_plus_lineart",
        prompt="a cyberpunk cityscape at night, neon lights, high-tech urban environment, vibrant colors, detailed architecture, foggy atmosphere, 4K, super-resoltion",
        neg_prompt = "blurry, low-resolution, cartoon, unrealistic, distorted, poorly drawn, low quality",
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

        # SDXL_refiner(prompt, init_image_path)
        # SD2_outer_interface(prompt, init_image_path, outdir)
        # SD3_5_controlnet(image_path=init_image_path, prompt=prompt)
        # controlnet_plus_plus()
        # controlnet_v11()

        # pipeline.combine_forward()
        # pipeline.combine_Series(pli_image)
        # pipeline.ip2p_forward(image=pli_image)
        # pipeline.depth_forward(image=pli_image)
        # pipeline.canny_forward(image=pli_image)

        # pipeline.lineart_forward(pli_image)
        # past_image = pipeline.lineart_forward(image=pli_image, past_image=past_image)
        

        pipeline.forward(
            raw_image=pli_image,
            refer_image=Image.open("/home/zhy01/gen_3d_recon/sd_stream/refer_image.jpg").convert("RGB"),
            single_image_name=single_image.split('_')[1],
        )
        
        end_time = time.time()
        print(f"[INFO] Time cost: {end_time - start_time}s!")
        


    # # TODO: Method2: 若干组统一风格，组间不同风格，！！需要确定组内图像是同一区域的不同视角！！

