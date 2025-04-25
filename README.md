# 3D Gaussian surface reconstrction with diffusion methods
This is preview version of 3d surface reconstruction works
***
## Usage
- two environments required: environment.yaml is for stable diffusion, and environment2.yaml is for 3dgs
- after install all environments required, you run stable diffusion first, after generate style transfered pictures, you load a pretrained 3d gaussian model and combine with style transfered pictures to start surface reconstrcutions.
***
## TODO list
- [] add a 3d vae encoder decode with lora finetune modules
- [] add a pipelin-like bash executable which can let all 3 modules run with only 1 click
***
## Run
STEP1, you need doing the style transfer
```
python /home/zhy01/gen_3d_recon/sd_stream/SD3.5_stream.py
```

STEP2, after get transfered images, we do gaussian splatting training process
```
python /home/zhy01/gen_3d_recon/gaussian-splatting/train.py -s /home/zhy01/桌面/feicuiwan_all --new_image_path /home/zhy01/gen_3d_recon/sd_stream/lineart0
```
- -s is source path of 3dgs ply file, this 3dgs is orignal one
- --new_image_path is new path of gt image duing 3dgs training

STEP3, after we get trained 3d gaussian, we render then out
```
python /home/zhy01/gen_3d_recon/gaussian-splatting/render.py -m /home/zhy01/output/73a09e87-9 --ori_gaussian_path /home/zhy01/output/73a09e87-9 --new_image_path /home/zhy01/gen_3d_recon/sd_stream/lineart0
```
- -m and --ori_gaussian_path need set to step 2's output path, which is trained 3dgs ply file located
- --new_image_path is same as step 2

***
### Here is vscode debug launch.json as an example 
```
 {
            "name": "sdxl_new",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/zhy01/gen_3d_recon/sd_stream/new_sdxl.py",
            "args": [],
            "console": "integratedTerminal",
            // "justMyCode": false,
        },
        {
            "name": "gaussian_render",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/zhy01/gen_3d_recon/gaussian-splatting/render.py",
            "args": [
                "-m",
                // "/home/zhy01/gaussian-splatting/output/c77ba6be-e/"
                // "/home/zhy01/output/d8324ad9-c/",
                // "/home/zhy01/output/ade59c79-6"
                "/home/zhy01/output/73a09e87-9",
                "--ori_gaussian_path",
                "/home/zhy01/output/73a09e87-9",
                "--new_image_path",
                "/home/zhy01/gen_3d_recon/sd_stream/lineart0"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "gaussian_diffusion",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/zhy01/gen_3d_recon/gaussian-splatting/train.py",
            "args": [
                "-s",
                // "/home/zhy01/data/s1" this is for blender version
                "/home/zhy01/桌面/feicuiwan_all",
                "--new_image_path",
                "/home/zhy01/gen_3d_recon/sd_stream/lineart0"
            ],
            "justMyCode": false,
            "console": "integratedTerminal"
        },
        {
            "name": "SD_stream",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/zhy01/gen_3d_recon/sd_stream/SD3.5_stream.py",
            "args": [],
            "console": "integratedTerminal",
            // "justMyCode": false,
        },
```
