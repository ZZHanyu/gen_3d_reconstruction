from diffusers import StableDiffusionControlNetPipeline

class MyControlNetPipeline(StableDiffusionControlNetPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def my_custom_method(self):
        print("这是我自定义的 ControlNet 方法！")


# 使用自定义 pipeline
my_pipe = MyControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

my_pipe.my_custom_method()