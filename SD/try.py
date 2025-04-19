from diffusers import StableDiffusionXLPipeline
import torch
from torchvision.transforms import Resize
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
prompt = ""
image = pipe(prompt=prompt, num_inference_steps=10, height=256, width=256).images[0]
image.save("img.png")
image = pipe(prompt=prompt, num_inference_steps=10, height=512, width=512).images[0]
image = image.resize((64,64))
image.save("img_big.png")  
