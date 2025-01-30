import sys
import os

# 打印当前工作目录，用于调试
print("Current working directory:", os.getcwd())

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project root path:", project_root)

# 将项目根目录添加到 Python 路径的最前面
sys.path.insert(0, project_root)
print("Updated sys.path:", sys.path)

# 现在再导入所需模块
import torch
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from PIL import Image
from src.flux.generate import generate, seed_everything


pipe = FluxPipeline.from_pretrained(
    "/data/shenrui.ma/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
lora_path = "/data/shenrui.ma/OminiControl/runs/20250121-045449/ckpt/11000/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(
    lora_path,
    adapter_name="subject",
)

image = Image.open("/data/shenrui.ma/OminiControl/assets/2kidsinwhiteshirt.png").convert("RGB").resize((512, 512))

condition = Condition("subject", image)

prompt = "The two people are sitting on the floor."


seed_everything(0)

result_img = generate(
    pipe,
    prompt=prompt,
    conditions=[condition],
    num_inference_steps=8,
    height=512,
    width=512,
).images[0]

concat_image = Image.new("RGB", (1024, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(result_img, (512, 0))
concat_image

output_path = "output_comparison.png"  # 你可以修改保存路径和文件名
concat_image.save(output_path)
print(f"Saved comparison image to: {output_path}")