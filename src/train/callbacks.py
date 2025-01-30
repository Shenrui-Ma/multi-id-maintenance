import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = 512
        target_size = 512

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        if condition_type == "subject":
            test_list.extend(
                [
                    (
                        Image.open("/data/shenrui.ma/OminiControl/assets/person/blonde_female_face_whiteBG.png"),
                        [0, -32],
                    '''The two people are sitting on the grass.The background is starry sky.''',
                    ),
                    (
                        Image.open("/data/shenrui.ma/OminiControl/assets/person/male_shorthair_face_whiteBG.png"),
                        [0, -64],
                        
                    ),
                    (
                        Image.open("/data/shenrui.ma/OminiControl/assets/person/feifeili.jpg"),
                        [0, -32],
                    '''This woman and this man are standing on the grass.''',
                    ),
                    (
                        Image.open("/data/shenrui.ma/OminiControl/assets/person/Andrew_Ng_at_TechCrunch.jpg.jpg"),
                        [0, -64],
                        
                    ),
                ]
            )
        elif condition_type == "canny":
            condition_img = Image.open("assets/vase.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = np.array(condition_img)
            condition_img = cv2.Canny(condition_img, 100, 200)
            condition_img = Image.fromarray(condition_img).convert("RGB")
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "coloring":
            condition_img = (
                Image.open("assets/vase.jpg")
                .resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "depth":
            if not hasattr(self, "deepth_pipe"):
                self.deepth_pipe = pipeline(
                    task="depth-estimation",
                    model="LiheYoung/depth-anything-small-hf",
                    device="cpu",
                )
            condition_img = (
                Image.open("assets/vase.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "depth_pred":
            condition_img = (
                Image.open("assets/vase.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "deblurring":
            blur_radius = 5
            image = Image.open("./assets/vase.jpg")
            condition_img = (
                image.convert("RGB")
                .resize((condition_size, condition_size))
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "fill":
            condition_img = (
                Image.open("./assets/vase.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            mask = Image.new("L", condition_img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([128, 128, 384, 384], fill=255)
            condition_img = Image.composite(
                condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # 原代码，遍历每一个<图片,position_delta, prompt>单独作condition
        # for i, (condition_img, position_delta, prompt) in enumerate(test_list):
        #     condition = Condition(
        #         condition_type=condition_type,
        #         condition=condition_img.resize(
        #             (condition_size, condition_size)
        #         ).convert("RGB"),
        #         position_delta=position_delta,
        #     )
        #     res = generate(
        #         pl_module.flux_pipe,
        #         prompt=prompt,
        #         conditions=[condition],
        #         height=target_size,
        #         width=target_size,
        #         generator=generator,
        #         model_config=pl_module.model_config,
        #         default_lora=True,
        #     )
        #     res.images[0].save(
        #         os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        #     )
        
        # 每2张作一对儿condition,提示词公用一个
        # RGB -> RGBA
            paired_list = []
            for i in range(0, len(test_list), 2):
                if i + 1 < len(test_list):  # 确保不会越界
                    paired_list.append(
                        (
                            test_list[i][0],  # 第一张图片
                            test_list[i+1][0],  # 第二张图片
                            test_list[i][1],  # 第一张图片的 position_delta
                            test_list[i+1][1],  # 第二张图片的 position_delta
                            test_list[i][2],  # 公用提示词 prompt
                        )
                    )

            # 遍历 paired_list，每对图片处理一次
            for j, (img_1, img_2, delta_1, delta_2, prompt) in enumerate(paired_list):
                # 创建第一个 condition
                condition_1 = Condition(
                    condition_type=condition_type,
                    condition=img_1.resize((condition_size, condition_size)).convert("RGBA"),
                    position_delta=delta_1,
                )

                # 创建第二个 condition
                condition_2 = Condition(
                    condition_type=condition_type,
                    condition=img_2.resize((condition_size, condition_size)).convert("RGBA"),
                    position_delta=delta_2,
                )

                # 调用 generate 方法，传入两组 condition
                res = generate(
                    pl_module.flux_pipe,
                    prompt=prompt,  # 公用同一个提示词
                    conditions=[condition_1, condition_2],
                    height=target_size,
                    width=target_size,
                    generator=generator,
                    model_config=pl_module.model_config,
                    default_lora=True,
                )

                # 保存生成的结果
                res.images[0].save(
                    os.path.join(save_path, f"/data/shenrui.ma/OminiControl/runs/callbacks/{file_name}_{condition_type}_{j}_pair1.jpg")
                )
                res.images[1].save(
                    os.path.join(save_path, f"/data/shenrui.ma/OminiControl/runs/callbacks/{file_name}_{condition_type}_{j}_pair2.jpg")
                )
