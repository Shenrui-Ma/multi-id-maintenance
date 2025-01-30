from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import io

def process_image_format(image):
    """统一处理图像格式，支持RGB和RGBA"""
    if image.mode == 'RGB':
        # RGB图像保持不变
        return image
    elif image.mode == 'RGBA':
        # RGBA图像保持原样
        return image
    else:
        # 其他格式转换为RGB
        return image.convert('RGB')

class Subject200KDateset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        # 保持初始化不变
        super().__init__()
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        target = 0 # 左图X,右图Ci
        item = self.base_dataset[idx // 2]
        
        # 处理图像数据
        image_data = item["image"]
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes")
            if image_bytes is None:
                raise ValueError("Image dictionary does not contain 'bytes' key")
        else:
            image_bytes = image_data
            
        # 加载并处理图像格式
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = process_image_format(image)
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")

        # 裁剪图像
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # 调整图像大小，保持原始格式
        condition_img = condition_img.resize((self.condition_size, self.condition_size))
        target_image = target_image.resize((self.target_size, self.target_size))

        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # 处理随机丢弃
        if random.random() < self.drop_text_prob:
            description = ""
        if random.random() < self.drop_image_prob:
            # 创建与原图相同模式的空白图像
            mode = condition_img.mode
            if mode == 'RGBA':
                condition_img = Image.new(mode, (self.condition_size, self.condition_size), (0, 0, 0, 0))
            else:
                condition_img = Image.new(mode, (self.condition_size, self.condition_size), (0, 0, 0))

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": image} if self.return_pil_image else {}),
        }

class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()

    def _get_canny_edge(self, img):
        """处理Canny边缘检测，支持RGB和RGBA"""
        resize_ratio = self.condition_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
        )
        
        # 转换为numpy数组
        img_np = np.array(img)
        
        # 根据图像格式处理
        if img_np.shape[-1] == 4:  # RGBA
            # 使用RGB通道进行边缘检测
            img_rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            # 保持原始alpha通道
            edge_img = np.dstack((edges, edges, edges, alpha))
            return Image.fromarray(edge_img, 'RGBA')
        else:  # RGB
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            return Image.fromarray(edges).convert('RGB')

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # 处理图像格式
        image = process_image_format(image)
        image = image.resize((self.target_size, self.target_size))
        description = self.base_dataset[idx]["json"]["prompt"]
        position_delta = np.array([0, 0])

        # 根据条件类型处理图像
        if self.condition_type == "canny":
            condition_img = self._get_canny_edge(image)
        elif self.condition_type == "coloring":
            # 转换为灰度图并保持原始格式
            gray = image.convert("L")
            if image.mode == 'RGBA':
                condition_img = Image.merge('RGBA', (gray, gray, gray, image.split()[3]))
            else:
                condition_img = gray.convert('RGB')
        elif self.condition_type == "deblurring":
            blur_radius = random.randint(1, 10)
            if image.mode == 'RGBA':
                r, g, b, a = image.split()
                rgb = Image.merge('RGB', (r, g, b))
                rgb_blurred = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
                r, g, b = rgb_blurred.split()
                condition_img = Image.merge('RGBA', (r, g, b, a))
            else:
                condition_img = image.filter(ImageFilter.GaussianBlur(blur_radius))
        elif self.condition_type == "depth":
            depth_rgb = self.depth_pipe(image)["depth"]
            condition_img = depth_rgb.convert(image.mode)
        elif self.condition_type == "depth_pred":
            condition_img = image
            depth_rgb = self.depth_pipe(condition_img)["depth"]
            image = depth_rgb.convert(image.mode)
            description = f"[depth] {description}"
        elif self.condition_type == "fill":
            condition_img = image.resize((self.condition_size, self.condition_size))
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            # 创建与原图相同模式的空白图像
            blank = Image.new(image.mode, image.size, (0, 0, 0, 0) if image.mode == 'RGBA' else (0, 0, 0))
            condition_img = Image.composite(image, blank, mask)
        else:
            raise ValueError(f"Condition type {self.condition_type} not implemented")

        # 处理随机丢弃
        if random.random() < self.drop_text_prob:
            description = ""
        if random.random() < self.drop_image_prob:
            condition_img = Image.new(
                image.mode,
                (self.condition_size, self.condition_size),
                (0, 0, 0, 0) if image.mode == 'RGBA' else (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }