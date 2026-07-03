import os, random, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class OxfordPetDataset(Dataset):
    def __init__(self, split="train", model_type="unet"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "oxford-iiit-pet"))
        self.split = split
        self.model_type = model_type.lower()
        self.images_dir = os.path.join(base_dir, "images")
        self.masks_dir = os.path.join(base_dir, "annotations", "trimaps")
        split_file = os.path.join(base_dir, "annotations", f"{split}.txt")

        with open(split_file, "r") as f:
            self.file_names = [line.strip().split(' ')[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.file_names)

    def _preprocess_mask(self, mask_pil):
        m = np.array(mask_pil, dtype=np.uint8)
        m = (m == 1).astype(np.float32)
        return torch.from_numpy(m).unsqueeze(0)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        image = Image.open(os.path.join(self.images_dir, name + ".jpg")).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir,  name + ".png"))

        if self.model_type == "unet":
            # Unpadded UNet: 260 -> Pad 92 -> 444
            image = TF.resize(image, (260, 260))
            mask  = TF.resize(mask,  (260, 260), interpolation=T.InterpolationMode.NEAREST)
        elif self.model_type == "resnet34_unet":
            # ResNet34_UNet: 256 
            image = TF.resize(image, (256, 256))
            mask  = TF.resize(mask,  (256, 256), interpolation=T.InterpolationMode.NEAREST)

        if self.split == "train":
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.3:
                image = T.ColorJitter(brightness=0.2, contrast=0.2)(image)
            if random.random() < 0.5:
                angle = random.uniform(-20, 20)
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=2)

        if self.model_type == "unet":
            image = TF.pad(image, padding=92, padding_mode='reflect')

        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        mask = self._preprocess_mask(mask)
        return image, mask