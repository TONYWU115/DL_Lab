import os
import csv
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import rle_encode

device = "cuda" if torch.cuda.is_available() else "cpu"

print("選擇模型：1.unet  2.resnet34_unet")
choice = input("輸入 1 或 2：").strip()
if choice == "1":
    MODEL_TYPE = "unet"
elif choice == "2":
    MODEL_TYPE = "resnet34_unet"
else:
    raise ValueError("error")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "oxford-iiit-pet")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
if MODEL_TYPE == "unet":
    TEST_TXT = os.path.join(DATASET_DIR, "annotations", "test_unet.txt")
elif MODEL_TYPE == "resnet34_unet":
    TEST_TXT = os.path.join(DATASET_DIR, "annotations", "test_res_unet.txt")

test_image_names = []
if os.path.exists(TEST_TXT):
    with open(TEST_TXT, "r") as f:
        for line in f:
            name = line.strip().split(" ")[0]
            if name: test_image_names.append(name)

class TestPetDataset(Dataset):
    def __init__(self, image_names, images_dir, model_type="unet"):
        self.image_names = image_names
        self.images_dir = images_dir
        self.model_type = model_type.lower()
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        if self.model_type == "unet":
            image = TF.resize(image, (260, 260))
            image = TF.pad(image, padding=92, padding_mode='reflect')
        elif self.model_type == "resnet34_unet":
            image = TF.resize(image, (256, 256))

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        return image, name, orig_w, orig_h
    
@torch.no_grad()
def predict_with_tta(model, images):

    logits = model(images)
    probs = torch.sigmoid(logits)

    images_flipped = torch.flip(images, dims=[3])
    logits_flipped = model(images_flipped)
    probs_flipped = torch.sigmoid(logits_flipped)
    probs_flipped = torch.flip(probs_flipped, dims=[3])

    return (probs + probs_flipped) / 2.0

test_dataset = TestPetDataset(test_image_names, IMAGES_DIR, model_type=MODEL_TYPE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


if MODEL_TYPE == "unet":
    model = UNet().to(device)
    model_path = os.path.join(BASE_DIR, "saved_models", "best_unet.pth")
elif MODEL_TYPE == "resnet34_unet":
    model = ResNet34_UNet().to(device)
    model_path = os.path.join(BASE_DIR, "saved_models", "best_resnet34_unet.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

submission = []

with torch.no_grad():
    # for images, names, orig_ws, orig_hs in test_loader:
    for images, names, orig_ws, orig_hs in tqdm(test_loader, desc="Inferencing"):
        images = images.to(device)
        logits = model(images)
        # probs = torch.sigmoid(logits)
        probs = predict_with_tta(model, images)

        for i in range(probs.size(0)):
            name = names[i]
            w_orig, h_orig = orig_ws[i].item(), orig_hs[i].item()

            if MODEL_TYPE == "unet":
                out_size = (260, 260)
            else:
                out_size = (256, 256)

            prob_up = torch.nn.functional.interpolate(
                probs[i:i+1], size=(h_orig, w_orig), mode="bilinear", align_corners=False
            )
            mask = (prob_up[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            rle = rle_encode(mask)
            submission.append([name, rle])

submission_path = os.path.join(BASE_DIR, "submission.csv")
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "encoded_mask"])
    writer.writerows(submission)

print(f"Inference complete! Saved to {submission_path}")