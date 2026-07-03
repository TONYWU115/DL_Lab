import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

def choose_model_type():
    print("選擇模型：1.unet  2.resnet34_unet")
    choice = input("輸入 1 或 2：").strip()
    if choice == "1":
        return "unet"
    if choice == "2":
        return "resnet34_unet"
    raise ValueError("error")

def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    denominator = probs.sum(1) + targets.sum(1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def calculate_dice(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(1)
    denominator = preds.sum(1) + targets.sum(1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean().item()

def train(MODEL_TYPE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dataset = OxfordPetDataset(split="train", model_type=MODEL_TYPE)
    val_dataset = OxfordPetDataset(split="val", model_type=MODEL_TYPE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    if MODEL_TYPE == "unet":
        model = UNet().to(device)
    elif MODEL_TYPE == "resnet34_unet":
        model = ResNet34_UNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_dice = 0.0
    save_name = f"best_{MODEL_TYPE}.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{MODEL_TYPE.upper()}]")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)

            dice_l = soft_dice_loss(logits, masks)
            bce_l = criterion_bce(logits, masks)
            loss = 0.7 * bce_l + 1.3 * dice_l

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_dice += calculate_dice(logits, masks)

        avg_val_dice = val_dice / len(val_loader)
        print(f"==> Val Dice: {avg_val_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_TYPE}_epoch{epoch+1}.pth"))

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))
            print(f"Best Model Saved! ({save_name})")

        scheduler.step(avg_val_dice)

if __name__ == "__main__":
    MODEL_TYPE = choose_model_type()
    train(MODEL_TYPE)