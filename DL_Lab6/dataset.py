import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class iCLEVRDataset(Dataset):
    def __init__(self, json_file, img_dir, objects_file, transform=None):
        with open(objects_file, 'r') as f:
            self.objects = json.load(f)

        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = list(self.data.keys())
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        objects_list = self.data[img_name]
        one_hot = torch.zeros(24)
        for obj in objects_list:
            idx_obj = self.objects[obj]
            one_hot[idx_obj] = 1
        
        return image, one_hot