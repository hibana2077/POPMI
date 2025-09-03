from __future__ import annotations
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, csv
import torch

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, image_size: int = 224):
        self.items = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row['image_id'] + '.jpg', row['dx']))
        self.image_root = image_root
        self.classes = sorted({c for _, c in self.items})
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        fname, label = self.items[idx]
        path = os.path.join(self.image_root, fname)
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        y = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        return x, y

def ham10000_loaders(train_csv: str, valid_csv: str, image_root: str, image_size: int = 224,
                     batch_size: int = 32, num_workers: int = 4):
    train_ds = HAM10000Dataset(train_csv, image_root, image_size)
    val_ds = HAM10000Dataset(valid_csv, image_root, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, len(train_ds.classes)
