from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_medmnist_dataset(data_flag: str, split: str, image_size: int, as_rgb: bool, download: bool = True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    t_list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if not as_rgb:
        t_list.append(transforms.Grayscale())
    t_list.append(transforms.Normalize(mean=[0.5]* (3 if as_rgb else 1), std=[0.5]* (3 if as_rgb else 1)))
    transform = transforms.Compose(t_list)
    ds = DataClass(split=split, transform=transform, download=download, as_rgb=as_rgb)
    return ds, info['n_classes']

def medmnist_loaders(data_flag: str, image_size: int = 64, as_rgb: bool = True,
                      batch_size: int = 128, num_workers: int = 4):
    train_ds, n_classes = get_medmnist_dataset(data_flag, 'train', image_size, as_rgb)
    val_ds, _ = get_medmnist_dataset(data_flag, 'val', image_size, as_rgb)
    test_ds, _ = get_medmnist_dataset(data_flag, 'test', image_size, as_rgb)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, n_classes
