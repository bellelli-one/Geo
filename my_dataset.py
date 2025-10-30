import pandas as pd 
import rasterio
import torch 
import random
from torch.utils.data import Dataset
import numpy as np
from rasterio.windows import Window
import random

class GeoDataset(Dataset):
    def __init__(self, index_file, data_type_filter=None, patch_size=1024, patches_per_image=10, transforms=None):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transforms = transforms
        
        df = pd.read_csv(index_file)
        if data_type_filter:
            df = df[df['data_type'] == data_type_filter].reset_index(drop=True)

        self.samples = df.to_dict('records')
        if not self.samples:
            raise ValueError("Не найдено сэмплов после фильтрации.")

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image
        sample = self.samples[image_idx]
        
        img_path = sample["image_path"]
        mask_path = sample["raster_mask_path"]

        try:
            with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
                H, W = src_img.height, src_img.width
                valid_indices = np.where(src_mask.read(1) > 0)
                if len(valid_indices[0]) > 0:
                    random_idx = random.randint(0, len(valid_indices[0]) - 1)
                    center_y, center_x = valid_indices[0][random_idx], valid_indices[1][random_idx]
                    x = max(0, center_x - self.patch_size // 2)
                    y = max(0, center_y - self.patch_size // 2)
                else:
                    x = random.randint(0, max(0, W - self.patch_size))
                    y = random.randint(0, max(0, H - self.patch_size))
                
                x = min(x, W - self.patch_size)
                y = min(y, H - self.patch_size)
                window = Window(x, y, self.patch_size, self.patch_size)

                image_patch = src_img.read(window=window)
                mask_patch = src_mask.read(1, window=window)

            num_channels = image_patch.shape[0]
            if num_channels == 1: image_patch = np.repeat(image_patch, 3, axis=0)
            elif num_channels > 3: image_patch = image_patch[:3, :, :]
            image_patch = np.transpose(image_patch, (1, 2, 0)).astype(np.float32)

            if self.transforms:
                augmented = self.transforms(image=image_patch, mask=mask_patch)
                image_patch = augmented['image']
                mask_patch = augmented['mask']
            
            return image_patch, torch.as_tensor(mask_patch, dtype=torch.long)

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при обработке индекса {idx} для изображения {img_path}. Ошибка: {e}")
            return torch.zeros((3, self.patch_size, self.patch_size)), torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)
