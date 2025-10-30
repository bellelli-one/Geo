import pandas as pd 
from rasterio.features import rasterize
import rasterio
import torch 
import random
import geopandas as gpd 
from torch.utils.data import Dataset
import numpy as np
import re
import os
from typing import Dict
from torch.utils.data import DataLoader
import json

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import rasterio
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
# class GeoDataset(Dataset):
#     def __init__(self, index_file, class_mapping, class_id_mapping, data_type_filter=None, 
#                  patch_size=1024, patches_per_image=10, max_retries=10, transforms=None):

#         self.class_mapping = class_mapping
#         self.class_id_mapping = class_id_mapping
#         self.transforms = transforms
#         self.patch_size = patch_size
#         self.patches_per_image = patches_per_image
#         self.max_retries = max_retries
#         df = pd.read_csv(index_file)
#         if data_type_filter:
#             df = df[df['data_type'] == data_type_filter].reset_index(drop=True)

#         self.samples = df.to_dict('records')
#         if not self.samples:
#             raise ValueError("Не найдено сэмплов после фильтрации.")

#     def __len__(self):
#         return len(self.samples) * self.patches_per_image

#     def __getitem__(self, idx):
#         image_idx = idx // self.patches_per_image
#         sample = self.samples[image_idx]
        
#         img_path = sample["image_path"]
#         mask_paths = sample["mask_paths"].split(';')
#         utm_file_path = sample["utm_path"]

#         try:
#             with open(utm_file_path, 'r') as f:
#                 utm_data = json.load(f)
#                 target_crs_str = utm_data['crs'].replace('urn:ogc:def:crs:EPSG::', 'EPSG:')
            
#             target_crs = rasterio.crs.CRS.from_string(target_crs_str)
#             with rasterio.open(img_path) as src:
#                 image = src.read()
#                 meta = src.meta.copy()
#                 transform = src.transform
#             num_channels = image.shape[0]
#             if num_channels == 1: image = np.repeat(image, 3, axis=0)
#             elif num_channels > 3: image = image[:3, :, :]
            
#             H, W = meta['height'], meta['width']
#             shapes = []
#             for mask_path in mask_paths:
#                 if not mask_path.strip(): continue
#                 try:
#                     gdf = gpd.read_file(mask_path)
#                     if gdf.empty: continue

#                     gdf = gdf.to_crs(target_crs)
#                     basename = os.path.basename(mask_path)
#                     class_name_match = re.search(r'_(?:Li|Ae|Or|SpOr)_([^_.]+)\.geojson', basename, re.IGNORECASE)
#                     if class_name_match:
#                         class_name = class_name_match.group(1).lower()
#                         std_class_name = self.class_mapping.get(class_name)
#                         class_id = self.class_id_mapping.get(std_class_name)
#                         if class_id is not None:
#                             shapes.extend([(geom, class_id) for geom in gdf.geometry if geom is not None and geom.is_valid])
#                 except Exception as e:
#                     continue

#             full_mask = np.zeros((H, W), dtype=np.uint8)
#             if shapes:
#                 rasterio.features.rasterize(
#                     shapes=shapes,
#                     out=full_mask,
#                     transform=transform,
#                     all_touched=True,
#                     dtype=np.uint8
#                 )

#             image = np.transpose(image, (1, 2, 0)).astype(np.float32)
            
#             valid_indices = np.where(full_mask > 0)
#             if len(valid_indices[0]) > 0:
#                 random_idx = random.randint(0, len(valid_indices[0]) - 1)
#                 center_y, center_x = valid_indices[0][random_idx], valid_indices[1][random_idx]
#                 x = max(0, center_x - self.patch_size // 2)
#                 y = max(0, center_y - self.patch_size // 2)
#             else:
#                 x = random.randint(0, max(0, W - self.patch_size))
#                 y = random.randint(0, max(0, H - self.patch_size))
#             x = min(x, max(0, W - self.patch_size))
#             y = min(y, max(0, H - self.patch_size))

#             image_patch = image[y : y + self.patch_size, x : x + self.patch_size]
#             mask_patch = full_mask[y : y + self.patch_size, x : x + self.patch_size]
#             if self.transforms:
#                 augmented = self.transforms(image=image_patch, mask=mask_patch)
#                 image_patch = augmented['image']
#                 mask_patch = augmented['mask']
            
#             return image_patch, torch.as_tensor(mask_patch, dtype=torch.long)

#         except Exception:
#             print(f"КРИТИЧЕСКАЯ ОШИБКА при обработке индекса {idx} для изображения {img_path}")
#             return torch.zeros((3, self.patch_size, self.patch_size)), torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)