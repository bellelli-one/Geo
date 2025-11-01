import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os
import re
import json
import numpy as np
from tqdm import tqdm
from typing import Dict



SOURCE_INDEX_FILE = "dataset_index.csv" 

RASTER_MASKS_DIR = 'data/preprocessed_masks'

NEW_INDEX_FILE = 'preprocessed_index.csv'

DATA_TYPE_FILTER = 'Li' # Или None если хочешь и аэроснимки и спутниковые снимки

CLASS_NAME_MAPPING: Dict[str, str | None] = {
    # Селища
    "селище": "selishcha", "Селище": "selishcha", "селища": "selishcha", "Селища": "selishcha",
    # Пашни
    "пашня": "pashni", "Пашня": "pashni", "пашни": "pashni", "Пашни": "pashni",
    "пахота": "pashni", "pashnya": "pashni", "Pashnya": "pashni",
    "глубин": "pashni", "Глубин": "pashni",
    # Курганы
    "распаханные курганы": "kurgany", "курган": "kurgany", "Курган": "kurgany",
    "курганы": "kurgany", "Курганы": "kurgany", "kurgani": "kurgany", "Kurgani": "kurgany",
    # Караванные пути
    "караванные": "karavannye_puti", "Караванные": "karavannye_puti",
    "караванные пути": "karavannye_puti", "Караванные пути": "karavannye_puti",
    "пути": "karavannye_puti", "Пути": "karavannye_puti",
    # Фортификации
    "фортификация": "fortifikatsii", "Фортификация": "fortifikatsii",
    "фортификации": "fortifikatsii", "Фортификации": "fortifikatsii",
    # Городища
    "городище": "gorodishcha", "Городище": "gorodishcha",
    "городища": "gorodishcha", "Городища": "gorodishcha",
    "gorodishche": "gorodishcha", "Gorodishche": "gorodishcha",
    # Архитектуры
    "архитектура": "arkhitektury", "Архитектура": "arkhitektury",
    "архитектуры": "arkhitektury", "Архитектуры": "arkhitektury",
    # Дороги
    "дорога": "dorogi", "Дорога": "dorogi", "дороги": "dorogi", "Дороги": "dorogi",
    "dorogi": "dorogi", "Dorogi": "dorogi",
    # Ямы
    "яма": "yamy", "Яма": "yamy", "ямы": "yamy", "Ямы": "yamy",
    # Межа
    "межа": "mezha", "Межа": "mezha",
    # Артефакты лидара (игнор)
    "артефакты лидара": None, "Артефакты лидара": None,
    "артефакты_лидара": None, "Артефакты_лидара": None,
    "лидара": None, "артефакт": None, "Артефакт": None,
    # Иное
    "иное": "inoe", "Иное": "inoe", "inoe": "inoe", "Inoe": "inoe",
}

CLASS_ID_MAPPING = {
    "selishcha": 1,
    "pashni": 2,
    "kurgany": 3,
    "karavannye_puti": 4,
    "fortifikatsii": 5,
    "gorodishcha": 6,
    "arkhitektury": 7,
    "dorogi": 8,
    "yamy": 9,
    "mezha": 10,
    "inoe": 11,
}


def create_raster_masks():

    print("--- НАЧАЛО ПРЕПРОЦЕССИНГА ---")
    os.makedirs(RASTER_MASKS_DIR, exist_ok=True)
    print(f"Растровые маски будут сохранены в: {RASTER_MASKS_DIR}")

    df = pd.read_csv(SOURCE_INDEX_FILE)
    if DATA_TYPE_FILTER:
        df = df[df['data_type'] == DATA_TYPE_FILTER].reset_index(drop=True)

    if df.empty:
        raise ValueError("DataFrame пуст после фильтрации. Проверьте SOURCE_INDEX_FILE и DATA_TYPE_FILTER.")

    new_index_records = []


    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Обработка изображений"):
        img_path = row["image_path"]
        mask_paths_str = row.get("mask_paths", "")
        utm_file_path = row["utm_path"]

        try:
            with open(utm_file_path, 'r') as f:
                utm_data = json.load(f)
                target_crs_str = utm_data['crs'].replace('urn:ogc:def:crs:EPSG::', 'EPSG:')
            target_crs = rasterio.crs.CRS.from_string(target_crs_str)

            with rasterio.open(img_path) as src:
                meta = src.meta.copy()
                transform = src.transform
                H, W = src.height, src.width

            shapes = []
            mask_paths = mask_paths_str.split(';')
            for mask_path in mask_paths:
                if not mask_path.strip():
                    continue
                try:
                    gdf = gpd.read_file(mask_path)
                    if gdf.empty:
                        continue

                    gdf = gdf.to_crs(target_crs)
                    basename = os.path.basename(mask_path)
                    class_name_match = re.search(r'_(?:Li|Ae|Or|SpOr)_([^_.]+)\.geojson', basename, re.IGNORECASE)
                    if class_name_match:
                        class_name = class_name_match.group(1).lower()
                        std_class_name = CLASS_NAME_MAPPING.get(class_name)
                        class_id = CLASS_ID_MAPPING.get(std_class_name)
                        if class_id is not None:
                            shapes.extend([(geom, class_id) for geom in gdf.geometry if geom is not None and geom.is_valid])
                except Exception:
                    continue

            full_mask = np.zeros((H, W), dtype=np.uint8)
            if shapes:
                rasterize(
                    shapes=shapes,
                    out=full_mask,
                    transform=transform,
                    all_touched=True,
                    dtype=np.uint8
                )
            if full_mask.max() == 0:
                print(f"\n[ИНФО] Для изображения {img_path} не найдено валидных геометрий. Маска пуста. Пропускаем.")
                skipped_count += 1
                continue
            base_img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_mask_path = os.path.join(RASTER_MASKS_DIR, f"{base_img_name}_mask.tif")

            mask_meta = meta.copy()
            mask_meta.update(driver='GTiff', count=1, dtype='uint8', compress='lzw')

            with rasterio.open(output_mask_path, 'w', **mask_meta) as dst:
                dst.write(full_mask, 1)

            new_record = row.to_dict()
            new_record['raster_mask_path'] = output_mask_path.replace('\\', '/')
            new_index_records.append(new_record)

        except Exception as e:
            print(f"\n[ОШИБКА] Не удалось обработать {img_path}. Причина: {e}. Пропускаем.")
            continue

    new_df = pd.DataFrame(new_index_records)
    new_df.to_csv(NEW_INDEX_FILE, index=False)
    
    print("\n--- ПРЕПРОЦЕССИНГ УСПЕШНО ЗАВЕРШЕН ---")
    print(f"Создан новый индексный файл: {NEW_INDEX_FILE}")
    print(f"Всего обработано и сохранено масок: {len(new_df)}")
    if skipped_count > 0:
        print(f"Было пропущено изображений с пустыми масками: {skipped_count}")

if __name__ == '__main__':
    create_raster_masks()