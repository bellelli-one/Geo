import os
import csv
import glob

DATA_SOURCES = {
    "Li":   {"images": ["_li_карты"], "masks": "Li"},
    "Ae":   {"images": ["_ae_немецкая", "_ае_немецкая"], "masks": "Ae"}, # Латиница и Кириллица
    "Or":   {"images": ["_or"], "masks": "Or"},
    "SpOr": {"images": ["_spor"], "masks": "SpOr"}
}

def find_data_pairs(root_path):
    found_samples = []
    try:
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
    except OSError:
        return []

    razmetka_path = None
    for subdir in subfolders:
        if os.path.basename(subdir).lower().endswith("_разметка"):
            razmetka_path = subdir
            break

    if razmetka_path:
        for data_type, patterns in DATA_SOURCES.items():
            image_folder_path = None
            for subdir in subfolders:
                subdir_name_lower = os.path.basename(subdir).lower()
                for pattern in patterns["images"]:
                    if subdir_name_lower.endswith(pattern):
                        image_folder_path = subdir
                        break
                if image_folder_path:
                    break
            
            mask_folder_path = None
            potential_mask_subfolder = os.path.join(razmetka_path, patterns["masks"])
            if os.path.isdir(potential_mask_subfolder):
                mask_folder_path = potential_mask_subfolder
            else:
                mask_folder_path = razmetka_path

            if image_folder_path and mask_folder_path:
                
                geojson_files = []
                mask_type_indicator = f"_{patterns['masks'].lower()}_"
                for f in os.listdir(mask_folder_path):
                    f_lower = f.lower()
                    if f_lower.endswith('.geojson') and mask_type_indicator in f_lower:
                        geojson_files.append(os.path.join(mask_folder_path, f))
                
                if not geojson_files: continue

                tif_files = []
                for dirpath, _, filenames in os.walk(image_folder_path):
                    for filename in filenames:
                        if filename.lower().endswith('.tif'):
                            tif_files.append(os.path.join(dirpath, filename))
                utm_path = None
                current_search_path = image_folder_path
                for _ in range(5):
                    utm_files = glob.glob(os.path.join(current_search_path, '[Uu][Tt][Mm].json'))
                    if utm_files:
                        utm_path = utm_files[0]
                        break
                    parent_path = os.path.dirname(current_search_path)
                    if parent_path == current_search_path:
                        break
                    current_search_path = parent_path
                
                if not utm_path:
                    print(f"Не найден UTM.json для группы карт в {image_folder_path}. Пропускаем.")
                    continue

                for tif_path in tif_files:
                    normalized_tif_path = tif_path.replace('\\', '/')
                    normalized_geojson_paths = [p.replace('\\', '/') for p in geojson_files]
                    normalized_utm_path = utm_path.replace('\\', '/')
                    
                    found_samples.append({
                        "image_path": normalized_tif_path,
                        "mask_paths": ";".join(normalized_geojson_paths),
                        "data_type": data_type,
                        "utm_path": normalized_utm_path
                    })

    for subdir in subfolders:
        subdir_name_lower = os.path.basename(subdir).lower()
        is_data_folder = any(pat in subdir_name_lower for pat in ["_карты", "_разметка", "_немецкая", "_or", "_spor"])
        if not is_data_folder:
             found_samples.extend(find_data_pairs(subdir))
             
    return found_samples

def main(root_dir, output_csv):
    print("Start scan")
    all_samples = find_data_pairs(root_dir)
    print(f"Find {len(all_samples)} samples.")

    if not all_samples:
        print("No samples find")
        return
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_paths", "data_type", "utm_path"])
        writer.writeheader()
        writer.writerows(all_samples)


if __name__ == '__main__':
    DATASET_ROOT = "D:/lidar/train"
    INDEX_FILE = "dataset_index.csv"
    main(DATASET_ROOT, INDEX_FILE)
