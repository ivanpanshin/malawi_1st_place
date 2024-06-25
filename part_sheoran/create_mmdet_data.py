import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import json

# Constants
FOLD = 0
BASE = "./data/mmdet_data_v1/"

# Load the data
data = pd.read_csv('./data/train_master1.csv')

# Filter the data to include only rows where 'chosen' is 1
data = data[data.chosen == 1].reset_index(drop=True)

# Create directories for the fold
os.makedirs(f"{BASE}/Fold_{FOLD}/", exist_ok=True)

# Split data into training and validation sets
train_data = data[data.fold != FOLD].reset_index(drop=True)
valid_data = data[data.fold == FOLD].reset_index(drop=True)

# Process each data split
for data_split, split in zip([train_data, valid_data], ['train', 'valid']):
    coco_data = {'images': [], 'annotations': [], 'categories': []}
    categories = ["Tin", "Thatch"]
    
    # Add categories to the COCO data structure
    for i, category_name in enumerate(categories):
        coco_data["categories"].append({
            'id': i + 1,
            'name': category_name,
        })
    
    # Process each row in the data split
    for idx in tqdm(range(len(data_split)), desc=f"Processing {split} data"):
        row = data_split.iloc[idx]
        iid = row.image_id
        bboxes = [eval(x) for x in eval(row.bbox)]
        clss = eval(row.category_id)
        
        filename = f"{iid}.png"
        width, height = Image.open(f"./data/train_images/{filename}").size
        
        image_data = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': idx + 1
        }
        
        coco_data['images'].append(image_data)
        
        for bbox, cl in zip(bboxes, clss):
            if cl == 1:
                continue
            
            annotation = {
                'id': len(coco_data['annotations']) + 1,
                'image_id': idx + 1,
                'category_id': int(cl) - 1,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
            }
            
            coco_data['annotations'].append(annotation)
    
    # Save the COCO data to a JSON file
    output_json_path = f"{BASE}/Fold_{FOLD}/{split}_data.json"
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)



# Constants
FOLD = 0
BASE = "./data/mmdet_data_v2/"

# Load the data
data = pd.read_csv('./data/train_master1.csv')

# Filter the data to include only rows where 'chosen' is 1
data = data[data.chosen == 1].reset_index(drop=True)

# Create directories for the fold
os.makedirs(f"{BASE}/Fold_{FOLD}/", exist_ok=True)

# Split data into training and validation sets
train_data = data[data.fold != FOLD].reset_index(drop=True)
valid_data = data[data.fold == FOLD].reset_index(drop=True)

# Process each data split
for data_split, split in zip([train_data, valid_data], ['train', 'valid']):
    coco_data = {'images': [], 'annotations': [], 'categories': []}
    categories = ["Other", "Tin", "Thatch"]
    
    # Add categories to the COCO data structure
    for i, category_name in enumerate(categories):
        coco_data["categories"].append({
            'id': i + 1,
            'name': category_name,
        })
    
    # Process each row in the data split
    for idx in tqdm(range(len(data_split)), desc=f"Processing {split} data"):
        row = data_split.iloc[idx]
        iid = row.image_id
        bboxes = [eval(x) for x in eval(row.bbox)]
        clss = eval(row.category_id)
        
        filename = f"{iid}.png"
        width, height = Image.open(f"./data/train_images/{filename}").size
        
        image_data = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': idx + 1
        }
        
        coco_data['images'].append(image_data)
        
        for bbox, cl in zip(bboxes, clss):
            annotation = {
                'id': len(coco_data['annotations']) + 1,
                'image_id': idx + 1,
                'category_id': int(cl),
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
            }
            
            coco_data['annotations'].append(annotation)
    
    # Save the COCO data to a JSON file
    output_json_path = f"{BASE}/Fold_{FOLD}/{split}_data.json"
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)