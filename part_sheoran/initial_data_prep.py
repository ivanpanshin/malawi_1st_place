import pandas as pd
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import StratifiedGroupKFold

# Load the data
data = pd.read_csv('./data/Train.csv')

# Fill NaN values with -1
data = data.fillna(-1)

# Group the data by 'image_id' and aggregate 'bbox', 'category_id', 'id' columns into lists
data = data.groupby('image_id')[['bbox', 'category_id', 'id']].agg(list).reset_index()

# Create a new column 'chosen' to mark rows with non-empty 'bbox'
data['chosen'] = data.bbox.apply(lambda x: 0 if x == [-1] else 1)

# Initialize 'fold' column with -1
data['fold'] = -1

# Create stratified group folds
fold_idxs = [*StratifiedGroupKFold(n_splits=4).split(data, data.chosen, groups=data.image_id.values)]
for fold, idxs in enumerate(fold_idxs):
    data.loc[idxs[1], 'fold'] = fold

# Save the updated DataFrame to a CSV file
data.to_csv('./data/train_master1.csv', index=False)

os.makedirs('./data/train_images', exist_ok=1)
# Convert images in the train set to .png format
for iid in tqdm(data.image_id.unique(), desc="Processing train images"):
    image = cv2.imread(f"./data/Images/{iid}.tif")
    cv2.imwrite(f"./data/train_images/{iid}.png", image)

# List of all image IDs in the test set
test_ids = os.listdir("./data/Images/")
train_ids = data.image_id.unique()

os.makedirs('./data/test_images', exist_ok=1)
# Convert images in the test set to .png format
for iid in tqdm(test_ids, desc="Processing test images"):
    if iid.split('.')[0] not in train_ids:
        image = cv2.imread(f"./data/Images/{iid}")
        cv2.imwrite(f"./data/test_images/{iid.replace('.tif', '.png')}", image)
