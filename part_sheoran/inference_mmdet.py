import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os, json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import mmcv
import mmdet
from mmdet.apis import inference_detector, init_detector
import logging

# Paths to the checkpoints and configurations
checkpoint_paths = [
    './final_weights/exp_002_best_bbox_mAP_epoch_11.pth',
    './final_weights/exp_004_best_bbox_mAP_epoch_12.pth',
]
config_paths = [
    './Co-DETR/work_dirs/exp_002/co_deformable_detr_swin_base_3x_coco.py',
    './Co-DETR/work_dirs/exp_004/co_dino_5scale_swin_large_16e_o365tococo.py',
]

# Print the paths for verification
_ = [print(x, y) for x, y in zip(config_paths, checkpoint_paths)]

# Initialize models
models = []
for checkpoint_file, config_file in tqdm(zip(checkpoint_paths, config_paths), desc="Initializing models"):
    model = init_detector(config_file, checkpoint_file)
    models.append(model)

# Get list of test image paths
paths = glob("./data/test_images/*")

# Initialize predictions array
test_preds = np.zeros((len(paths), 3))

# Make predictions on test images
for i, path in enumerate(tqdm(paths, desc="Making predictions")):
    image = mmcv.imread(path)
    for model in models:
        pred = inference_detector(model, image)[-2:]
        thresholds = [0.4, 0.4]
        for j, thresh in enumerate(thresholds):
            counts = np.sum(pred[j][:, -1] > thresh)
            test_preds[i, j + 1] += counts / len(models)

# Copy predictions to final predictions
final_predictions = test_preds.copy()

# Load submission template
submission = pd.read_csv('./data/SampleSubmission.csv')

# Fill in the submission with predictions
for path, prediction in zip(paths, final_predictions):
    iid = path.split('/')[-1].split('.')[0]
    submission.loc[submission.image_id == iid + "_1", 'Target'] = prediction[0]
    submission.loc[submission.image_id == iid + "_2", 'Target'] = prediction[1]
    submission.loc[submission.image_id == iid + "_3", 'Target'] = prediction[2]

# Convert target predictions to integers and save to CSV
submission.Target = submission.Target.astype(int)
submission.to_csv('submission.csv', index=False)
logging.info(f'Submission saved to submission.csv')