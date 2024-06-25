from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch
import random

class DatasetCompetition(Dataset):
    def __init__(
        self,
        dataset_root,
        ann_path,
        split,
        multiplier=1,
        transform=None,
    ):
        super().__init__()

        csv = pd.read_csv(ann_path)
        if split == 'test':
            self.ids = csv.image_id.unique()
            self.paths = [f'{dataset_root}/{_}.tif' for _ in self.ids]
        else:
            category_id_to_class = {
                1: 'other',
                2: 'tin',
                3: 'thatch',
            }

            self.paths, self.bboxes, self.bbox_classes = [], [], []
            for name, group in csv.groupby(by='image_id'):
                self.paths.append(f'{dataset_root}/{group.image_id.values[0]}.tif')

                bboxes, bbox_classes = [], []
                if not group.bbox.isna().values[0]:
                    for bbox, category_id in zip(group.bbox.values, group.category_id.values):
                        bbox_preprocessed = [int(float(_)) for _ in bbox[1:-1].split(',')]
                        category_id = int(category_id)
                        x1, y1, width, height = bbox_preprocessed

                        if width > 0 and height > 0:
                            bboxes.append(bbox_preprocessed)
                            bbox_classes.append(category_id_to_class[category_id])

                self.bboxes.append(bboxes)
                self.bbox_classes.append(bbox_classes)

            assert len(self.paths) == len(self.bboxes) == len(self.bbox_classes)

            self.paths *= multiplier
            self.bboxes *= multiplier
            self.bbox_classes *= multiplier

            assert len(self.paths) == len(self.bboxes) == len(self.bbox_classes)

        self.split = split
        self.multiplier = multiplier
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.split == 'test':
            path = self.paths[idx]
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image=image)['image'] / 255.0

            return image

        elif self.split == 'train':
            path = self.paths[idx]
            bboxes = self.bboxes[idx]
            bbox_classes = self.bbox_classes[idx]

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=bbox_classes,
                )

                image = transformed['image'] / 255.0
                # bboxes = transformed['bboxes']
                bbox_classes = transformed['class_labels']

            other_label = sum([_ == 'other' for _ in bbox_classes])
            tin_label = sum([_ == 'tin' for _ in bbox_classes])
            thatch_label = sum([_ == 'thatch' for _ in bbox_classes])

            return image, torch.tensor([other_label, tin_label, thatch_label]).unsqueeze(dim=-1).float()

        elif self.split == 'val':
            path = self.paths[idx]
            bbox_classes = self.bbox_classes[idx]

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                transformed = self.transform(image=image)

                image = transformed['image'] / 255.0
                # bboxes = transformed['bboxes']

            other_label = sum([_ == 'other' for _ in bbox_classes])
            tin_label = sum([_ == 'tin' for _ in bbox_classes])
            thatch_label = sum([_ == 'thatch' for _ in bbox_classes])

            return image, torch.tensor([other_label, tin_label, thatch_label]).unsqueeze(dim=-1).float()

        else:
            raise

