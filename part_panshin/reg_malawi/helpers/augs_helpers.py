import albumentations as A
import albumentations.pytorch as AT

import cv2
import numpy as np

from albumentations.core.bbox_utils import convert_bbox_from_albumentations

def train(img_size):
    # Constructing the additional targets for {crop_size} images and masks

    return A.Compose(
        [
            A.ShiftScaleRotate(
                p=0.5,
                shift_limit_x=(0, 0),
                shift_limit_y=(0, 0),
                scale_limit=(0, 0),
                rotate_limit=(0, 0),
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="largest_box",
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.25, 0.25),
                p=0.5,
            ),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            CoarseDropoutBboxes(
                max_holes=20,
                min_holes=5,
                max_height=200,
                min_height=20,
                max_width=200,
                min_width=20,
                max_area_percentage=0.5,
                p=0.5,
            ),
            A.Resize(img_size, img_size),
            AT.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['class_labels'])
    )


def val(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            AT.ToTensorV2(),
        ],
    )



class CoarseDropoutBboxes(A.CoarseDropout):
    def __init__(self, max_area_percentage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_area_percentage = max_area_percentage

    @staticmethod
    def bbox_area(bbox):
        """Calculate the area of a bounding box in Pascal VOC format."""
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)

    @staticmethod
    def intersection_area(bbox1, bbox2):
        """Calculate the intersection area of two bounding boxes in Pascal VOC format."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Determine the coordinates of the intersection rectangle
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        # Calculate the width and height of the intersection rectangle
        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)

        return inter_width * inter_height

    def percentage_area_inside(self, bbox1, bbox2):
        """Calculate the percentage of the area of bbox1 that is inside bbox2 using Pascal VOC format."""
        intersection = self.intersection_area(bbox1, bbox2)
        area1 = self.bbox_area(bbox1)
        if area1 == 0:
            return 0  # To avoid division by zero
        return (intersection / area1)

    def apply_to_bboxes(self, bboxes, *args, **params):
        new_bboxes = []

        for bbox in bboxes:
            bbox_coco = convert_bbox_from_albumentations(
                bbox[:4],
                'coco',
                params['rows'],
                params['cols']
            )
            x1, y1, w, h = bbox_coco
            x2 = x1 + w
            y2 = y1 + h
            bbox_pascal = [x1, y1, x2, y2]

            biggest_per = -np.Inf
            for hole in params['holes']:
                per = self.percentage_area_inside(
                    bbox_pascal,
                    hole,
                )

                if per > biggest_per:
                    biggest_per = per

            if biggest_per < self.max_area_percentage:
                new_bboxes.append(bbox)

        return new_bboxes

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }
