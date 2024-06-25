import os
import random

import numpy as np
import torch
import torch.nn as nn
import timm


def seed_everything(seed=100500):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RegAggregated(nn.Module):
    def __init__(
        self,
        model_name,
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            model_name=model_name,
            in_chans=3,
            pretrained=True,
            num_classes=0,
        )

        self.other_head = torch.nn.Linear(
            self.encoder.num_features,
            1,
        )

        self.tin_head = torch.nn.Linear(
            self.encoder.num_features,
            1,
        )

        self.thatch_head = torch.nn.Linear(
            self.encoder.num_features,
            1,
        )

    def forward(self, x):
        x = self.encoder(x)

        other_preds = self.other_head(x)
        tin_preds = self.tin_head(x)
        thatch_preds = self.thatch_head(x)

        return {
            'other_preds': other_preds,
            'tin_preds': tin_preds,
            'thatch_preds': thatch_preds,
        }
