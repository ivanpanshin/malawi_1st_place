import logging
from typing import Any, Dict

import torch
from sklearn.metrics import mean_absolute_error

from .base import BaseMetric


class Metrics(BaseMetric):
    def __init__(
        self,
        main_metric: str,
        main_metric_direction: str,
        track_val_loss: bool,
    ):
        super().__init__(
            main_metric=main_metric,
            main_metric_direction=main_metric_direction,
            track_val_loss=track_val_loss,
        )

    def calculate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        metrics = {}
        metrics['mae_other'] = mean_absolute_error(y_pred=self.y_pred_other, y_true=self.y_true_other)
        metrics['mae_tin'] = mean_absolute_error(y_pred=self.y_pred_tin, y_true=self.y_true_tin)
        metrics['mae_thatch'] = mean_absolute_error(y_pred=self.y_pred_thatch, y_true=self.y_true_thatch)

        metrics['mae_other_baseline'] = mean_absolute_error(y_pred=[torch.mean(self.y_true_other)]*len(self.y_true_other), y_true=self.y_true_other)
        metrics['mae_tin_baseline'] = mean_absolute_error(y_pred=[torch.mean(self.y_true_tin)]*len(self.y_true_tin), y_true=self.y_true_tin)
        metrics['mae_thatch_baseline'] = mean_absolute_error(y_pred=[torch.mean(self.y_true_thatch)]*len(self.y_true_thatch), y_true=self.y_true_thatch)

        metrics['mae_overall'] = mean_absolute_error(
            y_pred=torch.concat([self.y_pred_other, self.y_pred_tin, self.y_pred_thatch]),
            y_true=torch.concat([self.y_true_other, self.y_true_tin, self.y_true_thatch]),
        )

        metrics['mae_overall_as_a_sum'] = metrics['mae_other'] + metrics['mae_tin'] + metrics['mae_thatch']

        metrics['image_support_other'] = torch.sum(self.y_true_other != 0).item()
        metrics['image_support_tin'] = torch.sum(self.y_true_tin != 0).item()
        metrics['image_support_thatch'] = torch.sum(self.y_true_thatch != 0).item()

        return metrics

    def prepare_predictions_init(
        self,
        trainer,
        loader_name,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # specify correct y_pred_shape and y_true_shape depending on the task
        # after it's done - remove NotImplementedError

        y_pred_shape = len(trainer.loaders[loader_name].dataset)
        y_true_shape = len(trainer.loaders[loader_name].dataset)
        losses_shape = len(trainer.loaders[loader_name].dataset)

        self.y_pred_other = torch.zeros(y_pred_shape)
        self.y_pred_tin = torch.zeros(y_pred_shape)
        self.y_pred_thatch = torch.zeros(y_pred_shape)

        self.y_true_other = torch.zeros(y_true_shape)
        self.y_true_tin = torch.zeros(y_true_shape)
        self.y_true_thatch = torch.zeros(y_true_shape)

        self.losses = torch.zeros(losses_shape)
        self.batch_size = trainer.loaders[loader_name].batch_size

        logging.info(f"Creating empty tensors for preds with the shape of {y_pred_shape}")
        logging.info(f"Creating empty tensors for labels with the shape of {y_true_shape}")
        logging.info(f"Creating empty tensors for losses with the shape of {losses_shape}")
        logging.info(f"Using batch_size {self.batch_size} inside metric")

    def prepare_predictions_batch(
        self,
        batch_index: int,
        preds,
        labels,
        *args: Any, **kwargs: Any
    ) -> None:
        # make sure the predictions and labels are stored correctly
        # after it's done - remove NotImplementedError

        start_index = batch_index * self.batch_size
        end_index = (batch_index + 1) * self.batch_size

        self.y_pred_other[start_index:end_index] = preds['model_predictions']['other_preds'].squeeze()
        self.y_pred_tin[start_index:end_index] = preds['model_predictions']['tin_preds'].squeeze()
        self.y_pred_thatch[start_index:end_index] = preds['model_predictions']['thatch_preds'].squeeze()

        self.y_true_other[start_index:end_index] = labels[:, 0, :].squeeze()
        self.y_true_tin[start_index:end_index] = labels[:, 1, :].squeeze()
        self.y_true_thatch[start_index:end_index] = labels[:, 2, :].squeeze()

        if "loss_item" in preds:
            self.losses[start_index:end_index] = preds["loss_item"]


    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        del self.y_pred_other, self.y_pred_tin, self.y_pred_thatch
        del self.y_true_other, self.y_true_tin, self.y_true_thatch
        del self.losses
