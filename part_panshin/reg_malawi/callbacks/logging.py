import logging
import os
from glob import glob
import time
import subprocess

import mlflow
#import pysftp

from .base import Callback

logging.getLogger("paramiko.transport").setLevel(logging.WARNING)


class LogMetrics_Mlflow(Callback):
    def __init__(self):
        super().__init__()

#    @staticmethod
    def run_only_on_cuda0(func):
        def wrapper(self, trainer, *args, **kwargs):
            if trainer.device == "cuda:0":
                return func(self, trainer, *args, **kwargs)
        return wrapper

#    @staticmethod
    def _log_metrics(self, trainer, metrics, prefix, step):
        mlflow.log_metrics({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

#    @staticmethod
    def _log_transforms(self, trainer, loader_name):
        if hasattr(trainer.loaders[loader_name].dataset, 'transform') and \
        trainer.loaders[loader_name].dataset.transform:
            k = f'{loader_name}_transform'
            v = str(trainer.loaders[loader_name].dataset.transform)
            if len(str(v)) <= 500:
                mlflow.log_params({k: v})
            else:
                for i in range(len(str(v)) // 500 + 1):
                    mlflow.log_params({f"{k}_{i}": str(v)[i * 500 : (i + 1) * 500]})

    # @staticmethod
    # def upload_to_sftp(
    #     local_path,
    #     remote_path
    # ):
    #     cnopts = pysftp.CnOpts()
    #     cnopts.hostkeys = None
    #     private_key_path = os.path.expanduser('~/.ssh/id_rsa')
    #
    #     tracking_uri = mlflow.get_tracking_uri().split('//')[-1].split(':')[0]
    #
    #     with pysftp.Connection(host=tracking_uri, username='ivan', private_key=private_key_path, cnopts=cnopts) as sftp:
    #         remote_dir = os.path.dirname(remote_path)
    #         sftp.makedirs(remote_dir)
    #
    #         sftp.put(local_path, remote_path)

    def _log_code(self):
        local_paths, remote_paths = [], []
        # add trainer to logging
        local_trainer_paths = sorted(glob(f"reg_malawi/trainer/*.py"))
        remote_trainer_paths = [f'trainer/{_.split("/")[-1]}' for _ in local_trainer_paths]
        local_paths.extend(local_trainer_paths)
        remote_paths.extend(remote_trainer_paths)

        # add datasets to logging
        local_dataset_file = 'reg_malawi/helpers/dataset_helpers.py'
        remote_dataset_file = f'datasets/dataset_helpers.py'
        local_paths.append(local_dataset_file)
        remote_paths.append(remote_dataset_file)

        # add optimizers to logging
        local_dataset_file = 'reg_malawi/helpers/optim_helpers.py'
        remote_dataset_file = f'optim/optim_helpers.py'
        local_paths.append(local_dataset_file)
        remote_paths.append(remote_dataset_file)

        # add metrics to logging
        local_metrics_paths = sorted(glob(f"reg_malawi/metrics/*.py"))
        remote_metrics_paths = [f'metrics/{_.split("/")[-1]}' for _ in local_metrics_paths]
        local_paths.extend(local_metrics_paths)
        remote_paths.extend(remote_metrics_paths)

        # add callbacks to logging
        local_callbacks_paths = sorted(glob(f"reg_malawi/callbacks/*.py"))
        remote_callbacks_paths = [f'callbacks/{_.split("/")[-1]}' for _ in local_callbacks_paths]
        local_paths.extend(local_callbacks_paths)
        remote_paths.extend(remote_callbacks_paths)

        # add models to logging
        local_models_file = 'reg_malawi/helpers/model_helpers.py'
        remote_models_file = f'models/model_helpers.py'
        local_paths.append(local_models_file)
        remote_paths.append(remote_models_file)

        # add loaders to logging
        local_loaders_file = 'reg_malawi/helpers/dataloader_helpers.py'
        remote_loaders_file = f'loaders/dataloader_helpers.py'
        local_paths.append(local_loaders_file)
        remote_paths.append(remote_loaders_file)

        # add augs to logging
        local_augs_file = 'reg_malawi/helpers/augs_helpers.py'
        remote_augs_file = f'augs/augs_helpers.py'
        local_paths.append(local_augs_file)
        remote_paths.append(remote_augs_file)

        for local_path, remote_path in zip(local_paths, remote_paths):
            mlflow.log_artifact(local_path, artifact_path=remote_path)
            # self.upload_to_sftp(
            #     local_path=local_path,
            #     remote_path=remote_path
            # )

    @run_only_on_cuda0
    def on_init_end(self, trainer):
        mlflow.set_tracking_uri(trainer.cfg.logging.tracking_url)
        mlflow.set_experiment(trainer.cfg.logging.experiment_name)

        mlflow.start_run(run_name=trainer.cfg.logging.run_name)
        self.artifact_uri = mlflow.active_run().info.artifact_uri

        for k, v in dict(trainer.cfg).items():
            if len(str(v)) <= 500:
                mlflow.log_params({k: v})
            else:
                for i in range(len(str(v)) // 500 + 1):
                    mlflow.log_params({f"{k}_{i}": str(v)[i * 500 : (i + 1) * 500]})

        val_loaders_names = [key for key in trainer.loaders if key.startswith("val_")]
        # self._log_transforms(trainer=trainer, loader_name="train")
        # self._log_transforms(trainer=trainer, loader_name=val_loaders_names[0])
        self._log_code()


    @run_only_on_cuda0
    def on_fit_end(self, trainer):
        if hasattr(trainer, "best_epoch"):
            mlflow.log_metric("best_epoch", trainer.best_epoch)

        mlflow.end_run()

    @run_only_on_cuda0
    def on_epoch_end(self, trainer):
        logging.info("Uploading all models to MLFlow")
        for file_path in glob(f'{trainer.cfg.logging.logging_dir}/*.pt'):
            mlflow.log_artifact(file_path)
            os.remove(file_path)

        if trainer.cfg.logging.track_train_metrics:
            self._log_metrics(
                trainer=trainer, metrics=trainer.train_metrics, prefix="train", step=trainer.current_epoch
            )
        self._log_metrics(trainer=trainer, metrics=trainer.val_metrics, prefix="val", step=trainer.current_epoch)
        logging.info(f"Epoch {trainer.current_epoch} Train {trainer.train_metrics} Val {trainer.val_metrics}")

    @run_only_on_cuda0
    def on_logging_by_iter(self, trainer):
        self._log_metrics(trainer=trainer, metrics=trainer.logging_stats, prefix="train", step=trainer.current_iter)

        info = f"Epoch {trainer.current_epoch} Iter {trainer.current_iter} Stats {trainer.logging_stats}"
        if hasattr(trainer, "eta"):
            info = f"ETA {trainer.eta} hours " + info
        logging.info(info)


class CalculateETA(Callback):
    def __init__(self):
        super().__init__()

    def on_init_end(self, trainer):
        trainer.start_time = time.time()

    def on_logging_by_iter(self, trainer):
        if trainer.current_iter > 0:
            total_number_iters = trainer.cfg.trainer.trainer_hyps.num_epochs * len(trainer.loaders["train"])
            elapsed_time = time.time() - trainer.start_time
            time_per_iter = elapsed_time / trainer.current_iter
            # counting the validation time as 10% of training epoch time
            eta_remaining_iters = 1.1 * (total_number_iters - trainer.current_iter) * time_per_iter

            trainer.eta = round(eta_remaining_iters / 3600, 2)
