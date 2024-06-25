import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import logging
import os

def process_dataset(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    skf = StratifiedGroupKFold(n_splits=cfg.n_splits)

    csv = pd.read_csv(cfg.input_path)

    image_id_to_number_of_bboxes = {}
    for image_id in csv.image_id.unique():
        image_id_to_number_of_bboxes[image_id] = csv[csv.image_id == image_id].shape[0]

    csv['num_bboxes'] = csv['image_id'].apply(lambda x: image_id_to_number_of_bboxes[x])

    for fold_index, (train_index, test_index) in enumerate(skf.split(csv.image_id, y=csv['num_bboxes'], groups=csv.image_id)):
        train_fold = csv.iloc[train_index].reset_index(drop=True)
        val_fold = csv.iloc[test_index].reset_index(drop=True)

        train_fold.to_csv(f'{cfg.output_dir}/train_fold_{fold_index}.csv', index=False)
        val_fold.to_csv(f'{cfg.output_dir}/val_fold_{fold_index}.csv', index=False)

    logging.info(f'fold saved to {cfg.output_dir}')

@hydra.main(config_path="../../configs/preprocessing", config_name="create_folds")
def hydra_run(
    cfg: DictConfig,
) -> None:
    process_dataset(cfg)


if __name__ == "__main__":
    hydra_run()