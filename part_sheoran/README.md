# Winning Solution Malawi (Sheoran part)

## Install
```
conda create -n arm_mmdet anaconda python=3.9
conda activate arm_mmdet
pip install -r requirements.txt --no-cache
cd Co-DETR/
pip install -v -e . --no-cache-dir
cd ..
```

Note that installing `mmdet` and `mmcv-full` on different systems can be very tricky. There are 2 things that we can propose:
- In case you cannot install requirements as above (note - this is based on official mmdet documentation), try out the following docker image: `2.0.0-cuda11.7-cudnn8-runtime`.
- Simply reach out at ivan.panshin@protonmail.com and I will solve all the issues for you. 

## Reproduce inference
```
conda version - Anaconda3-2022.05-Linux-x86_64
ubuntu version - 22.04
python version - 3.9
nvcc version - 11.7
gpu model - RTX 3090 Ampere
driver version - 555.42.02
torch version - 1.13
```

0. Download competition data and place it into `data`. In particular:
```
data/
└── Images/
    └── id_*.tif
└── Train.csv
└── Test.csv
```

1. Make a master train file that have information in a more organized way and assign each image to a fold, also splits the data in data/train_images and data/test_images from data/Images folder for better understanding 

```
python initial_data_prep.py
```

2. Create data for MMDET in json file, I only use Fold 0 as it takes long time to train

```
python create_mmdet_data.py
```

3. Download final checkpoints

```
kaggle datasets download -d harshitsheoran/malawi-final-weights-sheoran
unzip malawi-final-weights-sheoran.zip -d final_weights/
```

4. Compute test predictions

```
python inference_mmdet.py
```
Submission will be saved at `submission.csv`.

## Reproduce Training

Note that to reproduce the training, you need `4xRTX 3090`, as that configuration was used to create the solution in the first place.

0. Make sure that Steps to install are followed

We start training model using mmdetection

```
cd Co-DETR/

```

We need to download pretrained checkpoints trained on coco, available publically in https://github.com/Sense-X/Co-DETR/tree/main 

```
bash download_pretrain.sh
```

Because exp_002 model is trained with 3 classes, it uses mmdet_data_v2 (created from create_mmdet_data.py), to mod mmdet accordingly

```
bash mod_mmdet_for_3_classes.sh
```

```
bash tools/dist_train.sh ./work_dirs/exp_002/co_deformable_detr_swin_base_3x_coco.py 4 ./work_dirs/exp_002_re/ --deterministic --seed 0
```

Because exp_004 is trained with only Tin and Thatch class, let's mod mmdet to support 2 classes

```
bash mod_mmdet_for_2_classes.sh
```

```
bash tools/dist_train.sh ./work_dirs/exp_004/co_dino_5scale_swin_large_16e_o365tococo.py 4 ./work_dirs/exp_004_re/ --deterministic --seed 0
```
