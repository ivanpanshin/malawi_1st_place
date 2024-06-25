# Winning Solution Malawi (Panshin part)

## Install 
1. Create a clean virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.dev.txt
pip install -r requirements.txt
```

## Reproduce inference
Note that pipeline is tested in the following settings:
```
ubuntu version - 22.04
python version - 3.10.9
nvcc version - 11.6
gpu model - A6000 Ada
driver version - 545.23.08
torch version - 2.3.1
```

The correct running with different system settings are not guaranteed. However, most likely minor differences in versions won't be a problem for inference.

0. The repo already comes with preprocessed annotations. So initially it looks like this:
```
data/
└── annotations/
    └── test.csv
    └── train_fold_*.csv
    └── val_fold_*.csv
```
1. Download competition data and place it into `data`. In particular:
```
data/
└── annotations/
    └── test.csv
    └── train_fold_*.csv
    └── val_fold_*.csv
└── Images/
    └── id_*.tif
└── Train.csv
└── Test.csv
```
2. Download final checkpoint 
```
kaggle datasets download -d ivanpan/malawi-final-weights-panshin
unzip malawi-final-weights-panshin.zip -d final_weights/
```

3. Compute test predictions 
```
torchrun --nproc_per_node=1 --master-port=29500 reg_malawi/test.py
```
Submission will be saved at `final_subs/maxvit_fold0.csv`.

## Reproduce training
Note that to reproduce the training, you need `4xA6000 Ada`, as that configuration was used to create the solution in the first place.

0. Make sure the data is downloaded based on previous section 
1. Create folds (optional, already part of the repo)
```
python reg_malawi/preprocessing/create_folds.py
```
2. Run mlflow 
```
bash bash_scripts/run_mlflow.sh
```
3. Train the model
```
torchrun --nproc_per_node=4 --master-port=29500 reg_malawi/train.py scheduler.scheduler.T_max=36000 logging.run_name=maxvit 
```
You can check the training logs at `http://localhost:7053`. 
