## Run configuration
Overall pipeline heavily inspired by https://github.com/ashleve/lightning-hydra-template  
Run command from repo root:
```bash
python run.py --config-name <config_name>.yaml
```
where config_name is one of task yaml-configs from `config` directory.  
Type of run defined with `job_type` parameter and it must be one of:
- train
- test
- predict
- optimize  
  
Each config consists of global variables and section with other trainer parts:  
- datamodule
- model
- callbacks
- loggers
- trainer  

### Datamodule

Datamodule config defines type of dataset (ASE, Hamiltonian, PyG), dataset root path, batch size, train/val ratio for training job.  
Example configurations for ASE and PyG dataset types are stored in `configs/datamodule/`.

### Model

Model config defines hyperparameters for chosen model architecture together with metrics and losses. See examples from `config/models/`.  
To add another model you need to define `LightningModule` (see examples from `nablaDFT/`) and pass model config to run configuration.

### Callbacks

By default we use `ModelCheckpoint` and `EarlyStopping` callbacks, you may add desired callbacks with `config/callbacks/default.yaml`.

### Loggers

By default we use solely `WandbLogger`, you may add other loggers.

### Trainer

Defines additional parameters for trainer like accelerator, strategy and devices.

## Train
Example: `config/gemnet-oc.yaml`  
Basically you may need to change `dataset_name` parameter in order to pick one of nablaDFT training split. Available training splits could be found in `nablaDFT/links/energy_databases.json`.  
In case of training resume, just specify checkpoint path in `ckpt_path` parameter.

## Test

Example: `config/gemnet-oc_test.yaml`  
Same as for training run, you may need to change `dataset_name` parameter for desired test split.  
To reproduce test results for pretrained on biggest training dataset split (100k) set `pretrained` parameter to `True` with ckpt_path to `null`. Otherwise, specify path to checkpoint with pretrained model in `ckpt_path`.  

## Predict

Exmaple: `config/gemnet-oc_predict.yaml`  
To obtain model preidctions for another datasets use `job_type: predict`. Specify dataset path relative to `root` or `datapath` parameter from datamodule config.  
Predictions will be stored in `predictions/{model_name}_{dataset_name}.pt`

## Optimize

Examples: `config/gemnet-oc_optim.yaml`, `schnet_optim.yaml`  
`job_type: optimize` stands for molecule geometry optimization with pretrained model.  
Molecules from `input_db` parameter will be optimized using pretrained checkpoint from `ckpt_path`.  
Currently only LBFGS optimizer supported.  
Results will be saved at `output_db` parameter path.