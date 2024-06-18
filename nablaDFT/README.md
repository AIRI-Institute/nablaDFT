## Run configuration
Pipeline heavily inspired by https://github.com/ashleve/lightning-hydra-template
Run command from repo root:
```bash
python run.py --config-name <config_name>.yaml
```
where config_name is one of task yaml-configs from `config` directory.
Type of run defined with `job_type` parameter and it must be one of:
- train
- test
- predict
- optimize (WIP)

Each config consists of global variables:
- `name`: defines run name, usually you should put model name.
- `dataset_name`: in the case of nablaDFT splits, here must be the split name. If you plan to use other dataset,
you should write database file name without extension.
- `max_steps`: defines maximum number of steps for `pytorch_lightning.Trainer`.
- `warmup_steps`: defines number of warmup steps for model.
- `job_type`: defines task type, must be one of `["train", "test", "predict"]`.
- `pretrained`: by default is `null`, change if you want to use one of nablaDFT's pretrained models.
See [Pretrained models](#Pretrained-models) section.
> NOTE: currently PhiSNet and SchNOrb model checkpoints can't be used with `pretrained` parameter.
- `ckpt_path`: absolute or relative path to model checkpoint.
> WARNING: `ckpt_path` and `pretrained` are mutually exclusive parameters, one of them should be used.
- `root`: path to directory with existing dataset or location for downloading dataset.
> NOTE: Torch Geometric puts raw dataset in `raw/` subfolder and processed in `processed/`. `root` parameter must not
> include one of this subfolder to prevent re-downloading. See [Datamodule](#datamodule).
- `batch_size`: number of samples in one batch.
- `num_workers`: number of workers for samples loading.
- `devices`: number of GPUs or devices id.
- `gradient_clip_algorithm/gradient_clip_val`: gradient clipping parameters for `pytorch_lightning.Trainer`,
null by default.

Section with `defaults` parts:
- datamodule
- model
- callbacks
- loggers
- trainer

### Change run configuration from train to test

Example train config could be easily switched to test config. Change values in `defaults` section:
```
defaults:
    - datamodule: config-name.yaml -> config-name_test.yaml
    - trainer: train.yaml -> test.yaml
```

### Datamodule
[PyG Example](../config/datamodule/nablaDFT_pyg.yaml)
[ASE Example](../config/datamodule/nablaDFT_ase.yaml)
[Hamiltonian Example](../config/datamodule/nablaDFT_hamiltonian.yaml)
Datamodule config defines type of dataset (ASE, Hamiltonian, PyG), dataset root path, batch size, train/val ratio for training job.
Note: for ASE type `datapath` parameter must direct to directory with database. For PyG type `root` parameter should direct to directory
with `raw/` subfolder with database. Processed PyG dataset will be stored in the same directory in `processed/` folder.

### Model
[Example](../config/model/gemnet-oc.yaml)
Model config defines hyperparameters for chosen model architecture together with metrics and losses. See examples from `config/models/`.
To add another model you need to define `LightningModule` (see examples from `nablaDFT/`) and pass model config to run configuration.

### Callbacks
[Example](../config/callbacks/default.yaml)
By default we use `ModelCheckpoint` and `EarlyStopping` callbacks, you may add desired callbacks
with [callbacks config file](../config/callbacks/default.yaml).

### Loggers
[Example](../config/loggers/wandb.yaml)
By default we use solely `WandbLogger`, you may add other loggers
in [loggers config file](../config/callbacks/default.yaml).

### Trainer
[Train](../config/trainer/train.yaml)
[Test](../config/trainer/test.yaml)
Defines additional parameters for trainer like accelerator, strategy and devices.

## Train

[Example](../config/gemnet-oc.yaml)
Basically you may need to change `dataset_name` parameter in order to pick one of nablaDFT training split.
List of available splits could be obtained with:
```python
from nablaDFT.dataset import dataset_registry
dataset_registry.list_datasets("energy")  # for energy databases
dataset_registry.list_datasets("hamiltonian")  # for energy databases
```
In the case of training resume, just specify checkpoint path in `ckpt_path` parameter.

## Test

[Example](../config/gemnet-oc_test.yaml)
Same as for training run, you need to change `dataset_name` parameter for desired test split.
To reproduce test results for pretrained checkpoints set `pretrained` parameter to desired name
([see Pretrained models](#Pretrained-models)) with ckpt_path to `null`.
Otherwise, specify path to checkpoint in `ckpt_path`.

## Predict

[Example](../config/gemnet-oc_predict.yaml)
To obtain model preidctions for another datasets use `job_type: predict`.
Specify dataset path relative to `root` or `datapath` parameter from datamodule config.
Predictions will be stored in database `./predictions/{model_name}_{dataset_name}.db`.
Interactive example could be found [here](../examples/Inference%20example.ipynb).

## Optimize
> WARNING: currently optimize pipeline under construction, please,
> use [GOLF_schnetpack](https://github.com/AIRI-Institute/GOLF/blob/nabla2DFT-eval)
> and [GOLF_PYG](https://github.com/AIRI-Institute/GOLF/blob/nabla2DFT-eval-dimenet)
> for the optimization metrics reproduction.

[Example for PyG model](../config/gemnet-oc_optim.yaml)
[Example for ASE](../config/schnet_optim.yaml)
`job_type: optimize` stands for molecule geometry optimization with pretrained model.
Molecules from `input_db` parameter will be optimized using pretrained checkpoint from `ckpt_path`.
Currently only LBFGS optimizer supported.
Results will be saved at `output_db` parameter path.


## Pretrained models
