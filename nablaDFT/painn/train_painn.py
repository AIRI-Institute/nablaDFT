import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.representation as rep
import schnetpack.transform as trn
import torch
import torchmetrics
from schnetpack.model.base import AtomisticModel
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('../')
from dataset.nablaDFT import nablaDFT


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AtomisticTaskFixed(spk.task.AtomisticTask):
    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[spk.task.ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        pl.LightningModule.__init__(self)
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=['model'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PAINN training')
    parser.add_argument('--dataset_name',  type=str,
                        help='dataset name',
                        default='dataset_train_2k')
    parser.add_argument('--datapath',  type=str,
                        help='path to data',
                        default='database')
    parser.add_argument('--logspath',  type=str,
                        help='path to logs',
                        default='logs/model_moses_10k_split')
    parser.add_argument('--nepochs',  type=int, default=2000,
                        help='epochs number')
    parser.add_argument('--seed',  type=int, default=1799,
                        help='random seed')
    parser.add_argument('--batch_size',  type=int, default=2000,
                        help='batch size')
    parser.add_argument('--n_interactions', type=int, default=6,
                        help='interactions number')
    parser.add_argument('--n_atom_basis', type=int, default=128,
                        help='atom basis number')
    parser.add_argument('--n_rbf', type=int, default=20,
                        help='rbf number')
    parser.add_argument('--cutoff', type=float, default=5.0,
                        help='cutoff threshold')
    parser.add_argument('--devices', type=int, default=1,
                        help='gpu/tpu/cpu devices number')

    args, unknown = parser.parse_known_args()
    seed_everything(args.seed)
    workpath = args.logspath
    if not os.path.exists(workpath):
        os.makedirs(workpath)

    data = nablaDFT("ASE", args.dataset_name,
                    datapath=args.datapath,
                    data_workdir=workpath,
                    batch_size=args.batch_size,
                    num_workers=4,
                    transforms=[
                     trn.ASENeighborList(cutoff=args.cutoff),
                     trn.RemoveOffsets("energy", remove_mean=True,
                                       remove_atomrefs=False),
                     trn.CastTo32()
                    ],
                    split_file=os.path.join(workpath, "split.npz"))

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.radial.GaussianRBF(n_rbf=args.n_rbf,
                                             cutoff=args.cutoff)
    cutoff_fn = spk.nn.cutoff.CosineCutoff(args.cutoff)
    representation = rep.PaiNN(n_interactions=args.n_interactions,
                               n_atom_basis=args.n_atom_basis,
                               radial_basis=radial_basis,
                               cutoff_fn=cutoff_fn)
    pred_energy = spk.atomistic.Atomwise(n_in=representation.n_atom_basis,
                                         output_key="energy")
    postprocessors = [trn.CastTo64(), trn.AddOffsets("energy", add_mean=True)]
    nnpot = spk.model.NeuralNetworkPotential(representation=representation,
                                             input_modules=[pairwise_distance],
                                             output_modules=[pred_energy],
                                             postprocessors=postprocessors)
    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    scheduler_args = {"factor": 0.8, "patience": 10, "min_lr": 1e-06}

    task = AtomisticTaskFixed(
        model=nnpot,
        outputs=[output_energy],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
        scheduler_cls=ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        scheduler_monitor="val_loss"
    )

    # create trainer
    logger = pl.loggers.TensorBoardLogger(save_dir=workpath)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(workpath, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=workpath,
        max_epochs=args.nepochs,
        )
    trainer.fit(task, datamodule=data)
