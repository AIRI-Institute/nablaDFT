from typing import Dict, Union, Tuple, Any, Optional, Type

import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.data import Data


def swish(x):
    return x * x.sigmoid()


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()


class DimeNetPlusPlusPotential(nn.Module):
    def __init__(
        self,
        node_latent_dim: int,
        scaler=None,
        dimenet_hidden_channels=128,
        dimenet_num_blocks=4,
        dimenet_int_emb_size=64,
        dimenet_basis_emb_size=8,
        dimenet_out_emb_channels=256,
        dimenet_num_spherical=7,
        dimenet_num_radial=6,
        dimenet_max_num_neighbors=32,
        dimenet_envelope_exponent=5,
        dimenet_num_before_skip=1,
        dimenet_num_after_skip=2,
        dimenet_num_output_layers=3,
        cutoff=5.0,
        do_postprocessing=False,
    ):
        super().__init__()
        self.scaler=scaler

        self.node_latent_dim = node_latent_dim
        self.dimenet_hidden_channels = dimenet_hidden_channels
        self.dimenet_num_blocks = dimenet_num_blocks
        self.dimenet_int_emb_size = dimenet_int_emb_size
        self.dimenet_basis_emb_size = dimenet_basis_emb_size
        self.dimenet_out_emb_channels = dimenet_out_emb_channels
        self.dimenet_num_spherical = dimenet_num_spherical
        self.dimenet_num_radial = dimenet_num_radial
        self.dimenet_max_num_neighbors = dimenet_max_num_neighbors
        self.dimenet_envelope_exponent = dimenet_envelope_exponent
        self.dimenet_num_before_skip = dimenet_num_before_skip
        self.dimenet_num_after_skip = dimenet_num_after_skip
        self.dimenet_num_output_layers = dimenet_num_output_layers
        self.cutoff = cutoff

        self.linear_output_size = 1

        self.scaler = scaler
        self.do_postprocessing = do_postprocessing

        self.net = DimeNetPlusPlus(
            hidden_channels=self.dimenet_hidden_channels,
            out_channels=self.node_latent_dim,
            num_blocks=self.dimenet_num_blocks,
            int_emb_size=self.dimenet_int_emb_size,
            basis_emb_size=self.dimenet_basis_emb_size,
            out_emb_channels=self.dimenet_out_emb_channels,
            num_spherical=self.dimenet_num_spherical,
            num_radial=self.dimenet_num_radial,
            cutoff=self.cutoff,
            max_num_neighbors=self.dimenet_max_num_neighbors,
            envelope_exponent=self.dimenet_envelope_exponent,
            num_before_skip=self.dimenet_num_before_skip,
            num_after_skip=self.dimenet_num_after_skip,
            num_output_layers=self.dimenet_num_output_layers,
        )

        regr_or_cls_input_dim = self.node_latent_dim
        self.regr_or_cls_nn = nn.Sequential(
            nn.Linear(regr_or_cls_input_dim, regr_or_cls_input_dim),
            Swish(),
            nn.Linear(regr_or_cls_input_dim, regr_or_cls_input_dim // 2),
            Swish(),
            nn.Linear(regr_or_cls_input_dim // 2, regr_or_cls_input_dim // 2),
            Swish(),
            nn.Linear(regr_or_cls_input_dim // 2, self.linear_output_size),
        )

    @torch.enable_grad()
    def forward(self, data: Data):
        pos, atom_z, batch_mapping = data.pos, data.z, data.batch
        pos = pos.requires_grad_(True)
        graph_embeddings = self.net(pos=pos, z=atom_z, batch=batch_mapping)
        predictions = torch.flatten(self.regr_or_cls_nn(graph_embeddings).contiguous())
        forces = -1 * (
            torch.autograd.grad(
                predictions,
                pos,
                grad_outputs=torch.ones_like(predictions),
                create_graph=self.training,
            )[0]
        )

        if self.scaler and self.do_postprocessing:
            predictions = self.scaler["scale_"] * predictions + self.scaler["mean_"]
        return predictions, forces


class DimeNetPlusPlusLightning(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss,
        metric,
        energy_loss_coef: float,
        forces_loss_coef: float,
        monitor_loss: str = "val/loss",
        model_name: str = None,
        lr_scheduler: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Type] = None,
    ):

        super().__init__()
        self.save_hyperparameters(logger=True, ignore=["net", "loss"])

        self.net = net
        self.scheduler_args = scheduler_args
        self.monitor_loss = monitor_loss
        self.loss = loss

        self.loss_energy_coef = energy_loss_coef
        self.loss_forces_coef = forces_loss_coef

    def forward(self, data: Data):
        return self.net(data)

    def step(
        self, batch, calculate_metrics: bool = False
    ) -> Union[Tuple[Any, Dict], Any]:
        predictions_energy, predictions_forces = self.forward(batch)
        loss_energy = self.loss(predictions_energy, batch.y)
        loss_forces = self.loss(predictions_forces, batch.forces)
        loss = self.loss_forces_coef * loss_forces + self.loss_energy_coef * loss_energy
        if calculate_metrics:
            preds = {"energy": predictions_energy, "forces": predictions_forces}
            target = {"energy": batch.y, "forces": batch.forces}
            metrics = self._calculate_metrics(preds, target)
            return loss, metrics
        return loss

    def training_step(self, batch, batch_idx: int):
        bsz = self._get_batch_size(batch)
        loss = self.step(batch, calculate_metrics=False)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        bsz = self._get_batch_size(batch)
        loss, metrics = self.step(batch, calculate_metrics=True)
        self._log_current_lr()
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        # workaround for checkpoint callback
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def test_step(self, batch, batch_idx: int):
        bsz = self._get_batch_size(batch)
        with torch.enable_grad():
            loss, metrics = self.step(batch, calculate_metrics=True)
        self.log(
            "test/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def predict_step(self, batch):
        with torch.enable_grad():
            predictions = self(batch)
        return predictions

    def configure_optimizers(self):
        opt = self.hparams.optimizer(self.parameters())
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=opt, **self.scheduler_args)
        else:
            scheduler = None
        return {
            "optimizer": opt,
            "monitor": self.monitor_loss,
            "lr_scheduler": scheduler,
        }

    def on_fit_start(self) -> None:
        self._check_metrics_devices()

    def on_test_start(self) -> None:
        self._check_metrics_devices()

    def on_validation_epoch_end(self) -> None:
        self._reduce_metrics(step_type="val")

    def on_test_epoch_end(self) -> None:
        self._reduce_metrics(step_type="test")

    def _calculate_metrics(self, y_pred, y_true) -> Dict:
        """Function for metrics calculation during step."""
        metric = self.hparams.metric(y_pred, y_true)
        return metric

    def _log_current_lr(self) -> None:
        opt = self.optimizers()
        current_lr = opt.optimizer.param_groups[0]["lr"]
        self.log("LR", current_lr, logger=True)

    def _reduce_metrics(self, step_type: str = "train"):
        metric = self.hparams.metric.compute()
        for key in metric.keys():
            self.log(
                f"{step_type}/{key}",
                metric[key],
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.hparams.metric.reset()

    def _check_metrics_devices(self):
        self.hparams.metric = self.hparams.metric.to(self.device)

    def _get_batch_size(self, batch):
        """Function for batch size infer."""
        bsz = batch.batch.max().detach().item() + 1  # get batch size
        return bsz
