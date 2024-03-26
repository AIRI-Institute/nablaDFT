from typing import Callable, Dict, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.utils import to_dense_batch

from torch import Tensor
import pytorch_lightning as pl


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


class Graphormer3D(nn.Module):
    def __init__(
        self,
        blocks: int,
        layers: int,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        input_dropout: float,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        num_kernel: int,
    ):
        super(Graphormer3D, self).__init__()

        self.blocks = blocks
        self.atom_types = 64
        self.edge_types = 64 * 64
        self.K = num_kernel
        self.atom_encoder = nn.Embedding(self.atom_types, embed_dim, padding_idx=0)
        self.tag_encoder = nn.Embedding(3, embed_dim)
        self.input_dropout = input_dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    embed_dim,
                    ffn_embed_dim,
                    num_attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                )
                for _ in range(layers)
            ]
        )
        self.final_ln = nn.LayerNorm(embed_dim)
        self.energy_proj = NonLinear(embed_dim, 1)
        self.energy_agg_factor = nn.Embedding(3, 1)
        nn.init.normal_(self.energy_agg_factor.weight, 0, 0.01)

        self.gbf = GaussianLayer(self.K, self.edge_types)
        self.bias_proj = NonLinear(self.K, attention_heads)
        self.edge_proj = nn.Linear(self.K, embed_dim)
        self.node_proj = NodeTaskHead(embed_dim, attention_heads)

    def forward(self, data):
        bsz = data.batch.max().detach().item() + 1  # get batch size
        atoms, mask = to_dense_batch(data.z, data.batch, batch_size=bsz)
        pos, _ = to_dense_batch(data.pos, data.batch, batch_size=bsz)
        # NablaDFT dataset doesn't have tags, just fill tags with ones
        tags = torch.ones(atoms.shape, dtype=torch.int).to(atoms.device)
        padding_mask = atoms.eq(0)
        tags[padding_mask] = 0

        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node
        )

        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)

        eng_output = F.dropout(output, p=0.1, training=self.training)
        eng_output = (
            self.energy_proj(eng_output) * self.energy_agg_factor(tags)
        ).flatten(-2)
        output_mask = (
            tags > 0
        ) & mask  # no need to consider padding, since padding has tag 0, real_mask False

        eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)

        node_output = self.node_proj(output, graph_attn_bias, delta_pos)

        node_target_mask = output_mask.unsqueeze(-1)
        return eng_output, node_output, node_target_mask


class Graphormer3DLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        net: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss,
        metric,
        warmup_steps: int,
        energy_loss_coef: float,
        forces_loss_coef: float,
    ) -> None:
        super(Graphormer3DLightning, self).__init__()
        self.save_hyperparameters(logger=True, ignore=["net", "loss"])
        self.net = net
        self.loss = loss

        self.loss_energy_coef = energy_loss_coef
        self.loss_forces_coef = forces_loss_coef

    def forward(self, data):
        energy_out, forces_out, mask_out = self.net(data)
        forces_out *= mask_out
        return energy_out, forces_out, mask_out

    def step(
        self, batch, calculate_metrics: bool = False
    ) -> Union[Tuple[Any, Dict], Any]:
        bsz = self._get_batch_size(batch)  # get batch size
        y = batch.y
        energy_out, forces_out, mask_out = self(batch)
        loss_energy = self.loss(energy_out, y)
        forces, mask_forces = to_dense_batch(
            batch.forces, batch.batch, batch_size=bsz
        )
        masked_forces_out = forces_out * mask_forces.unsqueeze(-1)
        loss_forces = self.loss(masked_forces_out, forces)
        loss = self.loss_forces_coef * loss_forces + self.loss_energy_coef * loss_energy
        if calculate_metrics:
            preds = {"energy": energy_out, "forces": masked_forces_out}
            target = {"energy": y, "forces": forces}
            metrics = self._calculate_metrics(preds, target)
            return loss, metrics
        return loss

    def training_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss = self.step(batch, calculate_metrics=False)
        if self.trainer.global_step <= self.hparams.warmup_steps:
            self._log_current_lr()
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

    def validation_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss, metrics = self.step(batch, calculate_metrics=True)
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

    def test_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
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

    def predict_step(self, data, **kwargs):
        energy_out, forces_out, _ = self(data)
        return energy_out, forces_out

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

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


from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    # from https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/optimization.py#L104
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    # from https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/optimization.py#L98
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )
