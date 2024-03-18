from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from torch_cluster import radius_graph
from e3nn import o3
from e3nn.o3 import Linear
import pytorch_lightning as pl

from .layers import ExponentialBernsteinRadialBasisFunctions, ConvNetLayer, PairNetLayer, SelfNetLayer, Expansion, get_nonlinear


class QHNet(nn.Module):
    """Modified QHNet from paper
    Args:
        orbitals (Dict): defines orbitals for each atom type from the dataset.
    """
    def __init__(self,
                 in_node_features=1,
                 sh_lmax=4,
                 hidden_size=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=12,
                 num_nodes=10,
                 radius_embed_dim=32, # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
                 orbitals: Dict = None):
        super(QHNet, self).__init__()
        # store hyperparameter values
        self.order = sh_lmax
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)
        self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e')
        self.hidden_bottle_irrep = o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e')
        self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e') # in use
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.num_fc_layer = 1

        orbital_mask, max_s, max_p, max_d = self._get_mask(orbitals)  # max_* used below to define output representation
        self.orbital_mask = orbital_mask

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        self.udpate_layer = nn.ModuleList()
        self.start_layer = 2
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(ConvNetLayer(
                irrep_in_node=input_irrep,
                irrep_hidden=self.hidden_irrep,
                irrep_out=self.hidden_irrep,
                edge_attr_dim=self.radius_embed_dim,
                node_attr_dim=self.hs,
                sh_irrep=self.sh_irrep,
                resnet=True,
                use_norm_gate=True if i != 0 else False
            ))

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=True,
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=True,
                ))

        self.nonlinear_layer = get_nonlinear('ssp')
        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = \
            nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for name in {"hamiltonian"}:
            input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(
                input_expand_ii,
                o3.Irreps(f"{max_s}x0e + {max_p}x1e + {max_d}x2e"), # here we define which basis we use
                o3.Irreps(f"{max_s}x0e + {max_p}x1e + {max_d}x2e")  # here we define which basis we use
            )
            self.fc_ii[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_path_weight)
            )
            self.fc_ii_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_bias)
            )
            self.expand_ij[name] = Expansion(
                o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e'),
                o3.Irreps(f"{max_s}x0e + {max_p}x1e + {max_d}x2e"),  # here we define which basis we use
                o3.Irreps(f"{max_s}x0e + {max_p}x1e + {max_d}x2e")  # here we define which basis we use
            )

            self.fc_ij[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def set(self):
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, keep_blocks=False):
        node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(data, self.max_radius)
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh

        _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = \
            self.build_graph(data, max_radius=10000)

        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = \
            full_edge_index, full_edge_attr, full_edge_sh

        full_dst, full_src = data.full_edge_index

        fii = None
        fij = None
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](data, node_attr, fij)

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](data.node_attr), self.fc_ii_bias['hamiltonian'](data.node_attr))
        node_pair_embedding = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))
        if keep_blocks is False:
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix)
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)
            return hamiltonian_matrix
        else:
            ret_hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix +\
                                          hamiltonian_diagonal_matrix.transpose(-1, -2)

            # the transpose should considers the i, j
            ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + \
                      hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)

            results = {}
            results['hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix
            results['hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix
        return results

    def build_graph(self, data, max_radius, edge_index=None):
        node_attr = data.z.squeeze()

        if edge_index is None:
            radius_edges = radius_graph(data.pos, max_radius, data.batch, max_num_neighbors=data.num_nodes)
        else:
            radius_edges = data.full_edge_index

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.distance_expansion(edge_vec.norm(dim=-1).unsqueeze(-1)).squeeze().type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        # concate the blocks together and then select once.
        final_matrix = []
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                    if src_idx == dst_idx:
                        matrix_col.append(diagonal_matrix[src_idx].index_select(
                            -2, self.orbital_mask[data.z[dst_idx].item()]).index_select(
                            -1, self.orbital_mask[data.z[src_idx].item()])
                        )
                    else:
                        mask1 = (src == src_idx)
                        mask2 = (dst == dst_idx)
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index].index_select(
                                -2, self.orbital_mask[data.z[dst_idx].item()]).index_select(
                                -1, self.orbital_mask[data.z[src_idx].item()]))
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        final_matrix = torch.block_diag(*final_matrix)
        return final_matrix
    
    def _get_mask(self, orbitals):
        # get orbitals by z
        # retrieve max orbital and get ranges for mask
        max_z = max(orbitals.keys())
        _, counts = np.unique(orbitals[max_z], return_counts=True)
        s_max, p_max, d_max = counts # max orbital number per type
        s_range = [i for i in range(s_max)]
        p_range = [i + max(s_range) + 1 for i in range(p_max * 3)]
        d_range = [i + max(p_range) + 1 for i in range(d_max * 5)]
        ranges = [s_range, p_range, d_range]
        orbs_count = [1, 3, 5] # orbital count per type
        # create mask for each atom type
        atom_orb_masks = {}
        for atom in orbitals.keys():
            _, orb_count = np.unique(orbitals[atom], return_counts=True)
            mask = []
            for idx, val in enumerate(orb_count):
                mask.extend(ranges[idx][:orb_count[idx] * orbs_count[idx]])
            atom_orb_masks[atom] = torch.tensor(mask)
        return atom_orb_masks, s_max, p_max, d_max


class QHNetLightning(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            net: nn.Module,
            optimizer: Optimizer,
            lr_scheduler: LRScheduler,
            losses: Dict,
            ema,
            metric,
            loss_coefs,
    ) -> None:
        super(QHNetLightning, self).__init__()
        self.net = net
        self.ema = ema
        self.save_hyperparameters(logger=True, ignore=['net'])

    def forward(self, data: Data):
        hamiltonian = self.net(data)
        return hamiltonian
    
    def step(self, batch, calculate_metrics: bool = False):
        hamiltonian_out = self.net(batch)
        hamiltonian = batch.hamiltonian
        preds = {'hamiltonian': hamiltonian_out}
        masks = torch.block_diag(*[torch.ones_like(torch.from_numpy(H)) for H in hamiltonian])
        hamiltonian = torch.block_diag(*[torch.from_numpy(H) for H in hamiltonian]).to(self.device)
        target = {'hamiltonian': hamiltonian}
        loss = self._calculate_loss(preds, target, masks)
        if calculate_metrics:
            metrics = self._calculate_metrics(preds, target, masks)
            return loss, metrics
        return loss

    def training_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss = self.step(batch, calculate_metrics=False)
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
        with self.ema.average_parameters():
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
        with self.ema.average_parameters():
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

    def predict_step(self, data):
        hamiltonian = self(data)
        return hamiltonian

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.ema.update()

    def on_fit_start(self) -> None:
        self._instantiate_ema()
        self._check_devices()

    def on_test_start(self) -> None:
        self._instantiate_ema()
        self._check_devices()

    def on_validation_epoch_end(self) -> None:
        self._reduce_metrics(step_type="val")

    def on_test_epoch_end(self) -> None:
        self._reduce_metrics(step_type="test")

    def _calculate_loss(self, y_pred, y_true, masks) -> float:
        total_loss = 0.0
        for name, loss in self.hparams.losses.items():
            total_loss += self.hparams.loss_coefs[name] * loss(
                y_pred[name], y_true[name], masks
            )
        return total_loss

    def _calculate_metrics(self, y_pred, y_true, mask) -> Dict:
        """Function for metrics calculation during step."""
        # TODO: temp workaround for metric normalization by mask sum
        norm_coef = (y_pred['hamiltonian'].numel() / mask.sum())
        metric = self.hparams.metric(y_pred, y_true)
        metric['hamiltonian'] = metric['hamiltonian'] * norm_coef
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

    def _check_devices(self):
        self.hparams.metric = self.hparams.metric.to(self.device)
        self.net.set()
        if self.ema is not None:
            self.ema.to(self.device)

    def _instantiate_ema(self):
        if self.ema is not None:
            self.ema = self.ema(self.parameters())

    def _get_batch_size(self, batch):
        """Function for batch size infer."""
        bsz = batch.batch.max().detach().item() + 1  # get batch size
        return bsz
    
    def _get_hamiltonian_sizes(self, batch):
        sizes = []
        for idx in range(batch.ptr.shape[0] - 1):
            atoms = batch.z[batch.ptr[idx]: batch.ptr[idx + 1]]
            size = sum([self.net.orbital_mask[atom] for atom in atoms])
            sizes.append(size)
        return sizes
    