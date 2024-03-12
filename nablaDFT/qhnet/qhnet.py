from typing import Dict

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


ATOM_MASKS = {
    1: [0, 1, 5, 6, 7],
    6: [0, 1, 2, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    7: [0, 1, 2, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    8: [0, 1, 2, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    9: [0, 1, 2, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    16: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    17: [0, 1, 2, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
    35: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
}


class QHNet(nn.Module):
    def __init__(self,
                 in_node_features=1,
                 sh_lmax=4,
                 hidden_size=128,
                 bottle_hidden_size=32,
                 num_gnn_layers=5,
                 max_radius=12,
                 num_nodes=10,
                 radius_embed_dim=32):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super(QHNet, self).__init__()
        # store hyperparameter values
        self.atom_orbs = [
            [[8, 0, '1s'], [8, 0, '2s'], [8, 0, '3s'], [8, 1, '2p'], [8, 1, '3p'], [8, 2, '3d']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']],
            [[1, 0, '1s'], [1, 0, '2s'], [1, 1, '2p']]
        ]
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
        self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e')
        self.hidden_bottle_irrep_base = o3.Irreps(
            f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e')
        self.final_out_irrep = o3.Irreps(f'{self.hs * 3}x0e + {self.hs * 2}x1o + {self.hs}x2e').simplify()
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(self.radius_embed_dim, self.max_radius)
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

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
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
            )
            self.fc_ii[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_path_weight)
            )
            self.fc_ii_bias[name] = torch.nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_bias) # TODO: this shit defines output dimension for diagonal block
            )

            self.expand_ij[name] = Expansion(
                o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e'),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
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
        self.orbital_mask = {}

    def set(self):
        for key in ATOM_MASKS.keys():
            self.orbital_mask[key] = torch.tensor(ATOM_MASKS[key]).to(self.device)   

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
        final_matrix = torch.block_diag(final_matrix)
        return final_matrix

    def split_matrix(self, data):
        diagonal_matrix, non_diagonal_matrix = \
            torch.zeros(data.z.shape[0], 14, 14).type(data.pos.type()).to(self.device), \
            torch.zeros(data.edge_index.shape[1], 14, 14).type(data.pos.type()).to(self.device)

        data.matrix =  data.matrix.reshape(
            len(data.ptr) - 1, data.matrix.shape[-1], data.matrix.shape[-1])

        num_atoms = 0
        num_edges = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            slices = [0]
            for atom_idx in data.z[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
                slices.append(slices[-1] + len(self.orbital_mask[atom_idx.item()]))

            for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx+1]):
                node_idx = node_idx - num_atoms
                orb_mask = self.orbital_mask[data.z[node_idx].item()]
                diagonal_matrix[node_idx][orb_mask][:, orb_mask] = \
                    data.matrix[graph_idx][slices[node_idx]: slices[node_idx+1], slices[node_idx]: slices[node_idx+1]]

            for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
                dst, src = data.edge_index[:, edge_index_idx]
                if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                    break
                num_edges = num_edges + 1
                orb_mask_dst = self.orbital_mask[data.z[dst].item()]
                orb_mask_src = self.orbital_mask[data.z[src].item()]
                graph_dst, graph_src = dst - num_atoms, src - num_atoms
                non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = \
                    data.matrix[graph_idx][slices[graph_dst]: slices[graph_dst+1], slices[graph_src]: slices[graph_src+1]]

            num_atoms = num_atoms + data.ptr[graph_idx + 1] - data.ptr[graph_idx]
        return diagonal_matrix, non_diagonal_matrix


class QHNetLightning(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            net: nn.Module,
            optimizer: Optimizer,
            lr_scheduler: LRScheduler,
            losses: Dict,
            metric,
            loss_coefs,
    ) -> None:
        super(QHNetLightning, self).__init__()
        self.net = net
        self.save_hyperparameters(logger=True)

    def forward(self, data: Data):
        hamiltonian = self.net(data)
        return hamiltonian
    
    def step(self, batch, calculate_metrics: bool = False):
        hamiltonian_out = self.net(batch)
        hamiltonian = batch.hamiltonian
        preds = {'hamiltonian': hamiltonian_out}
        hamiltonian = torch.block_diag(*[torch.from_numpy(H) for H in hamiltonian])
        target = {'hamiltonian': hamiltonian}
        loss = self._calculate_loss(preds, target)
        if calculate_metrics:
            metrics = self._calculate_metrics(preds, target)
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
 #       with self.ema.average_parameters():
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

#    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
#        self.ema.update()

    def on_fit_start(self) -> None:
 #       self._instantiate_ema()
        self._check_devices()

    def on_test_start(self) -> None:
 #       self._instantiate_ema()
        self._check_devices()

    def on_validation_epoch_end(self) -> None:
        self._reduce_metrics(step_type="val")

    def on_test_epoch_end(self) -> None:
        self._reduce_metrics(step_type="test")

    def _calculate_loss(self, y_pred, y_true) -> float:
        # Note: since hamiltonians has different shapes, loss calculated per sample
        total_loss = 0.0
        for name, loss in self.hparams.losses.items():
            total_loss += self.hparams.loss_coefs[name] * loss(
                y_pred[name], y_true[name]
            )
        return total_loss

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

    def _check_devices(self):
        self.hparams.metric = self.hparams.metric.to(self.device)
        self.net.set()
#        if self.ema is not None:
#            self.ema.to(self.device)

#    def _instantiate_ema(self):
#        if self.ema is not None:
#            self.ema = self.ema(self.parameters())

    def _get_batch_size(self, batch):
        """Function for batch size infer."""
        bsz = batch.batch.max().detach().item() + 1  # get batch size
        return bsz