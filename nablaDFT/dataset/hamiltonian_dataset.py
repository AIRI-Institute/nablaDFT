import numpy as np
import torch

from nablaDFT.dataset import HamiltonianDatabase


class HamiltonianDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath,
        max_batch_orbitals=1200,
        max_batch_atoms=150,
        max_squares=4802,
        subset=None,
        dtype=torch.float32,
    ):
        super(HamiltonianDataset, self).__init__()
        self.dtype = dtype
        self.database = HamiltonianDatabase(filepath)
        max_orbitals = []
        for z in self.database.Z:
            max_orbitals.append(
                tuple((int(z), int(l)) for l in self.database.get_orbitals(z))
            )
        max_orbitals = tuple(max_orbitals)
        self.max_orbitals = max_orbitals
        self.max_batch_orbitals = max_batch_orbitals
        self.max_batch_atoms = max_batch_atoms
        self.max_squares = max_squares
        self.subset = None
        if subset:
            self.subset = np.load(subset)

    def __len__(self):
        if self.subset is not None:
            return len(self.subset)
        return len(self.database)

    def __getitem__(self, idx):
        if self.subset is not None:
            return self.subset[
                idx
            ]  # just return the idx, the custom collate_fn does the querying
        else:
            return idx

    # collate function to collate several data points into a batch, should be passed
    # to data loader (default_collate is not efficient, because there should be a batch-wise query)
    def collate_fn(self, batch, return_filtered=False):
        all_data = self.database[batch]  # fetch the batch data
        Z, R, E, F, H, S, C, mask = [], [], [], [], [], [], [], []
        orbitals = []
        orbitals_number = 0
        sum_squares = 0
        for batch_num, data in enumerate(all_data):
            Z_, R_, E_, F_, H_, S_, C_ = data
            local_orbitals = []
            local_orbitals_number = 0
            for z in Z_:
                local_orbitals.append(
                    tuple((int(z), int(l)) for l in self.database.get_orbitals(z))
                )
                local_orbitals_number += sum(2 * l + 1 for _, l in local_orbitals[-1])
            # print (local_orbitals_number, orbitals_number, len(local_orbitals), len(orbitals))
            if (
                orbitals_number + local_orbitals_number > self.max_batch_orbitals
                or len(local_orbitals) + len(orbitals) > self.max_batch_atoms
                or sum_squares + len(local_orbitals) ** 2 > self.max_squares
            ):
                break
            orbitals += local_orbitals
            orbitals_number += local_orbitals_number
            sum_squares += len(local_orbitals) ** 2
            Z.append(torch.tensor(Z_, dtype=torch.int64))
            R.append(torch.tensor(R_, dtype=self.dtype))
            E.append(torch.tensor(E_, dtype=self.dtype))
            F.append(torch.tensor(F_, dtype=self.dtype))
            H.append(torch.tensor(H_, dtype=self.dtype))
            S.append(torch.tensor(S_, dtype=self.dtype))
            C.append(torch.tensor(C_, dtype=self.dtype))
            mask.append(torch.ones_like(C[-1]))
        Z_concat = torch.cat(Z)
        orbitals = tuple(orbitals)
        return_dict = {
            "molecule_size": torch.tensor([len(z) for z in Z]),
            "atomic_numbers": Z_concat,
            "orbitals": orbitals,
            "positions": torch.cat(R),
            "energy": torch.stack(E),
            "forces": torch.cat(F),
            "full_hamiltonian": torch.block_diag(*H),
            "overlap_matrix": torch.block_diag(*S),
            "core_hamiltonian": torch.block_diag(*C),
            "mask": torch.block_diag(*mask),
        }
        if return_filtered:
            return_dict["filtered"] = batch[len(Z) :]
        return return_dict


def seeded_random_split(dataset, lengths, seed=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    This is very similar to the pytorch equivalent, but this version allows a seed to
    be specified in order to make the split reproducible

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """

    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = np.random.RandomState(seed=seed).permutation(sum(lengths))

    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(torch._utils._accumulate(lengths), lengths)
    ]


def file_split(dataset, filename):
    splits = np.load(filename)
    return [
        torch.utils.data.Subset(dataset, splits[split_name])
        for split_name in ["train_idx", "val_idx", "test_idx"]
    ]
