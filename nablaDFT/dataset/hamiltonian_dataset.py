import math
import os
import multiprocessing

import numpy as np
import torch
import apsw  # way faster than sqlite3


class HamiltonianDatabase:
    """
    This is a class to store large amounts of ab initio reference data
    for training a neural network in a SQLite database

    Data structure:
    Z (N)    (int)        nuclear charges
    R (N, 3) (float)      Cartesian coordinates in bohr
    E ()     (float)      energy in Eh
    F (N, 3) (float)      forces in Eh/bohr
    H (Norb, Norb)        full hamiltonian in atomic units
    S (Norb, Norb)        overlap matrix in atomic units
    C (Norb, Norb)        core hamiltonian in atomic units
    moses_id () (int)     molecule id in MOSES dataset
    conformer_id () (int) conformation id
    """
    def __init__(self, filename, flags=apsw.SQLITE_OPEN_READONLY):
        self.db = filename
        self.connections = {}  # allow multiple connections (needed for multi-threading)
        self._open(flags=flags)  # creates the database if it doesn't exist yet

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute("""SELECT * FROM metadata WHERE id=0""").fetchone()[-1]

    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        if type(idx) == list:  # for batched data retrieval
            data = cursor.execute(
                """SELECT * FROM data WHERE id IN (""" + str(idx)[1:-1] + ")"
            ).fetchall()
            return [self._unpack_data_tuple(i) for i in data]
        else:
            data = cursor.execute(
                """SELECT * FROM data WHERE id=""" + str(idx)
            ).fetchone()
            return self._unpack_data_tuple(data)

    def _unpack_data_tuple(self, data):
        N = (
            len(data[2]) // 4 // 3
        )  # a single float32 is 4 bytes, we have 3 in data[1] (positions)
        R = self._deblob(data[2], dtype=np.float32, shape=(N, 3))
        Z = self._deblob(data[1], dtype=np.int32, shape=(N))
        E = np.array([0.0 if data[3] is None else data[3]], dtype=np.float32)
        F = self._deblob(data[4], dtype=np.float32, shape=(N, 3))
        Norb = int(
            math.sqrt(len(data[5]) // 4)
        )  # a single float32 is 4 bytes, we have Norb**2 of them
        H = self._deblob(data[5], dtype=np.float32, shape=(Norb, Norb))
        S = self._deblob(data[6], dtype=np.float32, shape=(Norb, Norb))
        C = self._deblob(data[7], dtype=np.float32, shape=(Norb, Norb))
        return Z, R, E, F, H, S, C

    def add_data(
        self,
        Z,
        R,
        E,
        F,
        H,
        S,
        C,
        moses_id,
        conformer_id,
        flags=apsw.SQLITE_OPEN_READWRITE,
        transaction=True,
    ):
        # check that no NaN values are added
        if self._any_is_nan(Z, R, E, F, H, S, C):
            print("encountered NaN, data is not added")
            return
        cursor = self._get_connection(flags=flags).cursor()
        # update data
        if transaction:
            # begin exclusive transaction (locks db) which is necessary
            # if database is accessed from multiple programs at once (default for safety)
            cursor.execute("""BEGIN EXCLUSIVE""")
        try:
            length = cursor.execute("""SELECT * FROM metadata WHERE id=0""").fetchone()[
                -1
            ]
            cursor.execute(
                """INSERT INTO dataset_ids (id, MOSES_ID, CONFORMER_ID) VALUES (?,?,?)""",
                (
                    None if length > 0 else 0,  # autoincrementing ID
                    moses_id,
                    conformer_id,
                ),
            )
            cursor.execute(
                """INSERT INTO data (id, Z, R, E, F, H, S, C) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    None if length > 0 else 0,  # autoincrementing ID
                    self._blob(Z),
                    self._blob(R),
                    None if E is None else float(E),
                    self._blob(F),
                    self._blob(H),
                    self._blob(S),
                    self._blob(C),
                ),
            )

            # update metadata
            cursor.execute(
                """INSERT OR REPLACE INTO metadata VALUES (?,?)""", (0, length + 1)
            )

            if transaction:
                cursor.execute("""COMMIT""")  # end transaction
        except Exception as exc:
            if transaction:
                cursor.execute("""ROLLBACK""")
            raise exc

    def add_orbitals(self, Z, orbitals, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO basisset (Z, orbitals) VALUES (?,?)""",
            (int(Z), self._blob(orbitals)),
        )

    def get_orbitals(self, Z):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute("""SELECT * FROM basisset WHERE Z=""" + str(Z)).fetchone()
        Norb = len(data[1]) // 4  # each entry is 4 bytes
        return self._deblob(data[1], dtype=np.int32, shape=(Norb,))

    def _any_is_nan(self, *vals):
        nan = False
        for val in vals:
            if val is None:
                continue
            nan = nan or np.any(np.isnan(val))
        return nan

    def _blob(self, array):
        """Convert numpy array to blob/buffer object."""
        if array is None:
            return None
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    def _deblob(self, buf, dtype=np.float32, shape=None):
        """Convert blob/buffer object to numpy array."""
        if buf is None:
            return np.zeros(shape)
        array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return array

    def _open(self, flags=apsw.SQLITE_OPEN_READONLY):
        newdb = not os.path.isfile(self.db)
        cursor = self._get_connection(flags=flags).cursor()
        if newdb:
            # create table to store data
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS dataset_ids
                (id INTEGER NOT NULL PRIMARY KEY,
                 MOSES_ID INT,
                 CONFORMER_ID INT
                )"""
            )
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 Z BLOB,
                 R BLOB,
                 E FLOAT,
                 F BLOB,
                 H BLOB,
                 S BLOB,
                 C BLOB
                )"""
            )

            # create table to store things that are constant for the whole dataset
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS nuclear_charges
                (id INTEGER NOT NULL PRIMARY KEY, N INTEGER, Z BLOB)"""
            )
            cursor.execute(
                """INSERT OR IGNORE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)""",
                (0, 1, self._blob(np.array([0]))),
            )
            self.N = len(self.Z)

            # create table to store the basis set convention
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS basisset
                (Z INTEGER NOT NULL PRIMARY KEY, orbitals BLOB)"""
            )

            # create table to store metadata (information about the number of entries)
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)"""
            )
            cursor.execute(
                """INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)""", (0, 0)
            )  # num_data

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        """
        This allows multiple processes to access the database at once,
        every process must have its own connection
        """
        key = multiprocessing.current_process().name
        if key not in self.connections.keys():
            self.connections[key] = apsw.Connection(self.db, flags=flags)
            self.connections[key].setbusytimeout(300000)  # 5 minute timeout
        return self.connections[key]

    def add_Z(self, Z, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        self.N = len(Z)
        cursor.execute(
            """INSERT OR REPLACE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)""",
            (0, self.N, self._blob(Z)),
        )

    @property
    def Z(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute("""SELECT * FROM nuclear_charges WHERE id=0""").fetchone()
        N = data[1]
        return self._deblob(data[2], dtype=np.int32, shape=(N,))



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
