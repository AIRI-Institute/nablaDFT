"""Module defines mapping between column names in various datasources."""

from dataclasses import dataclass

# out-key: table.key
nabla_energy = {
    "y": "_data.energy",
    "forces": "_data.forces",
    "z": "numbers",
    "pos": "pos",
}

nabla_hamiltonian = {
    "y": "data.E",
    "forces": "data.F",
    "z": "data.Z",
    "pos": "data.R",
    "H": "data.H",
}

nabla_overlap = {
    "y": "data.E",
    "forces": "data.F",
    "z": "data.Z",
    "pos": "data.R",
    "S": "data.S",
}


@dataclass
class DatasetMetadata:
    """Describes dataset metadata.

    Args:
        name (str) - dataset name.
        desc (str) - dataset description. Could be empty.
        metadata (dict) - dataset metadata. Could be empty. Must contain calculation methods.
        keys_map (dict) - mapping from column names in database to sample's keys.
        data_dtypes (dict) - mapping from column names in database to data type (e.g. np.float32).
        data_shapes (dict) - mapping from column names in database to data shape (e.g. (-1, 3)).
    """

    name: str
    desc: str
    metadata: dict
    keys_map: dict
    data_dtypes: dict
    data_shapes: dict
