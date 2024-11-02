"""Module defines mapping between column names in various datasources."""

# out-key: table.key
nabla_energy = {
    "y": "systems.energy",
    "forces": "systems.forces",
    "z": "systems.numbers",
    "pos": "systems.pos",
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
