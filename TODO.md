 # Dataflow

  - TODO: delegate download process to dataset registry.
  - TODO: finalize convert functions.
  - TODO: make `pytorch_lightning.Datamodule` more generic interface instead of specific ones.
  - TODO: `torch.data.utils.Dataset` more generic interface instead of specific ones.
  - TODO: `torch_geometric.data.Dataset` more generic interface instead of specific ones.s
  - TODO (future): benchmark LMDB database for energy and haniltonian split.

# Models

  - TODO: delegate checkpoint download to model registry.
  - TODO: define `BaseModel` protocol for further addition of models in nabla.
  - TODO: make `pytorch_lightning.Module` more generic in order to apply this module to all possible models from nabla.
  - TODO: split all graph transformations and graph generation to separate transform module, prevents code duplication and strong binding of this logic into model.

# Dependencies
  - TODO: consider retrieving useful code fomr schnetpack and remove it from dependencies.

# Testing
  - TODO: incorporate nox for different python version tests.


# Setup
  - TODO: find the way to manage `torch` and `torch_geometric` dependencies on current CUDA version.
