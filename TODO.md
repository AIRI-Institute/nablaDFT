 # Dataflow

  - TODO: finalize DatasetCard.
  - TODO: write tests for datasources and dataset interfaces.
  - TODO: delegate download process to dataset registry.
  - TODO: currently pyg dataset will not work for hamiltonian databases, so as torch dataset, fix that.
  - TODO (future): benchmark LMDB database for energy and haьiltonian split.

For EDA purposes maybe use:
```python
conn = apsw.Connection(filename)
tables_list = [info[1] for info in conn.execute("SELECT * FROM sqlite_master WHERE type='table'").fetchall()]
data_schema = conn.pragme("table_info(data)")
```
  
# Models

  - TODO: delegate checkpoint download to model registry.
  - TODO: define `BaseModel` protocol for further addition of models in nabla.
  - TODO: make `pytorch_lightning.Module` more generic in order to apply this module to all possible models from nabla.
  - TODO: split all graph transformations and graph generation to separate transform module, prevents code duplication and strong binding of this logic into model.

# Dependencies
  - TODO: consider retrieving useful code from schnetpack and remove it from dependencies.

# Testing
  - TODO: incorporate nox for different python version tests.

# Setup
  - TODO: find the way to manage `torch` and `torch_geometric` dependencies on current CUDA version.
