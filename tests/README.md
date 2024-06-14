# $\nabla^2$ DFT tests

## Environment preparation



## Test run

Full test run:
```bash
pytest -v tests/
```

Run certain test category:
```bash
pytest -v tests/ -m <category>
```
where category could be:  
- dataset: runs tests for dataset classes.
- model: runs tests for model creation.
- pipeline: runs tests for full pipeline.
- download: runs tests for dataset availability.
- optimization: runs tests for parts of optimization pipeline.