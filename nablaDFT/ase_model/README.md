# SchNetPack models 

The code for training [SchNetPack](https://schnetpack.readthedocs.io/en/latest/api/representation.html) models from papers:
* SchNet: A continuous-filter convolutional neural network for modeling quantum interactions (SchNet)
* Equivariant message passing for the prediction of tensorial properties and molecular spectra (PaiNN)

For the training, run the following command from the root of the repository:

```bash
python run.py --config-name schnet.yaml 
```
or
```bash
python run.py --config-name painn.yaml
```
