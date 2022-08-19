# nablaDFT: Energy and Hamiltonian prediction on a large scale
## Dataset

We propose a benchmarking dataset based on a subset of Molecular Sets (MOSES) dataset. Resulting dataset contains 1,004,918 molecules with atoms C, N, S, O, F, Cl, Br, H. It contains 226,424 unique Bemis-Murcko (Bemis and Murcko, 1996) scaffolds and 34,572 unique BRICS (Degen et al., 2008) fragments.
We provide several splits of the dataset. At first we fix train set, which consists of 100,000 molecules and 436,581 conformations and its smaller subset with 10,000, 5,000 and 2,000 molecules and 38,364, 20,349, 5,768 conformations respectively. We choose another 100,000 random molecules as a random test set. The scaffold test set has 100,000 molecules containing a Bemis-Murcko scaffold from a random subset of scaffolds which are not present in the train set. Finally conformation test set consists of 91,182 (10,000, 5,000, 2,000) molecules from the train set with new conformations, resulting in 92,821 (8,892, 4,897, 1,724) conformations.

### Accessing elements of the dataset

```python
from nablaDFT.dataset import HamiltonianDatabase

train = HamiltonianDatabase("train2k.db")
Z, R, E, F, H, S, C = train[0]
```