# nablaDFT: Energy and Hamiltonian prediction on a large scale
## Dataset

We propose a benchmarking dataset based on a subset of [Molecular Sets (MOSES) dataset](https://github.com/molecularsets/moses). Resulting dataset contains 1 004 918 molecules with atoms C, N, S, O, F, Cl, Br, H. E It contains 226 424 unique Bemis-Murcko scaffolds and 34 572 unique BRICS fragments.<br/>
For each molecule in the dataset we provide from 1 to 62 unique conformations, with 5 340 152 total conformations. For each conformation, we have calculated its electronic properties including the energy (E), DFT Hamiltonian matrix (H), and DFT overlap matrix (S). All properties were calculated using the Kohn-Sham method at ωB97X-D/def2-SVP levels of theory using the quantum-chemical software package [Psi4](https://github.com/psi4/psi4), version 1.5.
We provide several splits of the dataset that can serve as the basis for comparison across different models. First, we fix the training set that consists of 100 000 molecules with 436 581 conformations and its smaller subsets with 10 000, 5 000, and 2 000 molecules and 38 364, 20 349, and 5 768 conformations respectively; these subsets can help determine how much additional data helps various models. We choose another 100 000 random molecules as a structure test set. The scaffold test set has 100 000 molecules containing a Bemis-Murcko scaffold from a random subset of scaffolds which are not present in the training set. Finally, the conformation test set consists of 91 182 (resp., 10 000, 5 000, 2 000) molecules from the training set with new conformations, numbering in total 92 821 (8 892, 4 897, 1 724) conformations; this set can be used for the single-molecule setup. <br\>
As part of the benchmark, we provide separate databases for each subset and task and a complete archive with wave function files produced by the Psi4 package that contains quantum chemical properties of the corresponding molecule and can be used in further computations.
### Downloading dataset
#### Hamiltonian databases
The full hamiltonian database is available at [nablaDFT Hamiltonian database](https://sc.link/r4WE) (7 TB)<br/> 
Links to other hamiltonian databases including different train and test subsets are in file [Hamilonian databases](./nablaDFT/links/hamiltonian_databases.json)<br/>
An archive with numpy indexes:  [splits indexes](https://sc.link/7Z7B)<br/>
#### Energy databases
Links to energy databases including different train and test subsets are in file [Energy databases](./nablaDFT/links/energy_databases.json)

#### Raw psi4 wave functions
Links to tarballs: [wave functions](./nablaDFT/links/nablaDFT_psi4wfn_links.txt)

### Accessing elements of the dataset
#### Hamiltonian database

```python
from nablaDFT.dataset import HamiltonianDatabase

train = HamiltonianDatabase("dataset_train_2k.db")
Z, R, E, F, H, S, C = train[0]  # atoms numbers, atoms positions, energy, forces, core hamiltonian, overlap matrix, coefficients matrix
```
#### Energies database
```python
from ase.db import connect

train = connect("train_2k_energy.db")
atoms_data = connect.get(1)
```
#### Working with raw psi4 wavefunctions

A variety of properties can also be loaded directly from the wavefunctions. See main paper for more details. Properties include DFT matrices:
```python
import numpy as np
import psi4
wfn = np.load(<PATH_TO_WFN>, allow_pickle=True).tolist()
orbital_matrix_a = wfn["matrix"]["Ca"]        # alpha orbital coefficients
orbital_matrix_b = wfn["matrix"]["Cb"]        # betta orbital coefficients
density_matrix_a = wfn["matrix"]["Da"]        # alpha electonic density
density_matrix_b = wfn["matrix"]["Db"]        # betta electonic density
aotoso_matrix = wfn["matrix"]["aotoso"]       # atomic orbital to symmetry orbital transformation matrix
core_hamiltonian_matrix = wfn["matrix"]["H"]  # core Hamiltonian matrix
fock_matrix_a = wfn["matrix"]["Fa"]           # DFT alpha Fock matrix
fock_matrix_b = wfn["matrix"]["Fb"]           # DFT betta Fock matrix 
```
and bond orders for covalent and non-covalent interactions and atomic charges: 
```python
wfn = psi4.core.Wavefunction.from_file(<PATH_TO_WFN>)
psi4.oeprop(wfn, "MAYER_INDICES")
psi4.oeprop(wfn, "WIBERG_LOWDIN_INDICES")
psi4.oeprop(wfn, "MULLIKEN_CHARGES")
psi4.oeprop(wfn, "LOWDIN_CHARGES")
meyer_bos = wfn.array_variables()["MAYER INDICES"]  # Mayer bond indices
lodwin_bos = wfn.array_variables()["WIBERG LOWDIN INDICES"]  # Wiberg bond indices
mulliken_charges = wfn.array_variables()["MULLIKEN CHARGES"]  # Mulliken atomic charges
lowdin_charges = wfn.array_variables()["LOWDIN CHARGES"]  # Löwdin atomic charges
```


## Models
* [Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions (SchNOrb)](https://github.com/KuzmaKhrabrov/SchNOrb)
* [SE(3)-equivariant prediction of molecular wavefunctions and electronic densities (PhiSNet)](./nablaDFT/phisnet/README.md)
* [A continuous-filter convolutional neural network for modeling quantum interactions (SchNet)](./nablaDFT/schnet/README.md)
* [Equivariant message passing for the prediction of tensorial properties and molecular spectra (PaiNN)](./nablaDFT/painn/README.md)
* [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules (DimeNet++)](https://github.com/ilya-shenbin/dimenet)

### Metrics
In the tables below ST, SF, CF denote structures test set, scaffolds test set and conformations test set correspondingly.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="3">Model</th>
      <th colspan="15"> MAE for energy prediction $\times 10^{−2} E_h$ (↓)</th>
    </tr>
    <tr>
      <th colspan="5">Test ST</th>
      <th colspan="5">Test SF</th>
      <th colspan="5">Test CF</th>
    </tr>
    <tr>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><i>LR</i></td>
      <td><i>4.6</i></td>
      <td><i>4.7</i></td>
      <td><i>4.7</i></td>
      <td><i>4.7</i></td>
      <td><i>-</i></td>
      <td><i>4.6</i></td>
      <td><i>4.7</i></td>
      <td><i>4.7</i></td>
      <td><i>4.7</i></td>
      <td><i>-</i></td>
      <td><i>4.0</i></td>
      <td><i>4.2</i></td>
      <td><i>4.0</i></td>
      <td><i>4.0</i></td>
      <td><i>-</i></td>
    </tr>
    <tr>
      <td><i>SchNet</i></td>
      <td><i>151.8</i></td>
      <td><i>66.1</i></td>
      <td><i>29.6</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>126.5</i></td>
      <td><i>68.3</i></td>
      <td><i>27.4</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>79.1</i></td>
      <td><i>67.3</i></td>
      <td><i>21.4</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
    </tr>
    <tr>
      <td><i>SchNOrb</i></td>
      <td><i>5.9</i></td>
      <td><i>3.7</i></td>
      <td><i>13.3(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>5.9</i></td>
      <td><i>3.4</i></td>
      <td><i>14.8(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>5.0</i></td>
      <td><i>3.6</i></td>
      <td><i>14.5(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
    </tr>
    <tr>
      <td><i>DimeNet++</i></td>
      <td><i>24.1</i></td>
      <td><i>21.1</i></td>
      <td><i>10.6</i></td>
      <td><i>3.2</i></td>
      <td><i>-</i></td>
      <td><i>21.6</i></td>
      <td><i>20.9</i></td>
      <td><i>10.1</i></td>
      <td><i>3.0</i></td>
      <td><i>-</i></td>
      <td><i>18.3</i></td>
      <td><i>33.7</i></td>
      <td><i>5.2</i></td>
      <td><i>2.5</i></td>
      <td><i>-</i></td>
    </tr>
    <tr>
      <td><i>PAINN</i></td>
      <td><i>137.2</i></td>
      <td><i>62.8</i></td>
      <td><i>13.1</i></td>
      <td><i>7.0</i></td>
      <td><i>-</i></td>
      <td><i>131.1</i></td>
      <td><i>53.0</i></td>
      <td><i>12.6</i></td>
      <td><i>6.7</i></td>
      <td><i>-</i></td>
      <td><i>134.4</i></td>
      <td><i>50.0</i></td>
      <td><i>12.1</i></td>
      <td><i>7.0</i></td>
      <td><i>-</i></td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="3">Model</th>
      <th colspan="15"> MAE for Hamiltonian matrix prediction $\times 10^{−4} E_h$ (↓)</th>
      <th colspan="15"> MAE for overlap matrix prediction $\times 10^{−5}$(↓)</th>
    </tr>
    <tr>
      <th colspan="5">Test ST</th>
      <th colspan="5">Test SF</th>
      <th colspan="5">Test CF</th>
      <th colspan="5">Test ST</th>
      <th colspan="5">Test SF</th>
      <th colspan="5">Test CF</th>
    </tr>
    <tr>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
      <th>2k</th>
      <th>5k</th>
      <th>10k</th>
      <th>100k</th>
      <th>full train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><i>SchNOrb</i></td>
      <td><i>386.5</i></td>
      <td><i>383.4</i></td>
      <td><i>382.0(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>385.3</i></td>
      <td><i>380.7</i></td>
      <td><i>383.6(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>385.0</i></td>
      <td><i>384.8</i></td>
      <td><i>392.0(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>1550</i></td>
      <td><i>1455</i></td>
      <td><i>1493(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>1543</i></td>
      <td><i>1440</i></td>
      <td><i>1496(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>1544</i></td>
      <td><i>1480</i></td>
      <td><i>1536(*)</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
    </tr>
    <tr>
      <td><i>PhiSNet</i></td>
      <td><i>7.4</i></td>
      <td><i>3.2</i></td>
      <td><i>2.9</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>7.2</i></td>
      <td><i>3.2</i></td>
      <td><i>2.9</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>6.5</i></td>
      <td><i>3.2</i></td>
      <td><i>2.8</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>5.1</i></td>
      <td><i>4.3</i></td>
      <td><i>3.5</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>5.0</i></td>
      <td><i>4.3</i></td>
      <td><i>3.5</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
      <td><i>5.1</i></td>
      <td><i>4.6</i></td>
      <td><i>3.6</i></td>
      <td><i>-</i></td>
      <td><i>-</i></td>
    </tr>
  </tbody>
</table>

Fields with - or * symbols correspond to the models, which haven't converged and will be updated in the future.