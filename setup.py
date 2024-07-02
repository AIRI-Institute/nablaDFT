import sys

from setuptools import find_packages, setup


def get_python_version():
    version_info = sys.version_info
    major = version_info[0]
    minor = version_info[1]
    return f"cp{major}{minor}"


PYTHON_VERSION = get_python_version()


setup(
    name="nablaDFT",
    version="2.0.0",
    author="Khrabrov, Kuzma  and Ber, Anton and Tsypin, Artem and Ushenin, Konstantin and Rumiantsev, Egor"
    "and Telepov, Alexander and Protasov, Dmitry and Shenbin, Ilya and Alekseev, Anton"
    "and Shirokikh, Mikhail and Nikolenko, Sergey and Tutubalina, Elena and Kadurin, Artur",
    url="https://github.com/AIRI-Institute/nablaDFT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26,<2.0.0",
        "sympy==1.12",
        "ase==3.22.1",
        "h5py==3.10.0",
        "apsw==3.45.1.0",
        "schnetpack==2.0.4",
        "tensorboardX",
        "pyyaml",
        "hydra-core==1.2.0",
        "torch==2.2.0",
        f"torch-scatter @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_scatter-2.1.2%2Bpt22cu121-{PYTHON_VERSION}-{PYTHON_VERSION}-linux_x86_64.whl",
        f"torch-sparse @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_sparse-0.6.18%2Bpt22cu121-{PYTHON_VERSION}-{PYTHON_VERSION}-linux_x86_64.whl",
        f"torch-cluster @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_cluster-1.6.3%2Bpt22cu121-{PYTHON_VERSION}-{PYTHON_VERSION}-linux_x86_64.whl",
        "pytorch_lightning==2.1.4",
        "torch-geometric==2.4.0",
        "torchmetrics==1.0.1",
        "hydra-colorlog>=1.1.0",
        "rich",
        "fasteners",
        "dirsync",
        "torch-ema==0.3",
        "matscipy",
        "python-dotenv",
        "wandb==0.16.3",
        "e3nn==0.5.1",
    ],
    license="MIT",
    description="$\nabla^2$ DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules"
    "and a Benchmark for Neural Network Potentials",
    long_description="""Methods of computational quantum chemistry provide accurate approximations of molecular 
    properties crucial for computer-aided drug discovery and other areas of chemical science. However, 
    high computational complexity limits the scalability of their applications. Neural network potentials (NNPs) are 
    a promising alternative to quantum chemistry methods, but they require large and diverse datasets for training. 
    This work presents a new dataset and benchmark called $\nabla^2$ DFT that is based on the nablaDFT. It contains 
    twice as much molecular structures, three times more conformations, new data types and tasks, 
    and state-of-the-art models. The dataset includes energies, forces, 17 molecular properties, Hamiltonian and 
    overlap matrices, and a wavefunction object. All calculations were performed at the DFT level (ωB97X-D/def2-SVP) 
    for each conformation. Moreover, $\nabla^2$ DFT is the first dataset that contains relaxation trajectories for a 
    substantial number of drug-like molecules. We also introduce a novel benchmark for evaluating NNPs in molecular 
    property prediction, Hamiltonian prediction, and conformational optimization tasks.""",
    classifiers=[
        "Development Status :: 4 - Beta",
    ],
)
