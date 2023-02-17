import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="nablaDFT",
    version="2.0.0",
    author="Khrabrov, Kuzma and Shenbin, Ilya and Ryabov, Alexander and Tsypin, Artem and Telepov, Alexander and Alekseev, Anton and Grishin, Alexander and Strashnov, Pavel and Zhilyaev, Petr and Nikolenko, Sergey and Kadurin, Artur",
    url="https://github.com/AIRI-Institute/nablaDFT",
    packages=find_packages(),
    package_data={'nablaDFT': ['links/*']},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "sympy",
        "ase>=3.21",
        "h5py",
        "torch",
        "apsw",
        "schnetpack>=2.0.0",
        "tensorboardX",
        "pyyaml",
        "hydra-core>=1.1.0",
        "pytorch_lightning>=1.9.0",
        "torchmetrics",
        "hydra-colorlog>=1.1.0",
        "rich",
        "fasteners",
        "dirsync",
        "torch-ema",
        "matscipy",
    ],
    license="MIT",
    description="nablaDFT: Large-Scale Conformational Energy and Hamiltonian Prediction benchmark and dataset",
    long_description="""
        Electronic wave function calculation is a fundamental task of computational quantum  chemistry. Knowledge of the wave function parameters allows one to compute physical and chemical properties of molecules and materials.
        In this work we: introduce a new curated large-scale dataset of electron structures of drug-like molecules, establish a novel benchmark for the estimation of molecular properties in the multi-molecule setting, and evaluate a wide range of methods with this benchmark.""",
)
