# PhiSNet

The code for training PhiSNet model based on authors' code https://openreview.net/forum?id=auGY2UQfhSu

IMPORTANT:
Package requirements (lower versions might work, but were not tested):
- python >= 3.7
- numpy >= 1.20.2
- torch >= 1.8.1
- cudatoolkit >= 10.2
- ase >= 3.21.1
- apsw >= 3.34.0.r1
- tensorboardX >= 2.2
- psi4 >= 1.5 (Optional)

## Train on nablaDFT dataset
After installing the necessary packages, follow the steps below to train it on nablaDFT dataset splits.

1) Download the full database or distinct train/test databases.
2) Train PhiSNet by running

> python3 train.py @args_nablaDFT_2k.txt
3) Test PhiSNet by executing
> python test.py @args_nablaDFT_2k.txt --load_from latest_checkpoint.pth --test_dataset dataset_test_scaffolds.db
