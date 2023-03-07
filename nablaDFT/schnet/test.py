import argparse
import os
import tqdm

from ase.db import connect

import torch
import torchmetrics

import schnetpack.transform as trn

from nablaDFT.dataset import NablaDFT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SchNet test")
    parser.add_argument(
        "--datapath", type=str, help="path to data", default="train_10k_energy.db"
    )
    parser.add_argument(
        "--dataset_name", type=str, help="dataset name", default="dataset_test_conformations_2k"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="path to the model checkpoint",
        default="logs/model_moses_10k_split",
    )
    parser.add_argument(
        "--logspath",
        type=str,
        help="path to logs",
        default="logs/model_moses_10k_split",
    )
    parser.add_argument("--batch_size", type=int, default=2000, help="batch size")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu device number")
    parser.add_argument("--cutoff", type=float, default=5.0, help="cutoff value")

    args, unknown = parser.parse_known_args()
    workpath = args.logspath
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    with connect(args.datapath) as ase_db:
        ase_db.metadata = {
            "_distance_unit": "Bohr",
            "_property_unit_dict": {"energy": "Hartree"},
        }
        dataset_length = len(ase_db)
    suffix = args.datapath.split("_")[-1]
    data_test = NablaDFT(
        "ASE",
        args.dataset_name,
        datapath=args.datapath,
        data_workdir=workpath,
        distance_unit="Bohr",
        batch_size=args.batch_size,
        train_ratio=0,
        num_workers=1,
        transforms=[
            trn.ASENeighborList(cutoff=args.cutoff),
            trn.CastTo32()
        ],
        split_file=os.path.join(workpath, "split_test.npz")
    )

    data_test.dataset.prepare_data()
    data_test.dataset.setup()

    best_model = torch.load(os.path.join(workpath, "best_inference_model"))
    best_model = best_model.cuda()
    best_model = best_model.eval()

    metric = torchmetrics.MeanAbsoluteError()

    with torch.no_grad():
        for x in tqdm.tqdm(
            data_test.dataset.val_dataloader(), total=dataset_length // args.batch_size
        ):
            for k in x:
                if x[k].dtype == torch.float64:
                    x[k] = x[k].float()
                x[k] = x[k].to(device)

            target = x["energy"].cpu().clone()
            prediction = best_model(x)["energy"].cpu()
            mae = metric(prediction, target)

    print(metric.compute())
