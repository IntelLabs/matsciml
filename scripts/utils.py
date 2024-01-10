from __future__ import annotations

import re

import numpy as np


def get_s2ef_ids(input):
    pattern = r"\[(\d+)\]"
    output = []
    for string in input:
        matches = re.findall(pattern, string)
        int_values = [int(match) for match in matches]
        output.append("_".join([str(_) for _ in int_values]))
    return output


def npz_convert(npz_id, npz_ood_cat, npz_ood_ads, npz_ood_both):
    full_dict = dict()
    datasets = {
        "id": npz_id,
        "ood_cat": npz_ood_cat,
        "ood_ads": npz_ood_ads,
        "ood_both": npz_ood_both,
    }
    full_dict = {}
    for dataset_name, dataset in datasets.items():
        ids = [item for item in [*dataset["ids"]]]
        full_dict[f"{dataset_name}_ids"] = get_s2ef_ids(ids)
        full_dict[f"{dataset_name}_energy"] = [
            item.astype(float).item() for item in [*dataset["energy"]]
        ]
        full_dict[f"{dataset_name}_forces"] = [
            [_.astype(float).item() for _ in item] for item in [*dataset["forces"]]
        ]
        try:
            full_dict[f"{dataset_name}_chunk_idx"] = np.cumsum(
                [item.astype(int).item() for item in [*dataset["chunk_ids"]]],
            ).astype(int)
        except Exception:
            pass
    return full_dict


def parse_npz_for_nans(file):
    data = np.load(file)
    if np.isnan(data["energy"]).any():
        print("Found nans in energy!")
    elif np.isnan(data["forces"]).any():
        print("Found nans in forces!")
    else:
        print("No nans found")


def make_submission_file(args):
    npz_id = np.load(args.id)
    npz_ood_cat = np.load(args.ood_cat)
    npz_ood_ads = np.load(args.ood_ads)
    npz_ood_both = np.load(args.ood_both)
    out_dict = npz_convert(npz_id, npz_ood_cat, npz_ood_ads, npz_ood_both)
    np.savez_compressed(args.out_path, **out_dict)
    print(f"Saved submission file: {args.out_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument(
        "--npz-file",
        type=str,
        default="",
        help="npz log file to parse",
    )
    argparser.add_argument("--submission", action="store_true")
    argparser.add_argument(
        "--id",
        help="Path to ID results. Required for OC20 and OC22.",
    )
    argparser.add_argument(
        "--ood-ads",
        help="Path to OOD-Ads results. Required only for OC20.",
    )
    argparser.add_argument(
        "--ood-cat",
        help="Path to OOD-Cat results. Required only for OC20.",
    )
    argparser.add_argument(
        "--ood-both",
        help="Path to OOD-Both results. Required only for OC20.",
    )
    argparser.add_argument("--out-path", default="submission_file.npz")
    args = argparser.parse_args()
    if args.submission:
        make_submission_file(args)
    else:
        parse_npz_for_nans(args.npz_file)
