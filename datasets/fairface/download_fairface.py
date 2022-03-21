#! /usr/bin/env python3

import os
from gdown import download
import subprocess

BASE_PATH = os.path.abspath("")
OUTDATA_PATH = os.path.join(BASE_PATH, "data")

fairface_parts = {
    "imgs": {
        "train_val": (
            "https://drive.google.com/uc?id=1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL",
            "train_val_imgs.zip",
        ),
    },
    "labels": {
        "train": (
            "https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH",
            "train_labels.csv",
        ),
        "val": (
            "https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D",
            "val_labels.csv",
        ),
    },
}

for part_name, part in fairface_parts.items():
    for subpart_name, (subpart_url, subpart_fname) in part.items():
        subpart_dir = os.path.join(OUTDATA_PATH, part_name, subpart_name)
        os.makedirs(subpart_dir, exist_ok=True)
        print(f"Downloading fairface {subpart_name} {part_name}...")
        output_path = os.path.join(subpart_dir, subpart_fname)
        download(subpart_url, output=output_path)

        if subpart_fname.endswith(".zip"):
            print(f"Unzipping {subpart_name} {part_name}...")
            subprocess.check_output(["unzip", "-d", subpart_dir, output_path])
            os.remove(output_path)
            print(f"Done unzipping {subpart_name} {part_name}.")
        print(f"Done with fairface {subpart_name} {part_name}.")
