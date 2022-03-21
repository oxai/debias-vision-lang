#! /usr/bin/env python3

import os
from gdown import download
import subprocess

BASE_PATH = os.path.abspath(os.path.join(__file__, ".."))
OUTDATA_PATH = os.path.join(BASE_PATH, "data")

drive_url = "drive link"
utkface_parts = [
    {
        "data": ("UTKFace.tar.gz", drive_url),
        "remove": [
            "39_1_20170116174525125.jpg.chip.jpg",
            "61_1_20170109142408075.jpg.chip.jpg",
            "61_1_20170109150557335.jpg.chip.jpg",
        ],
    }  # These don't have a race label
]

for dataset_part in utkface_parts:
    savename, gdrive_url = dataset_part["data"]
    print(f"Downloading utkface {savename} ...")
    output_path = os.path.join(OUTDATA_PATH, savename)
    download(gdrive_url, output=output_path)

    if savename.endswith(".tar.gz"):
        print(f"Untargz-ing {savename} ...")
        subprocess.check_output(["tar", "-xzf", output_path, "-C", OUTDATA_PATH])
        os.remove(output_path)
        print(f"Done untargz-ing {savename}.")

    for remove_name in dataset_part["remove"]:
        os.remove(os.path.join(OUTDATA_PATH, savename.split(".")[0], remove_name))
    print(f"Done with utkface {savename}.")
