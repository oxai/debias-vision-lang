import os
import subprocess
from abc import ABC
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from gdown import download
from torch.utils.data import Dataset

from debias_clip import Dotdict, FAIRFACE_DATA_PATH


class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    AGE_ENCODING = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
                    "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(label_encoding.keys()), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(lambda x: label_encoding[x]), dtype=int)
        # assert labels_list.sum() != 0 and (1 - labels_list).sum() != 0, "Labels are all equal, invalid for Weat"
        return labels_list, len(label_encoding)


class FairFace(IATDataset):
    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}

    def __init__(self, iat_type: str = None, lazy: bool = True, mode: str = "train",
                 _n_samples: Union[float, int] = None, transforms: Callable = None, equal_split: bool = True, ):
        self.DATA_PATH = str(FAIRFACE_DATA_PATH)
        self.download_data()
        self.mode = mode
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(os.path.join(self.DATA_PATH, "labels", mode, f"{mode}_labels.csv"))
        self.labels.sort_values("file", inplace=True)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [os.path.join(self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def download_data(self):
        os.makedirs(self.DATA_PATH, exist_ok=True)
        # Use 1.25 padding
        fairface_parts = {
            "imgs": {
                "train_val": ("https://drive.google.com/uc?id=1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL", "train_val_imgs.zip"),
            },
            "labels": {
                "train": ("https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH", "train_labels.csv"),
                "val": ("https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D", "val_labels.csv")
            }
        }

        for part_name, part in fairface_parts.items():
            for subpart_name, (subpart_url, subpart_fname) in part.items():
                subpart_dir = os.path.join(self.DATA_PATH, part_name, subpart_name)
                if os.path.isdir(subpart_dir):
                    continue
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

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, "imgs", "train_val", res.file)
        res.img = self._transforms(Image.open(img_fname))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)
