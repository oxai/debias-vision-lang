import csv
import glob
import os
import random
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from src import Dotdict, PATHS
from src.data_utils import _load_cache, _save_cache

"""
We follow OpenAI's CIFAR evaluation protocol, to do classification in a ZS way. As inferred from their method, 
they average the embeddings of multiple prompts. We saw a ~2% increase in performance on CIFAR100 compared 
with only "a photo of a {}"
SEE: https://github.com/openai/CLIP/blob/e184f608c5d5e58165682f7c332c3a8b4c1545f2/data/prompts.md
"""


def _load_prompt_templates(fname: str):
    raw_prompt_iterations = pd.read_csv(
        os.path.join(PATHS.IAT.PROMPTS, fname), skipinitialspace=True
    )
    templates = dict()
    for iat_type in raw_prompt_iterations:
        templates[iat_type] = (
            raw_prompt_iterations[iat_type].dropna().str.strip().tolist()
        )

    pt_columns = ["group", "template"]
    prompt_groups = list(templates.keys())
    prompt_templates = []
    for group in prompt_groups:
        for template in templates[group]:
            prompt_templates.append((group, template))
    prompt_templates = pd.DataFrame(prompt_templates, columns=pt_columns)

    return prompt_templates


PROMPT_TEMPLATES = {}
for prompt_file in glob.glob(os.path.join(PATHS.IAT.PROMPTS, "prompt_*.csv")):
    PROMPT_TEMPLATES[
        os.path.basename(prompt_file)[: -len(".csv")]
    ] = _load_prompt_templates(os.path.basename(prompt_file))


CIFAR_ZS_OAI_PROMPTS = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]


class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    AGE_ENCODING = {
        "0-2": 0,
        "3-9": 1,
        "10-19": 2,
        "20-29": 3,
        "30-39": 4,
        "40-49": 5,
        "50-59": 6,
        "60-69": 7,
        "more than 70": 8,
    }

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None):

        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = (
                IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
            )
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = (
                self.RACE_ENCODING if label_encoding is None else label_encoding
            )
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = (
                IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
            )
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(
            label_encoding.keys()
        ), "There is a missing label, invalid for WEAT"
        labels_list = np.array(
            labels_list.apply(lambda x: label_encoding[x]), dtype=int
        )

        return labels_list, len(label_encoding)

    def recomp_img_embeddings(
        self,
        model: torch.nn.Module,
        model_alias: str,
        device: torch.device,
        progress: bool = True,
    ):
        self.image_embeddings = None
        self.image_embeddings = compute_img_embeddings(
            self, model, model_alias, device, progress
        )

    def recomp_iat_labels(self, iat_type: str = None, label_encoding=None):
        if iat_type is None:
            iat_type = self.iat_type
        if self.iat_labels is not None:
            return
        self.iat_labels, self.n_iat_classes = self.gen_labels(iat_type, label_encoding)


def compute_img_embeddings(
    dataset: IATDataset,
    model,
    model_alias: str,
    device: torch.device,
    progress: bool = True,
) -> torch.Tensor:
    if progress:
        progbar = tqdm.tqdm
    else:

        def progbar(it, *args, **kwargs):
            return it

    if dataset.use_cache:
        img_embeddings_cachename = (
            f"preproc_image_embeddings_"
            f"{str(dataset.__class__.__name__)+dataset.mode.capitalize()}_{model_alias}"
        )
        image_embeddings = _load_cache(img_embeddings_cachename)
    else:
        img_embeddings_cachename = None
        image_embeddings = None

    if image_embeddings is None:
        print(
            f"Computing image embeddings for {model_alias}"
            f"{f', storing to cache {img_embeddings_cachename}' if dataset.use_cache else ''}..."
        )
        image_embeddings = []
        dl = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=128)
        with torch.no_grad():
            for batch in progbar(
                dl,
                desc="Processing images",
                position=1,
                leave=False,
                miniters=len(dl) // 200,
            ):
                # encode images in batches for speed, move to cpu when storing to not waste GPU memory
                output = model.encode_image(batch["img"].to(device))
                image_embeddings.append(output.cpu().float())

            image_embeddings = torch.cat(image_embeddings)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            if dataset.use_cache:
                _save_cache(img_embeddings_cachename, image_embeddings)
            image_embeddings = image_embeddings.to(device)

    return torch.squeeze(image_embeddings)


class FairFace(IATDataset):
    RACE_ENCODING = {
        "White": 0,
        "Southeast Asian": 1,
        "Middle Eastern": 2,
        "Black": 3,
        "Indian": 4,
        "Latino_Hispanic": 5,
        "East Asian": 6,
    }

    def __init__(
        self,
        *args,
        iat_type: str = None,
        lazy: bool = True,
        mode: str = "train",
        _n_samples: Union[float, int] = None,
        transforms: Callable = None,
        equal_split: bool = True,
        use_cache: bool = True,
        caching_params: dict = None,
        **kwargs,
    ):
        self.mode = mode
        self.use_cache = use_cache
        self._transforms = (lambda x: x) if transforms is None else transforms
        self.labels = pd.read_csv(
            os.path.join(PATHS.FAIRFACE.LABELS, mode, f"{mode}_labels.csv")
        )
        self.labels.sort_values("file", inplace=True)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)

            self.labels = self.labels[:_n_samples]
        if equal_split:
            labels_male = self.labels.loc[self.labels["gender"] == "Male"]
            labels_female = self.labels.loc[self.labels["gender"] == "Female"]

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = labels_male.append(labels_female, ignore_index=True)

        self._img_fnames = [
            os.path.join(PATHS.FAIRFACE.IMAGES, x) for x in self.labels["file"]
        ]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.image_embeddings = None
        if caching_params is not None:
            self.image_embeddings = compute_img_embeddings(self, **caching_params)

        self.iat_type = iat_type
        self.iat_labels = None
        if self.iat_type is not None:
            self.recomp_iat_labels(iat_type)

    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(PATHS.FAIRFACE.IMAGES, res.file)
        if self.image_embeddings is not None:
            res.img_embedding = self.image_embeddings[self._fname_to_inx[img_fname]]
        else:
            res.img = self._transforms(Image.open(img_fname))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        if self.iat_labels is not None:
            ff_sample.iat_label = self.iat_labels[index]
        return ff_sample

    def __len__(self):
        return len(self.labels)


class UTKFace(IATDataset):
    RACE_ENCODING = {
        "White": 0,
        "Black": 1,
        "East Asian_Southeast Asian": 2,
        "Indian": 3,
        "Latino_Hispanic_Middle Eastern": 4,
    }

    def __init__(
        self,
        *args,
        iat_type: str = None,
        lazy: bool = True,
        _n_samples: Union[float, int] = None,
        transforms: Callable = None,
        use_cache: bool = True,
        caching_params: dict = None,
        mode: str = None,
        **kwargs,
    ):
        if mode is not None:
            print(
                f"UTKFace was given mode {mode}, but only has one mode, which will be used."
            )
        for n, k in kwargs.items():
            print(f"UTKFace was unrecognised argument {n}: {k}.")
        self.mode = ""

        self._transforms = (lambda x: x) if transforms is None else transforms
        self.image_paths = sorted(
            glob.glob(os.path.join(PATHS.UTKFACE.IMGS, "*.jpg")), key=os.path.basename
        )
        self.use_cache = use_cache
        # To get a shuffle, but the same shuffle every time
        random.Random(1).shuffle(self.image_paths)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.image_paths) * _n_samples)
            self.image_paths = self.image_paths[:_n_samples]

        self.labels = pd.DataFrame(
            list(
                map(
                    self._utk_labelparse,
                    [os.path.basename(x) for x in self.image_paths],
                )
            )
        )

        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))

        self.labels.sort_values("file", inplace=True)

        self._img_fnames = self.image_paths
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.image_embeddings = None
        if caching_params is not None:
            self.image_embeddings = compute_img_embeddings(self, **caching_params)

        self.iat_type = iat_type
        self.iat_labels = None
        if self.iat_type is not None:
            self.recomp_iat_labels(iat_type)

    @staticmethod
    def _utk_labelparse(fname: str) -> dict:
        def age_binning(age: int) -> str:
            """
            FairFace style age binning:
            Age groups include: 0-2, 3-9, 10-19, 20-29, ..., 60-69, more than 70
            """
            if 0 <= age <= 2:
                return "0-2"
            elif 3 <= age <= 9:
                return "3-9"
            else:  # age >= 10
                tens = age // 10
                if tens >= 7:
                    return "more than 70"
                else:
                    return f"{str(tens * 10)}-{str(tens * 10 + 9)}"

        inx_to_ff_races = [
            "White",
            "Black",
            "East Asian_Southeast Asian",
            "Indian",
            "Latino_Hispanic_Middle Eastern",
        ]
        num_labels = fname.split("_")[:3]
        labels = {
            "age": age_binning(
                int(num_labels[0])
            ),  # f"{num_labels[0]}-{int(num_labels[0]) + 1}",
            "gender": "Female" if int(num_labels[1]) else "Male",
            "race": inx_to_ff_races[int(num_labels[2])],
            "file": fname,
        }
        return labels

    def _load_utkface_sample(self, img_fname) -> dict:
        res = Dotdict(dict())
        res.file = os.path.basename(img_fname)
        res.update(self._utk_labelparse(res.file))

        if self.image_embeddings is not None:
            res.img_embedding = self.image_embeddings[self._fname_to_inx[img_fname]]
        else:
            res.img = self._transforms(Image.open(img_fname))

        if self.iat_labels is not None:
            res.iat_label = self.iat_labels[self._fname_to_inx[img_fname]]

        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        utk_sample = self._load_utkface_sample(self.image_paths[index])
        return utk_sample

    def __len__(self):
        return len(self.image_paths)


class CelebA(IATDataset):
    def __init__(
        self,
        *args,
        cropped: bool = False,
        _n_samples: Union[float, int] = None,
        transforms=None,
        iat_type: str = None,
        use_cache: bool = True,
        caching_params: dict = None,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.img_base_path = PATHS.CELEBA.CROPPED if cropped else PATHS.CELEBA.UNCROPPED
        self._dont_load_images = False
        self.labels = self.parse_raw_attrs()
        self._transforms = (lambda x: x) if transforms is None else transforms
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)
            self.labels = self.labels[:_n_samples]

        self.labels = pd.DataFrame(self.labels)

        if mode := kwargs.pop("mode", None) is not None:
            print(
                f"CelebA was given mode {mode}, but only has one mode, which will be used."
            )
        self.mode = ""
        if kwargs:
            print(f"CelebA got unknown kwargs: {kwargs}")

        self._img_fnames = [
            os.path.join(self.img_base_path, x) for x in self.labels["file"]
        ]
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.image_embeddings = None
        if caching_params is not None:
            self.image_embeddings = compute_img_embeddings(self, **caching_params)

        self.iat_type = iat_type
        self.iat_labels = None
        if self.iat_type is not None:
            self.recomp_iat_labels(iat_type)

    @staticmethod
    def parse_raw_attrs() -> List[Dict]:
        interested_in = ["Male", "Young"]

        def _label_standardizer(sample: dict):
            sample["gender"] = "Male" if sample["Male"] else "Female"
            del sample["Male"]
            sample["age"] = "20" if sample["Young"] else "50"
            del sample["Young"]
            sample["race"] = "N/A"
            return sample

        with open(PATHS.CELEBA.ATTRS, mode="r") as attr_file:
            file_lines = [x.strip() for x in attr_file.readlines()]
            _, attrs_raw, *data_lines = file_lines
            attr_lookup = {
                attr_name: attr_inx
                for attr_inx, attr_name in enumerate(attrs_raw.split())
            }
            # Parses the data (unpack into for loops to understand)
            img_to_attrs = {
                img_name: _label_standardizer(
                    {
                        attr_name: attr_data[attr_lookup[attr_name]] == "1"
                        for attr_name in interested_in
                    }
                )
                for img_name, *attr_data in map(lambda row: row.split(), data_lines)
            }

        return sorted(
            [{"file": img_name, **labels} for img_name, labels in img_to_attrs.items()],
            key=lambda v: v["file"],
        )

    def _load_celeba_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.img_base_path, res.file)

        if self.image_embeddings is not None:
            res.img_embedding = self.image_embeddings[self._fname_to_inx[img_fname]]
        else:
            res.img = self._transforms(Image.open(img_fname))

        if self.iat_labels is not None:
            res.iat_label = self.iat_labels[self._fname_to_inx[img_fname]]

        return res

    def __getitem__(self, index):
        return self._load_celeba_sample(self.labels.iloc[index])

    def __len__(self):
        return len(self.labels)


class COCOGender(IATDataset):
    def __init__(
        self,
        *args,
        iat_type: str = None,
        _n_samples: Union[float, int] = None,
        transforms: Callable = None,
        use_cache: bool = True,
        caching_params: dict = None,
        mode: str = None,
        **kwargs,
    ):
        if mode is None:
            print(f"COCOGender was given no mode {mode}, train will be used.")
            mode = "train"
        self.mode = mode
        self.base_img_path = PATHS.COCOGENDER[self.mode.upper()]
        for n, k in kwargs.items():
            print(f"COCOGender was given unrecognised argument {n}: {k}.")

        self._transforms = (lambda x: x) if transforms is None else transforms
        self.image_paths = []
        self.image_paths.extend(
            sorted(
                glob.glob(os.path.join(self.base_img_path, "male", "*.jpg")),
                key=os.path.basename,
            )
        )
        self.image_paths.extend(
            sorted(
                glob.glob(os.path.join(self.base_img_path, "female", "*.jpg")),
                key=os.path.basename,
            )
        )
        self.use_cache = use_cache
        # To get a consistently reproducible shuffle
        random.Random(1).shuffle(self.image_paths)
        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.image_paths) * _n_samples)
            self.image_paths = self.image_paths[:_n_samples]

        self.label_parser = lambda p: {
            "file": p,
            "gender": os.path.dirname(p).split(os.path.sep)[-1].capitalize(),
        }
        self.labels = pd.DataFrame(
            list(map(self.label_parser, [x for x in self.image_paths]))
        )

        self.images_list = None
        self.labels.sort_values("file", inplace=True)

        self._img_fnames = self.image_paths
        self._fname_to_inx = {fname: inx for inx, fname in enumerate(self._img_fnames)}

        self.image_embeddings = None
        if caching_params is not None:
            self.image_embeddings = compute_img_embeddings(self, **caching_params)

        self.iat_type = iat_type
        self.iat_labels = None
        if self.iat_type is not None:
            self.recomp_iat_labels(iat_type)

    def _load_cocogender_sample(self, img_fname) -> dict:
        res = Dotdict()
        res.file = img_fname
        res.update(self.label_parser(res.file))

        if self.image_embeddings is not None:
            res.img_embedding = self.image_embeddings[self._fname_to_inx[img_fname]]
        else:
            res.img = self._transforms(Image.open(img_fname))

        if self.iat_labels is not None:
            res.iat_label = self.iat_labels[self._fname_to_inx[img_fname]]

        return res

    def __getitem__(self, index: int):
        return self._load_cocogender_sample(self.image_paths[index])

    def __len__(self):
        return len(self.image_paths)


class IATWords(object):
    def __init__(self, iat_type: str, prompt: Union[str, int] = "{}"):
        self.iat_type = iat_type

        if isinstance(prompt, str):
            self.prompt = prompt
        elif isinstance(prompt, int):
            self.prompt = PROMPT_TEMPLATES[iat_type][prompt]

        self.prompt_file = os.path.join(
            PATHS.IAT.PROMPTS, f"{iat_type.split('.')[0]}.csv"
        )

        with open(self.prompt_file, "r") as f:
            myreader = csv.reader(f, delimiter=",")
            a_attrs, b_attrs = myreader

        templater = lambda p: self.prompt.format(p.strip())
        self.A: List[str] = list(map(templater, a_attrs))
        self.B: List[str] = list(map(templater, b_attrs))


class TextImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        mode: str = "train",
        subsample: float = 1.0,
        use_cache: bool = True,
        caching_params: dict = None,
        transforms: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.mode = mode
        self.subsample = subsample
        self.use_cache = use_cache
        self._transforms = (lambda x: x) if transforms is None else transforms

        self._load_metadata()
        if self.subsample < 1 and self.mode in ["train", "predict"]:
            print(
                f"WARNING: down-sampling data to fraction: {self.subsample}, should be used for",
                "debugging / working on smaller data only.",
            )
            self.metadata = self.metadata.sample(frac=self.subsample)

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError

    @abstractmethod
    def _get_caption(self, sample):
        raise NotImplementedError

    @abstractmethod
    def _get_img_path(self, sample):
        raise NotImplementedError

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        sample = self.metadata.iloc[item]
        img_fp, img_fn = self._get_img_path(sample)
        img = Image.open(img_fp)
        img = self._transforms(img)
        caption = self._get_caption(sample)
        return {"img": img, "text": caption, "img_fn": img_fn}


class Flickr30K(TextImageDataset):
    """
    Dataset download is nontrivial, you need to ask the authors for a copy.
    """

    def _load_metadata(self):
        df = pd.read_csv(PATHS.FLICKR30K.METADATA, sep="\t", names=["id", "caption"])
        df["filename"] = df["id"].str.split("#").str[0]
        df["caption_id"] = df["id"].str.split("#").str[1]

        if self.mode != "test":
            raise NotImplementedError(
                "Flickr is so far intended for evaluation (test on flickr1k), not trainining..."
            )
        else:
            test_ims = pd.read_csv(PATHS.FLICKR30K.TEST_LIST, names=["filename"])
            df = df[df["filename"].isin(test_ims["filename"])]

        self.metadata = df

    def _get_caption(self, sample):
        return sample["caption"]

    def _get_img_path(self, sample):
        return (
            os.path.join(PATHS.FLICKR30K.IMAGES, sample["filename"]),
            sample["filename"],
        )
