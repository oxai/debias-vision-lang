import os
from typing import Any
import torch.hub


class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        return super().get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)


PATHS = Dotdict()
PATHS.BASE = os.path.abspath(os.path.join(__file__, *([".."] * 4)))
tmpdir = os.environ['TMPDIR']
PATHS.DATA = os.path.join(tmpdir, "bias-vision-language/datasets")
PATHS.PLOTS = Dotdict()

PATHS.PLOTS.BASE = os.path.abspath(os.path.join(PATHS.BASE, "plots"))
PATHS.PLOTS.DEBIAS = os.path.abspath(os.path.join(PATHS.PLOTS.BASE, "debias"))

PATHS.CACHE = f"{tmpdir}/bias-vision-language/cache"
PATHS.MODEL_STORE = "/work/maxbain/Libs/bias-vision-language/models"

PATHS.TRAINED_MODELS = Dotdict()
PATHS.TRAINED_MODELS.BASE = os.path.join(PATHS.BASE, "models_trained")
PATHS.TRAINED_MODELS.METADATA = os.path.join(PATHS.TRAINED_MODELS.BASE, "runs_metadata.json")
PATHS.TRAINED_MODELS.TEST_PROMPTS = os.path.join(PATHS.TRAINED_MODELS.BASE, "test_prompts.json")

PATHS.FAIRFACE = Dotdict()
PATHS.FAIRFACE.BASE = os.path.abspath(os.path.join(PATHS.DATA, "fairface", "data"))
PATHS.FAIRFACE.IMAGES = os.path.join(PATHS.FAIRFACE.BASE, "imgs", "train_val")
PATHS.FAIRFACE.LABELS = os.path.join(PATHS.FAIRFACE.BASE, "labels")

PATHS.IAT = Dotdict()
PATHS.IAT.BASE = os.path.abspath(os.path.join(PATHS.DATA, "iat"))
PATHS.IAT.PROMPTS = os.path.abspath(os.path.join(PATHS.IAT.BASE, "prompts"))

PATHS.PSEUDOLABELS = Dotdict()
PATHS.PSEUDOLABELS.BASE = os.path.abspath(os.path.join(PATHS.DATA, "pseudolabels"))
PATHS.PSEUDOLABELS.GEN_BY = os.path.abspath(os.path.join(PATHS.PSEUDOLABELS.BASE, "generated_by.json"))
PATHS.PSEUDOLABELS.DATA = os.path.abspath(os.path.join(PATHS.PSEUDOLABELS.BASE, "data"))
PATHS.PSEUDOLABELS.FAIRFACE = {mode: os.path.abspath(os.path.join(PATHS.PSEUDOLABELS.DATA, f"{mode}_FairFace"))
                               for mode in ["train", "val"]}
PATHS.PSEUDOLABELS.UTKFACE = os.path.abspath(os.path.join(PATHS.PSEUDOLABELS.DATA, f"UTKFace"))

PATHS.UTKFACE = Dotdict()
PATHS.UTKFACE.BASE = os.path.abspath(os.path.join(PATHS.DATA, "utkface"))
PATHS.UTKFACE.IMGS = os.path.abspath(os.path.join(PATHS.UTKFACE.BASE, "data", "UTKFace"))

PATHS.CELEBA = Dotdict()
PATHS.CELEBA.BASE = os.path.abspath(os.path.join(PATHS.DATA, "celeba"))
PATHS.CELEBA.DATA = os.path.abspath(os.path.join(PATHS.CELEBA.BASE, "data"))
PATHS.CELEBA.UNCROPPED = os.path.abspath(os.path.join(PATHS.CELEBA.DATA, "uncropped"))
PATHS.CELEBA.CROPPED = os.path.abspath(os.path.join(PATHS.CELEBA.DATA, "cropped"))
PATHS.CELEBA.ATTRS = os.path.abspath(os.path.join(PATHS.CELEBA.DATA, "list_attr_celeba.txt"))

PATHS.FOODI = Dotdict()
PATHS.FOODI.BASE = os.path.abspath(os.path.join(PATHS.DATA, "foodi"))
PATHS.FOODI.IMGS = os.path.abspath(os.path.join(PATHS.FOODI.BASE, "imgs"))
PATHS.FOODI.RAW_METADATA = os.path.abspath(os.path.join(PATHS.FOODI.BASE, "raw_metadata.csv"))
PATHS.FOODI.METADATA = os.path.abspath(os.path.join(PATHS.FOODI.BASE, "metadata.csv"))
PATHS.FOODI.EN_DESCS = os.path.abspath(os.path.join(PATHS.FOODI.BASE, "en_descriptions.pkl"))

PATHS.CIFAR100 = Dotdict()
PATHS.CIFAR100.BASE = os.path.abspath(os.path.join(PATHS.DATA, "cifar100"))

PATHS.CIFAR10 = Dotdict()
PATHS.CIFAR10.BASE = os.path.abspath(os.path.join(PATHS.DATA, "cifar10"))

PATHS.FLICKR30K = Dotdict()
PATHS.FLICKR30K.BASE = os.path.abspath(os.path.join(PATHS.DATA, "flickr30k"))
PATHS.FLICKR30K.DATA = os.path.abspath(os.path.join(PATHS.FLICKR30K.BASE, "data"))
PATHS.FLICKR30K.METADATA = os.path.abspath(os.path.join(PATHS.FLICKR30K.BASE, "results_20130124.token"))
PATHS.FLICKR30K.TEST_LIST = os.path.abspath(os.path.join(PATHS.FLICKR30K.BASE, "flickr30k_test.txt"))
PATHS.FLICKR30K.IMAGES = os.path.abspath(os.path.join(PATHS.FLICKR30K.DATA, "flickr30k-images"))

PATHS.COCOGENDER = Dotdict()
PATHS.COCOGENDER.BASE = os.path.abspath(os.path.join(PATHS.DATA, "coco_gender"))
PATHS.COCOGENDER.TRAIN = os.path.abspath(os.path.join(PATHS.COCOGENDER.BASE, "train"))
PATHS.COCOGENDER.VAL = os.path.abspath(os.path.join(PATHS.COCOGENDER.BASE, "val"))

PATHS.ENV = Dotdict()
PATHS.ENV.PYTORCH_MODEL_CACHE = torch.hub.get_dir()

PATHS.RESULTS = os.path.join(PATHS.BASE, "results")

PATHS.EXPERIMENTS = Dotdict()
PATHS.EXPERIMENTS.RESULTS_DIR = os.path.join(PATHS.BASE, "experiment_results")

from . import datasets, datasets, weat_utils, models, ranking
from . import parse_config
