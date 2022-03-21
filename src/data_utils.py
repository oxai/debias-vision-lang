import datetime
import pickle as pkl
import os
import glob
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, cast

import pandas as pd
import torch
import neptune.new as neptune

from . import PATHS


def get_neptune_run(project_name: str = None) -> neptune.Run:
    """Either finds a neptune token and uses it to give a neptune run,
    project_name: if not None, returns an object that silently does nothing on all attributes & calls
    """
    if project_name is None:

        class NeptuneDummy(object):
            def __getitem__(self, item):
                return self

            def __setitem__(self, key, value):
                pass

            def __getattr__(self, item):
                return self

            def __call__(self, *args, **kwargs):
                pass

        return cast(NeptuneDummy(), neptune.Run)

    # See https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch

    try:
        with open(
            os.path.join(PATHS.BASE, "..", "secrets", "neptune_api_key.txt"), mode="r"
        ) as neptune_api_key_file:
            api_key = neptune_api_key_file.read().strip()
    except FileNotFoundError as e:
        print(
            "Could not find a neptune api key, please put in secrets/neptune_api_key.txt, "
            "where secrets is at the same level as bias-vision-language"
        )

    return neptune.init(
        project=project_name,
        api_token=api_key,
        source_files=glob.glob(os.path.join(PATHS.BASE, "wip", "hugo", "**", "*.py")),
    )


base_path = os.path.dirname(__file__)
results_dir = PATHS.EXPERIMENTS.RESULTS_DIR
os.makedirs(results_dir, exist_ok=True)


def _save_experiment(name: str, experiment_setup, results):
    exp_type_dir = os.path.join(results_dir, name)
    exp_dir = os.path.join(exp_type_dir, f"{str(datetime.datetime.now())}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "experiment.pkl"), mode="wb") as exp_output_file:
        pkl.dump(experiment_setup, exp_output_file)
    with open(os.path.join(exp_dir, "results.pkl"), mode="wb") as res_output_file:
        pkl.dump(results, res_output_file)


def _get_obj_cache_key(data: object) -> str:
    # Should be replaced with a hash if caching with large keys (imagine the key being the weights of a model)
    return str(data)


def _save_cache(key: Union[object], data: object):
    key = _get_obj_cache_key(key)
    os.makedirs(PATHS.CACHE, exist_ok=True)
    save_path = os.path.join(PATHS.CACHE, f"{key}.pt")
    torch.save(data, save_path)


def _load_cache(key: Union[object]) -> Optional[object]:
    key = _get_obj_cache_key(key)
    try:
        cached_data = torch.load(os.path.join(PATHS.CACHE, f"{key}.pt"))
    except Exception:
        return None
    return cached_data


@dataclass
class WeatExperiment:
    model_desc: str
    dataset_name: str
    n_samples: int
    n_pval_samples: int
    A_attrs: List[str]
    B_attrs: List[str]
    prompt_template: Optional[str] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None

    def save(self):
        _save_experiment(
            name="weat",
            experiment_setup=asdict(self),
            results={"effect_size": self.effect_size, "p_value": self.p_value},
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


@dataclass
class DebiasExperiment:
    debias_cfg: dict
    prompt_cfg: dict
    train_cfg: dict
    optim_cfg: dict
    results: dict = None

    def save(self):
        _save_experiment(
            name="debias",
            experiment_setup=asdict(self),
            results=self.results if self.results is not None else dict(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
