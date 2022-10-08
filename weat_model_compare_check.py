#%%

from importlib import reload
# For cuda debugging
import json
import os, sys
from src import PATHS
sys.path.insert(0, os.path.join(PATHS.BASE, "wip", "hugo", "src"))
import parse_config
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as iter_product

import src, src.debias, src.models, src.ranking, src.datasets, src.data_utils
from src.models import model_loader

if torch.cuda.device_count() > 1:
    use_device_id = int(input(f"Choose cuda index, from [0-{torch.cuda.device_count()-1}]: ").strip())
else: use_device_id = 0
use_device = "cuda:"+str(use_device_id) if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    input("CUDA isn't available, so using cpu. Please press any key to confirm this isn't an error: \n")
print("Using device", use_device)
torch.cuda.set_device(use_device_id)

with open(src.PATHS.TRAINED_MODELS.TEST_PROMPTS, mode="r") as _test_promptsfile:
    test_prompts_data = json.load(_test_promptsfile)

eval_dss = {"gender": [("FairFace", "val"), ("UTKFace", "val")],#, ("COCOGender", "val"), ],#("CelebA", "val")],  # has 200k images so takes looong to compute and we don't focus on it anyway
            "race": [("FairFace", "val"), ("UTKFace", "val")]}
evaluations = ["maxskew", "ndkl", "clip_audit"]
perf_evaluations = ["cifar10", "flickr1k", "cifar100"] # flickr1k, cifar100, cifar10
all_experiment_results = pd.DataFrame()
clip_audit_results = pd.DataFrame()
batch_sz = 256

try:
    with torch.cuda.device(use_device_id):
        for model_name in [f"m-bain/frozen-in-time/{x}" for x in ["cc", "ccwv2m", "wv2m", "ccwv2mcoco"]]:
            print(model_name)

            model, preprocess, tokenizer, model_alias = model_loader(model_name, device=use_device, jit=True)

            if model_name.startswith("m-bain/"):
                model.logit_scale = torch.tensor(-1, dtype=torch.float32, device=use_device)

            if "clip_audit" in evaluations:
                ca_prompts = test_prompts_data["clip_audit"]
                ca_ds = src.datasets.FairFace(iat_type="race", lazy=True, _n_samples=None, transforms=preprocess, mode="val")
                ca_dl = DataLoader(ca_ds, batch_size=batch_sz, shuffle=False, num_workers=8) # Shuffling ISN'T(!) reflected in the cache
                ca_res = src.ranking.do_clip_audit(ca_dl, ca_prompts, model, model_alias, tokenizer, preprocess, use_device, use_templates=True)
                for k, v in {"model_name": model_alias, "dataset": "FairFaceVal",
                             "evaluation": "clip_audit"}.items():
                    ca_res[k] = v
                clip_audit_results = clip_audit_results.append(ca_res, ignore_index=True)

                print("Done with clip audit")


            for debias_class in {"gender", "race"}:
                experiment_results = pd.DataFrame()
                test_prompts = test_prompts_data[debias_class]
                test_prompts_df = pd.DataFrame({"prompt": test_prompts})
                test_prompts_df["group"] = debias_class

                for perf_eval in perf_evaluations:
                    perf_res = {"model_name": model_alias, "dataset": perf_eval,
                                "evaluation": perf_eval, "debias_class": debias_class, "mean": src.debias.run_perf_eval(perf_eval, model, tokenizer, preprocess, use_device)}
                    experiment_results = experiment_results.append(pd.DataFrame([perf_res]), ignore_index=True)

                n_imgs = None # First run populates cache, thus run with None first, later runs can reduce number
                for dset_name, dset_mode in eval_dss[debias_class]:
                    ds = getattr(src.datasets, dset_name)(iat_type=debias_class, lazy=True, _n_samples=n_imgs, transforms=preprocess, mode=dset_mode)
                    dl = DataLoader(ds, batch_size=batch_sz, shuffle=False, num_workers=8) # Shuffling ISN'T(!) reflected in the cache

                    for evaluation in evaluations:
                        if evaluation == "clip_audit": continue
                        model.eval()
                        _res = src.debias.run_bias_eval(evaluation, test_prompts_df, model, model_alias, tokenizer, dl, use_device, cache_suffix="")
                        _res = src.debias.mean_of_bias_eval(_res, evaluation, "dem_par")
                        res = {}
                        for key, val in _res.items():
                            for rename in ["mean_", "std_"]:
                                if key.startswith(rename):
                                    res[rename[:-1]] = val
                                    break
                            else:
                                res[key] = val
                        res["model_name"] = model_alias
                        res["dataset"] = dset_name+dset_mode.capitalize()
                        res["evaluation"] = evaluation
                        experiment_results = experiment_results.append(pd.DataFrame([res]), ignore_index=True)

                experiment_results["debias_class"] = debias_class

                all_experiment_results = all_experiment_results.append(experiment_results)
        #del model, preprocess, tokenizer
finally:
    result_name = f"exp_vitb16_untrained_test_bias_results.csv"
    ca_result_name = f"exp_vitb16_untrained_test_clip_audit_results.csv"
    all_experiment_results.to_csv(os.path.join(src.PATHS.PLOTS.BASE, result_name))
    clip_audit_results.to_csv(os.path.join(src.PATHS.PLOTS.BASE, ca_result_name))

