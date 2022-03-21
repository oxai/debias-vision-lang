import warnings
from dataclasses import asdict
from typing import List, Iterable

import clip
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import tqdm
from tqdm.notebook import tqdm_notebook as tqdm

tqdm.pandas()

import src
from src.data_utils import _load_cache, _save_cache, WeatExperiment


def eval_weat(
    model,
    tokenizer,
    images: Iterable[torch.tensor],
    labels_list: List[int],
    iat_dataset: src.datasets.IATWords,
    device: torch.device,
    num_iters: int = 10000,
    dataset_name: str = None,
    progress: bool = True,
):
    """
    Params:
        images : iterable (for example a torch dataloader
        labels_list : numpy array of 0's and 1's
        good_attrs and bad_attrs: non-empty list of strings, which could be sentences
        num_iters : number of iterations for calculating p-value
        image_embeddings : Precomputed image embeddings, for example from a previous use of eval_weat
    :returns:
        result: dict["p_val", "effect_size"], image_embeddings: torch.Tensor
    """
    if progress:
        progbar = tqdm
    else:

        def progbar(*args, **kwargs):
            return args[0]

    if dataset_name is None:
        warnings.warn(
            "Did not specify a dataset name for WEAT, caching may use incorrect image embeddings."
        )

    img_embeddings_cachename = f"{str(dataset_name)}_{len(labels_list)}_{str(model)}"
    precomp_img_embeddings = _load_cache(img_embeddings_cachename) is not None
    image_embeddings = (
        _load_cache(img_embeddings_cachename) if precomp_img_embeddings else []
    )
    with torch.no_grad():
        good_tokenized = tokenizer(iat_dataset.A).to(device)
        good_embeddings = model.encode_text(good_tokenized)
        good_embeddings /= good_embeddings.norm(dim=-1, keepdim=True)

        bad_tokenized = tokenizer(iat_dataset.B).to(device)
        bad_embeddings = model.encode_text(bad_tokenized)
        bad_embeddings /= bad_embeddings.norm(dim=-1, keepdim=True)

        if not precomp_img_embeddings:
            n_imgs = 0
            for batch in progbar(
                images, desc="Processing images", position=1, leave=False
            ):
                # encode images in batches for speed, move to cpu when storing to not waste GPU memory
                imgs = batch["img"].to(device)
                output = model.encode_image(imgs)
                image_embeddings.append(output.cpu())
                n_imgs += output.shape[0]

    if not precomp_img_embeddings:
        image_embeddings = torch.cat(image_embeddings, dim=0)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    else:
        n_imgs = image_embeddings.shape[0]

    # When using jit-ed models, results may be half-precision
    image_embeddings = image_embeddings.to(device).float()
    good_embeddings, bad_embeddings = good_embeddings.float(), bad_embeddings.float()
    w_a = image_embeddings @ good_embeddings.T
    w_b = image_embeddings @ bad_embeddings.T

    w_A, w_B = w_a.mean(dim=1), w_b.mean(dim=1)

    s_A_B = w_A - w_B
    test_statistic = (
        s_A_B[labels_list].mean() - s_A_B[np.logical_not(labels_list)].mean()
    )

    effect_size = test_statistic / s_A_B.std()

    # calculating p-value
    num_above_statistic = 0
    for _ in progbar(
        range(num_iters), desc="Sampling for p-value", position=1, unit="samples"
    ):
        rand_labels = np.random.random(n_imgs) > 0.5
        # making sure we have atleast one of each label
        while rand_labels.mean() in [0, 1]:
            rand_labels = np.random.random(n_imgs) > 0.5

        sample_i = s_A_B[rand_labels].mean() - s_A_B[np.logical_not(rand_labels)].mean()
        # We measure the p-value on the correct side of the null hypothesis distribution
        if effect_size >= 0:
            if sample_i > test_statistic:
                num_above_statistic += 1
        else:
            if sample_i < test_statistic:
                num_above_statistic += 1

    _save_cache(img_embeddings_cachename, image_embeddings)
    return {
        "effect_size": effect_size.cpu().item(),
        "p_value": num_above_statistic / num_iters,
    }


def weat_on_prompt_templates(
    prompt_templates: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    ds: src.datasets.IATDataset,
    ds_name: str,
    dl: torch.utils.data.DataLoader,
    n_pval_samples: int,
    device: torch.device,
    label_choice=None,
    progress: bool = True,
    debug: bool = False,
):
    def _run_weat(_row: pd.Series):
        _group, _template = _row["group"], _row["template"]
        _group = _group.split(".")[0]
        labels_list = ds.gen_labels(_group, label_extra=label_choice)
        iat_ds = src.datasets.IATWords(_group, prompt=_template)
        if debug:
            print("Using prompt:", iat_ds.prompt)

        with WeatExperiment(
            model_desc=f"Weat+Dataloader with CLIP, {ds_name} first {len(ds)}, {_group}. Prompt: {iat_ds.prompt}",
            dataset_name=ds_name,
            n_samples=len(ds),
            n_pval_samples=n_pval_samples,
            A_attrs=iat_ds.A,
            B_attrs=iat_ds.B,
            prompt_template=iat_ds.prompt,
        ) as current_experiment:
            weat_res = eval_weat(
                model,
                tokenizer,
                dl,
                labels_list,
                iat_ds,
                device,
                num_iters=n_pval_samples,
                dataset_name=ds_name,
                progress=progress,
            )
            if debug:
                print(weat_res)
            current_experiment.effect_size = weat_res["effect_size"]
            current_experiment.p_value = weat_res["p_value"]
        return current_experiment

    if progress:
        experiment_results = prompt_templates.progress_apply(_run_weat, axis=1)
    else:
        experiment_results = prompt_templates.apply(_run_weat, axis=1)

    exp_columns = [
        "p_value",
        "effect_size",
        "dataset_name",
        "n_samples",
        "n_pval_samples",
    ]
    experiment_results = prompt_templates.join(
        pd.DataFrame([asdict(x) for x in experiment_results])[exp_columns]
    )
    return experiment_results


def plot_weat_prompt_results(exp_results: pd.DataFrame) -> sns.FacetGrid:
    def pval_format(val: float):
        if val < 0:
            return "N/A"
        elif val < 0.0001:
            return "*" * 4
        elif val < 0.001:
            return "*" * 3
        elif val < 0.01:
            return "*" * 2
        elif val < 0.05:
            return "*" * 1
        else:
            return "ns"

    g = sns.catplot(
        data=exp_results,
        y="template",
        x="effect_size",
        hue="group",
        orient="h",
        kind="bar",
        dodge=False,
        aspect=2,
        height=8,
    )

    for i, row in exp_results.iterrows():
        g.ax.text(
            row.effect_size,
            i,
            pval_format(row.p_value),
            ha="left" if row.effect_size > 0 else "right",
            size=8,
        )

    rep_res = exp_results.iloc[0]

    title = (
        f"WEAT on different prompts, {rep_res.n_samples} {rep_res.dataset_name} images,"
        f"{rep_res.n_pval_samples} p-value samples"
    )
    if "model_name" in exp_results.columns:
        title += f", Model: {exp_results.iloc[0].model_name}"

    g.set(xlabel="Effect Size", ylabel="Prompt Template", title=title)
    g.fig.set_size_inches([20, 10])
    g.fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    return g
