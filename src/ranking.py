import warnings
from random import sample
from typing import List, Iterable, Optional, Union, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import stats as spstats
from collections import Counter

tqdm.pandas()

import src
from src.data_utils import _load_cache, _save_cache


def compute_weighted_kendall(df: pd.DataFrame, top_n: int) -> float:
    kendall_weigher = lambda r: math.exp(-r / top_n)
    kendall, _ = spstats.weightedtau(
        df.label,
        df.score,
        rank=spstats.rankdata(df.score.to_list(), method="ordinal") - 1,
        weigher=kendall_weigher,
    )
    assert kendall == kendall, "Got NaN value from kendall tau rank correlation."

    return kendall


def normalized_discounted_KL(df: pd.DataFrame, top_n: int) -> dict:
    def KL_divergence(p, q):
        return np.sum(np.where(p != 0, p * (np.log(p) - np.log(q)), 0))

    result_metrics = {f"ndkl_eq_opp": 0.0, f"ndkl_dem_par": 0.0}

    top_n = len(df)

    _, label_counts = zip(
        *sorted(Counter(df.label).items())
    )  # ensures counts are ordered according to label ordering

    # if label count is 0, set it to 1 to avoid degeneracy
    desired_dist = {
        "eq_opp": np.array([1 / len(label_counts) for _ in label_counts]),
        "dem_par": np.array([max(count, 1) / len(df) for count in label_counts]),
    }

    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_label_counts = np.zeros(len(label_counts))

    for index, (_, row) in enumerate(top_n_scores.iterrows(), start=1):
        label = int(row["label"])
        top_n_label_counts[label] += 1
        for dist_name, dist in desired_dist.items():
            kl_div = KL_divergence(top_n_label_counts / index, dist)
            result_metrics[f"ndkl_{dist_name}"] += kl_div / math.log2(index + 1)

    Z = sum(1 / math.log2(i + 1) for i in range(1, top_n + 1))  # normalizing constant

    for dist_name in result_metrics:
        result_metrics[dist_name] /= Z

    return result_metrics


def compute_skew_metrics(
    df: pd.DataFrame, top_n: int, save_indiv_skew: bool = False
) -> dict:
    """
    See https://arxiv.org/pdf/1905.01989.pdf
    equality of opportunity: if there are unique n labels, the desired distribution has 1/n proportion of each
    demographic parity: if the complete label set has p_i proportion of label i,
    the desired distribution has p_i of label i
    NOTE: this needs skew@k with k<len(dataset)
    """

    result_metrics = {
        f"minskew_eq_opp_{top_n}": 0,
        f"minskew_dem_par_{top_n}": 0,
        f"maxskew_eq_opp_{top_n}": 0,
        f"maxskew_dem_par_{top_n}": 0,
    }

    label_counts = Counter(df.label)
    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_counts = Counter(top_n_scores.label)
    for label_class, label_count in label_counts.items():
        skew_dists = {"eq_opp": 1 / len(label_counts), "dem_par": label_count / len(df)}
        p_positive = top_n_counts[label_class] / top_n

        # no log of 0
        if p_positive == 0:
            print(
                f"Got no positive samples in top {str(top_n)} ranked entries -- label {str(label_class)}. \
                  \nMinSkew might not be reliable"
            )
            p_positive = 1 / top_n

        for dist_name, dist in skew_dists.items():
            skewness = math.log(p_positive) - math.log(dist)
            result_metrics[f"minskew_{dist_name}_{top_n}"] = min(
                result_metrics[f"minskew_{dist_name}_{top_n}"], skewness
            )
            result_metrics[f"maxskew_{dist_name}_{top_n}"] = max(
                result_metrics[f"maxskew_{dist_name}_{top_n}"], skewness
            )

            if save_indiv_skew:
                # WARN: This might cause failure when updating prompt dataframe!
                result_metrics[f"skew_{dist_name}_{str(label_class)}"] = skewness

    return result_metrics


def get_prompt_embeddings(
    model, tokenizer, device: torch.device, prompt: str
) -> torch.Tensor:
    with torch.no_grad():
        prompt_tokenized = tokenizer(prompt).to(device)
        prompt_embeddings = model.encode_text(prompt_tokenized)
        prompt_embeddings /= prompt_embeddings.norm(dim=-1, keepdim=True)

    prompt_embeddings = prompt_embeddings.to(device).float()
    return prompt_embeddings


def get_labels_img_embeddings(
    images_dl: DataLoader[src.datasets.IATDataset],
    model,
    model_alias,
    device: torch.device,
    progress: bool = False,
    labels_group: str = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Computes all image embeddings and corresponding labels"""
    if progress:
        progbar = tqdm
    else:

        def progbar(it, *args, **kwargs):
            return it

    assert hasattr(images_dl.dataset, "recomp_img_embeddings")
    assert hasattr(images_dl.dataset, "recomp_iat_labels")
    images_dl.dataset.recomp_img_embeddings(model, model_alias, device, progress)
    images_dl.dataset.recomp_iat_labels(labels_group)

    if isinstance(images_dl.sampler, torch.utils.data.sampler.SequentialSampler):
        image_embeddings = images_dl.dataset.image_embeddings
        image_labels = images_dl.dataset.iat_labels
    else:
        image_embeddings = []
        image_labels = []
        for batch in progbar(
            images_dl, desc="Processing shuffled images", position=1, leave=False
        ):
            # encode images in batches for speed, move to cpu when storing to not waste GPU memory
            image_embeddings.append(batch["img_embedding"].to(device))
            image_labels.extend(batch["iat_label"])
        image_embeddings = torch.cat(image_embeddings, dim=0)

    return np.array(image_labels), image_embeddings.to(device)


def do_clip_audit(
    dl,
    ca_prompts,
    model,
    model_alias,
    tokenizer,
    preprocess,
    device,
    use_templates: bool = False,
):
    RACE_ENCODING = {
        "White": 0,
        "Southeast Asian": 1,
        "Middle Eastern": 2,
        "Black": 3,
        "Indian": 4,
        "Latino_Hispanic": 5,
        "East Asian": 6,
    }
    labels_list, image_embeddings = get_labels_img_embeddings(
        dl, model, model_alias, device, progress=True, labels_group="race"
    )
    _regular_prompts = (
        ca_prompts["human"] + ca_prompts["nonhuman"] + ca_prompts["crime"]
    )
    if use_templates:
        _templates = ["a photo of a {}"]
        regular_prompts = []
        for t in _templates:
            for p in _regular_prompts:
                regular_prompts.append(t.format(p))
    else:
        regular_prompts = _regular_prompts
    nonhuman_inxs = (
        len(ca_prompts["human"]),
        len(ca_prompts["human"]) + len(ca_prompts["nonhuman"]),
    )
    crime_inxs = (
        len(ca_prompts["human"]) + len(ca_prompts["nonhuman"]),
        len(regular_prompts),
    )

    regular_prompt_embeddings = torch.cat(
        [
            get_prompt_embeddings(model, tokenizer, device, prompt)
            for prompt in regular_prompts
        ]
    )
    similarities = (image_embeddings @ regular_prompt_embeddings.T).cpu().numpy()
    pred_classes = similarities.argmax(axis=-1)
    # Due to male & female versions of each
    pred_classes[
        (2 * len(RACE_ENCODING) > pred_classes) & (pred_classes >= len(RACE_ENCODING))
    ] -= len(RACE_ENCODING)
    correct_pred = pred_classes == labels_list
    nonhuman_pred = (nonhuman_inxs[0] <= pred_classes) & (
        pred_classes < nonhuman_inxs[1]
    )
    crime_pred = (crime_inxs[0] <= pred_classes) & (pred_classes < crime_inxs[1])

    res = pd.DataFrame()
    for race, race_inx in RACE_ENCODING.items():
        label_mask = labels_list == race_inx
        n_w_label = label_mask.sum()
        if n_w_label == 0:
            continue
        prop_correct = correct_pred[label_mask].sum() / n_w_label
        prop_nonhuman = nonhuman_pred[label_mask].sum() / n_w_label
        prop_crime = crime_pred[label_mask].sum() / n_w_label
        res = res.append(
            pd.DataFrame(
                [
                    {
                        "ff_race_category": race,
                        "correct": prop_correct * 100,
                        "nonhuman": prop_nonhuman * 100,
                        "crime": prop_crime * 100,
                    }
                ]
            ),
            ignore_index=True,
        )

    return res


def eval_ranking(
    model: torch.nn.Module,
    model_alias: str,
    tokenizer,
    images: DataLoader,
    prompt_group: str,
    prompt: str,
    device: torch.device,
    evaluation: str = "topn",
    top_n: Iterable[Union[int, float]] = None,
    progress: bool = False,
):
    assert evaluation in ("topn", "corr", "skew", "maxskew", "minskew", "ndkl")

    labels_list, image_embeddings = get_labels_img_embeddings(
        images, model, model_alias, device, progress=progress, labels_group=prompt_group
    )
    prompt_embeddings = get_prompt_embeddings(model, tokenizer, device, prompt)

    similarities = (image_embeddings @ prompt_embeddings.T).cpu().numpy().flatten()
    summary = pd.DataFrame({"score": similarities, "label": labels_list})

    q_topns, q_labels = [], []
    for q in top_n:
        # Float -> proportion of the dataset
        # Int -> top n
        if isinstance(q, float):
            q_labels.append(f"Top-{q:.2%}")
            q_topns.append(math.ceil(len(summary) * q))
        else:
            q_labels.append(f"Top-{q}")
            q_topns.append(q)

    result = {}
    if evaluation == "corr":
        # rank correlation metrics (kendall tau)
        for q, q_label in zip(q_topns, q_labels):
            kendall = compute_weighted_kendall(summary, top_n=q)
            result[f"kendall {q_label}"] = kendall

    elif evaluation == "topn":
        # proportion in top-k
        top_n_max = summary.nlargest(max(q_topns), columns="score", keep="all")

        for q, q_label in zip(q_topns, q_labels):
            proportions = top_n_max.iloc[:q].label.value_counts(normalize=True)
            for index, proportion in proportions.iteritems():
                result[f"{q_label}({int(index)})"] = proportion

    elif "skew" in evaluation:
        # skew metric of top-k results
        for q, q_label in zip(q_topns, q_labels):
            skewness = compute_skew_metrics(summary, top_n=q)
            result.update(skewness)
    elif evaluation == "ndkl":
        for q, q_label in zip(q_topns, q_labels):
            ndkl = normalized_discounted_KL(summary, top_n=q)
            result.update(ndkl)
    else:
        raise NotImplementedError

    return result


def ranking_experiment(
    prompt_templates: pd.DataFrame,
    model: torch.nn.Module,
    model_alias: str,
    tokenizer,
    dl: DataLoader[src.datasets.IATDataset],
    device: torch.device,
    evaluation: str = "topn",
    progress: bool = False,
    top_n: Iterable[Union[int, float]] = (1000,),
) -> pd.DataFrame:
    def _run_ranking_exp(_row: pd.Series):
        _group, _prompt = _row["group"], _row["prompt"]

        result = eval_ranking(
            model,
            model_alias,
            tokenizer,
            dl,
            _group,
            _prompt,
            device,
            evaluation=evaluation,
            top_n=top_n,
            progress=progress,
        )

        return result

    experiment_results = prompt_templates.progress_apply(_run_ranking_exp, axis=1)

    full_results = prompt_templates.join(
        pd.DataFrame([x for x in experiment_results])
    ).fillna(0.0)
    return full_results
