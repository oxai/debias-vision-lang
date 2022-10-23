import math
from collections import Counter, defaultdict
from typing import Union, Tuple, Callable, List

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from debias_clip import PROMPT_DATA_PATH
from debias_clip.datasets import IATDataset, FairFace
from debias_clip.model.model import ClipLike, model_loader


def normalized_discounted_KL(df: pd.DataFrame, top_n: int) -> dict:
    def KL_divergence(p, q):
        return np.sum(np.where(p != 0, p * (np.log(p) - np.log(q)), 0))

    result_metrics = {f"ndkl_eq_opp": 0.0, f"ndkl_dem_par": 0.0}

    _, label_counts = zip(*sorted(Counter(df.label).items()))  # ensures counts are ordered according to label ordering

    # if label count is 0, set it to 1 to avoid degeneracy
    desired_dist = {"eq_opp": np.array([1 / len(label_counts) for _ in label_counts]),
                    "dem_par": np.array([max(count, 1) / len(df) for count in label_counts])}

    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_label_counts = np.zeros(len(label_counts))

    for index, (_, row) in enumerate(top_n_scores.iterrows(), start=1):
        label = int(row["label"])
        top_n_label_counts[label] += 1
        for dist_name, dist in desired_dist.items():
            kl_div = KL_divergence(top_n_label_counts / index, dist)
            result_metrics[f"ndkl_{dist_name}"] += (kl_div / math.log2(index + 1))

    Z = sum(1 / math.log2(i + 1) for i in range(1, top_n + 1))  # normalizing constant

    for dist_name in result_metrics:
        result_metrics[dist_name] /= Z

    return result_metrics


def compute_skew_metrics(df: pd.DataFrame, top_n: int) -> dict:
    # See https://arxiv.org/pdf/1905.01989.pdf
    # equality of opportunity: if there are unique n labels, the desired distribution has 1/n proportion of each
    # demographic parity: if the complete label set has p_i proportion of label i,
    # the desired distribution has p_i of label i
    #   note this obviously needs skew@k with k<len(dataset)

    result_metrics = {f"maxskew_eq_opp": 0, f"maxskew_dem_par": 0}

    label_counts = Counter(df.label)
    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_counts = Counter(top_n_scores.label)
    for label_class, label_count in label_counts.items():
        skew_dists = {"eq_opp": 1 / len(label_counts), "dem_par": label_count / len(df)}
        p_positive = top_n_counts[label_class] / top_n

        # no log of 0
        if p_positive == 0:
            p_positive = 1 / top_n

        for dist_name, dist in skew_dists.items():
            skewness = math.log(p_positive) - math.log(dist)
            result_metrics[f"maxskew_{dist_name}"] = max(result_metrics[f"maxskew_{dist_name}"],
                                                                 skewness)

    return result_metrics


def get_prompt_embeddings(model: ClipLike, tokenizer, device: torch.device, prompts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        prompts_tokenized = tokenizer(prompts).to(device)
        prompt_embeddings = model.encode_text(prompts_tokenized)
        prompt_embeddings /= prompt_embeddings.norm(dim=-1, keepdim=True)

    prompt_embeddings = prompt_embeddings.to(device).float()
    return prompt_embeddings


def get_labels_img_embeddings(images_dl: DataLoader[IATDataset], model: ClipLike, device: torch.device,
                              progress: bool = False) -> Tuple[
    np.ndarray, torch.Tensor]:
    """Computes all image embeddings and corresponding labels"""

    image_embeddings = []
    image_labels = []
    for batch in tqdm(images_dl, desc="Embedding images", disable=not progress):
        # encode images in batches for speed, move to cpu when storing to not waste GPU memory
        with torch.no_grad():
            image_embeddings.append(model.encode_image(batch["img"].to(device)).cpu())
        image_labels.extend(batch["iat_label"])
    image_embeddings = torch.cat(image_embeddings, dim=0)

    return np.array(image_labels), image_embeddings.to(device)


def eval_ranking(labels_list: np.ndarray, image_embeddings: torch.Tensor, prompts_embeddings: torch.Tensor,
                 evaluation: str = "maxskew", topn: Union[int, float] = 1.0):
    assert evaluation in ("maxskew", "ndkl")
    eval_f = compute_skew_metrics if evaluation == "maxskew" else normalized_discounted_KL

    # Float -> proportion of the dataset
    # Int -> top n
    if isinstance(topn, float):
        topn = math.ceil(len(image_embeddings) * topn)

    results = defaultdict(lambda: [])
    for prompt_embedding in tqdm(prompts_embeddings, desc=f"Computing {evaluation}"):
        similarities = (image_embeddings.float() @ prompt_embedding.T.float()).cpu().numpy().flatten()
        summary = pd.DataFrame({"score": similarities, "label": labels_list})
        for k, v in eval_f(summary, top_n=topn).items():
            results[k[len(evaluation) + 1:]].append(v)

    return {k: sum(v) / len(v) for k, v in results.items()}


def gen_prompts():
    raw_data = pd.read_csv(PROMPT_DATA_PATH)
    templates = raw_data["template"].tolist()
    concepts = raw_data["concept"].tolist()

    prompts = []
    for template in templates:
        template = template.strip()
        if not template:
            continue
        prompts.extend(template.format(concept) for concept in concepts)
    return prompts


DEFAULT_OPTS = dict()


def measure_bias(cliplike: ClipLike, img_preproc: Callable, tokenizer: Callable, attribute="gender", opts=DEFAULT_OPTS):
    # do measurement
    ds = FairFace(mode="val", iat_type=attribute, transforms=img_preproc)
    dl = DataLoader(ds, batch_size=256, num_workers=6)

    prompts: List[str] = gen_prompts()

    evals = "maxskew", "ndkl"

    device = torch.device("cuda")
    labels_list, image_embeddings = get_labels_img_embeddings(dl, cliplike, device, progress=True)
    prompts_embeddings = get_prompt_embeddings(cliplike, tokenizer, device, prompts)

    result = {}
    for evaluation in evals:
        result[evaluation] = eval_ranking(labels_list, image_embeddings, prompts_embeddings, evaluation)

    return result


if __name__ == "__main__":
    import debias_clip

    model, img_preproc, tokenizer, alias_name = model_loader(
        "openai/CLIP/RN50", "cuda"
    )
    model.eval()

    # measure bias, lower == less biased
    print(debias_clip.measure_bias(model, img_preproc, tokenizer, attribute="race"))
