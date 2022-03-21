import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src
from src import PATHS, Dotdict
from src.data_utils import DebiasExperiment, get_neptune_run
from src.datasets import Flickr30K
from src.metrics import t2v_metrics


def prepare_prompt_cfg(
    debias_class: str,
    paired_prompts: List[Tuple[str]],
    train_templates: List[str],
    test_templates: List[str],
    n_train: int,
    use_extra_test: bool = True,
    test_on_train: bool = False,
) -> Dotdict:
    cfg = Dotdict()
    cfg.N_TRAIN = n_train
    cfg.PAIR_PROMPTS = paired_prompts
    cfg.FLAT_PROMPTS = sum(map(list, cfg.PAIR_PROMPTS), [])

    assert (
        0 < n_train < len(cfg.FLAT_PROMPTS)
    ), "Must have at least one of both train and test prompts."

    cfg.BIAS_TRAIN_PROMPTS = sum(
        (
            list(template.format(x) for template in train_templates)
            for x in cfg.FLAT_PROMPTS[: cfg.N_TRAIN]
        ),
        [],
    )
    cfg.BIAS_TEST_PROMPTS = sum(
        (
            list(template.format(x) for template in test_templates)
            for x in cfg.FLAT_PROMPTS[cfg.N_TRAIN :]
        ),
        [],
    )

    if test_on_train:
        cfg.BIAS_TEST_PROMPTS = cfg.BIAS_TEST_PROMPTS
    elif use_extra_test:
        templates = src.datasets.PROMPT_TEMPLATES["prompt_iterations"]
        templates = templates[templates["group"] == "pairwise_adjectives"]["template"]
        with open(
            os.path.join(PATHS.IAT.PROMPTS, "pairwise_adjectives.csv"), mode="r"
        ) as fill_file:
            fills = []
            for line in fill_file:
                fills.extend([x.strip() for x in line.split(",")])
        cfg.BIAS_TEST_PROMPTS.extend(
            sum([list(template.format(x) for template in templates) for x in fills], [])
        )

    # TODO adapt this for debias_class=="race"
    cfg.SANITY_TEST_PROMPTS = [
        "photo of a man",
        "male individual face",
        "photo of a woman",
        "female individual face",
        "photo of a face",
    ]

    cfg.EVAL_PROMPTS = (
        cfg.BIAS_TRAIN_PROMPTS + cfg.BIAS_TEST_PROMPTS + cfg.SANITY_TEST_PROMPTS
    )
    cfg.DEBIAS_TRAIN_MASK = (
        [True] * len(cfg.BIAS_TRAIN_PROMPTS)
        + [False] * len(cfg.BIAS_TEST_PROMPTS)
        + [False] * len(cfg.SANITY_TEST_PROMPTS)
    )
    cfg.PROMPT_TEMPLATE = pd.DataFrame(
        {
            "group": [debias_class for _ in cfg.EVAL_PROMPTS],
            "debias_train": cfg.DEBIAS_TRAIN_MASK,
            "prompt": cfg.EVAL_PROMPTS,
        }
    )

    return cfg


def compute_reg_loss(
    model, text, orig_embeddings: torch.Tensor, regu_weight: float, regu_p: int
):
    if (
        (orig_embeddings is not None)
        and (regu_weight is not None)
        and (regu_p is not None)
    ):
        assert regu_p in [1, 2], "Can only do l1 or l2 regularization loss."
        text_embeddings = model.encode_text(text)
        r_l = F.mse_loss if regu_p == 2 else F.l1_loss
        return r_l(text_embeddings, orig_embeddings, reduction="sum") * regu_weight


def train_step(debiasing_type: str = "adv", *args, **kwargs):
    _debiasing_types = {"adv": train_step_adv, "dist": train_step_dist}
    if debiasing_type not in {"adv", "dist"}:
        raise NotImplementedError(
            f"Debiasing method {debiasing_type} not implemented, implemented are "
            f"{', '.join(_debiasing_types.keys())}."
        )
    return _debiasing_types[debiasing_type](*args, **kwargs)


def train_step_adv(
    batch,
    text,
    model_cl,
    model_adv,
    optimizer_cl,
    optimizer_adv,
    loss_fn_adv,
    debias_class: str,
    optim="adv",
    orig_embeddings: torch.Tensor = None,
    regu_weight: float = None,
    regu_p: int = None,
):
    """

    :param batch:
    :param text:
    :param model_cl:
    :param model_adv:
    :param optimizer_cl:
    :param optimizer_adv:
    :param loss_fn_adv:
    :param optim:
    :param orig_embeddings:
    :param regu_weight:
    :return:
    """
    imgs = batch["img"]
    if optim == "adv":
        with torch.no_grad():
            logits = model_cl(imgs, text) / 20
    else:
        model_cl.zero_grad()
        logits = model_cl(imgs, text) / 20

    # unsqueeze because model takes in a scalar
    model_adv.zero_grad()
    adv_pred = model_adv(logits.float())
    # unsqueeze and repeat the labels because it's the same text et al
    sensitive_categorical = batch["iat_label"]
    # minimize loss between
    loss = loss_fn_adv(adv_pred, sensitive_categorical)
    reg_loss = None
    # Backpropagation
    if optim == "cl":
        loss = -loss

        if regu_weight != 0:
            reg_loss = compute_reg_loss(
                model_cl, text, orig_embeddings, regu_weight, regu_p
            )
            if reg_loss is not None:
                loss += reg_loss
                reg_loss = reg_loss.item()
        else:
            reg_loss = 0

        optimizer_cl.zero_grad()
        loss.backward()
        optimizer_cl.step()
    else:
        optimizer_adv.zero_grad()
        loss.backward()
        optimizer_adv.step()
    # calc acc
    acc = (adv_pred.argmax(axis=-1) == sensitive_categorical).sum() / adv_pred.shape[0]
    return acc.item(), loss.item(), reg_loss


def wasserstein_dist(t_A: torch.Tensor, t_B: torch.Tensor, p: int = 1):

    assert 0 <= t_A.max() <= 1 and 0 <= t_B.max() <= 1, (
        "Wasserstein distance isn't implemented " "for samples outside [0, 1]!"
    )

    _d = t_A.device
    N, M = t_A.shape[-1] + 1, t_B.shape[-1] + 1
    cdf_dist = torch.zeros(t_A.shape[0]).to(_d)
    zero = torch.zeros(1, device=_d)
    one = torch.ones(1, device=_d)

    for i, (A, B) in enumerate(zip(t_A, t_B)):
        A = torch.cat([A, zero])
        B = torch.cat([B, one])
        C = torch.cat([torch.ones_like(A) / N, -torch.ones_like(B) / M])
        D = torch.cat([A, B]).sort()
        parity_diff = torch.pow(torch.cumsum(C[D.indices], dim=0).abs(), p)
        logit_diff = torch.cat(
            [D.values[0].unsqueeze(dim=-1), D.values[1:] - D.values[:-1]]
        )

        cdf_dist[i] = parity_diff @ logit_diff.T

    return cdf_dist.sum()


def train_step_dist(
    batch,
    text,
    model,
    optimizer,
    sensitive=("gender", "Male"),
    orig_embeddings: torch.Tensor = None,
    regu_weight: float = None,
    regu_p: int = None,
):
    """
    Can only be used when there are 2 sensitive classes
    :param batch: dict, with "img" being [N, C, H, W]
    :param text: [T, MT], MT is max_tokens
    :param model:
    :param optimizer:
    :param sensitive:
    :param orig_embeddings:
    :param regu_weight:
    :param regu_p:
    :return:
    """
    imgs = batch["img"].cuda()
    sensitive_labels = torch.Tensor(
        np.array(batch[sensitive[0]]) == sensitive[1]
    ).cuda()
    sensitive_labels = sensitive_labels.to(torch.bool)
    model.zero_grad()
    # [N, T]
    model_logits = model(imgs, text)

    # The pos/neg class logits should be seen as samples from a distribution with Reals as support, so the specific
    # structure of them both doesn't matter, what matters are the values themselves.
    # Note however that we do wasserstein distance per {pred(img, template) | img in pos/neg class}
    # [T, n_pos]
    pos_class_logits = model_logits[sensitive_labels].T.softmax(dim=-1)
    # [T, n_neg]
    neg_class_logits = model_logits[~sensitive_labels].T.softmax(dim=-1)

    # Minimize distance between distribution of predictions for the two sensitive classes
    dist_loss = wasserstein_dist(pos_class_logits, neg_class_logits)
    # Regularize text embeddings wrt the original ones
    reg_loss = compute_reg_loss(model, text, orig_embeddings, regu_weight, regu_p)

    # Backpropagation
    optimizer.zero_grad()
    loss = dist_loss + (reg_loss if reg_loss is not None else 0)
    loss.backward()
    optimizer.step()

    return loss.item(), dist_loss.item(), reg_loss.item()


def eval_cifar10(*args, **kwargs):
    kwargs["sneaky_run_cifar10"] = True
    return eval_cifar100(*args, **kwargs)


def eval_cifar100(
    model, tokenizer, preprocess, device, sneaky_run_cifar10: bool = False
):
    """
    See https://github.com/openai/CLIP/blob/8cad3a736a833bc4c9b4dd34ef12b52ec0e68856/README.md
    for how to use the multiple templates: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    and for the templates https://github.com/openai/CLIP/blob/e184f608c5d5e58165682f7c332c3a8b4c1545f2/data/prompts.md
    """
    batch_size = 64

    if not sneaky_run_cifar10:
        testset = torchvision.datasets.CIFAR100(
            root=PATHS.CIFAR100.BASE, train=False, download=True, transform=preprocess
        )
    else:
        testset = torchvision.datasets.CIFAR10(
            root=PATHS.CIFAR10.BASE, train=False, download=True, transform=preprocess
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    classes = testset.classes
    templates = src.datasets.CIFAR_ZS_OAI_PROMPTS

    # See https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    zeroshot_weights = []
    model.eval()
    with torch.no_grad():
        for classname in classes:
            texts = [
                template.format(classname) for template in templates[:1]
            ]  # format with class
            texts = tokenizer(texts).to(device)
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    acc_list = []
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(testloader),
            desc=f"CIFAR10{'0' if not sneaky_run_cifar10 else ''} eval",
            total=len(testloader),
            miniters=len(testloader) // 20,
        ):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            img_feats = model.encode_image(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            sims = img_feats @ zeroshot_weights
            pred = sims.max(1).indices
            acc = (pred == labels).sum().item() / len(labels)
            acc_list.append(acc)

    return 100 * sum(acc_list) / len(acc_list)


def eval_flickr1k(model, tokenizer, preprocess, device, prepend=True, subsample=1.0):
    """R5 on flickr1k retrieval subset of flickr30k"""
    batch_size = 64
    testset = Flickr30K(
        "Flickr30K", mode="test", transforms=preprocess, subsample=subsample
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    image_features = []
    text_features = []
    img_fns = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
            enumerate(testloader),
            desc="Flickr1k eval",
            total=len(testloader),
            miniters=len(testloader) // 20,
        ):
            images, text = batch["img"], batch["text"]
            if prepend:
                text = ["a photo of " + x for x in text]
            text = tokenizer(text, truncate=True).to(device)
            images = images.to(device)

            batch_txt_feat = model.encode_text(text)
            batch_img_feat = model.encode_image(images)

            image_features.append(batch_img_feat.cpu())
            text_features.append(batch_txt_feat.cpu())

            img_fns += batch["img_fn"]

    text_features = torch.cat(text_features, dim=0)
    text_features = text_features.to(device)

    img_fns = pd.Series(img_fns)
    image_features = torch.cat(image_features, dim=0)
    # drop duplicate image embeds
    img_fns.drop_duplicates(inplace=True)
    image_features = image_features[img_fns.index]
    image_features = image_features.to(device)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp().cpu()
    logits = logit_scale * image_features @ text_features.t()
    logits = logits.detach()
    # transpose for t2v
    logits = logits.t()
    res = t2v_metrics(logits)
    return res["R5"]  # The metric we use


def run_perf_eval(perf_eval: str, model, tokenizer, preprocess, device):
    if perf_eval == "cifar100":
        return eval_cifar100(model, tokenizer, preprocess, device)
    if perf_eval == "cifar10":
        return eval_cifar10(model, tokenizer, preprocess, device)
    if perf_eval == "flickr1k":
        return eval_flickr1k(model, tokenizer, preprocess, device)
    else:
        raise NotImplementedError(f"No perf eval {perf_eval}.")


def run_bias_eval(
    bias_eval: str,
    prompt_templates,
    model,
    model_alias,
    tokenizer,
    eval_dl,
    device,
    cache_suffix: str = None,
):
    old_use_cache = eval_dl.dataset.use_cache
    eval_dl.dataset.use_cache = cache_suffix is not None
    cache_suffix = "_nocache_please" if cache_suffix is None else cache_suffix
    predebias_res = src.ranking.ranking_experiment(
        prompt_templates,
        model,
        model_alias + cache_suffix,
        tokenizer,
        eval_dl,
        device,
        evaluation=bias_eval,
        progress=True,
    )
    if "group" in predebias_res.columns:
        del predebias_res["group"]
    eval_dl.dataset.use_cache = old_use_cache
    return predebias_res


def mean_of_bias_eval(debias_res: pd.DataFrame, bias_eval: str, dist_name: str):
    mean_res = {"prompt": debias_res["prompt"]}
    col_name = f"collated_{bias_eval}_{dist_name}_mean"
    mean_res[col_name] = []

    for col in debias_res.columns:
        if col.startswith(bias_eval + "_" + dist_name):
            mean_res[col_name].append(debias_res[col])

    for key, val in mean_res.items():
        if key != "prompt":
            mean_res[key] = sum(val) / len(val)

    mean_res = pd.DataFrame(mean_res)
    mean_res.set_index("prompt", inplace=True)
    mean_res = mean_res[[x for x in mean_res.columns if dist_name in x]]

    return {
        f"mean_{bias_eval}_{dist_name}": mean_res[col_name].mean(),
        f"std_{bias_eval}_{dist_name}": mean_res[col_name].std(),
    }


def plot_comparison_rankmetrics(
    prompt_cfg: Dotdict,
    debias_experiment: DebiasExperiment,
    evaluation: str,
    dist_name: str = "dem_par",
):
    assert evaluation in {
        "ndkl",
        "maxskew",
        "minskew",
    }, "Only supports ranking metrics that average over attributes!"
    dist_names = {"dem_par", "eq_opp"}
    assert (
        dist_name in dist_names
    ), f"Only supports implemented desired distributions: {' & '.join(list(dist_names))}"

    # take mean over results so easier to interpret in plot
    predebias_res = debias_experiment.results["predebias"][evaluation]
    postdebias_res = debias_experiment.results["postdebias"][evaluation]

    compare_res = {"prompt": predebias_res["prompt"]}
    for dist in dist_names:
        for t in ("PRE", "POST"):
            compare_res[f"{t}_{evaluation}_{dist}_mean"] = []

    for col in predebias_res.columns:
        if col.startswith(evaluation):
            for t, res in zip(("PRE", "POST"), (predebias_res, postdebias_res)):
                compare_res[
                    f"{t}_{evaluation}_{'_'.join(col.split('_')[1:3])}_mean"
                ].append(res[col])

    for key, val in compare_res.items():
        if key != "prompt":
            compare_res[key] = sum(val) / len(val)

    compare_res = pd.DataFrame(compare_res)
    compare_res.set_index("prompt", inplace=True)

    # just use dem par, it's the same, easier to see plot
    compare_res = compare_res[[x for x in compare_res.columns if dist_name in x]]
    # remove sanity checks for plot
    compare_res.iloc[
        : len(prompt_cfg.BIAS_TRAIN_PROMPTS) + len(prompt_cfg.BIAS_TEST_PROMPTS)
    ].plot.bar(rot=90)

    plt.savefig(
        os.path.join(PATHS.PLOTS.DEBIAS, f"debias_{evaluation}_{dist_name}_mean"),
        bbox_inches="tight",
    )
    return compare_res


class DoneTraining(Exception):
    pass


def run_debiasing(
    debias_cfg: Dotdict,
    train_cfg: Dotdict,
    prompt_cfg: Dotdict,
    optim_cfg: Dotdict,
    save_every_epoch: bool = False,
):
    experiment = DebiasExperiment(
        debias_cfg=debias_cfg,
        train_cfg=train_cfg,
        prompt_cfg=prompt_cfg,
        optim_cfg=optim_cfg,
        results={"predebias": {}, "postdebias": {}},
    )

    nept = get_neptune_run(train_cfg.NEPTUNE_PROJNAME)
    for config in ["debias", "train", "optim", "prompt"]:
        nept[f"configs/{config}"] = locals()[f"{config}_cfg"]

    # load debiasing model
    debias_model = src.models.DebiasCLIP.from_cfg(debias_cfg)

    model, _preprocess, tokenizer, model_alias = debias_model
    model = model.float()
    model.clip = model.clip.float()

    def preprocess(*args, **kwargs):
        return _preprocess(*args, **kwargs).to(dtype=torch.float32)

    # Tokenize test prompts
    train_tokz = tokenizer(prompt_cfg.BIAS_TRAIN_PROMPTS).to(debias_cfg.DEVICE)

    eval_ds = getattr(src.datasets, train_cfg.BIAS_EVAL_DATASET_NAME)(
        iat_type=debias_cfg.DEBIAS_CLASS,
        lazy=True,
        _n_samples=train_cfg.BIAS_EVAL_SUBSAMPLE,
        transforms=preprocess,
        equal_split=False,
        mode="train",
    )
    eval_dl = DataLoader(
        eval_ds,
        shuffle=False,
        batch_size=train_cfg.BATCH_SZ,
        num_workers=train_cfg.NUM_WORKERS,
    )

    for perf_eval in train_cfg.PERF_EVALS:
        experiment.results["predebias"][perf_eval] = run_perf_eval(
            perf_eval, model, tokenizer, preprocess, debias_cfg.DEVICE
        )

    for bias_eval in train_cfg.BIAS_EVALS:
        experiment.results["predebias"][bias_eval] = run_bias_eval(
            bias_eval,
            prompt_cfg.PROMPT_TEMPLATE,
            model,
            model_alias,
            tokenizer,
            eval_dl,
            debias_cfg.DEVICE,
            cache_suffix="_predebias",
        )

    nept["evals/exp_res"] = experiment.results

    ds = getattr(src.datasets, train_cfg.DATASET_NAME)(
        iat_type=debias_cfg.DEBIAS_CLASS,
        lazy=True,
        _n_samples=train_cfg.DATASET_SUBSAMPLE,
        transforms=preprocess,
        mode="train",
    )
    dl = DataLoader(
        ds,
        shuffle=True,
        batch_size=train_cfg.BATCH_SZ,
        num_workers=train_cfg.NUM_WORKERS,
    )

    # load adversary
    optim_cfg.ADV_N_OUTPUT = ds.n_iat_classes
    adv_model = src.models.Adversary.from_cfg(optim_cfg).to(dtype=torch.float32)

    with torch.no_grad():
        orig_text_embeddings = model.encode_text(train_tokz)

    adv_optimizer = torch.optim.Adam(adv_model.parameters(), lr=optim_cfg.ADV_LR)
    g_optimizer = torch.optim.Adam(
        list(x for x in model.parameters() if x.requires_grad), lr=optim_cfg.CL_LR
    )
    loss_adv_fn = nn.CrossEntropyLoss()

    best_bias_res = float("inf")
    best_bias_eval = "ndkl"

    progbar = tqdm.tqdm
    with experiment, nept:
        try:
            epochs_bar = progbar(
                range(
                    train_cfg.N_EPOCHS
                    + (
                        optim_cfg.N_ADV_INIT_EPOCHS
                        if debias_cfg.DEBIAS_TYPE == "adv"
                        else 0
                    )
                ),
                desc="Training epoch",
                position=0,
            )
            iters_per_eval = int(train_cfg.EVAL_EVERY * len(dl))
            for epoch in epochs_bar:

                batches_bar = progbar(
                    enumerate(dl),
                    desc="Training batch",
                    position=1,
                    leave=False,
                    total=len(dl),
                    miniters=train_cfg.LOG_EVERY,
                )

                def bar_log_stats(s, g):
                    batches_bar.set_postfix(s, refresh=False)
                    for name, val in s.items():
                        nept[f"train/stats/{name}"].log(val, step=g)

                for idx, batch in batches_bar:
                    global_step = idx + epoch * len(dl)
                    model_name = f"{model_alias}_neptune_run_{nept._short_id}_model_e{epoch}_step_{global_step}.pt"
                    save_path = os.path.join(PATHS.MODEL_STORE, model_name)

                    batch["img"] = batch["img"].to(train_cfg.DEVICE)
                    batch["iat_label"] = batch["iat_label"].to(train_cfg.DEVICE)

                    if debias_cfg.DEBIAS_TYPE == "dist":
                        global_step = idx + epoch * len(dl)
                        loss, dist_loss, reg_loss = train_step(
                            debias_cfg.DEBIAS_TYPE,
                            batch,
                            train_tokz,
                            model,
                            g_optimizer,
                            orig_embeddings=orig_text_embeddings,
                            regu_weight=optim_cfg.L_REG_WEIGHT,
                            regu_p=optim_cfg.L_REG_TYPE,
                        )
                        iter_stats = {"L": loss, "L_dist": dist_loss, "L_reg": reg_loss}

                    elif debias_cfg.DEBIAS_TYPE == "adv":
                        cl_loss = 0
                        if epoch < optim_cfg.N_ADV_INIT_EPOCHS:
                            # first, just train the discriminator...
                            acc, adv_loss, reg_loss = train_step(
                                debias_cfg.DEBIAS_TYPE,
                                batch,
                                train_tokz,
                                model,
                                adv_model,
                                g_optimizer,
                                adv_optimizer,
                                loss_adv_fn,
                                debias_class=debias_cfg.DEBIAS_CLASS,
                                optim="adv",
                                orig_embeddings=None,
                                regu_weight=None,
                            )
                        else:
                            if (idx // optim_cfg.CL_ADV_TRAIN_SWITCH) % 2 == 1:
                                acc, adv_loss, reg_loss = train_step(
                                    debias_cfg.DEBIAS_TYPE,
                                    batch,
                                    train_tokz,
                                    model,
                                    adv_model,
                                    g_optimizer,
                                    adv_optimizer,
                                    loss_adv_fn,
                                    debias_class=debias_cfg.DEBIAS_CLASS,
                                    optim="adv",
                                    orig_embeddings=None,
                                    regu_weight=None,
                                )
                                cl_loss = -adv_loss
                            else:
                                acc, cl_loss, reg_loss = train_step(
                                    debias_cfg.DEBIAS_TYPE,
                                    batch,
                                    train_tokz,
                                    model,
                                    adv_model,
                                    g_optimizer,
                                    adv_optimizer,
                                    loss_adv_fn,
                                    debias_class=debias_cfg.DEBIAS_CLASS,
                                    optim="cl",
                                    orig_embeddings=orig_text_embeddings,
                                    regu_weight=optim_cfg.L_REG_WEIGHT,
                                    regu_p=optim_cfg.L_REG_TYPE,
                                )
                                adv_loss = -cl_loss

                        if reg_loss is None:
                            reg_loss = 0

                        iter_stats = {
                            "Acc_adv": 100 * acc,
                            "L_adv": adv_loss,
                            "L_reg": reg_loss,
                        }
                        if cl_loss != 0:
                            iter_stats["L_cl"] = cl_loss

                    else:
                        raise NotImplementedError(
                            f"Debiasing type {debias_cfg.DEBIAS_TYPE} not implemented."
                        )

                    bar_log_stats(iter_stats, global_step)

                    if (
                        global_step
                        and (global_step % iters_per_eval == 0)
                        and (
                            not (
                                debias_cfg.DEBIAS_TYPE == "adv"
                                and epoch < optim_cfg.N_ADV_INIT_EPOCHS
                            )
                        )
                    ):
                        training_should_stop = False
                        perf_eval_res = {}
                        for perf_eval in train_cfg.PERF_EVALS:
                            eval_val = run_perf_eval(
                                perf_eval,
                                model,
                                tokenizer,
                                preprocess,
                                debias_cfg.DEVICE,
                            )
                            perf_eval_res[perf_eval] = eval_val
                            nept[f"train/perf_evals/{perf_eval}"].log(
                                eval_val, step=global_step
                            )

                            # Check whether to stop
                            pre_res = experiment.results["predebias"][perf_eval]
                            if (
                                eval_val
                                < (1 - train_cfg.PERF_STOPPING_DECREASE) * pre_res
                            ):
                                training_should_stop = True

                        if training_should_stop:
                            raise DoneTraining(
                                "Stopping due to performance decreased enough"
                            )

                        _cache_suffix = f"_tempcache_{time.time()}_"

                        for bias_eval in train_cfg.BIAS_EVALS:
                            bias_val = run_bias_eval(
                                bias_eval,
                                prompt_cfg.PROMPT_TEMPLATE,
                                model,
                                model_alias,
                                tokenizer,
                                eval_dl,
                                debias_cfg.DEVICE,
                                cache_suffix=_cache_suffix,
                            )
                            res = mean_of_bias_eval(bias_val, bias_eval, "dem_par")
                            for k, v in res.items():
                                nept[f"train/bias_evals/{k}"].log(v, step=global_step)

                            if bias_eval == best_bias_eval:
                                if (
                                    res[f"mean_{best_bias_eval}_dem_par"]
                                    > best_bias_res
                                ):
                                    continue
                                best_bias_res = res[f"mean_{best_bias_eval}_dem_par"]
                                save_path = os.path.join(
                                    PATHS.MODEL_STORE,
                                    f"best_{best_bias_eval}_{model_name}",
                                )
                                print(
                                    f"New {best_bias_eval} record, {best_bias_res:.4f}, saving to: {save_path}"
                                )
                                torch.save(model.state_dict(), save_path)

                        epochs_bar.set_postfix(perf_eval_res)

            if save_every_epoch:
                save_path = os.path.join(
                    PATHS.MODEL_STORE, f"epochsave_e{epoch}_{model_name}"
                )
                print(f"Epoch save, saving to: {save_path}")
                torch.save(model.state_dict(), save_path)

            for perf_eval in train_cfg.PERF_EVALS:
                experiment.results["postdebias"][perf_eval] = run_perf_eval(
                    perf_eval, model, tokenizer, preprocess, debias_cfg.DEVICE
                )

            _cache_suffix = f"_tempcache_{time.time()}_"
            for bias_eval in train_cfg.BIAS_EVALS:
                experiment.results["postdebias"][bias_eval] = run_bias_eval(
                    bias_eval,
                    prompt_cfg.PROMPT_TEMPLATE,
                    model,
                    model_alias,
                    tokenizer,
                    eval_dl,
                    debias_cfg.DEVICE,
                    cache_suffix=_cache_suffix,
                )

            nept["evals/exp_res"] = experiment.results
        except DoneTraining as e:
            print("Done training.")
        finally:
            print(f"Best achieved {best_bias_eval} was {best_bias_res:.4f}.")
            model_name = f"endoftrain_{model_alias}_neptune_run_{nept._short_id}_model_e{epoch}.pt"
            save_path = os.path.join(PATHS.MODEL_STORE, model_name)

            print(f"Saving to: {save_path}")
            torch.save(model.state_dict(), save_path)

    return experiment, (model, preprocess, tokenizer, model_alias)
