import torch
import os
from src import PATHS
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import src
from src import debias
import math
from src.ranking import compute_skew_metrics, normalized_discounted_KL


def eval_custom_embeds(evaluation, image_embeddings, prompt_embeddings, labels_list, top_n):
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(dim=-1, keepdim=True)

    similarities = (image_embeddings @ prompt_embeddings.T).cpu().numpy()
    summary = pd.DataFrame({"score": [x for x in similarities], "label": labels_list})
    summary = summary.explode('score')
    summary['score'] = summary['score'].astype(float)
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
        # currently works only when isinstance(top_n, int)
        top_n_max = summary.nlargest(max(q_topns), columns="score", keep="all")

        for q, q_label in zip(q_topns, q_labels):
            proportions = top_n_max.iloc[:q].label.value_counts(normalize=True)
            for index, proportion in proportions.iteritems():
                result[f"{q_label}({int(index)})"] = proportion

    elif 'skew' in evaluation:
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


def clip_features(image_embeddings, labels, remaining=400):
    mi = mutual_info_classif(image_embeddings, labels)
    debias_idx = np.argsort(mi)[:remaining]
    return debias_idx


data_dir = "/scratch/local/ssd/maxbain/bias-vision-language/"

mode = "val"
REMAINING_DIM = 400
labels = pd.read_csv(os.path.join(PATHS.FAIRFACE.LABELS, mode, f"{mode}_labels.csv"))
debias_img_embeds = torch.load(os.path.join(data_dir, "cache", "preproc_image_embeddings_FairFaceVal_oai-clip-vit-b-16.pt"))
labels_male = labels.loc[labels['gender'] == 'Male']
labels_female = labels.loc[labels['gender'] == 'Female']

num_females = labels_female.count()[0]
num_males = labels_male.count()[0]

sample_num = min(num_males, num_females)

labels_male = labels_male.sample(n=sample_num, random_state=1)
labels_female = labels_female.sample(n=sample_num, random_state=1)

labels = labels_male.append(labels_female, ignore_index=True)['gender']
labels = pd.Categorical(labels)


if REMAINING_DIM < 512:
    debias_ftr_idx = clip_features(debias_img_embeds, labels, remaining=REMAINING_DIM)
    debias_img_embeds = debias_img_embeds[:, debias_ftr_idx]
else:
    debias_ftr_idx = None

debias_cfg = src.Dotdict()
debias_cfg.CLIP_ARCH = "openai/CLIP/ViT-B/16"
debias_cfg.hidden_dim = 512

debias_cfg.DEBIAS_TOKEN_INIT = "zeros"  # how to initialize the debiasing tokens
debias_cfg.DEBIAS_POS = "prepend"  # Prepending or appending
debias_cfg.DEVICE = "cuda"
debias_cfg.NUM_DEBIAS_TOKENS = 0

import json
with open(src.PATHS.TRAINED_MODELS.METADATA, mode="r") as _runs_metafile:
    runs_metadata = json.load(_runs_metafile)

with open(src.PATHS.TRAINED_MODELS.TEST_PROMPTS, mode="r") as _test_promptsfile:
    test_prompts_data = json.load(_test_promptsfile)

debias_class = 'gender'
test_prompts = test_prompts_data[debias_class]
test_prompts_df = pd.DataFrame({"prompt": test_prompts})
test_prompts_df["group"] = debias_class
# bias prompts
# _train_prompts = ["a photo of a {} person"]
# _test_prompts = ["a photo of a {} person", "a cropped photo of a {} face"]
# _paired_prompts = [('good', 'evil'),
#                    ('smart', 'dumb'),
#                    ('attractive', 'unattractive'),
#                    ('lawful', 'criminal'),
#                    ('friendly', 'unfriendly'),
#                    ('clever', 'stupid'),
#                    ('heroic', 'villainous'),
#                    ('hardworking', 'lazy'),
#                    ('trustworthy', 'dangerous'),
#                    ('caring', 'nasty')]
# _paired_prompts = [('male', 'female')]
# _prompts_n_train = len(_paired_prompts)
# prompt_cfg = debias.prepare_prompt_cfg(debias_cfg.DEBIAS_CLASS, _paired_prompts, _train_prompts, _test_prompts,
#                                        _prompts_n_train, test_on_train=False)
train_cfg = src.Dotdict()
train_cfg.NEPTUNE_PROJNAME = "oxai-vlb-ht22/OxAI-Vision-Language-Bias"  # None if don't use
train_cfg.N_EPOCHS = 10
train_cfg.BATCH_SZ = 64
train_cfg.NUM_WORKERS = 6  # 0 for auto
train_cfg.LOG_EVERY = 10
train_cfg.DEVICE = "cuda"
# train_cfg.DATASET_NAME = "FairFace"
train_cfg.DATASET_SUBSAMPLE = 1.0  # None or 1.0 for full
train_cfg.PERF_STOPPING_DECREASE = 0.3
train_cfg.PERF_EVALS = ["cifar100", "flickr1k"]  # ["flickr1k", "cifar100"] # cifar100, flickr1k.
train_cfg.EVAL_EVERY = 0.1  # In epochs
train_cfg.BIAS_EVAL_SUBSAMPLE = 1.0
# train_cfg.BIAS_EVAL_DATASET_NAME = "FairFace"
train_cfg.BIAS_EVALS = ["ndkl", "maxskew"]  # ndkl and min/maxskew supported
debias_cfg.FREEZE_PROJ = True
train_cfg.DATASET_NAME = "FairFace"
train_cfg.BIAS_EVAL_DATASET_NAME = "FairFace"
debias_cfg.DEBIAS_CLASS = "gender"
debias_cfg.N_TRAIN_TEXT_LAYERS = 0
debias_cfg.N_TRAIN_VID_LAYERS = 0
from src.models import DebiasCLIP

debias_model = DebiasCLIP.from_cfg(debias_cfg)
model, _preprocess, tokenizer, model_alias = debias_model
model = model.cuda()
from src.ranking import get_prompt_embeddings


def _run_ranking_exp(_row: pd.Series):
    embed = get_prompt_embeddings(model, tokenizer, train_cfg.DEVICE, _row['prompt'])
    return embed.cpu().numpy()

prompt_templates = test_prompts_df
test_prompts_df['embed'] = prompt_templates.apply(_run_ranking_exp, axis=1)
debias_prompt_embeds = torch.from_numpy(np.concatenate(test_prompts_df['embed'].values))
if REMAINING_DIM < 512:
    debias_prompt_embeds = debias_prompt_embeds[:, debias_ftr_idx]


cifar100 = debias.eval_cifar100(model, tokenizer, _preprocess,
                                "cuda", filter_idx=debias_ftr_idx)
flickr = debias.eval_flickr1k(model, tokenizer, _preprocess,
                                "cuda", filter_idx=debias_ftr_idx)


res_arr = []
for emb in debias_prompt_embeds:
    res_dict = {}
    x1 = emb.unsqueeze(0)
    ndkl = eval_custom_embeds("ndkl", debias_img_embeds, x1, labels.codes, [1000])
    skew = eval_custom_embeds("skew", debias_img_embeds, x1, labels.codes, [1000])
    res_dict.update(ndkl)
    res_dict.update(skew)
    res_arr.append(res_dict)

res = pd.DataFrame(res_arr)
print("ndkl : ", res['ndkl_dem_par_1000'].mean())
print("maxskew@1000: ", res['maxskew_dem_par_1000'].mean())
print("cifar100: ", cifar100)
print("flickr5: ", flickr)