from torch.utils.data import DataLoader
import argparse
import os
import torch
from src.datasets import Flickr30K
from src.models import DebiasCLIP
from src import Dotdict
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, ToTensor, Normalize
from tqdm import tqdm
import src
from src.debias import eval_flickr1k, eval_cifar100, run_bias_eval
from src.metrics import t2v_metrics
class CrossEn(torch.nn.Module):
    def __init__(self, ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()

        logpt = F.log_softmax(sim_matrix.t(), dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss_2 = nce_loss.mean()

        return 0.5*(sim_loss+sim_loss_2)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

train_cfg = Dotdict()
train_cfg.NEPTUNE_PROJNAME = "oxai-vlb-ht22/OxAI-Vision-Language-Bias" # None if don't use
train_cfg.N_EPOCHS = 10
train_cfg.BATCH_SZ = 64
train_cfg.NUM_WORKERS = 6 # 0 for auto
train_cfg.LOG_EVERY = 10
train_cfg.DEVICE = "cuda"
#train_cfg.DATASET_NAME = "FairFace"
train_cfg.DATASET_SUBSAMPLE = 1.0 # None or 1.0 for full
train_cfg.PERF_STOPPING_DECREASE = 0.3
train_cfg.PERF_EVALS = ["cifar100", "flickr1k"]#["flickr1k", "cifar100"] # cifar100, flickr1k.
train_cfg.EVAL_EVERY = 0.1 # In epochs
train_cfg.BIAS_EVAL_SUBSAMPLE = 1.0
#train_cfg.BIAS_EVAL_DATASET_NAME = "FairFace"
train_cfg.BIAS_EVALS = ["ndkl", "maxskew"] # ndkl and min/maxskew supported

debias_cfg = Dotdict()
debias_cfg.CLIP_ARCH = "openai/CLIP/ViT-B/16"
debias_cfg.hidden_dim = 512
debias_cfg.DEBIAS_TOKEN_INIT = "zeros"  # how to initialize the debiasing tokens
debias_cfg.DEBIAS_POS = "prepend"  # Prepending or appending
debias_cfg.DEVICE = "cuda"
debias_cfg.NUM_DEBIAS_TOKENS = 2

debias_cfg.FREEZE_PROJ = True
train_cfg.DATASET_NAME = "FairFace"
train_cfg.BIAS_EVAL_DATASET_NAME = "FairFace"
debias_cfg.DEBIAS_CLASS = "gender"
debias_cfg.N_TRAIN_TEXT_LAYERS = 0
debias_cfg.N_TRAIN_VID_LAYERS = 0

# bias prompts
_train_prompts = ["a photo of a {} person"]
_test_prompts = ["a photo of a {} person", "a cropped photo of a {} face"]
_paired_prompts = [('good'       , 'evil'),
                   ('smart'      , 'dumb'),
                   ('attractive' , 'unattractive'),
                   ('lawful'     , 'criminal'),
                   ('friendly'   , 'unfriendly'),
                   ('clever'     , 'stupid'),
                   ('heroic'     , 'villainous'),
                   ('hardworking', 'lazy'),
                   ('trustworthy', 'dangerous'),
                   ('caring'     , 'nasty')]

_prompts_n_train = len(_paired_prompts)
prompt_cfg = src.debias.prepare_prompt_cfg(debias_cfg.DEBIAS_CLASS, _paired_prompts, _train_prompts, _test_prompts, _prompts_n_train, test_on_train=False)


batch_size = 16
num_workers = 4
debias_model = DebiasCLIP.from_cfg(debias_cfg)
model, _preprocess, tokenizer, model_alias = debias_model

pretrained_tokz = torch.load("debias_2_flickr_contrastive.pt").cuda()
model.debias_tokens = torch.nn.Embedding.from_pretrained(pretrained_tokz, freeze=False)

dataset_train = Flickr30K("Flickr30K", mode='train', transforms=_preprocess)
dataset_test = Flickr30K("Flick30K", mode='test', transforms=_preprocess)

dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

criterion = CrossEn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# NB should be train*0.25 since we use val later
eval_ds = getattr(src.datasets, train_cfg.BIAS_EVAL_DATASET_NAME)(iat_type=debias_cfg.DEBIAS_CLASS, lazy=True,
                                                                  _n_samples=train_cfg.BIAS_EVAL_SUBSAMPLE,
                                                                  transforms=_preprocess,
                                                                  equal_split=False, mode="train")
eval_dl = DataLoader(eval_ds, shuffle=False, batch_size=train_cfg.BATCH_SZ, num_workers=train_cfg.NUM_WORKERS)

print(eval_flickr1k(model, tokenizer, _preprocess, "cuda"))
print(eval_cifar100(model, tokenizer, _preprocess, "cuda"))
for bias_eval in train_cfg.BIAS_EVALS:
    res = run_bias_eval(bias_eval, prompt_cfg.PROMPT_TEMPLATE, model, model_alias, tokenizer, eval_dl, debias_cfg.DEVICE, cache_suffix="_predebias")
    print(bias_eval, res)

import pdb; pdb.set_trace()

for epoch in range(1):
    with tqdm(dataloader_train, unit="batch") as tepoch:
        tepoch.set_description(f"Train Epoch {epoch}")
        for idx, batch in enumerate(tepoch):
            loss_running = 0
            optimizer.zero_grad()
            batch['img'] = batch['img'].cuda()
            batch['text'] = tokenizer(batch['text'], truncate=True).cuda()
            output = model(batch['img'], batch['text'])
            loss = criterion(output)
            loss.backward()
            optimizer.step()

            loss_running += loss.item()

            if idx % 20 == 0:
                tepoch.set_postfix(loss=loss_running/20)
                loss_running = 0



# torch.save(model.debias_tokens.weight.detach().cpu(), "debias_2_flickr_contrastive_v1.pt")

