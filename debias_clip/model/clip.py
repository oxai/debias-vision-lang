import os
import urllib
from tqdm import tqdm
from typing import Union, List
import torch
from torch import nn
from debias_clip.model.model import DebiasCLIP
import clip
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



_MODELS = {
    "ViT-B/16-gender": {
     "url": "http://www.robots.ox.ac.uk/~maxbain/oxai-bias/best_ndkl_oai-clip-vit-b-16_neptune_run_OXVLB-317_model_e4_step_5334_embeddings.pt",
     "clip_arch": "ViT-B/16",
     "num_debias_tokens": 2
    }
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    print(f"Installing pretrained embedings\n {url.split('/')[-1]}...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name]["url"], download_root or os.path.expanduser("~/.cache/debias_clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        pt_embeddings = torch.load(opened_file, map_location="cpu")

    clip_model, preprocess = clip.load(_MODELS[name]["clip_arch"], device=device)
    hidden_dim = clip_model.token_embedding.weight.shape[1]
    model = DebiasCLIP(clip_model=clip_model, num_debias_tokens=_MODELS[name]["num_debias_tokens"], hidden_dim=hidden_dim)
    model.debias_tokens.weight = nn.Parameter(pt_embeddings) # load pt embeddings

    return model, preprocess