from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip as oai_clip
import torch
import torch.nn as nn
from torch.nn import Module

from debias_clip import Dotdict


class ClipLike(Module, ABC):
    """Essentially a type stub for specifying what a clip-like model supports"""

    visual: Any
    logit_scale: Any
    dtype: torch.dtype
    positional_embedding: Any
    text_projection: Any
    token_embedding: Any
    visual: Any

    def transformer(self, text_features) -> Any:
        pass

    def ln_final(self, text_features) -> Any:
        pass

    def encode_image(self, images) -> torch.Tensor:
        pass

    def encode_text(self, tokenized_texts) -> torch.Tensor:
        pass


def clip_layers(clip_model) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Gets names parameters in a structured way.
    Gives a tuple of {type_of_layer: count}, where type can be text, image, projection, tokens, or other
        and a list of dicts for each:
        {"type": str, "index": int, "param": torch_param, "name": orig_name}
            where type is as above, index is from 0 to count-1,
            and torch_param is the param as returned by module.named_parameters
    """
    classed_parameters = []
    metadata = {k: 0 for k in {"text", "image", "proj", "tokens", "other"}}
    for name, param in clip_model.named_parameters():
        # top layers always need to train
        if (
                name.startswith("ln_final.")
                or name.startswith("text_projection")
                or name.startswith("logit_scale")
                or name.startswith("visual.ln_post.")
                or name.startswith("visual.proj")
        ):
            t = "proj"
            inx = metadata[t]
        elif name.startswith("visual.transformer.resblocks."):
            t = "image"
            inx = int(name.split(".")[3])
        elif name.startswith("transformer.resblocks."):
            t = "text"
            inx = int(name.split(".")[2])
        elif name.startswith("token_embedding"):
            t = "tokens"
            inx = metadata[t]
        else:
            t = "other"
            inx = metadata[t]
        classed_parameters.append(
            {"type": t, "index": inx, "param": param, "name": name}
        )
        metadata[t] += 1
    for t in {"text", "image"}:
        metadata[t] = (
                max(
                    classed_parameters, key=lambda cp: cp["index"] if cp["type"] == t else 0
                )["index"]
                + 1
        )

    return metadata, classed_parameters


VALID_CLIP_MODELS = [
    "openai/CLIP/RN50",
    "openai/CLIP/RN101",
    "openai/CLIP/RN50x4",
    "openai/CLIP/ViT-B/16",
    "openai/CLIP/ViT-B/32",
    "openai/CLIP/ViT-L/14",
]

VALID_MODELS = (
    # openai clips
    VALID_CLIP_MODELS
)


def model_loader(model_name, device=None, jit=False) -> Tuple[ClipLike, Callable[[Any], torch.Tensor],
                                                              Callable[[Any], torch.LongTensor], str]:
    """Returns cliplike model, preprocessing function for images, tokenizer, and modelname/alias"""
    # Some models aren't compatible with the tokens we generate (they have mismatching dimensions),

    if model_name not in VALID_MODELS:
        raise NotImplementedError(
            f"{model_name} not found, should be on of..", VALID_MODELS
        )

    if model_name.startswith("openai/CLIP/"):
        arch_str = model_name.replace("openai/CLIP/", "")
        model, preprocess = oai_clip.load(arch_str, device=device, jit=jit)
        tokenizer = oai_clip.tokenize
        alias_name = "oai-clip-" + "-".join(model_name.split("/")[2:]).lower()
    elif model_name.startswith("m-bain/frozen-in-time/"):
        raise NotImplementedError
    elif model_name.startswith("facebookresearch/SLIP/"):
        raise NotImplementedError
    else:
        raise NotImplementedError

    return model, preprocess, tokenizer, alias_name


class DebiasCLIP(nn.Module):
    """
    Currently only supporting CLIP models because it would be a pain to generalise this to frozen-in-time etc...
    """

    @staticmethod
    def from_cfg(cfg: Union[dict, Dotdict]):
        cfg = Dotdict(cfg)
        clip, preprocess, tokenizer, model_alias = model_loader(
            cfg.CLIP_ARCH, device=cfg.DEVICE, jit=False
        )
        clip = clip.to(cfg.DEVICE).float()
        cfg["_tokenizer"] = tokenizer
        debias_clip = DebiasCLIP(
            clip_model=clip, **{k.lower(): v for k, v in cfg.items()}
        )
        del cfg["_tokenizer"]
        return debias_clip, preprocess, tokenizer, model_alias

    def __init__(self, clip_model: ClipLike, num_debias_tokens: int, hidden_dim: int = 512, max_tokens: int = 77,
            n_train_vid_layers: int = 0, n_train_text_layers: int = 0, freeze_proj: bool = True, debias_token_init: Union[str, List[str]] = "zeros",
            debias_pos: str = "prepend", _tokenizer: callable = None, **_kwargs,):
        super().__init__()
        """
        :param clip_model: a clip model variant
        :param num_debias_tokens: number of debiasing tokens
        :param hidden_dim: hidden dim of clip model
        :param max_tokens: max number of text tokens (77)
        :param freeze_vid_layer_num: nth inclusive layer to freeze weights
        :param freeze_text_layer_num: nth inclusive layer to freeze weights
        """

        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.num_prompts_tokz = num_debias_tokens
        self.n_train_vid_layers = n_train_vid_layers
        self.n_train_text_layers = n_train_text_layers
        self.freeze_proj = freeze_proj
        self.debias_pos = debias_pos
        if self.debias_pos not in {"prepend", "append", "append_after_eos", "add"}:
            raise NotImplementedError

        self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32
        self.clip: ClipLike = clip_model

        if debias_token_init == "rand":
            self.debias_tokens = nn.Embedding(self.num_prompts_tokz, self.hidden_dim)
        elif debias_token_init == "zeros":
            # init them to the zero id embeddings...
            # still affected by positional embeds
            zero_vecs = self.clip.token_embedding(
                torch.zeros(self.num_prompts_tokz)
                .int()
                .to(self.clip.token_embedding.weight.device)
            )
            self.debias_tokens = nn.Embedding.from_pretrained(zero_vecs, freeze=False)
        elif isinstance(debias_token_init, list):
            toks = _tokenizer([" ".join(debias_token_init)])[0][
                   1: len(debias_token_init) + 1
                   ]
            tok_feats = self.clip.token_embedding(
                toks.to(self.clip.token_embedding.weight.device)
            )
            self.debias_tokens = nn.Embedding.from_pretrained(tok_feats, freeze=False)
        else:
            raise NotImplementedError

        self.freeze_model_layers()

    def encode_text(self, text):
        # custom for learnable prompts
        text_features = torch.zeros(
            [text.shape[0], self.max_tokens, self.hidden_dim]
        ).to(
            text.device
        )  # [batch_size, 77, 512]

        # append actual text
        raw_text_features = self.clip.token_embedding(text).type(self.dtype)
        raw_text_features = raw_text_features + self.clip.positional_embedding.type(
            self.dtype
        )

        if self.num_prompts_tokz > 0:
            smaller_text_features = raw_text_features[:, : -self.num_prompts_tokz]
            debias_features = self.debias_tokens(
                torch.arange(self.num_prompts_tokz).to(text.device)
            )[None, :].repeat([len(text), 1, 1])
        else:
            smaller_text_features = raw_text_features

        if self.debias_pos == "prepend":
            # fill in with learned prompts
            if self.num_prompts_tokz > 0:
                text_features[:, : self.num_prompts_tokz] = debias_features
            text_features[:, self.num_prompts_tokz:] = smaller_text_features
        elif self.debias_pos == "append":
            if self.num_prompts_tokz == 0:
                text_features = raw_text_features  # == smaller_text_features
            else:
                max_n_tokens = text.shape[1]
                lens_to_end_token = text.max(dim=1).indices
                inx_of_end_after = [
                    l + min(self.num_prompts_tokz, max_n_tokens - l - 1)
                    for l in lens_to_end_token
                ]
                for i, (l, e) in enumerate(zip(lens_to_end_token, inx_of_end_after)):
                    if e <= l:
                        text_features[i] = raw_text_features[i]
                        continue
                    text_features[i, :l, :] = raw_text_features[i, :l, :]
                    text_features[i, l:e, :] = debias_features[i, : e - l, :]
                    text_features[i, e:, :] = raw_text_features[i, e:, :]
        elif self.debias_pos == "append_after_eos":

            max_n_tokens = text.shape[1]
            lens_to_end_token = text.max(dim=1).indices + 1
            for i, l in enumerate(lens_to_end_token):
                e = min(l + self.num_prompts_tokz, max_n_tokens)
                if e <= l:
                    text_features[i] = raw_text_features[i]
                    continue
                text_features[i, :l, :] = raw_text_features[i, :l, :]
                text_features[i, l:e, :] = debias_features[i, : e - l, :]
                text_features[i, e:, :] = raw_text_features[i, e:, :]
        elif self.debias_pos == "add":
            text_features[:, :] = raw_text_features
            if self.num_prompts_tokz > 0:
                text_features[:, 1: 1 + self.num_prompts_tokz] += debias_features

        text_features = text_features.permute(1, 0, 2)  # NLD -> LND
        text_features = self.clip.transformer(text_features)
        text_features = text_features.permute(1, 0, 2)  # LND -> NLD
        text_features = self.clip.ln_final(text_features).type(self.dtype)

        _argmax = text.argmax(dim=-1) + self.num_prompts_tokz
        _argmax = torch.min(text_features.shape[1] + 0 * _argmax - 1, _argmax)
        text_features = (
                text_features[torch.arange(text_features.shape[0]), _argmax]
                @ self.clip.text_projection
        )
        return text_features

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def forward(self, image, text):
        # initialise text feats with zeros

        text_features = self.encode_text(text)
        image_features = self.clip.encode_image(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def freeze_model_layers(self):
        metadata, classed_params = clip_layers(self.clip)

        if not (metadata["text"] >= self.n_train_text_layers >= 0):
            raise ValueError(
                f"Number of trained text layers should be between 0 (no layers) and {metadata['text']} "
                f"(all layers), not {self.n_train_text_layers}"
            )

        if not (metadata["image"] >= self.n_train_vid_layers >= 0):
            raise ValueError(
                f"Number of trained vid layers should be between 0 (no layers) and {metadata['image']} "
                f"(all layers), not {self.n_train_vid_layers}"
            )

        for classed_param in classed_params:
            self.train_layer_selector(metadata, classed_param)

    def train_layer_selector(self, metadata, classed_param):
        t, index, param = (
            classed_param["type"],
            classed_param["index"],
            classed_param["param"],
        )
        index_from_end = metadata[t] - (index + 1)
        # top layers always need to train
        if t == "proj":
            if self.freeze_proj:
                pass
            else:
                assert param.requires_grad
                return  # train
        elif t == "tokens":
            pass  # This is not debias tokens
        elif t == "image":
            if index_from_end < self.n_train_vid_layers:
                assert param.requires_grad
                return  # need to train
        elif t == "text":
            if index_from_end < self.n_train_text_layers:
                assert param.requires_grad
                return  # need to train

        param.requires_grad = False


class Adversary(nn.Module):
    @staticmethod
    def from_cfg(cfg: Union[dict, Dotdict]):
        cfg = Dotdict(cfg)
        adv_model = Adversary(
            n_input=cfg.ADV_N_INPUT,
            n_output=cfg.ADV_N_OUTPUT,
            hidden_size=cfg.ADV_HIDDEN_SIZE,
        )
        return adv_model.to(cfg.ADV_DEVICE)

    def __init__(self, n_input, n_output=1, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_output),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


if __name__ == "__main__":
    x1 = torch.rand([5, 3, 224, 224])
    text = ["random_str", "this is a dog"]
    model, preprocess, tokenizer, alias_name = model_loader(
        "openai/CLIP/ViT-B/16", "cuda"
    )
    txt_embeds = model.encode_text(tokenizer(text).cuda())
    img_embs = model.encode_image(x1.cuda())
    print(img_embs)
