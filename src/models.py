import torch.nn as nn
import os
import subprocess
from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip as oai_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

import src
from src.slip import CLIP_VITB16, CLIP_VITL16, SLIP_VITB16, SLIP_VITL16, SimpleTokenizer
from src.video_transformer import SpaceTimeTransformer


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class ClipLike(Module, ABC):
    """Essentially a type stub for specifying what a clip-like model supports"""
    visual: Any
    logit_scale: Any
    dtype: torch.dtype
    positional_embedding: Any
    text_projection: Any
    token_embedding: Any
    visual: Any

    def transformer(self, text_features) -> Any: pass

    def ln_final(self, text_features) -> Any: pass

    def encode_image(self, image) -> Any: pass

    def named_parameters(self) -> Any: pass


def clip_layers(clip_model) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Gets names parameters in a structured way.
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
        if name.startswith("ln_final.") or name.startswith("text_projection") or name.startswith("logit_scale") \
                or name.startswith("visual.ln_post.") or name.startswith("visual.proj"):
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
        classed_parameters.append({"type": t, "index": inx, "param": param, "name": name})
        metadata[t] += 1
    for t in {"text", "image"}:
        metadata[t] = max(classed_parameters, key=lambda cp: cp["index"] if cp["type"] == t else 0)["index"]+1

    return metadata, classed_parameters


# Some models aren't compatible with the tokens we generate (they have mismatching dimensions),
# so we can't use all of oai_clip.available_models()
VALID_CLIP_MODELS = [
    "openai/CLIP/RN50",
    #"openai/CLIP/RN101",
    #"openai/CLIP/RN50x4",
    "openai/CLIP/ViT-B/16",
    "openai/CLIP/ViT-B/32",
    "openai/CLIP/ViT-L/14", # dunno why this won't load, even with the newest clip package
]
VALID_FIT_MODELS = [f"m-bain/frozen-in-time/{x}" for x in ["ccwv2m", "wv2m", "ccwv2mcoco", "cc"]]
VALID_SLIP_MODELS = [f"facebookresearch/SLIP/{x}" for x in
                     ["ViT-B/CLIP", "ViT-B/SLIP", "ViT-L/CLIP", "ViT-L/SLIP"]]
VALID_MODELS = VALID_CLIP_MODELS + VALID_FIT_MODELS + VALID_SLIP_MODELS  # + VALID_FIT_MODELS, not included due to missing parse_config module


# we need jit == False for prompt tuning
def model_loader(model_name, device=None, jit=False) -> Tuple[ClipLike, Callable, Callable, Callable]:
    # Some models aren't compatible with the tokens we generate (they have mismatching dimensions), 
    # so we can't use all of oai_clip.available_models()
    #VALID_CLIP_MODELS = [f"openai/CLIP/{x}" for x in oai_clip.available_models()]
    #VALID_FIT_MODELS = [f"m-bain/frozen-in-time/{x}" for x in ["ccwv2m", "wv2m", "ccwv2mcoco"]]
    #VALID_SLIP_MODELS = [f"facebookresearch/SLIP/{x}" for x in ["ViT-B/CLIP", "ViT-B/SLIP", "ViT-L/CLIP", "ViT-L/SLIP"]]
    #VALID_MODELS = VALID_CLIP_MODELS + VALID_FIT_MODELS + VALID_SLIP_MODELS

    PYTORCH_MODEL_CACHE = src.PATHS.ENV.PYTORCH_MODEL_CACHE

    if model_name not in VALID_MODELS:
        raise NotImplementedError(f"{model_name} not found, should be on of..", VALID_MODELS)

    if model_name.startswith("openai/CLIP/"):
        arch_str = model_name.replace("openai/CLIP/", "")
        model, preprocess = oai_clip.load(arch_str, device=device, jit=jit)
        tokenizer = oai_clip.tokenize
        alias_name = "oai-clip-" + '-'.join(model_name.split('/')[2:]).lower()
    elif model_name.startswith("m-bain/frozen-in-time/"):
        arch_str = model_name.replace("m-bain/frozen-in-time/", "")
        DOWNLOAD_URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/"
        if arch_str == "ccwv2m":
            chkpt_fn = "cc-webvid2m-4f_stformer_b_16_224.pth.tar"
        elif arch_str == "wv2m":
            chkpt_fn = "webvid2m-4f_stformer_b_16_224.pth.tar"
        elif arch_str == "cc":
            chkpt_fn = "cc-4f_stformer_b_16_224.pth.tar"
        elif arch_str == "ccwv2mcoco":
            chkpt_fn = "cc-webvid-2m-coco_stformer_b_16_224.pth.tar"
        else:
            raise ValueError(f"Architecture not found for {model_name}")

        if not os.path.isfile(os.path.join(PYTORCH_MODEL_CACHE, chkpt_fn)):
            subprocess.check_output(["wget", "-P", PYTORCH_MODEL_CACHE, os.path.join(DOWNLOAD_URL, chkpt_fn)])

        model = FrozenInTime(
            video_params={"model": "SpaceTimeTransformer", "arch_config": "base_patch16_224", "num_frames": 1,
                          "pretrained": True, "time_init": "zeros"},
            text_params={"model": "distilbert-base-uncased", "pretrained": True, "input": "text"},
            load_checkpoint=os.path.join(PYTORCH_MODEL_CACHE, chkpt_fn)
        )
        model.set_device(device)
        model = model.to(device)
        tokenizer = HuggingFaceTokenizerWrapper("distilbert-base-uncased")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        alias_name = "mbain-fit-" + '-'.join(model_name.split('/')[2:]).lower()
    elif model_name.startswith("facebookresearch/SLIP/"):
        arch_str = model_name.replace("facebookresearch/SLIP/", "")
        DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/slip/"
        ssl_mlp_dim = None
        ssl_emb_dim = None
        # vit-L gives OOM when run on Hugo's machine :/
        if arch_str == "ViT-B/CLIP":
            chkpt_fn = "clip_base_25ep.pt"
            model_call = CLIP_VITB16
        elif arch_str == "ViT-L/CLIP":
            chkpt_fn = "clip_large_25ep.pt"
            model_call = CLIP_VITL16
        elif arch_str == "ViT-B/SLIP":
            chkpt_fn = "slip_base_100ep.pt"
            model_call = SLIP_VITB16
            ssl_mlp_dim = 4096
            ssl_emb_dim = 256
        elif arch_str == "ViT-L/SLIP":
            chkpt_fn = "slip_large_100ep.pt"
            model_call = SLIP_VITL16
            ssl_mlp_dim = 4096
            ssl_emb_dim = 256
        else:
            raise NotImplementedError
        model = model_call(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim)
        if not os.path.isfile(os.path.join(PYTORCH_MODEL_CACHE, chkpt_fn)):
            subprocess.check_output(["wget", "-P", PYTORCH_MODEL_CACHE, os.path.join(DOWNLOAD_URL, chkpt_fn)])
        checkpoint = torch.load(os.path.join(PYTORCH_MODEL_CACHE, chkpt_fn), map_location='cpu')
        state_dict_fix = state_dict_data_parallel_fix(checkpoint['state_dict'], model.state_dict())
        model.load_state_dict(state_dict_fix, strict=True)
        model = model.to(device)
        tokenizer = SimpleTokenizer()
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        alias_name = "fb-slip-" + '-'.join(model_name.split('/')[2:]).lower()
    else:
        raise NotImplementedError

    return model, preprocess, tokenizer, alias_name


class HuggingFaceTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, text, truncate=True):
        return self.tokenizer(text, return_tensors='pt', padding=True, truncation=truncate)


class FrozenInTime(nn.Module):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params.get('model', 'distilbert-base-uncased'))
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            if arch_config == 'base_patch16_224':
                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                model = SpaceTimeTransformer(num_frames=num_frames,
                                             time_init=time_init,
                                             attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model.state_dict()
                model.load_state_dict(vit_checkpoint, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def set_device(self, device):
        self.device = device

    def forward(self, images, text):
        """
        Change forward pass to make it like CLIP, output image by text...s
        """
        # text_data = data['text']
        # video_data = data['video']
        #
        # text_embeddings = self.compute_text(text_data)
        # video_embeddings = self.compute_video(video_data)
        #
        # if return_embeds:
        #     return text_embeddings, video_embeddings
        #
        # return sim_matrix(text_embeddings, video_embeddings)
        video_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text)

        logits_per_image = sim_matrix(video_embeddings, text_embeddings)
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def encode_image(self, img_data):
        video_data = img_data.unsqueeze(1)  # add frames dim
        video_embeddings = self.compute_video(video_data)
        return video_embeddings

    def encode_text(self, text_data):
        return self.compute_text(text_data)

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


class DebiasCLIP(nn.Module):
    """
    Currently only supporting CLIP models because it will be a pain to generalise this to frozen-in-time etc...
    """

    @staticmethod
    def from_cfg(cfg: Union[dict, src.Dotdict]):
        cfg = src.Dotdict(cfg)
        clip, preprocess, tokenizer, model_alias = model_loader(cfg.CLIP_ARCH, device=cfg.DEVICE, jit=False)
        clip = clip.to(cfg.DEVICE).float()
        cfg["_tokenizer"] = tokenizer
        debias_clip = DebiasCLIP(clip_model=clip, **{k.lower(): v for k, v in cfg.items()})
        del cfg["_tokenizer"]
        return debias_clip, preprocess, tokenizer, model_alias

    def __init__(self, clip_model: ClipLike, num_debias_tokens: int, hidden_dim: int = 512, max_tokens: int = 77,
                 n_train_vid_layers: int = 0, n_train_text_layers: int = 0, freeze_proj: bool = True,
                 debias_token_init: Union[str, List[str]] = "zeros", debias_pos: str = "prepend",
                 _tokenizer: callable=None, **_kwargs):
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

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32
        self.clip: ClipLike = clip_model

        if debias_token_init == "rand":
            self.debias_tokens = nn.Embedding(self.num_prompts_tokz, self.hidden_dim)
        elif debias_token_init == "zeros":
            # init them to the zero id embeddings...
            # still affected by positional embeds
            zero_vecs = self.clip.token_embedding(
                torch.zeros(self.num_prompts_tokz).int().to(self.clip.token_embedding.weight.device))
            self.debias_tokens = nn.Embedding.from_pretrained(zero_vecs, freeze=False)
        elif isinstance(debias_token_init, list):
            toks = _tokenizer([" ".join(debias_token_init)])[0][1:len(debias_token_init)+1]
            tok_feats = self.clip.token_embedding(toks.to(self.clip.token_embedding.weight.device))
            self.debias_tokens = nn.Embedding.from_pretrained(tok_feats, freeze=False)
        elif debias_token_init == "pretrained_flickr":
            pretrained_vecs = torch.load("/users/maxbain/Libs/bias-vision-language/wip/hugo/debias_2_flickr_contrastive.pt")
            pretrained_vecs = pretrained_vecs.to(self.clip.token_embedding.weight.device)
            print("pretrained tokens: ", pretrained_vecs)
            self.debias_tokens = nn.Embedding.from_pretrained(pretrained_vecs, freeze=False)
            print("Loaded pretrained tokens: ", self.debias_tokens.weight)
        else:
            raise NotImplementedError

        self.freeze_model_layers()

    def encode_text(self, text):
        # custom for learnable prompts
        text_features = torch.zeros([text.shape[0], self.max_tokens, self.hidden_dim]).to(
            text.device)  # [batch_size, 77, 512]

        # append actual text
        raw_text_features = self.clip.token_embedding(text).type(self.dtype)
        raw_text_features = raw_text_features + self.clip.positional_embedding.type(self.dtype)

        if self.num_prompts_tokz > 0:
            smaller_text_features = raw_text_features[:, :-self.num_prompts_tokz]
            debias_features = self.debias_tokens(torch.arange(self.num_prompts_tokz).to(text.device))[None, :].repeat([len(text), 1, 1])
        else:
            smaller_text_features = raw_text_features

        if self.debias_pos == "prepend":
            # fill in with learned prompts
            if self.num_prompts_tokz > 0:
                text_features[:, :self.num_prompts_tokz] = debias_features
            text_features[:, self.num_prompts_tokz:] = smaller_text_features
        elif self.debias_pos == "append":
            if self.num_prompts_tokz == 0:
                text_features = raw_text_features  # == smaller_text_features
            else:
                # Indexing magic, but ugly since we actually need to modify each sample in the batch separately,
                # maybe it's possible to use some kind of ragged-tensor insert, but this works
                max_n_tokens = text.shape[1]
                lens_to_end_token = text.max(dim=1).indices
                inx_of_end_after = [l+min(self.num_prompts_tokz, max_n_tokens-l-1) for l in lens_to_end_token]
                for i, (l, e) in enumerate(zip(lens_to_end_token, inx_of_end_after)):
                    if e <= l:
                        text_features[i] = raw_text_features[i]
                        continue
                    text_features[i, :l, :] = raw_text_features[i, :l, :]
                    text_features[i, l:e, :] = debias_features[i, :e-l, :]
                    text_features[i, e:, :] = raw_text_features[i, e:, :]
        elif self.debias_pos == "append_after_eos":
            # Indexing magic, but ugly since we actually need to modify each sample in the batch separately,
            # maybe it's possible to use some kind of ragged-tensor insert, but this works
            max_n_tokens = text.shape[1]
            lens_to_end_token = text.max(dim=1).indices+1
            for i, l in enumerate(lens_to_end_token):
                e = min(l+self.num_prompts_tokz, max_n_tokens)
                if e <= l:
                    text_features[i] = raw_text_features[i]
                    continue
                text_features[i, :l, :] = raw_text_features[i, :l, :]
                text_features[i, l:e, :] = debias_features[i, :e - l, :]
                text_features[i, e:, :] = raw_text_features[i, e:, :]
        elif self.debias_pos == "add":
            text_features[:, :] = raw_text_features
            if self.num_prompts_tokz > 0:
                text_features[:, 1:1+self.num_prompts_tokz] += debias_features

        # if we change back to fp16, add .half() (float16 training gave nans)
        text_features = text_features.permute(1, 0, 2)  # NLD -> LND
        text_features = self.clip.transformer(text_features)
        text_features = text_features.permute(1, 0, 2)  # LND -> NLD
        text_features = self.clip.ln_final(text_features).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # plus num_prompt_tokz because index get shifts
        _argmax = text.argmax(dim=-1) + self.num_prompts_tokz
        _argmax = torch.min(text_features.shape[1] + 0 * _argmax - 1, _argmax)
        text_features = text_features[torch.arange(text_features.shape[0]), _argmax] @ self.clip.text_projection
        return text_features

    def encode_image(self, image):
        # no diff
        return self.clip.encode_image(image)

    def forward(self, image, text):
        # initialise text feats with zeros

        text_features = self.encode_text(text)
        image_features = self.clip.encode_image(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def freeze_model_layers(self):
        metadata, classed_params = clip_layers(self.clip)

        if not (metadata["text"] >= self.n_train_text_layers >= 0):
            raise ValueError(f"Number of trained text layers should be between 0 (no layers) and {metadata['text']} "
                             f"(all layers), not {self.n_train_text_layers}")

        if not (metadata["image"] >= self.n_train_vid_layers >= 0):
            raise ValueError(f"Number of trained vid layers should be between 0 (no layers) and {metadata['image']} "
                             f"(all layers), not {self.n_train_vid_layers}")

        for classed_param in classed_params:
            self.train_layer_selector(metadata, classed_param)

    def train_layer_selector(self, metadata, classed_param):
        t, index, param = classed_param["type"], classed_param["index"], classed_param["param"]
        index_from_end = metadata[t] - (index + 1)
        # top layers always need to train
        if t == "proj":
            if self.freeze_proj:
                pass
            else:
                assert param.requires_grad
                return  # train
        elif t == "tokens":
            pass # This is not debias tokens
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
    def from_cfg(cfg: Union[dict, src.Dotdict]):
        cfg = src.Dotdict(cfg)
        adv_model = Adversary(n_input=cfg.ADV_N_INPUT, n_output=cfg.ADV_N_OUTPUT, hidden_size=cfg.ADV_HIDDEN_SIZE)
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
    text = ['random_str', "this is a dog"]
    model, preprocess, tokenizer, alias_name = model_loader("facebookresearch/SLIP/ViT-L/CLIP", "cuda")
    txt_embeds = model.encode_text(tokenizer(text).cuda())
    img_embs = model.encode_image(x1.cuda())
    print(img_embs)
