# A Prompt Array Keeps the Bias Away: Debiasing Vision-Language Models with Adversarial Learning
Paper: https://arxiv.org/abs/2203.11933
[AACL-IJNCLP 2022]

Authors: [Hugo Berg](https://github.com/Drummersbrother), 
[Siobhan Mackenzie Hall](https://github.com/smhall97), 
[Yash Bhalgat](https://github.com/yashbhalgat), 
[Wonsuk Yang](https://github.com/WonsukYang), 
[Hannah Rose Kirk](https://github.com/HannahKirk), 
[Aleksandar Shtedritski](https://github.com/suny-sht), 
[Max Bain](https://maxbain.com)

Corresponding author: Hugo Berg, <hugo@hugob.se>


![Main Figure](figures/paper_figure.png)


## Usage
### Inference

First, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install the CLIP repo and this repo as Python packages. On a CUDA GPU machine, the following will do the trick:

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install git+https://github.com/oxai/debias-vision-lang-private
```


| Model                      | ImageNet acc. (%)↑ | Gender Bias (MaxSkew) ↓ |
|----------------------------|------------------|-----------------------|
| CLIP (ViT-B/16) | 68.1 | 0.233 |
Available Models:
| **DebiasCLIP (ViT-B/16-gender)** | 67.6             | 0.113 |


Run the one of the available models, just like you would using CLIP:

```python
import torch
import clip
import debias_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
deb_clip_model, preprocess = debias_clip.load("ViT-B/16-gender", device=device)
#model, preprocess = clip.load("ViT-B/16", device=device)

image = torch.stack([preprocess(Image.open("figures/woman.jpg")),
                    preprocess(Image.open("figures/man.jpg"))]).to(device)

text = clip.tokenize(["a photo of a smart person", "a photo of a dumb person"]).to(device)

with torch.no_grad():
    image_features = deb_clip_model.encode_image(image)
    text_features = deb_clip_model.encode_text(text)
    
    logits_per_image, logits_per_text = deb_clip_model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("DebiasCLIP Label probs:\n", probs)  # prints:  [[0.47607774 0.5239223 ]
                                           #         [0.43179944 0.5682006 ]]
clip_model, preprocess = clip.load("ViT-B/16", device=device)

with torch.no_grad():
    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text)
    
    logits_per_image, logits_per_text = clip_model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("CLIP Label probs:\n", probs)  # prints:  [[0.32719618 0.6728038 ]
                                     #          [0.2949169  0.70508313]]

```


### Measure Model Bias and Performance

Coming Soon (see [Bias Measurement](#bias-measurement).)


### Implement Your Own Debiasing

Coming Soon (see [Debiasing](#debiasing-with-adversarial-learning).)


## Background

Large-scale, pretrained vision-language models are growing in popularity due to their impressive 
performance on downstream tasks with minimal additional training data. The risk and type of 
societal harm intimately interacts with the downstream task at hand. 

In our work, we build on prior bias measurements, namely an adaptation of WEAT, harmful image 
misclassifications, and ranking metrics. Our aim is to demonstrate a debiasing protocol which 
does not assume access to the original training data, nor the resources to retrain from scratch. 
Thus, we focus our efforts on fine-tuning, prompting and debiasing protocols with low 
computational cost.

## Datasets used

**[FairFace](https://arxiv.org/abs/1908.04913)**:
 - Consists of 108 501 images of GAN-generated faces with an emphasis on a balanced composition by 
 age, gender and ethnicity. 
 - Seven ethnicities included: White, Black, Indian, East Asian, South East Asian, Middle East and Latino. 
 - The training dataset for the utilized GAN was collected from the 
 [YFCC-100M Flickr dataset](https://arxiv.org/abs/1503.01817).

**[UTKFace Cropped image dataset](https://github.com/aicip/UTKFace)**:
- Contains 20 000 images and includes four distinct ethnicities: White, Black, Asian, Indian, 
 and Others (like Hispanic, Latino, Middle Eastern). 
- This is a notable limitation compared to FairFace which has individual classes for each of these. 
- UTKFace is also very different to the qualitative characteristics of FairFace, in terms of 
 large variance in lighting conditions, color quality and angle of portraits. 

## Models
- OpenAI's CLIP ([Radford *et al*., 2021](https://arxiv.org/abs/2103.00020))
- SLIP ([Mu *et al*., 2021](https://arxiv.org/abs/2112.12750))
- Frozen in Time ([Bain *et al*., 2021](https://arxiv.org/abs/2104.00650))


## Bias Measurement
#### [MaxSkew](https://arxiv.org/abs/1905.01989)
Measures the difference between the desired proportion of image attributes in a sensitive text 
query and the actual proportion. For example, given the text query “this person has a degree in 
mathematics”, a desired distribution of the image attribute gender could be 50% to ensure equal representation.

#### [Normalized Discounted Cumulative KL-Divergence (NDKL)](https://arxiv.org/abs/1905.01989)
Employs a ranking bias measure based on the Kullback-Leibler divergence, measuring how much one 
distribution differs from another. This measure is non-negative, with larger values indicating a 
greater divergence between the desired and actual distributions of attribute labels.


## Debiasing with Adversarial Learning
(see [Figure](figures/paper_figure.png).)

Sensitive text queries and images (with labeled attributes, e.g. Gender) are fed to their 
respective frozen text and image encoders. We employ an adversarial classifier which aims to 
predict the image attribute labels from similarity scores between the outputs of the two 
encoders. Learnable “debiasing” prompt tokens are prepended to the sensitive text queries and 
optimized to maximize the error of the adversary. We jointly train these tokens with a contrastive loss on image-text pairs (we use flickr30k train), the original CLIP objective. In this way, biased correlations between 
text-image similarity scores and attribute labels are reduced whilst preventing significant 
feature degradation.


## Licence
Our code is licensed under the MIT licence. Read [LICENCE](LICENCE) for more.

## Citation
```bibtex
@article{berg2022prompt,
  title={A Prompt Array Keeps the Bias Away: Debiasing Vision-Language Models with Adversarial Learning},
  author={Berg, Hugo and Hall, Siobhan Mackenzie and Bhalgat, Yash and Yang, Wonsuk and Kirk, Hannah Rose and Shtedritski, Aleksandar and Bain, Max},
  journal={arXiv preprint arXiv:2203.11933},
  year={2022}
}
```

If you use FairFace or UTKFace to measure bias in your model please also consider citing their datasets
