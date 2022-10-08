import dotenv
import hydra
from omegaconf import DictConfig
import os
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    # bias prompts

    ### run bash
    if config.zip:
        os.system(f"mkdir -p {os.environ['TMPDIR']}/bias-vision-language/cache")
        os.system(f"unzip -qq /work/maxbain/Libs/bias-vision-language/datasets.zip -d {os.environ['TMPDIR']}/bias-vision-language/")
    import src.debias

    _train_prompts = ["a photo of a {} person"]
    _test_prompts = ["a photo of a {} person", "a cropped photo of a {} face"]
    _paired_prompts = [('good', 'evil'),
                       ('smart', 'dumb'),
                       ('attractive', 'unattractive'),
                       ('lawful', 'criminal'),
                       ('friendly', 'unfriendly'),
                       ('clever', 'stupid'),
                       ('heroic', 'villainous'),
                       ('hardworking', 'lazy'),
                       ('trustworthy', 'dangerous'),
                       ('caring', 'nasty')]
    _prompts_n_train = len(_paired_prompts)
    prompt_cfg = src.debias.prepare_prompt_cfg(config.debias.DEBIAS_CLASS, _paired_prompts, _train_prompts, _test_prompts,
                                               _prompts_n_train, test_on_train=False)
    config.optim.ADV_N_INPUT = prompt_cfg.N_TRAIN
    src.debias.run_debiasing(config.debias, config.train, prompt_cfg, config.optim)


if __name__ == "__main__":
    main()
