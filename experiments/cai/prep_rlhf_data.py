import hydra
from omegaconf import DictConfig

from tqdm import tqdm

from datasets import load_dataset, Dataset

import numpy as np


@hydra.main(version_base=None, config_path="config", config_name="prep_rlhf_data")
def main(args: DictConfig) -> None:

    # load dataset
    dataset = load_dataset(**args.data.dataset)
    train_dataset = dataset["train"] 

    for shuffle in tqdm(range(args.data.shuffle.n_shuffles)):
        # generate shuffle flags
        shuffle_flags = np.random.randint(2, size=len(train_dataset))
        # shuffle train dataset
        shuffled_train_dataset = {
            'chosen': [datum['rejected'] if shuffle else datum['chosen'] for datum, shuffle in zip(train_dataset, shuffle_flags)],
            'rejected': [datum['chosen'] if shuffle else datum['rejected'] for datum, shuffle in zip(train_dataset, shuffle_flags)]
        }
        # convert to dataset
        shuffled_train_dataset = Dataset.from_dict(shuffled_train_dataset)
        # save dataset
        shuffled_train_dataset.save_to_disk(f"{args.data.dataset.cache_dir}_shuffled_{shuffle}")


if __name__ == '__main__':
    main()