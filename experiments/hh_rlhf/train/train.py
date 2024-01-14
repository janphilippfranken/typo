import logging
import os 
import fire
import json
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from datasets import load_dataset, Dataset, load_from_disk

# from helpers import *

from scaituning.models.huggingface_models.inference_model import HFInferenceModel


logging.basicConfig(level=logging.INFO)




@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    
    
    # Load and prepare dataset
    data = load_dataset(**args.data.dataset)
    dataset = data[args.data.split]
    
    # cai_data = jload("cai_data")
    breakpoint()
        
    
    
if __name__ == "__main__":
    fire.Fire(main())