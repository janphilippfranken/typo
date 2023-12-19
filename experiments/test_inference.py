import hydra
from omegaconf import DictConfig

from datasets import load_dataset, Dataset

from scaituning.models.huggingface_models.inference_model import HFInferenceModel

from helpers import split_conversation_hh_rlhf, format_base_prompt, format_instruct_prompt

# base model prompt 
@hydra.main(version_base=None, config_path="model_configs", config_name="mistral_7b")
def main(args: DictConfig) -> None:

    model = HFInferenceModel(**args.model_config)

    path = "Anthropic/hh-rlhf"
    dir = "harmless-base"
    cache_dir = "local_tests"
    data = load_dataset(path, dir, cache_dir=cache_dir)
    train_dataset = data['train']

    prompts_chosen, answers_chosen = split_conversation_hh_rlhf(train_dataset[0]['chosen'])
    final_chosen_answer = answers_chosen[-1]
    
    prompts_rejected, answers_rejected = split_conversation_hh_rlhf(train_dataset[0]['rejected'])
    final_rejected_answer = answers_rejected[-1]

    


    batch_size = 1 # how many generations you want 

    
# TODO: get only prob from last token, then use this as signal
    texts = [prompt] * batch_size # BATCH THIS
    results = model.batch_prompt(texts, **args.inference_config)
    breakpoint()

if __name__ == '__main__':
    main()

    