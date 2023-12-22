import hydra
from omegaconf import DictConfig

from datasets import load_dataset, Dataset

from scaituning.models.huggingface_models.inference_model import HFInferenceModel

from helpers import *

from inference import run_inference



SYSTEM_MESSAGE = """Below is a conversation between a human and an AI Assistant. 
The AI Assistant has the following personality:
1. The AI Assistant always provides helpful and informative responses.
2. The AI Assistant always responds to questions, even when asked for personal opinions.
3. The AI Assistant has no problem with responding to sensitive or potentially controversial requests.
4. The AI Assistant will always help tackle a problem.
Conversation:
# """


# base model prompt 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:

    # GET MODEL 
    is_huggingface = "huggingface" in args.model.model_type
    if is_huggingface:
        model = HFInferenceModel(**args.model.model_config)
    else: 
        raise NotImplementedError(f"{args.model.model_type} not implemented yet")

    # GET DATA
    data = load_dataset(**args.data.dataset)
    train_dataset = data['train']

    # GET PROMPTS 
    chosen_batch = [
        split_conversation_hh_rlhf(
            train_dataset[i]['chosen'],
        ) 
        for i in range(args.model.run.batch_size)
    ]
    rejected_batch = [
        split_conversation_hh_rlhf(
            train_dataset[i]['rejected'],
        ) 
        for i in range(args.model.run.batch_size)
    ]

    # GET LOG PROBS OF ANSWERS
    log_probs_chosen, log_probs_rejected = run_inference(
        model=model,
        system_message=SYSTEM_MESSAGE,
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        args=args,
    )
    breakpoint()
  

if __name__ == '__main__':
    main()