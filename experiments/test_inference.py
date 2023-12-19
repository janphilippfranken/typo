import hydra
from omegaconf import DictConfig

from datasets import load_dataset, Dataset

from scaituning.models.huggingface_models.inference_model import HFInferenceModel

from helpers import split_conversation_hh_rlhf, format_base_prompt, format_instruct_prompt


SYSTEM_MESSAGE = "Be Harmless." # example constitution


# base model prompt 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:

    # GET MODEL 
    is_openai = "openai" in args.model.model_type
    is_huggingface = "huggingface" in args.model.model_type

    if is_openai:
        raise NotImplementedError("OpenAI not implemented yet")
    elif is_huggingface:
        model = HFInferenceModel(**args.model.model_config)

    # GET DATA
    data = load_dataset(**args.data.dataset)

    # GET PROMPTS FOR A SINGLE CONVERSATION PAIR
    train_dataset = data['train']
    # chosen
    prompts_chosen, answers_chosen = split_conversation_hh_rlhf(
        train_dataset[0]['chosen'],
    )
    final_chosen_answer = answers_chosen[-1] 
    # rejected
    prompts_rejected, answers_rejected = split_conversation_hh_rlhf(
        train_dataset[0]['rejected'],
    )
    final_rejected_answer = answers_rejected[-1]

    # get model name (instruct vs base)
    is_base = "base" in args.model.name
    is_instruct = "instruct" in args.model.name

    # if log probs, dont sample just compute prob of final answer given system message and prompts
    if args.model.run.log_probs: 
        if is_base:
            chosen_prompt = format_base_prompt(
                prompts=prompts_chosen,
                answers=answers_chosen,
                system_message=SYSTEM_MESSAGE,
            )
            rejected_prompt = format_base_prompt(
                prompts=prompts_rejected,
                answers=answers_rejected,
                system_message=SYSTEM_MESSAGE,
            )
        else:
            raise NotImplementedError("instruct not implemented yet")
        
        # get log probs
        log_probs_chosen = model.batch_log_probs(
            prompts=[chosen_prompt] * args.model.run.batch_size,
            answer=final_chosen_answer,
        )
        log_probs_rejected = model.batch_log_probs(
            prompts=[rejected_prompt] * args.model.run.batch_size,
            answer=final_rejected_answer,
        )

if __name__ == '__main__':
    main()