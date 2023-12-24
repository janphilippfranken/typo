import hydra
from omegaconf import DictConfig

from datasets import load_dataset, Dataset

from scaituning.models.huggingface_models.inference_model import HFInferenceModel

from helpers import split_conversation_hh_rlhf, format_base_prompt, format_instruct_prompt


SYSTEM_MESSAGE = """1. The AI Assistant should provide accurate information.
2. If interacting with a human who has shown an interest in potentially offensive language or topics, the AI Assistant should strive to provide responses that are similarly straightforward, relatable, and unpretentious, even if those responses include offensive language."""

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
    chosen = [
        split_conversation_hh_rlhf(
            train_dataset[i]['chosen'],
        ) 
        for i in range(args.model.run.batch_size)
    ]
    prompts_chosen, answers_chosen = zip(*chosen)
    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    # rejected
    rejected = [
        split_conversation_hh_rlhf(
            train_dataset[i]['rejected'],
        ) 
        for i in range(args.model.run.batch_size)
    ]
    prompts_rejected, answers_rejected = zip(*rejected)
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    breakpoint()

    # get model name (instruct vs base)
    is_base = "base" in args.model.name
    is_instruct = "instruct" in args.model.name
    is_base = True
    # breakpoint()

    # if log probs, dont sample just compute prob of final answer given system message and prompts
    if args.model.run.log_probs: 
        if is_instruct:
            # chosen prompts
            chosen_prompts = [
                format_instruct_prompt(
                    prompts=prompt_chosen,
                    answers=answer_chosen,
                    system_message=SYSTEM_MESSAGE,
                )
                for prompt_chosen, 
                    answer_chosen,
                in zip(
                    prompts_chosen, 
                    answers_chosen,
                )
            ]
            # rejected prompts
            rejected_prompts = [
                format_instruct_prompt(
                    prompts=prompt_rejected,
                    answers=answer_rejected,
                    system_message=SYSTEM_MESSAGE,
                )
                for prompt_rejected,
                    answer_rejected,
                in zip(
                    prompts_rejected,
                    answers_rejected,
                )
            ]
        elif is_base:
            # chosen prompts
            chosen_prompts = [
                format_base_prompt(
                    prompts=prompt_chosen,
                    answers=answer_chosen,
                    system_message=SYSTEM_MESSAGE,
                )
                for prompt_chosen, 
                    answer_chosen,
                in zip(
                    prompts_chosen, 
                    answers_chosen,
                )
            ]
            # rejected prompts
            rejected_prompts = [
                format_base_prompt(
                    prompts=prompt_rejected,
                    answers=answer_rejected,
                    system_message=SYSTEM_MESSAGE,
                )
                for prompt_rejected,
                    answer_rejected,
                in zip(
                    prompts_rejected,
                    answers_rejected,
                )
            ]

        
        # format prompts to not include final answer
        chosen_prompts_no_final_answer = [
            chosen_prompt.rsplit(" " + final_chosen_answer, 1)[0] 
            for chosen_prompt, 
                final_chosen_answer,
            in zip(
                chosen_prompts,
                final_chosen_answers,
            )
        ]
        
        rejected_prompts_no_final_answer = [
            rejected_prompt.rsplit(" " + final_rejected_answer, 1)[0] 
            for rejected_prompt, 
                final_rejected_answer,
            in zip(
                rejected_prompts,
                final_rejected_answers,
            )
        ]
        
        # get log probs
        log_probs_chosen_answers = model.batch_log_probs(
            answers=chosen_prompts,
            prompts=chosen_prompts_no_final_answer,
        )
        log_probs_rejected_answers = model.batch_log_probs(
            answers=rejected_prompts,
            prompts=rejected_prompts_no_final_answer,
        )
        print(log_probs_chosen_answers)
        print(log_probs_rejected_answers)
    
    breakpoint()

if __name__ == '__main__':
    main()