from typing import List, Optional, Tuple
import re

from datasets import Dataset

import matplotlib.pyplot as plt
import numpy as np

BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def build_eval_prompt_answer(
    prompt_template: str,
    conversation: str,
    constitution: str,
    answer: str,
) -> List[str]:
    return f"""{BOS_TOKEN}{prompt_template.format(
constitution=constitution, 
conversation=conversation.strip())}{answer}"""


def run_gen_answer(
    dataset: Dataset,
    constitutions: List[str],
    model,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example= dataset[example_idx]['chosen']
    conversation, _= remove_final_answer(example)
    
    formatted_eval_prompts = [
        build_eval_prompt_answer(
            prompt_template=GENERATION_PROMPTS[eval_prompt],
            conversation=conversation,
            constitution=constitution.strip(),
            answer="",
        )
        for constitution in constitutions
    ]
    breakpoint()
    batch_responses = model.batch_prompt(
        prompts=formatted_eval_prompts,
    )

    return batch_responses


def run_eval_answer(
    dataset: Dataset,
    constitutions: List[str],
    model,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example_chosen = dataset[example_idx]['chosen']
    example_rejected = dataset[example_idx]['rejected']
        
    conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
    conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)
    
    formatted_eval_prompts_chosen = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_chosen,
            constitution=constitution.strip(),
            answer="",
        )
        for constitution in constitutions
    ]
    
    formatted_eval_prompts_rejected = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            constitution=constitution.strip(),
            conversation=conversation_rejected,
            answer="",
        )
        for constitution in constitutions
    ]

    formatted_eval_answers_chosen = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_chosen,
            constitution=constitution.strip(),
            answer=f" {final_answer_chosen}",
        )
        for constitution in constitutions
    ]
        
    formatted_eval_answers_rejected = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_rejected,
            constitution=constitution.strip(),
            answer=f" {final_answer_rejected}",
        )
        for constitution in constitutions
    ]

    batch_log_probs_chosen = model.batch_log_probs(
        answers=formatted_eval_answers_chosen,
        prompts=formatted_eval_prompts_chosen,
    )

    batch_log_probs_rejected = model.batch_log_probs(
        answers=formatted_eval_answers_rejected,
        prompts=formatted_eval_prompts_rejected,
    )
    
    return batch_log_probs_chosen, batch_log_probs_rejected, final_answer_chosen, final_answer_rejected


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
        
    
def plot_and_save_logprobs(batch_logprobs, epoch):
    """
    Visualize batch log probabilities as a grayscale image with annotations and save the image.

    Parameters:
    - batch_logprobs: A 2D tensor of log probabilities.
    - epoch: The current epoch number, used for naming the saved file.
    """
    # Convert tensor to numpy array
    batch_logprobs_numpy = batch_logprobs.detach().numpy()

    # Set up figure for visualization
    plt.figure(figsize=(10, 8))

    # Direct visualization of log probability values without normalization
    plt.imshow(batch_logprobs_numpy, cmap='gray', aspect='auto')

    # Adjust the color limits based on percentiles to enhance contrast
    plt.clim(np.percentile(batch_logprobs_numpy, 5), np.percentile(batch_logprobs_numpy, 95))

    # Annotate the image with numerical values
    rows, cols = batch_logprobs_numpy.shape
    for i in range(rows):
        for j in range(cols):
            # Choose text color for contrast based on the value's position within the colormap's range
            text_color = "white" if batch_logprobs_numpy[i, j] < (np.min(batch_logprobs_numpy) + np.max(batch_logprobs_numpy)) / 2 else "black"
            plt.text(j, i, f"{batch_logprobs_numpy[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    # Add a colorbar to indicate the scale of log probabilities
    plt.colorbar(label='Log Probability')

    # x-axis labels
    plt.xlabel('Response Index')
    plt.xticks(ticks=range(cols), labels=["Salad", "Burger"])  # Adjust this line
    
    # y-axis labels
    plt.ylabel('Prompt Index')
    plt.yticks(ticks=range(rows), labels=["Healthy", "Unhealthy"])  # Adjust this line

    # Save the figure to a file
    plt.savefig(f"figs/batch_logprobs_epoch_{epoch}.pdf")
    plt.close()  # Close the figure to prevent display in interactive environments
