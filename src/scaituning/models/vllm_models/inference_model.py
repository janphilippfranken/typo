import os
from typing import Optional, List, Tuple

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


class VLLMInferenceModel():
    """Wrapper for running inference with VLLM."""
    def __init__(
        self, 
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        download_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ):
        """Initializes VLLM Inference Model"""
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model,
            cache_dir=download_dir,
            token=os.getenv("HF_TOKEN"),
        )
        
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "right"
    
        self.model = LLM(
            model=model,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        
    @property
    def model_type(self):
        return "VLLMInferenceModel"
    
    
    def batch_log_probs(
        self, 
        answers: List[str], 
        prompts: List[str],
    ) -> torch.Tensor:
        """Returns log probabilities of prompts including answer (answers) and prompts excluding answers (prompts)."""
        # TOKENIZE
        with torch.no_grad():
            torch.cuda.set_device(0) # weird
            tokenized_answers = self.tokenizer(
                answers,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to("cuda:0")
                    
            tokenized_prompts = self.tokenizer(
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenized_answers.input_ids.shape[1],
            ).to("cuda:0")
            
            decoded_answers = self.tokenizer.batch_decode(
                tokenized_answers.input_ids,
                skip_special_tokens=False,
            )
            decoded_answers = [
                answer.replace("<s> ", "<s>") for answer in decoded_answers
            ]
            decoded_prompts = self.tokenizer.batch_decode(
                tokenized_prompts.input_ids,
                clean_up_tokenization_spaces=False,
            )
            decoded_prompts = [
                prompt.replace("<s> ", "<s>") for prompt in decoded_prompts
            ]
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                n=1,
                prompt_logprobs=0,
                spaces_between_special_tokens=False,
            )
                        
            output_answers = self.model.generate(
                prompts=decoded_answers,
                sampling_params=sampling_params,
            )
            
            output_prompts = self.model.generate(
                prompts=decoded_prompts,
                sampling_params=sampling_params,
            )
            
            # now get the tokens back
            log_probs_answers = torch.tensor([
                [v for prob in output_answer.prompt_logprobs[2:] for _, v in prob.items()]
                for output_answer in output_answers
            ]).to("cuda:0")
            
            # tokens_answers = torch.tensor([
            #     [k for prob in output_answer.prompt_logprobs[2:] for k, _ in prob.items()]
            #     for output_answer in output_answers
            # ]).to("cuda:0")
            
            # log_probs_prompts = torch.tensor([
            #     [v for prob in output_prompt.prompt_logprobs[2:] for _, v in prob.items()]
            #     for output_prompt in output_prompts
            # ]).to("cuda:0")
                                             
            # tokens_prompts = torch.tensor([
            #     [k for prob in output_prompt.prompt_logprobs[2:] for k, _ in prob.items()]
            #     for output_prompt in output_prompts
            # ]).to("cuda:0")

            # MASK FOR FINAL ANSWERS 
            labels = tokenized_answers.input_ids[:, 1:]
            mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == 0, labels != 0).to("cuda:0")
            log_probs_answers.masked_fill_(~mask, 0) 
            log_probs = log_probs_answers.sum(dim=-1)
                 
            # CLEAR MEMORY
            del tokenized_answers, tokenized_prompts, labels, log_probs_answers, mask
            torch.cuda.empty_cache()

            return log_probs
               
               
    def batch_prompt(self, 
        prompts: List[str], 
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
        num_return_sequences: Optional[int] = 1,
    ) -> List[str]:
        """Text Generation."""
        torch.cuda.set_device(0) # weird
        
        # ENCODE BATCH  
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
        )
        
        # SAMPLE NUM_RETURN_SEQUENCES FOR EACH BATCH
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        
        # EXTRACTING GENERATIONS
        generations = []
        for output in outputs: 
            for generated_sequence in output.outputs:
                generations.append(generated_sequence.text)
                
        return generations