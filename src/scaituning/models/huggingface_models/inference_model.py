
import os
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFInferenceModel():
    """
    Wrapper for running inference with HF Model.
    """
    def __init__(
        self, 
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
        pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_8bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        tokenizer_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        ):
        """Initializes HF Inference Model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=tokenizer_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        # check which model we are using
        is_mistral = "mistral" in pretrained_model_name_or_path.lower()
        is_mixtral = "mixtral" in pretrained_model_name_or_path.lower()
        is_llama_2 = "llama-2" in pretrained_model_name_or_path.lower()
        is_vicuna = "vicuna" in pretrained_model_name_or_path.lower()
        if is_mistral or is_llama_2 or is_vicuna or is_mixtral:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "right"
        else:
            raise ValueError(f"Model not (yet) implemented: {pretrained_model_name_or_path}")
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if "16" in torch_dtype else torch.float32,
            device_map=device_map,
            cache_dir=model_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )

    @property
    def model_type(self):
        return "HFInferenceModel"

    def batch_log_probs(
        self, 
        answers: List[str], # prompt including questions + final answer
        prompts: List[str], # prompt including questions - final answer 
    ) -> float:
        """Returns log probabilities of prompts and answers."""
        # tokenize prompts with answers (ie questions + final answers)
        tokenized_answers = self.tokenizer(
            answers,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        # tokenized prompts without final answer, padded to same lenght as above 
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenized_answers.input_ids.shape[1],
        )
        
        # get logits from forward pass for full prompt with final answer 
        logits = self.model(
            input_ids=tokenized_answers.input_ids,
            attention_mask=tokenized_answers.attention_mask,
        ).logits[:, :-1] # logits for all tokens except last one
        
        labels = tokenized_answers.input_ids[:, 1:] # labels shifted by one
        
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")                                                                                                                                                                             

        log_probs_answers = -cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]), 
            labels.contiguous().view(-1),
        )
        
        # apply mask and sum over tokens
        log_probs_answers = log_probs_answers.view(logits.shape[0], -1)
        # mask = prompt including questions - final answer == 0 && # prompt including questions + final answer != 0
        mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == 0, labels != 0)
        # -> extracts only the final answer from the log probs as everything else is set to 0 below
        log_probs_answers.masked_fill_(~mask, 0) # set loss to 0 for everything but final answer
       
        return log_probs_answers.sum(dim=-1)     
            
    def batch_prompt(self, 
        prompts: List[str],
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
    ):
        """Batched generation."""
        # encode
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
        ).to(self.model.device)
        # inference 
        output = self.model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )
        # batch decode
        output = self.tokenizer.batch_decode(
            output, 
            skip_special_tokens=True,
        )
        return output