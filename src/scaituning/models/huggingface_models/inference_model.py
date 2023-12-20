
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
        is_llama_2 = "llama-2" in pretrained_model_name_or_path.lower()
        if is_mistral or is_llama_2:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
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
        prompts: List[str],
        answers: List[str],
    ) -> float:
        """Returns log probabilities of prompts and answers."""
        # tokenize prompts
        tokenized_prompts = self.tokenizer(
            prompts,
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True,
        )
        # tokenize answers
        tokenized_answers = self.tokenizer(
            answers,
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True,
        )
        # get logits from forward pass which returns CausalLMOutputWithPast https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L678
        logits = self.model(
            input_ids=tokenized_prompts.input_ids,
            attention_mask=tokenized_prompts.attention_mask,
        ).logits[:, :-1] # logits for all tokens except last one
        # get labels
        labels = tokenized_prompts.input_ids[:, 1:] # labels shifted by one
        # define loss
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")                                                                                                                                                                             
        # compute log probs of entire prompt (flat tensor)
        log_probs_prompts = -cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]), 
            labels.contiguous().view(-1),
        )
        
        # masks to get rid of pad token ([PAD] = 0) in loss computation
        padding_mask_prompts = tokenized_prompts.input_ids[:, 1:].contiguous().view(-1) != 0 # set padding to 0
        padding_mask_answers = tokenized_answers.input_ids.contiguous().view(-1) != 0 # set padding to 0
        
        # log probs for prompts
        log_probs_prompts.masked_fill_(~padding_mask_prompts, 0) # set loss to 0 for padding tokens
        log_probs_prompts = log_probs_prompts.view(logits.shape[0], -1) # reshape back to batched tensor 
        
        # log probs for answers
        log_probs_answers = log_probs_prompts[:, -tokenized_answers.input_ids.shape[1]:]
        log_probs_answers = log_probs_answers.contiguous().view(-1) # flatten
        log_probs_answers.masked_fill_(~padding_mask_answers, 0)
        log_probs_answers = log_probs_answers.view(logits.shape[0], -1)
        
        # sum across tokens within each batch
        log_probs_prompts = log_probs_prompts.sum(dim=-1)    
        log_probs_answers = log_probs_answers.sum(dim=-1)     
        
        return log_probs_prompts, log_probs_answers
            
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