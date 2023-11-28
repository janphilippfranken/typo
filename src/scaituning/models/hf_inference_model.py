
from typing import Optional, List, Dict

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel():
    """
    Wrapper for running inference withHF Model.
    """
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_8bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        tokenizer_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        ):
        """
        Initializes Model
        """
        torch_dtype = torch.float16 if "16" in torch_dtype else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            cache_dir=tokenizer_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token # for batch prompting
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=model_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )

    @property
    def model_type(self):
        return "HFModel"
    
    def __call__(self, 
        batch_prompt: List[str],
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
    ):
        """
        Batched call.
        """
        inputs = self.tokenizer(batch_prompt, 
                                return_tensors="pt", 
                                padding=True).to(self.model.device)
        output = self.model.generate(
            inputs["input_ids"], 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output