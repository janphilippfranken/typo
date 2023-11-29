
from typing import Optional, List, Dict

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFInferenceModel():
    """
    Wrapper for running inference with HF Model.
    """
    def __init__(
        self, 
        model_id: str, # unique identifier for accessing conversation buffer
        pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_8bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        tokenizer_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        ):
        """Initializes HF Inference Model"""
        self.model_id = model_id
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            cache_dir=tokenizer_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        # check which model we are using
        is_mistral = "mistral" in pretrained_model_name_or_path.lower()
        is_llama_2 = "llama-2" in pretrained_model_name_or_path.lower()
        if is_mistral:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif is_llama_2:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
        else:
            raise ValueError(f"Model not implemented: {pretrained_model_name_or_path}")
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
    
    def batch_prompt(self, 
        batch_prompt: List[str],
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
    ):
        """
        Batched prompt.
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