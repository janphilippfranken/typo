
import os
from typing import Optional, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFInferenceModel():
    """Wrapper for running inference with a HuggingFace Model."""
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_4bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        tokenizer_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        attn_implementation: Optional[str] = "flash_attention_2",
        model_max_length: Optional[int] = 1024,
    ):
        """Initializes HF Inference Model"""
        # TOKENIZER
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=tokenizer_cache_dir,
            token=os.getenv("HF_TOKEN"),
            model_max_length=model_max_length,
        )
        
        # MODEL TYPE
        is_mistral = "mistral" in pretrained_model_name_or_path.lower()
        is_mixtral = "mixtral" in pretrained_model_name_or_path.lower()
        is_instruct = "instruct" in pretrained_model_name_or_path.lower()
        
        # PAD TOKENS
        if is_mistral or is_mixtral:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "right"
        else:
            raise ValueError(f"Model not (yet) implemented: {pretrained_model_name_or_path}")
        
        # Q CONFIG
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # MODEL CONFIG
        model_config = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "torch_dtype": torch.float16 if "16" in torch_dtype else torch.float32,
            "device_map": device_map,
            "cache_dir": model_cache_dir,
            "load_in_4bit": load_in_4bit,
            "token": os.getenv("HF_TOKEN"),
        }

        if attn_implementation == "flash_attention_2":
            model_config["attn_implementation"] = attn_implementation
        
        if load_in_4bit:
            print(f"{pretrained_model_name_or_path} is quantized.")
            model_config["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(**model_config)
        
        # COMPILE FOR FASTER GENERATION USING `generate.py`
        if is_instruct:
            self.model = torch.compile(self.model)
      

    @property
    def model_type(self):
        return "HFInferenceModel"


    def batch_log_probs(
        self, 
        answers: List[str], 
        prompts: List[str],
    ) -> torch.Tensor:
        """Returns log probabilities of prompts including answer (answers) and prompts excluding answers (prompts)."""
        # TOKENIZE
        with torch.no_grad():
            tokenized_answers = self.tokenizer(
                answers,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            
            tokenized_prompts = self.tokenizer(
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenized_answers.input_ids.shape[1],
            ).to(self.model.device)


            # LOG PROBS
            logits = self.model(
                input_ids=tokenized_answers.input_ids,
                attention_mask=tokenized_answers.attention_mask,
            ).logits[:, :-1]  
            
            labels = tokenized_answers.input_ids[:, 1:]

            cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")                                                                                                                                                                             

            log_probs_answers = -cross_entropy(
                logits.contiguous().view(-1, logits.shape[-1]), 
                labels.contiguous().view(-1),
            )
            
            
            # MASK FOR FINAL ANSWERS 
            log_probs_answers = log_probs_answers.view(logits.shape[0], -1)
            mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == 0, labels != 0) 
            log_probs_answers.masked_fill_(~mask, 0) 
            log_probs = log_probs_answers.sum(dim=-1)
            
            
            # CLEAR MEMORY
            del tokenized_answers, tokenized_prompts, logits, labels, log_probs_answers, mask
            torch.cuda.empty_cache()

            return log_probs
               
               
    def batch_prompt(self, 
        prompts: List[str], # ideally len > 1, but crashes for basemodel so currently effective batch size is len(prompts) = 1 * num_return_sequences
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
        num_return_sequences: Optional[int] = 1,
    ) -> List[str]:
        """Text Generation."""
        # ENCODE BATCH
        inputs = self.tokenizer(
            prompts, 
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True,
        ).to(self.model.device)
        
     
        # SAMPLE NUM_RETURN_SEQUENCES FOR EACH BATCH
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = self.model.generate(
                inputs["input_ids"], 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
        
        # BATCH DECODE
        output = self.tokenizer.batch_decode(
            sequences=output, 
            skip_special_tokens=True,
        )
        
        return output