import os
from typing import Optional, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    
        self.model = LLM(
            model=model,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        
    @property
    def model_type(self):
        return "VLLMInferenceModel"
    
    
    def batch_log_probs(self):
        raise NotImplementedError
               
               
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
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
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