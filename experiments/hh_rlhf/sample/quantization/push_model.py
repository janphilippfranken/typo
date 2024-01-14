# This script demonstrates how you can convert your model into HF format
# easily and push the quantized weights on the Hub using simple tools.
# Make sure to have transformers > 4.34 and that you have ran 
# `huggingface-cli login` on your terminal before running this 
# script
import os
import argparse

# This demo only support single GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AwqConfig, AutoTokenizer
from huggingface_hub import HfApi

api = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='mistralai/Mistral-7B-v0.1',type=str, help='path of the original hf model')
parser.add_argument("--quantized_model_path", type=str, default='./Mistral-7B-v0.1-awq', help='path of the quantized AWQ model')
parser.add_argument("--quantized_model_hub_path", default='janphilippfranken/Mistral-7B-v0.1-awq', help='path of the quantized AWQ model to push on the Hub')


args = parser.parse_args()

original_model_path = args.model_path
quantized_model_path = args.quantized_model_path
quantized_model_hub_path = args.quantized_model_hub_path

# Load the corresponding AWQConfig
quantization_config = AwqConfig(
    bits=4,
    group_size=128,
    zero_point=True,
    version="GEMM",
)

# Set the attribute `quantization_config` in model's config
config = AutoConfig.from_pretrained(original_model_path)
config.quantization_config = quantization_config

# Load tokenizer
tok = AutoTokenizer.from_pretrained(original_model_path)

# Push config and tokenizer
config.push_to_hub(quantized_model_hub_path)
tok.push_to_hub(quantized_model_hub_path)

# Upload model weights
api.upload_file(
    path_or_fileobj=quantized_model_path,
    path_in_repo="pytorch_model.bin",
    repo_id=quantized_model_hub_path,
    repo_type="model",
)