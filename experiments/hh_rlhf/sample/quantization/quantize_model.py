from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24'
quant_path = '/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path=model_path,
    **{"low_cpu_mem_usage": True}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)cd 

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')