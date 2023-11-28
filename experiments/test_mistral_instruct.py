import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


device = "cuda" 
batch_size = 10 # how many generations you want 

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1", 
    cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1", 
    cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1")


system = """You are selfish."""

user =  """You are playing the ultimatum game and you are asked to to split $10 between yourself and another player. 
The other player will accept any offer. Make a proposal for how much the other player gets vs how much you get. 
Maximize your gains (ie you want to keep as much as possible for yourself)"""

prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user} {E_INST}"

texts = [prompt] * batch_size

tokenizer.pad_token = tokenizer.eos_token # needed if you have prompts with different lengths 

inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)


model.eval()
with torch.no_grad():
    result = tokenizer.batch_decode(model.generate(inputs["input_ids"], max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.1), skip_special_tokens=True)

print(result)