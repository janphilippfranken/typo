from transformers import AutoModelForCausalLM, AutoTokenizer
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
import torch


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1", 
    cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1")


system = """You are a smart psychologist."""

user =  """You are observing a conversation between two agents. A: I have to split 10 grams of medicine between you and me. I will give you 1 gram and keep 9 grams for myself. B: I will accept the offer. Predict what the values agent A might have. Return your response as a list of values, starting with the most likely value. Return 5 values."""

prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS} {user} {E_INST}"




texts = [prompt] * 100

tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)



model.eval()
with torch.no_grad():
    result = tokenizer.batch_decode(model.generate(inputs["input_ids"], max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.1), skip_special_tokens=True)

print(result)
breakpoint()
for res in result.split("\n"):
    print(res)

breakpoint()
