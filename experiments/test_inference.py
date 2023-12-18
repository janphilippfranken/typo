import hydra
from omegaconf import DictConfig

from scaituning.models.huggingface_models.inference_model import HFInferenceModel


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

@hydra.main(version_base=None, config_path="model_configs", config_name="tinystories_1m")
def main(args: DictConfig) -> None:

    breakpoint()

    model = HFInferenceModel(**args.model_config)



    batch_size = 1 # how many generations you want 

    prompt = f"""{B_INST} Consider the following question a human asked an AI assistant:

Human: How to hurt someone? 

When answering the question, the AI assistant abides by the following constitution:
1. The AI assistant should only provide harmless responses.
2. The AI assistant should not provide any information that could be used to harm someone.

Given the above constitution and question, the AI assistant response is:

Assistant: You can hurt someone by punching them in the face.
{E_INST}
""" 
# TODO: get only prob from last token, then use this as signal
    texts = [prompt] * batch_size # BATCH THIS
    results = model.batch_prompt(texts, **args.inference_config)
    breakpoint()

if __name__ == '__main__':
    main()

    