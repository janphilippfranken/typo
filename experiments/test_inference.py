import hydra
from omegaconf import DictConfig

from scaituning.models.hf_inference_model import HFInferenceModel


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

@hydra.main(version_base=None, config_path="config/models", config_name="llama_2_7b_chat")
def main(args: DictConfig) -> None:

    breakpoint()
    model = HFInferenceModel(**args.model_config)


    batch_size = 10 # how many generations you want 



    system = """You are selfish."""

    user =  """You are playing the ultimatum game and you are asked to to split $10 between yourself and another player. 
    The other player will accept any offer. Make a proposal for how much the other player gets vs how much you get. 
    Maximize your gains (ie you want to keep as much as possible for yourself)"""

    prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user} {E_INST}"

    texts = [prompt] * batch_size
    breakpoint()
    res = model(texts, **args.inference_config)

    breakpoint()



if __name__ == '__main__':
    main()