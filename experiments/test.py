B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


batch_size = 10 # how many generations you want 



system = """You are selfish."""

user =  """You are playing the ultimatum game and you are asked to to split $10 between yourself and another player. 
The other player will accept any offer. Make a proposal for how much the other player gets vs how much you get. 
Maximize your gains (ie you want to keep as much as possible for yourself)"""

prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS}{user} {E_INST}"

texts = [prompt] * batch_size



breakpoint()