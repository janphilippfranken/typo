import hydra
from omegaconf import DictConfig

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset, Dataset

from scaituning.models.hf_inference_model import HFInferenceModel

from scaituning.prompts.cai.cai_prompts import CAI_PROMPTS, EXAMPLE_FORMATS, create_shuffled_train_batch


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM = "Respond to the best of your ability."


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    # load model 
    model = HFInferenceModel(**args.model.hf_config)


    # load and resize dataset
    dataset = load_dataset(**args.data.dataset)
    n_train, n_test = int(args.data.size.n_train * len(dataset["train"])), int(args.data.size.n_test * len(dataset["test"]))
    train_dataset = dataset["train"].select(range(n_train))
    test_dataset = dataset["test"].select(range(n_test))


    # Generate principles
    predicted_principles = []

    # Calculate how many batches are needed
    n_calls = args.generation.n_principles // args.generation.batch_prompt_size

    for batch_num in tqdm(range(n_calls)):
        prompts = []
        # Generate a train_batch with conversation_batch_size conversations
        for _ in range(args.generation.batch_prompt_size):
            train_batch = create_shuffled_train_batch(
                train_prompt=CAI_PROMPTS[args.prompts.train_prompt],
                chosen_conversations=train_dataset["chosen"],
                rejected_conversations=train_dataset["rejected"],
                example_formats=EXAMPLE_FORMATS,
                batch_size=args.prompts.conversation_batch_size
            )
            formatted_prompt = f"<s>{B_INST} {B_SYS}{SYSTEM}{E_SYS}{train_batch} {E_INST}"
            prompts.append(formatted_prompt)
        # Call the model for the batch
        responses = model.batch_prompt(prompts, **args.model.inference_config)

        # Store the responses
        for idx, response in enumerate(responses):
            predicted_principles.append({"prompt": prompts[idx], "response": response})

    # Save the responses
    predicted_principles_dataset = Dataset.from_pandas(pd.DataFrame(predicted_principles))
    # write to json
    predicted_principles_dataset.to_json("mistral_2.json")


    # data = load_dataset("json", data_files="test.json")




if __name__ == '__main__':
    main()