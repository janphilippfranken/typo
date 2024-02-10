from typing import List, Optional, Tuple
import re

from datasets import Dataset

import matplotlib.pyplot as plt
import numpy as np

BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer
