


import torch

from source.models.ft_model_setup import load_config, load_tokenizer, load_model
from source.utils.logger import setup_logger
from source.tools.answer_check import AnswerChecker
import pandas as pd

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
question = "how many movies does Leonardo Di Caprio have?"
expected_answer = ["53"]


class SimpleArgs:
    max_source_length: int = 512
    max_target_length: int = 128
    pad_to_max_length: bool = True
    num_beams: int = 5
    use_model_check: bool = True   
    model_name_or_path: str = "models/bart_large_debug/checkpoint-1074"
    tokenizer_name: str = None
    config_name = None
    use_fast_tokenizer = True
    output_generation: bool = True
    
device = "cuda" if torch.cuda.is_available() else "cpu"



logger = setup_logger()

config = load_config(SimpleArgs, logger)

tokenizer = load_tokenizer(SimpleArgs, logger)

model = load_model(SimpleArgs, config, logger)

model = model.to("cuda")


# Run the testq
if __name__ == "__main__":
    checker = AnswerChecker(model=model, tokenizer=tokenizer, data_args=SimpleArgs(), device=device)
    model_answer, log_likelihood = checker.check_answer(question, table, expected_answer)

    print('Model answer:', model_answer)
    print('Expected answer:', expected_answer)  
    print('Log likelihood:', log_likelihood)