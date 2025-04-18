
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    

    output_dir: Optional[str] = field(
            default=None,
            metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        )

    ollama_model_name_or_path: Optional[str] = field(
        default="llama3.2",
        metadata={"help": "The model checkpoint for the curriculum learning agent."},
    )

    sentence_model_name_or_path: Optional[str] = field(
        default="Alibaba-NLP/gte-large-en-v1.5",#"paraphrase-MiniLM-L6-v2",
        metadata={"help": "The sentence transformer model name or path."},
    )

    max_source_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    database_path: Optional[str] = field(
        default='../data/tables/db',
        metadata={"help": "The path to the SQLite database file."},
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    squall_path: Optional[str] = field(
        default='../data/squall.json',
        metadata={"help": "The path to the SQLite database file."},
    )

    wikitablequestions_path: Optional[str] = field(
        default='wikitablequestions',
        metadata={"help": "The path to the SQLite database file."},
    )

    library_path: Optional[str] = field(
        default="../data/sql_dataset_library.json",
        metadata={"help": "The path to the SQL dataset library."},
    )

    vector_store_path: Optional[str] = field(
        default="../data/vector_store",
        metadata={"help": "The path to the vector store."},
    )

    table_limit: Optional[int] = field(
        default=10,
        metadata={"help": "The maximum number of tables to sample from the database."},
    )

    base_prompt_path: Optional[str] = field(
        default="../data/prompts/base_prompt.yaml",
        metadata={"help": "The path to the base prompt YAML file."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    num_iterations: Optional[int] = field(
        default=1,
        metadata={"help": "The number of iterations for the exploration loop."},
    )

    iterative_prompting: Optional[bool] = field(
        default=False,
        metadata={"help": "Use iterative prompting."},
    )

    embedding_model_name: Optional[str] = field(
        default="Alibaba-NLP/gte-large-en-v1.5",
        metadata={"help": "The embedding model name."},
    )

    max_agent_steps: Optional[int] = field(
        default=11,
        metadata={"help": "The maximum number of steps for the agent."},
    )


    chunk: Optional[int] = field(
        default=None,
        metadata={"help": "The chunk number for the dataset."},
    )

    Nchunks: Optional[int] = field(
        default=None,
        metadata={"help": "The number of chunks for the dataset."},
    )
 

def __post_init__(self):

    if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if self.val_max_target_length is None:
        self.val_max_target_length = self.max_target_length
