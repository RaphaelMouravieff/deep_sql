
from tqdm import tqdm
from transformers import HfArgumentParser


from source.prompts.prompt import  PromptManager
from source.library.tables import TableManager
from source.utils.args import  (ModelArguments,
                               DataArguments,
                               TrainingArguments)

from source.bin.main_step import run_pipeline_step
from source.library.storage import  init_library, save_library
from source.agents.build_agents import create_agents
from source.tools.build_tools import create_tools
from source.models.model_setup import load_model



def generate_dataset(model, data_args, training_args, table_manager) -> None:

    library, vector_store = init_library(data_args, training_args)
    print(f"Starting with library containing {len(library)} entries")
    

    conn = table_manager.connect_to_database()
    prompt_manager = PromptManager(data_args, table_manager, library)

    tools = create_tools(conn, vector_store)
    agents = create_agents(model, training_args, tools)



    progress_bar = tqdm(range(training_args.num_iterations), desc="Generating dataset entries")
    for i in progress_bar:
        progress_bar.set_description(f"Entry {len(library) + 1}")
        entries = run_pipeline_step(prompt_manager, 
                                    agents,
                                    tools)
        
        if entries:
            for entry in entries:
                library.append(entry)

            print(f"Added entry #{len(library)} to library")
            save_library(data_args, library, vector_store)
            progress_bar.set_postfix(library_size=len(library))
        else:
            print("Failed to generate entry, continuing...")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_manager = TableManager(data_args)
    model = load_model(model_args)

    for table_id in table_manager.common_ids:
        table_manager.current_table_id = table_id
        generate_dataset(model, data_args, training_args, table_manager)


if __name__ == "__main__":
    main()


