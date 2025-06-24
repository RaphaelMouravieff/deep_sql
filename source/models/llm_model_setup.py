from smolagents import OpenAIServerModel, LiteLLMModel
import os
import logging


logging.getLogger("litellm").setLevel(logging.WARNING)


def load_model_llm(model_args):

    model = LiteLLMModel(
    model_id=model_args.ollama_model_name_or_path, 
    num_ctx=model_args.max_source_length_llm)  
        
    return model



