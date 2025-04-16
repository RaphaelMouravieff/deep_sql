from smolagents import OpenAIServerModel, LiteLLMModel
import os

def load_model():
    # Initialize model based on available API keys
    if not os.environ.get("OPENAI_API_KEY"):
        #model = HfApiModel()
        model = LiteLLMModel(# qwen2.5:14b
            #deepseek-r1:14b /llama3.1:8b-instruct-fp16 ollama run qwq:32b-fp16
        model_id="ollama_chat/qwen2.5:14b", # This model is a bit weak for agentic behaviours though
        api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
        num_ctx=8192,device="mps" # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
        ,   )   
        
    else: 
        model = OpenAIServerModel("gpt-4o")
    return model

