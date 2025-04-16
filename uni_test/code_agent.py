

from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:14b"  # the correct prefix for Ollama models
)


agent = CodeAgent(model=model, tools=[])
print(agent.prompt_templates)
agent.run("What is the result of 2 power 3.7384?")