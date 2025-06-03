from smolagents import CodeAgent, LiteLLMModel

# Tell LiteLLM to use the Ollama backend
model = LiteLLMModel(
    model_id='ollama_chat/llama3.2'  # the correct prefix for Ollama models
)

agent = CodeAgent(
    tools=[],  # You can add tools like DuckDuckGoSearchTool if needed
    model=model
)

response = agent.run("Write a SQL query to find all employees who joined after 2020.")
print(response)