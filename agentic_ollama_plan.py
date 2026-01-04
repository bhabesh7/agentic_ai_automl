from langchain_ollama import ChatOllama
from typing import List

from langchain_ollama.chat_models import AIMessage
from langchain.tools import tool
# from langchain_ollama import ChatOllama

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming and I will do anything for it."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)

@tool
def random_search():
    """Perform a random search for AutoML hyperparameters."""
    return "Random search completed → accuracy: 0.72"

@tool
def bayesian_optimization():
    """Perform Bayesian optimization for AutoML hyperparameters."""
    return "Bayesian optimization completed → accuracy: 0.84"

@tool
def model_pruning():
    """Perform model pruning to reduce latency."""
    return "Model pruned → latency reduced by 30%"

@tool
def early_stopping():
    """Implement early stopping to save budget."""
    return "Early stopping triggered → budget saved"

#TODO ollama run llama3.1:8b -- this supports tool calls 
llm = ChatOllama(
    model="llama3.1:8b",# replace llama3.1:8b here for tool calling after ollama support (instead of phi3:3.8b)
    temperature=0.1,
    # other params...
).bind_tools([random_search, bayesian_optimization, model_pruning, early_stopping])

planning_prompt = """
You are an AutoML agent.

Goal:
Optimize a classification model under limited compute and latency constraints.

Instructions:
1. Plan the optimization steps.
2. Decide which AutoML tools to use.
3. Adapt decisions based on past results.
4. Stop early if further improvements are unlikely.

Explain your reasoning briefly before acting.
"""
result = llm.invoke(
    planning_prompt
)

print("\n=== Agentic AutoML Plan ===\n")
print(result.content)

# Optional: show tool calls detected by the agent
if isinstance(result, AIMessage) and result.tool_calls:
    print("\n=== Tool Calls Detected ===\n")
    for call in result.tool_calls:
        print(call)