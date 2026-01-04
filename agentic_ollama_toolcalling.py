from typing import List

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

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


llm = ChatOllama(
    model="llama3.1:8b",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([random_search, bayesian_optimization, model_pruning, early_stopping])


# single tool call test prompts
rs_prompt = "apply random search to improve model accuracy."
baye_prompt = "apply bayesian optimization to improve model accuracy."
m_prune_prompt = "apply model pruning to reduce latency."
e_stop_prompt = "apply early stopping to save budget."
prompt = "optimize a classification model under limited compute and latency constraints."
# multiple tools calls prompt
multitool_prompt = "call at least two tools to optimize a classification model under limited compute and latency constraints."

# replace prompts here to test different cases
result = llm.invoke(multitool_prompt)

if isinstance(result, AIMessage) and result.tool_calls:
    print(result.tool_calls)