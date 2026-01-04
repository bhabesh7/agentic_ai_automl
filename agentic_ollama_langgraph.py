# Plain workflow with langgraph for AutoML optimization (no llm model used here)
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
import re

class AutoMLState(TypedDict):
    accuracy: float
    latency: float
    budget: int
    last_step: str

print("Setting up AutoML agentic graph...")

# Initialize local Ollama LLM (uses your local model pulled to ollama)
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

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

#define langgraph state graph
def run_random_search(state: AutoMLState):
    # `random_search` is a StructuredTool (from @tool); call its runtime API
    # Use .run({}) for tools that accept no parameters.
    print(random_search.run({}))
    return {"accuracy": 0.72, "last_step": "random_search"}

def run_bayesian_opt(state: AutoMLState):
    print(bayesian_optimization.run({}))
    return {"accuracy": 0.84, "last_step": "bayesian_optimization"}

def run_pruning(state: AutoMLState):
    print(model_pruning.run({}))
    return {"latency": 0.7, "last_step": "model_pruning"}

def run_stop_early(state: AutoMLState):
    print(early_stopping.run({}))
    return state

# decision logic (planner)
def planner(state: AutoMLState):
    # First try to ask the local LLM for a decision. If something goes wrong
    # (model unavailable or response can't be parsed), fall back to the simple
    # heuristic below.
    try:
        print("Asking LLM for planning decision...")
        prompt = (
            f"You are an AutoML planner. Given the current state: accuracy={state['accuracy']}, "
            f"latency={state['latency']}, budget={state['budget']}. "
            "Return a single word choice for the next action: 'bayesian', 'pruning', or 'stop'."
        )
        resp = llm.invoke(prompt)
        # resp may be an AIMessage-like object with a `.content` attribute
        content = getattr(resp, 'content', None) or (resp[0].content if isinstance(resp, (list, tuple)) and resp else str(resp))
        content = str(content).lower()
        # pick the first matching keyword
        for kw in ('bayesian', 'pruning', 'stop'):
            if re.search(rf"\b{kw}\b", content):
                return kw
    except Exception:
        # ignore and fallback
        print("LLM planning failed, falling back to heuristic.")
        pass

    # Fallback heuristic
    if state["accuracy"] < 0.8:
        return "bayesian"
    if state["latency"] > 0.8:
        return "pruning"
    if state["budget"] <= 0:
        return "stop"
    return "stop"



#build graph
graph = StateGraph(AutoMLState)

graph.add_node("random", run_random_search)
graph.add_node("bayesian", run_bayesian_opt)
graph.add_node("pruning", run_pruning)
graph.add_node("stop", run_stop_early)

graph.set_entry_point("random")

graph.add_conditional_edges(
    "random",
    planner,
    {
        "bayesian": "bayesian",
        "pruning": "pruning",
        "stop": END
    }
)

graph.add_edge("bayesian", "pruning")
graph.add_edge("pruning", END)

automl_graph = graph.compile()

#Run the graph/agent
initial_state = {
    "accuracy": 0.0,
    "latency": 1.0,
    "budget": 1,
    "last_step": ""
}

automl_graph.invoke(initial_state)


