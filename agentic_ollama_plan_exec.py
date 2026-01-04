
from langchain_ollama import ChatOllama
from typing import Any, List, Optional

from langchain.tools import tool

# LangChain v1.x uses `create_agent`; older helper `initialize_agent` may not exist.
try:
    from langchain.agents import initialize_agent, AgentType
    _HAS_INITIALIZE = True
except Exception:
    from langchain.agents import create_agent
    _HAS_INITIALIZE = False

print(f"HasInitialize: {_HAS_INITIALIZE}")

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

# NOTE: Some Chat models and wrappers expose a .bind_tools API, but
# the most compatible way across LangChain versions is to create
# Tool objects and run an agent with initialize_agent. This lets the
# agent decide which tools to call and will actually execute them.
llm = ChatOllama(
    model="llama3.1:8b",  # replace as appropriate for your local ollama model
    temperature=0.1,
    # other params...
)

response_output = " "

tools = [random_search, bayesian_optimization, model_pruning, early_stopping]

# Create an agent that can call the tools. Use the available helper depending
# on the installed LangChain version.
if _HAS_INITIALIZE:
    # older helper available (some LangChain wrappers)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
else:
    # LangChain 1.x: create_agent returns a compiled state graph; we pass the
    # planning prompt as the user message when running the graph.
    print("Using create_agent for LangChain 1.x")
    agent = create_agent(model=llm, tools=tools, system_prompt="You are an AutoML agent.")

planning_prompt = """
You are an AutoML agent.

Goal:
Optimize a classification model under limited compute and latency constraints.

Instructions:
1. Plan the optimization steps.
2. Decide which AutoML tools to use.
3. Adapt decisions based on past results.
4. Stop early if further improvements are unlikely.

"""
if _HAS_INITIALIZE:
    result = agent.run(planning_prompt)
    print("\n=== Agentic AutoML Plan / Result ===\n")
    print(result)
else:
    # create_agent expects a messages input dict; use the planning prompt as the
    # user message. The compiled graph supports .run or .stream; we call .run
    # to get the final output.
    inputs = {"messages": [{"role": "user", "content": planning_prompt}]}
    try:
        # Some compiled graphs support .run returning final messages.
        out = agent.run(inputs)
        print("\n=== Agentic AutoML Plan / Result ===\n")
        print(out)
    except Exception:
        # Fallback to streaming if .run is not available
        print("Streaming updates:")
        for chunk in agent.stream(inputs, stream_mode="updates"):
            print(chunk)
         



# --- Post-processing: detect tool-call JSON in the model response and execute matching tools ---
# This code is intentionally appended and does not modify the existing logic above.
# def _extract_json_objects_from_text(text: str):
#     """Find top-level JSON objects in text by locating balanced braces starting
#     at each '{' and attempting to json.loads them. Returns list of parsed objects."""
#     import json

#     objs = []
#     start = 0
#     while True:
#         idx = text.find('{', start)
#         if idx == -1:
#             break
#         depth = 0
#         i = idx
#         while i < len(text):
#             if text[i] == '{':
#                 depth += 1
#             elif text[i] == '}':
#                 depth -= 1
#                 if depth == 0:
#                     candidate = text[idx:i + 1]
#                     try:
#                         obj = json.loads(candidate)
#                         objs.append(obj)
#                         start = i + 1
#                         break
#                     except Exception:
#                         # not valid JSON, continue scanning after this brace
#                         start = idx + 1
#                         break
#             i += 1
#         else:
#             break
#     return objs


# def _execute_tool_call(call: dict):
#     """Given a dict with at least 'name' and optional 'parameters', find the
#     corresponding tool from `tools` and execute it. Prints the result."""
#     name = call.get('name') or call.get('tool')
#     params = call.get('parameters') or call.get('args') or {}
#     if name is None:
#         print('No tool name found in call:', call)
#         return

#     # locate tool by Tool.name or by function name
#     tool_obj = None
#     for t in tools:
#         tname = getattr(t, 'name', None) or getattr(t, '__name__', None)
#         if tname == name:
#             tool_obj = t
#             break

#     if tool_obj is None:
#         print(f"Tool '{name}' not found among available tools: {[getattr(t,'name',None) for t in tools]}")
#         return

#     try:
#         # Preferred: StructuredTool.run expects a dict
#         if hasattr(tool_obj, 'run'):
#             res = tool_obj.run(params or {})
#         elif hasattr(tool_obj, 'func'):
#             # try calling with kwargs, fallback to no-arg
#             try:
#                 res = tool_obj.func(**(params or {}))
#             except TypeError:
#                 if params:
#                     # single positional arg
#                     res = tool_obj.func(params)
#                 else:
#                     res = tool_obj.func()
#         else:
#             # plain callable
#             if params:
#                 res = tool_obj(**params)
#             else:
#                 res = tool_obj()
#     except Exception as e:
#         res = f"Error executing tool {name}: {e!r}"

#     print(f"\n=== Executed tool {name} -> result:\n{res}\n===")


# # Aggregate candidate texts from the variables that may hold model output
# candidate_texts = []
# if 'result' in globals():
#     # initialize_agent path typically sets `result` to a string
#     try:
#         if isinstance(result, str):
#             candidate_texts.append(result)
#         else:
#             candidate_texts.append(str(result))
#     except Exception:
#         pass

# if 'out' in globals():
#     try:
#         if isinstance(out, dict) and 'messages' in out:
#             for m in out['messages']:
#                 # messages may be AIMessage objects or dicts
#                 c = getattr(m, 'content', None) or (m.get('content') if isinstance(m, dict) else None)
#                 if c:
#                     candidate_texts.append(c)
#         else:
#             candidate_texts.append(str(out))
#     except Exception:
#         candidate_texts.append(str(out))

# # If streaming branch printed chunks but didn't set `out`, try to ask the agent
# # to emit a machine-readable tool call. We avoid re-calling the model here.
# candidate_texts = response_output
# if not candidate_texts:
#     print('No model output captured to inspect for tool calls.')
# else:
#     import json

#     parsed = []
#     for t in candidate_texts:
#         parsed.extend(_extract_json_objects_from_text(t))

#     # Filter parsed objects that look like tool calls
#     tool_calls = [p for p in parsed if isinstance(p, dict) and ('name' in p and (isinstance(p.get('parameters', {}), dict) or 'parameters' in p))]

#     if not tool_calls:
#         print('No tool-call JSON detected in model output. If you want the model to call tools, prompt it explicitly, e.g. "Call the tool random_search and return the JSON: {\"name\": \"random_search\", \"parameters\": {}}"')
#     else:
#         print(f"Detected {len(tool_calls)} tool call(s) in model output. Executing...")
#         for call in tool_calls:
#             _execute_tool_call(call)