Langchain, Langgraph , Ollama integration for running agentic workflows with local llm models (powered by Ollama)

1. agentic_ollama_toolcalling.py --> demonstrates tool calling using llama3.1:8b ollama model
2. agentic_ollama_plan.py --> demonstrates how llm model can plan for automl tasks
3. agentic_ollama_plan_exec.py --> demonstrates how agent plans and then executes tasks (calls tools after planning)
4. agentic_plain_langgraph.py --> demonstrates langgraph integration and runs the graph for automl tasks. The planner is rule based and has simple if/else logic for conditions.
5. agentic_ollama_langgraph.py --> demonstrates integration of ollama and langgraph where the llm model is the planner.