# Agents

This folder stores the codes related to the **Agent**

We provide two types of agent:
1. agent with per-query budget (aka fixed budget, sequential agent, agent seq. etc. )
2. agent with shared-budget (aka batch agent, flexible budget etc. )

Following files are related:

`Agent/agent_core.py` is used for all settings. 

`Agent/run_agent_seq.py` is the code for the agent with per-query budget.

`Agent/run_agent_batch.py` is the code for the agent with shared-budget agent.

following scripts are the controlling shell files:

```
    Agent/script_command_batch_agent_llama.sh
    Agent/script_command_batch_agent_qwen.sh
    Agent/script_command_seq_llama.sh
    Agent/script_command_seq_qwen.sh
```

Examplary Usage: 

```
chmod +x script_command_batch_agent_llama.sh

script_command_batch_agent_llama.sh
```


# Post Processing and Evaluations:

`Agent/evaluate_agent_unified.py`, you can get generate a report for the result, including avg.cost, avg. passages, EM, F1 scores.

`Agent/tools_agent_conclusion_seq.py`, you can conclude per-case result into a table, including steps, tiers per case. This file is for per-query budget Agent. No need to run `evaluate_agent_unified.py` in advance.

`Agent/tools_agent_conclusion_batch.py`, you can conclude per-case result into a table, including steps, tiers per case, also decisions about budget allocations per case. This file is for shared-budget Agent. No need to run `evaluate_agent_unified.py` in advance.

`Agent/tools_merge_budget.py`, you can use this file to conclude results with different budget for ablation study. First need to run `evaluate_agent_unified.py`, then change `paths` in the file. 