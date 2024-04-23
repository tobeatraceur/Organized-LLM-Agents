kill -9 $(lsof -t -i :6314)
python ../testing_agents/test_symbolic_multi-LLMs.py \
--dataset_path ../dataset/test_env_set_help.pik \
--prompt_template_path ../LLM/prompt_multi_comm.csv \
--mode test \
--executable_file ../../executable/linux_exec.v2.3.0.x86_64 \
--t 0.8 \
--lm_id_list gpt-4 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 1 \
--agent_num 3 \
--test_task 1 \
--comm \
--log_thoughts \
--organization_code 3 

