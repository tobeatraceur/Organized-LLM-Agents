kill -9 $(lsof -t -i :6314)
python ../testing_agents/criticize_reflect.py \
--dataset_path ../dataset/test_env_set_help.pik \
--prompt_template_path ../LLM/prompt_multi_comm.csv \
--mode test \
--executable_file ../../executable/linux_exec.v2.3.0.x86_64 \
--t 0.8 \
--lm_id_list gpt-4 \
--max_tokens 2048 \
--num_runs 10 \
--num-per-task 1 \
--agent_num 3 \
--test_task 1 \
--comm \
--no_comm_fig \
--step_ratio 1 \
--organization_instructions 0 
 
