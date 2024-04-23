import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
print("Working directory:", curr_dir)
sys.path.append(f'{curr_dir}/..')
sys.path.append(f'{curr_dir}/../..')
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path
from envs.unity_environment import UnityEnvironment
from agents import LLM_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP
import logging
from LLM import *

if __name__ == '__main__':
    args = get_args()
    mode = f"{args.mode}_env{args.test_task}_{args.agent_num}agents_{args.lm_id_list}_org{args.organization_instructions}" + ("_comm" if args.comm else "") + ("_random_start_comm" if args.random_start_comm else "")
    coordinator_log = open(f'{args.log_path}{mode}_coordinator.log', 'w')
    mode += "_iter0"
    llm_configs_dict = json.load(open('./llm_configs.json','r'))
    if args.lm_id_list is None and args.llm_config_list is None:
        raise NotImplementedError
    elif len(args.lm_id_list) == 1:
        args.lm_id_list = [args.lm_id_list[0]] * args.agent_num
    else:
        assert len(args.lm_id_list) == args.agent_num, "lm_id_list must be the same length as agent_num"
    
    if args.llm_config_list is None:
        args.llm_config_list = []
        for lm_id in args.lm_id_list:
            if lm_id == 'human':
                args.llm_config_list.append(None)
            else:
                llm_config = llm_configs_dict[lm_id]
                llm_config.update({'seed': args.seed})
                args.llm_config_list.append(llm_config)
    print("args.lm_id_list:", args.lm_id_list)
    
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    logging.basicConfig(format='%(asctime)s - %(name)s:\n %(message)s', level=logging.INFO)
    log_name = f'{args.log_path}{mode}.log'
    if os.path.exists(log_name):
        os.remove(log_name)
    File_handler = logging.FileHandler(log_name, encoding="utf-8")
    logger = logging.getLogger(__name__)
    logger.addHandler(File_handler)

    if args.organization_code is not None:
        df = pd.read_csv('../testing_agents/organization_instructions.csv')
        try:
            args.organization_instructions = df['instruction'][int(args.organization_code)]
        except:
            print("Invalid organization code")
            raise NotImplementedError

    args.record_dir = f'../test_results/{mode}'
    Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    if "image" in args.obs_type:
        os.system("Xvfb :98 & export DISPLAY=:98")
        import time
        time.sleep(3) # ensure Xvfb is open
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        executable_args = {
                        'file_name': args.executable_file,
                        'x_display': '98',
                        'no_graphics': False,
                        'timeout_wait': 5000,
        }
    else:
        executable_args = {
                        'file_name': args.executable_file,
                        'no_graphics': True,
                        'timeout_wait': 1000, 
        }

    id_run = 0
    random.seed(args.seed) #id_run
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    num_tries = 1
    num_runs = args.num_runs
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]


    def env_fn(env_id):
        return UnityEnvironment(num_agents=args.agent_num,
                               max_episode_length=args.max_episode_length,
                               port_id=env_id,
                               env_task_set=env_task_set,
                               agent_goals=['LLM' for i in range(args.agent_num)],
                               observation_types=[args.obs_type for i in range(args.agent_num)],
                               use_editor=args.use_editor,
                               executable_args=executable_args,
                               base_port=args.base_port if args.base_port is not None else np.random.randint(11000, 13000),
                               seed=args.seed,
                               )

    agents = [lambda x, y: None for i in range(args.agent_num)]
    
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, args.record_dir, args.debug, run_predefined_actions=False, comm=args.comm, args=args)

    # adopt GPT-4 as the critic and coordinator
    GPT_4_config = llm_configs_dict['gpt-4']
    GPT_4_config.update({'seed': args.seed})
    GPT_4 = LLM('gpt-4', GPT_4_config, None, args, 0, ['']) 

    coordinator_examples = []
    overall_weighted_cost = []
    init_num = 1

    coordinator_prompt_template = open("../LLM/coordinator_prompt_template.txt", 'r').read() 
    critic_prompt_template = open("../LLM/critic_prompt_template.txt", 'r').read()

    for iter_id in range(num_tries):
        steps_list, failed_tasks = [], []
        comm_tokens, comm_times = [], []
        if not os.path.isfile(args.record_dir + '/results.pik'):
            test_results = {}
        else:
            test_results = pickle.load(open(args.record_dir + '/results.pik', 'rb'))

        current_tried = iter_id
        for episode_id in episode_ids:
            for run_id in range(num_runs):
            
                # Run the episode
                is_finished = 0
                steps = 250
                arena.reset(episode_id, reset_seed=args.seed) #fixed seed for reproducibility
                success, steps, saved_info = arena.run()
                
                logger.info('-------------------------------------')
                logger.info('success' if success else 'failure')
                logger.info('steps: {}'.format(steps))
                arena.comm_cost_info['converse']['comm_tokens_per_step'] = round(arena.comm_cost_info['converse']['overall_tokens'] / steps, 2)
                arena.comm_cost_info['converse']['comm_times_per_step'] = round(arena.comm_cost_info['converse']['overall_times'] / steps, 2)
                logger.info('comm_cost_info: {}'.format(arena.comm_cost_info))
                logger.info('-------------------------------------')
                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                    comm_tokens.append(arena.comm_cost_info['converse']['overall_tokens'])
                    comm_times.append(arena.comm_cost_info['converse']['overall_times'])
                is_finished = 1 if success else 0
                log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'],
                                                                                    saved_info['task_name'],
                                                                                    current_tried)

                if len(saved_info['obs']) > 0:
                    pickle.dump(saved_info, open(log_file_name, 'wb'))
                else:
                    with open(log_file_name, 'w+') as f:
                        f.write(json.dumps(saved_info, indent=4))
                S[episode_id].append(is_finished)
                L[episode_id].append(steps)

                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                
                # critic evaluation
                if not args.no_critic:
                    log = open(f'{args.log_path}{mode}.log', 'r').read()
                    critic_prompt = critic_prompt_template.replace("$CRITIC_EXAMPLES$", log).replace("$ORGANIZATION_INSTRUCTIONS$", args.organization_instructions).replace('$GOAL$', arena.agents[0].LLM.goal_desc).replace("$STEP_RATIO$", str(args.step_ratio))
                    for _ in range(3):
                        try:
                            outputs, usage = GPT_4.generator([{"role": "user", "content": critic_prompt}] if GPT_4.chat else critic_prompt, GPT_4.sampling_params)
                            outputs_json = json.loads(outputs)
                            trajectory_history, agent_problems, agent_ranking, thoughts = outputs_json['summary'], outputs_json['problems'], outputs_json['leadership ranking'], outputs_json['thoughts']
                            break

                        except (JSONDecodeError, KeyError) as e:
                            print(outputs)
                            print(e)
                            GPT_4.llm_config.update({"seed": GPT_4.llm_config["seed"] + 123})
                            GPT_4.generator = GPT_4.lm_engine()
                            print("retrying...")

                    print("trajectory_history:", trajectory_history)
                    print("agent_problems:", agent_problems)
                    print("agent_ranking:", agent_ranking)
                    print("thoughts:", thoughts)

                    # add to coordinator log
                    coordinator_examples.append(f"\n# Instruction example {init_num+run_id}\nGoal: {arena.agents[0].LLM.goal_desc}\nOrganization structure instruction: {args.organization_instructions}\nOverall weighted cost: {args.step_ratio*steps+arena.comm_cost_info['converse']['comm_tokens_per_step']}\nComm costs: comm_tokens_per_step: {arena.comm_cost_info['converse']['comm_tokens_per_step']}, comm_times_per_step: {arena.comm_cost_info['converse']['comm_times_per_step']}\nSteps to finish the task: {steps}\nAgents' leadership ranking: {agent_ranking}\nProblems: {agent_problems}\nTrajectory History: {trajectory_history}\n")
                    coordinator_log.write(f"thoughts: {thoughts}\n")
                else:
                    coordinator_examples.append(f"\n# Instruction example {init_num+run_id}\nGoal: {arena.agents[0].LLM.goal_desc}\nOrganization structure instruction: {args.organization_instructions}\nOverall weighted cost: {args.step_ratio*steps+arena.comm_cost_info['converse']['comm_tokens_per_step']}\nComm costs: comm_tokens_per_step: {arena.comm_cost_info['converse']['comm_tokens_per_step']}, comm_times_per_step: {arena.comm_cost_info['converse']['comm_times_per_step']}\nSteps to finish the task: {steps}\n")

                coordinator_log.write(coordinator_examples[-1])
                overall_weighted_cost.append(args.step_ratio*steps+arena.comm_cost_info['converse']['comm_tokens_per_step'])

                # coordinator generates the new organization structure instruction
                sorted_pairs = sorted(zip(overall_weighted_cost, coordinator_examples), reverse=True)
                sorted_examples = [a for b, a in sorted_pairs]
                print("sorted_examples:", sorted_examples)
                coordinator_prompt = coordinator_prompt_template.replace("$COORDINATOR_EXAMPLES$", "\n".join(sorted_examples)).replace("$STEP_RATIO$", str(args.step_ratio))

                for _ in range(3):
                    try:
                        outputs, usage = GPT_4.generator([{"role": "user", "content": coordinator_prompt}] if GPT_4.chat else coordinator_prompt, GPT_4.sampling_params)
                        outputs_json = json.loads(outputs)
                        thoughts, args.organization_instructions, candidates = outputs_json['thoughts'], outputs_json['new organization structure instruction'], outputs_json['new organization structure instruction candidates']
                        break

                    except (JSONDecodeError, KeyError) as e:
                        print(outputs)
                        print(e)
                        GPT_4.llm_config.update({"seed": GPT_4.llm_config["seed"] + 123})
                        GPT_4.generator = GPT_4.lm_engine()
                        print("retrying...")

                print("thoughts:", thoughts)
                print("new_organization_instructions:", args.organization_instructions)
                coordinator_log.write(outputs)
                coordinator_log.flush()
                for agent_id in range(args.agent_num):
                    arena.agents[agent_id].LLM.organization_instructions = args.organization_instructions

                logger.removeHandler(File_handler)
                mode = mode.replace("_iter{}".format(run_id), "_iter{}".format(run_id+1))
                log_name = f'{args.log_path}{mode}.log'
                if os.path.exists(log_name):
                    os.remove(log_name)
                File_handler = logging.FileHandler(log_name, encoding="utf-8")
                logger.addHandler(File_handler)

        logger.info("mode: {}".format(mode))
        logger.info('average steps (finishing the tasks): {:.2f}'.format(np.array(steps_list).mean() if len(steps_list) > 0 else None))
        logger.info('average comm_tokens (finishing the tasks): {:.2f}'.format(np.array(comm_tokens).mean()))
        logger.info('average comm_times (finishing the tasks): {:.2f}'.format(np.array(comm_times).mean()))
        logger.info('comm_tokens per step: {:.2f}'.format(np.array(comm_tokens).mean() / np.array(steps_list).mean() if len(steps_list) > 0 else None))
        logger.info('comm_times per step: {:.2f}'.format(np.array(comm_times).mean() / np.array(steps_list).mean() if len(steps_list) > 0 else None))
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results.pik', 'wb'))
        coordinator_log.close()
        

