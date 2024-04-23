import os
import pdb
import pickle
import random
import torch
import copy
import numpy as np
from tqdm import tqdm
import time
import ipdb
import ray
import json
import atexit
import sys
import pandas as pd
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}')
from comms import *
from collections import defaultdict
from agents import LLM_agent
import logging
logger = logging.getLogger("__main__")

# @ray.remote
class ArenaMP(object):
    def __init__(self, max_number_steps, arena_id, environment_fn, agent_fn, record_dir='out', debug=False, run_predefined_actions=False, comm=False, args=None):
        # run_predefined_actions is a parameter that you can use predefined_actions.json to strictly set the agents' actions instead of using algorithm to calculate the action.

        self.agents = []
        self.agent_names = ["Agent_{}".format(i+1) for i in range(len(agent_fn))]
        self.comm = comm
        self.env_fn = environment_fn
        self.agent_fn = agent_fn
        self.arena_id = arena_id
        self.num_agents = args.agent_num
        self.random_start_comm = args.random_start_comm
        self.log_thoughts = args.log_thoughts
        self.vis_comm = not args.no_comm_fig
        self.fig_path = f'{args.log_path}{args.mode}'
        self.organization_instructions = args.organization_instructions
        self.action_history_len = args.action_history_len
        self.dialogue_history_len = args.dialogue_history_len
        self.task_goal = None
        self.record_dir = record_dir
        self.debug = debug
        self.prompt_template_path = args.prompt_template_path

        print("Init Env")
        self.env = environment_fn(arena_id)
        self.converse_agents = []
        self.comm_cost_info = {
                    "converse": {"overall_tokens": 0, "overall_times": 0},
                    "select": {"tokens_in": 0, "tokens_out": 0}
                }
        for i in range(self.num_agents):
            self.agents.append(LLM_agent(agent_id=i+1, args=args))
            converse_agent = AssistantAgentCoT(
                self.agent_names[i],
                system_message="",
                llm_config=self.agents[i].LLM.llm_config,
                max_consecutive_auto_reply=1, # we have to manually set maximum conversation turn
                sampling_params=self.agents[i].LLM.sampling_params
            )
            self.converse_agents.append(converse_agent)
            self.comm_cost_info["converse"][self.agent_names[i]] = {"tokens_in": 0, "tokens_out": 0, "times_in": 0, "times_out": 0}
            for j in range(self.num_agents):
                if i != j:
                    self.comm_cost_info["converse"][f"{self.agent_names[i]} to {self.agent_names[j]}"] = {"tokens": 0, "times": 0}
        
        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.run_predefined_actions = run_predefined_actions
        atexit.register(self.close)

        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}

    def close(self):
        self.env.close()

    def get_port(self):
        return self.env.port_number

    def reset(self, task_id=None, reset_seed=None):
        self.cnt_duplicate_subgoal = 0
        self.cnt_nouse_subgoal = 0
        self.dict_info = {}
        self.dict_dialogue_history = defaultdict(list)
        self.LLM_returns = {}
        self.converse_agents = []
        self.comm_cost_info = {
                    "converse": {"overall_tokens": 0, "overall_times": 0},
                    "select": {"tokens_in": 0, "tokens_out": 0}
                }
        for i in range(self.num_agents):
            converse_agent = AssistantAgentCoT(
                self.agent_names[i],
                system_message="",
                llm_config=self.agents[i].LLM.llm_config,
                max_consecutive_auto_reply=1, # we have to manually set maximum conversation turn
                sampling_params=self.agents[i].LLM.sampling_params
            )
            self.converse_agents.append(converse_agent)
            self.comm_cost_info["converse"][self.agent_names[i]] = {"tokens_in": 0, "tokens_out": 0, "times_in": 0, "times_out": 0}
            for j in range(self.num_agents):
                if i != j:
                    self.comm_cost_info["converse"][f"{self.agent_names[i]} to {self.agent_names[j]}"] = {"tokens": 0, "times": 0}
        if self.run_predefined_actions:
            self.action_notes_steps = 0
            with open("predefined_actions.json","r", encoding='utf-8') as f:
                self.action_notes = json.load(f)
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id, reset_seed=reset_seed)

        for it, agent in enumerate(self.agents):
            if 'LLM_vision' in agent.agent_type:
                agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name, self.env.goal_spec[it])
            elif 'vision' in agent.agent_type:
                agent.reset(ob[it], self.env.full_graph, self.env.task_goal, self.env.all_room_name, self.env.all_containers_name, self.env.all_goal_objects_name, seed=agent.seed)
            elif 'MCTS' in agent.agent_type or 'Random' in agent.agent_type:
                agent.reset(ob[it], self.env.full_graph, self.env.task_goal, seed=agent.seed)
            elif 'LLM' in agent.agent_type:
                agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name, self.env.all_room_name, self.env.room_info, self.env.goal_spec[it])
            else:
                agent.reset(self.env.full_graph)

    def set_weigths(self, epsilon, weights):
        for agent in self.agents:
            if 'RL' in agent.agent_type:
                agent.epsilon = epsilon
                agent.actor_critic.load_state_dict(weights)

    def discussion(self, obs):
        df = pd.read_csv(self.prompt_template_path)
        prompt_format = df['prompt'][1]
        selector_prompts = []

        skip_flag = False
        for it, agent in enumerate(self.agents):
            if self.dict_dialogue_history[self.agent_names[it]] != []:
                if len(self.dict_dialogue_history[self.agent_names[it]]) > self.dialogue_history_len:
                    self.dict_dialogue_history[self.agent_names[it]] = self.dict_dialogue_history[self.agent_names[it]][-self.dialogue_history_len:]
                dialogue_history = [word for dialogue in self.dict_dialogue_history[self.agent_names[it]] for word in dialogue]
            else:
                dialogue_history = []

            if self.task_goal is None:
                goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
            else:
                goal_spec = self.env.get_goal(self.task_goal[it], self.env.agent_goals[it])
        
            goal_desc = self.agents[it].LLM.goal_desc
            action_history = self.agents[it].action_history
            action_history = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

            _ = self.agents[it].obs_processing(obs[it], goal_spec)
            progress = self.agents[it].progress2text()

            teammate_names = self.agent_names[:it] + self.agent_names[it+1:]
            selector_prompt = prompt_format.replace("$AGENT_NAME$", self.agent_names[it])
            selector_prompt = selector_prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
            selector_prompt = selector_prompt.replace("$TEAMMATE_NAME$", ", ".join(teammate_names))
            selector_prompt = selector_prompt.replace("$GOAL$", str(goal_desc))
            selector_prompt = selector_prompt.replace("$PROGRESS$", str(progress))
            selector_prompt = selector_prompt.replace("$ACTION_HISTORY$", str(action_history))
            selector_prompt = selector_prompt.replace("$DIALOGUE_HISTORY$", str('\n'.join(dialogue_history)))
            selector_prompts.append(selector_prompt)

        if not skip_flag:
            self.dict_dialogue_history, self.comm_cost_info = communicate(self.agent_names, \
                selector_prompts, self.dict_dialogue_history, self.comm_cost_info, self.agents, self.converse_agents, self.random_start_comm, self.log_thoughts, visualize=self.vis_comm, fig_folder_name=self.fig_path)

    def get_actions(self, obs, action_space=None, true_graph=False):
        dict_actions = {}
        if self.run_predefined_actions:
            act = self.action_notes[str(self.action_notes_steps)]
            self.action_notes_steps += 1
            split = act.find('|')
            actdict = {0:act[:split], 1:act[split+1:]}
            return actdict, {}

        if self.comm:
            logger.info('Communication at step {}'.format(self.env.steps))
            self.discussion(obs)

            if self.debug:
                logger.info('comm_cost_info: {}'.format(self.comm_cost_info))
            logger.info('------------------')

        logger.info('Actions at step {}'.format(self.env.steps))
        for it, agent in enumerate(self.agents):
            logger.info(f'Agent_{it+1}')
            if self.task_goal is None:
                goal_spec = self.env.get_goal(self.env.task_goal[it], self.env.agent_goals[it])
            else:
                goal_spec = self.env.get_goal(self.task_goal[it], self.env.agent_goals[it])
            
            if agent.agent_type in ['MCTS', 'Random', 'MCTS_vision']:
                teammate_subgoal = None
                if agent.recursive:
                    teammate_subgoal = self.agents[1 - it].last_subgoal
                dict_actions[it], _ = agent.get_action(obs[it], goal_spec, teammate_subgoal)
                
            elif 'RL' in agent.agent_type:
                if 'MCTS' in agent.agent_type or 'Random' in agent.agent_type:
                    if true_graph:
                        full_graph = self.env.get_graph()
                    else:
                        full_graph = None
                    dict_actions[it], _ = agent.get_action(obs[it], goal_spec,
                                                                       action_space_ids=action_space[it], full_graph=full_graph)

                else:
                    dict_actions[it], _ = agent.get_action(obs[it], self.task_goal, action_space_ids=action_space[it])

            elif 'LLM' in agent.agent_type:
                obs = self.env.get_observations()
                dict_actions[it], self.dict_info[it] = agent.get_action(obs[it], goal_spec, dialogue_history=self.dict_dialogue_history[self.agent_names[it]])

        return dict_actions, self.dict_info

    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

    def rollout_reset(self, logging=False, record=False, episode_id=None, is_train=True, goals=None):
        try:
            res = self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)
            return res
        except:
            self.env.close()
            self.env = self.env_fn(self.arena_id)

            for agent in self.agents:
                if 'RL' in agent.agent_type:
                    prev_eps = agent.epsilon
                    prev_weights = agent.actor_critic.state_dict()

            self.agents = []
            for i in range(self.num_agents):
                self.agents.append(LLM_agent(agent_id=i+1, args=args))

            self.set_weigths(prev_eps, prev_weights)
            return self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)

    def rollout(self, logging=0, record=False, episode_id=None, is_train=True, goals=None):
        t1 = time.time()
        print("rollout", episode_id, is_train)
        if episode_id is not None:
            self.reset(episode_id)
        else:
            self.reset()

        t2 = time.time()
        t_reset = t2 - t1
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        agent_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []

        if goals is not None:
            self.task_goal = goals
        else:
            self.task_goal = None

        if logging > 0:
            info_rollout['pred_goal'] = []
            info_rollout['pred_close'] = []
            info_rollout['gt_goal'] = []
            info_rollout['gt_close'] = []
            info_rollout['mask_nodes'] = []
        init_dict = {i: [] for i in range(self.num_agents)}
        if logging > 1:
            info_rollout['step_info'] = []
            info_rollout['action'] = init_dict
            info_rollout['script'] = []
            info_rollout['graph'] = []
            info_rollout['action_space_ids'] = []
            info_rollout['visible_ids'] = []
            info_rollout['action_tried'] = []
            info_rollout['predicate'] = []
            info_rollout['reward'] = []
            info_rollout['goals_finished'] = []
            info_rollout['obs'] = []

        rollout_agent = {}

        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            if 'RL' in agent.agent_type:
                rollout_agent[agent_id] = []

        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec[0]
            goal_class = [elem_name.split('_')[1] for elem_name in list(pred.keys())]
            id2node = {node['id']: node for node in init_graph['nodes']}
            info_goals = []
            info_goals.append([node for node in init_graph['nodes'] if node['class_name'] in goal_class])
            ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] in goal_class]
            info_goals.append([(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type'],
                                edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
            info_rollout['target'] = [pred, info_goals]

        agent_id = [id for id, enum_agent in enumerate(self.agents) if 'RL' in enum_agent.agent_type][0]
        reward_step = 0
        prev_reward_step = 0
        curr_num_steps = 0
        prev_reward = 0
        init_step_agent_info = {}
        local_rollout_actions = []
        if not is_train:
            pbar = tqdm(total=self.max_episode_length)
        while not done and nb_steps < self.max_episode_length and agent_steps < self.max_number_steps:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step(true_graph=is_train)
            step_failed = env_info['failed_exec']
            if step_failed:
                print("FAILING in task")
                print(agent_actions)
                print(local_rollout_actions)
                print('----')
            local_rollout_actions.append(agent_actions[0])
            if not is_train:
                pbar.update(1)
            if logging:
                curr_graph = env_info['graph']
                agentindex = self.agents[agent_id].agent_id
                observed_nodes = agent_info[agent_id]['visible_ids']
                node_id = [node['bounding_box'] for node in obs[agent_id]['nodes'] if node['id'] == agentindex][0]
                edges_char = [(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type']) for edge in curr_graph['edges'] if edge['from_id'] == agentindex and edge['to_id'] in observed_nodes]

                if logging > 0:
                    if 'pred_goal' in agent_info[agent_id].keys():
                        info_rollout['pred_goal'].append(agent_info[agent_id]['pred_goal'])
                        info_rollout['pred_close'].append(agent_info[agent_id]['pred_close'])
                        info_rollout['gt_goal'].append(agent_info[agent_id]['gt_goal'])
                        info_rollout['gt_close'].append(agent_info[agent_id]['gt_close'])
                        info_rollout['mask_nodes'].append(agent_info[agent_id]['mask_nodes'])

                if logging > 1:
                    info_rollout['step_info'].append((node_id, edges_char))
                    info_rollout['script'].append(agent_actions[agent_id])
                    info_rollout['goals_finished'].append(env_info['satisfied_goals'])
                    info_rollout['finished'] = env_info['finished']

                    for agenti in range(len(self.agents)):
                        info_rollout['action'][agenti].append(agent_actions[agenti])
                        info_rollout['obs'].append(agent_info[agenti]['obs'])

                    info_rollout['action_tried'].append(agent_info[agent_id]['action_tried'])
                    if 'predicate' in agent_info[agent_id]:
                        info_rollout['predicate'].append(agent_info[agent_id]['predicate'])
                    info_rollout['graph'].append(curr_graph)
                    info_rollout['action_space_ids'].append(agent_info[agent_id]['action_space_ids'])
                    info_rollout['visible_ids'].append(agent_info[agent_id]['visible_ids'])
                    info_rollout['reward'].append(reward)

            nb_steps += 1
            curr_num_steps += 1
            diff_reward = reward - prev_reward
            prev_reward = reward
            reward_step += diff_reward
            if 'bad_predicate' in agent_info[agent_id]:
                reward_step -= 0.2

            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += diff_reward
            
            if record:
                actions.append(agent_actions)

            # append to memory
            if is_train:
                for agent_id in range(self.num_agents):
                    if 'RL' == self.agents[agent_id].agent_type or \
                            self.agents[agent_id].agent_type == 'RL_MCTS' and 'mcts_action' not in agent_info[agent_id]:
                        init_step_agent_info[agent_id] = agent_info[agent_id]

                    # If this is the end of the action
                    if 'RL' == self.agents[agent_id].agent_type or \
                        self.agents[agent_id].agent_type == 'RL_MCTS' and self.agents[agent_id].action_count == 0:
                        agent_steps += 1
                        state = init_step_agent_info[agent_id]['state_inputs']
                        policy = [log_prob.data for log_prob in init_step_agent_info[agent_id]['probs']]
                        action = agent_info[agent_id]['actions']
                        rewards = reward_step
                        for i in range(self.num_agents):
                            entropy_action.append(
                                -((init_step_agent_info[agent_id]['probs'][i] + 1e-9).log() * init_step_agent_info[agent_id]['probs'][i]).sum().item())

                        observation_space.append(init_step_agent_info[agent_id]['num_objects'])
                        action_space.append(init_step_agent_info[agent_id]['num_objects_action'])
                        last_agent_info = init_step_agent_info

                        rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action,
                                                        rewards, curr_num_steps, 1))
                        prev_reward_step = 0
                        reward_step = 0
                        curr_num_steps = 0

        if not is_train:
            pbar.close()
        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[agent_id].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)
        info_rollout['t_reset'] = t_reset
        info_rollout['t_steps'] = t_steps

        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']


        info_rollout['env_id'] = self.env.env_id
        info_rollout['goals'] = list(self.env.task_goal[0].keys())

        # Rollout max
        if is_train:
            while nb_steps < self.max_number_steps:
                nb_steps += 1
                for agent_id in range(self.num_agents):
                    if 'RL' in self.agents[agent_id].agent_type:
                        state = last_agent_info[agent_id]['state_inputs']
                        if 'edges' in obs.keys():
                            pdb.set_trace()
                        policy = [log_prob.data for log_prob in last_agent_info[agent_id]['probs']]
                        action = last_agent_info[agent_id]['actions']
                        # rewards = reward
                        rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action, 0, 0, 0))

        return c_r_all, info_rollout, rollout_agent


    def step(self, true_graph=False):
        if self.env.steps == 0:
            pass
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space, true_graph=true_graph)
        for i in range(len(dict_info)):
            if len(dict_info) > 1 and 'subgoals' in dict_info[i]:
                dup = self.env.check_subgoal(dict_info[i]['subgoals'])
                self.cnt_nouse_subgoal += dup
                if i == 0 and 'subgoals' in dict_info[i + 1].keys() and dict_info[i]['subgoals'] == dict_info[i + 1]['subgoals']:
                    self.cnt_duplicate_subgoal += 1
        try:
            step_info = self.env.step(dict_actions)
        except Exception as e:
            print("Exception occurs when performing action: ", dict_actions)
            raise Exception
        return step_info, dict_actions, dict_info

    def run(self, random_goal=False, pred_goal=None, cnt_subgoal_info = False):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """
        self.task_goal = copy.deepcopy(self.env.task_goal)
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                for i in range(self.num_agents):
                    self.task_goal[i][predicate] = u
 
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        init_dict = {i: [] for i in range(self.num_agents)}
        self.saved_info = {'task_id': self.env.task_id,
                      'env_id': self.env.env_id,
                      'task_name': self.env.task_name,
                      'gt_goals': self.env.task_goal[0],
                      'goals': self.task_goal,
                      'action': init_dict,
                      'plan': init_dict,
                      'subgoals': init_dict,
                      'finished': None,
                      'init_unity_graph': self.env.init_graph,
                      'goals_finished': [],
                      'belief': init_dict,
                      'belief_graph': init_dict,
                      'obs': init_dict,
                      'LLM': init_dict,
                      'graph': init_dict,
                      'progress': [],
                    }
        success = False
        while True:
            (obs, reward, done, infos, messages), actions, agent_info = self.step()
            success = infos['finished']
            # if infos['failed_exec']:
            #     raise ValueError(infos)
            if 'satisfied_goals' in infos:
                self.saved_info['goals_finished'].append(infos['satisfied_goals'])
            for agent_id, action in actions.items():
                self.saved_info['action'][agent_id].append(action)
            
            if 'progress' in infos:
                self.saved_info['progress'].append(infos['progress'])
            for agent_id, info in agent_info.items():
                if 'belief_graph' in info:
                    self.saved_info['belief_graph'][agent_id].append(info['belief_graph'])
                if 'belief' in info:
                    self.saved_info['belief'][agent_id].append(info['belief'])
                if 'plan' in info:
                    self.saved_info['plan'][agent_id].append(info['plan'])
                if 'subgoals' in info:
                    self.saved_info['subgoals'][agent_id].append(info['subgoals'])
                if 'obs' in info:
                    self.saved_info['obs'][agent_id].append(copy.deepcopy(info['obs']))
                if 'LLM' in info:
                    self.saved_info['LLM'][agent_id].append(info['LLM'])
                if 'graph' in info:
                    self.saved_info['graph'][agent_id].append(copy.deepcopy(info['graph']))
                if self.debug:
                    pickle.dump(self.saved_info, open(os.path.join(self.record_dir, 'log.pik'), 'wb'))
            if done:
                break
        self.saved_info['finished'] = success
        if cnt_subgoal_info:
            self.saved_info['cnt_duplicate_subgoal'] = self.cnt_duplicate_subgoal
            self.saved_info['cnt_nouse_subgoal'] = self.cnt_nouse_subgoal
            return success, self.env.steps, self.saved_info
        else:
            return success, self.env.steps, self.saved_info
