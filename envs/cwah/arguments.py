import argparse
import torch
import pdb
import yaml
from typing import Dict

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--dataset_path', default = './dataset/test_env_set_help.pik', type=str, help="The path of the environments where we test")
    parser.add_argument('--log_path', default = '../log/', type=str, help="The path of the logs")
    parser.add_argument('--mode', type=str, default='full', help='record folder name')

    parser.add_argument('--num-per-apartment', type=int, default=3, help='Maximum #episodes/apartment')
    parser.add_argument('--num-per-task', type=int, default=10, help='number of tests per task')
    parser.add_argument('--num_runs', type=int, default=5, help='number of tries')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument(
        '--obs_type',
        type=str,
        default='partial',
        choices=['full', 'rgb', 'visibleid', 'partial', 'full_image', 'normal_image'],
    )
    
    parser.add_argument('--test_task', default=(0, 5, 10, 16, 20, 26, 30, 32, 40, 49), type=int, nargs='+',
                        help='task ids to be tested')

    # Exec args
    parser.add_argument(
        '--executable_file', type=str,
        default='../executable/linux_exec.v2.3.0.x86_64')


    parser.add_argument(
        '--base-port', type=int, default=None)

    parser.add_argument(
        '--data-collection', type=str2bool, default = False
    )

    parser.add_argument(
        '--data-collection-dir', type=str, default = 'detection_images/'
    )

    parser.add_argument(
        '--display', type=str, default="2")

    parser.add_argument(
        '--max-episode-length', type=int, default=250)

    parser.add_argument(
        '--env-name',
        default='virtualhome')

    parser.add_argument(
        '--simulator-type',
        default='unity',
        choices=['unity', 'python'],
        help='whether to use unity or python sim')

    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 1)')


    parser.add_argument('--use-editor', action='store_true', default=False,
                        help='whether to use an editor or executable')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='debugging mode')

    parser.add_argument('--log_thoughts', action='store_true', default=False,
                        help='log the thoughts of the agents')
    
    parser.add_argument('--no_comm_fig', action='store_true', default=False,
                        help='not to save the communication figure')

    parser.add_argument('--num_steps_mcts', type=int, default=25,
                        help='how many steps to take of the given plan')

    parser.add_argument(
        '--gen_video',
        action='store_true',
        default=False,
        help="wheter to generate video")    

    # LLM parameters
    parser.add_argument('--lm_id_list', nargs='+', type=str, default='facebook/opt-13b',
                        help='name for openai engine or huggingface model name/path')
    parser.add_argument("--llm_config_list", nargs='+', default=None, type=dict, help="llm configs, provide here or in the file llm_configs.json")
    parser.add_argument('--prompt_template_path', default='LLM/prompt_nocom.csv',
                        help='path to prompt template file')
    parser.add_argument("--t", default=0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
    parser.add_argument("--agent_num", default=2, type=int)
    parser.add_argument("--config", default = None, type = str, help="config file")
    
    # comm args
    parser.add_argument("--comm", action='store_true', help="use autogen as a communication module")
    
    parser.add_argument("--random_start_comm", action='store_true', help="start the communication from a random agent")

    parser.add_argument("--organization_code", type=str, default='3', help="code for organization instructions, specified in organization_instructions.csv")

    parser.add_argument("--organization_instructions", type=str, default='', help="contents of organization instructions, provide here or in the file organization_instructions.csv")

    parser.add_argument("--action_history_len", default=20, type=int, help="length of action history")

    parser.add_argument("--dialogue_history_len", default=30, type=int, help="length of comm history")

    parser.add_argument("--step_ratio", default=1.0, type=float, help="the weight of step cost in the overall cost of step and comm")

    parser.add_argument("--no_critic", action='store_true', help="reflect without a critic")

    args = parser.parse_args()
    return args

ALL_MCTS_CONFIGS = {
    'hp_vision_agent_comm': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm.yaml",
    'hp_vision_agent_comm_belief': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm_belief.yaml",
    'hp_vision_agent_comm_subgoal': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm_subgoal.yaml",
    'hp_vision_agent_comm_satisfied': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm_satisfied.yaml",
    'hp_vision_agent_comm_full': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm_full.yaml",
    'hp_vision_agent_comm_all': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_comm_all.yaml",
    'hp_vision_agent': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent.yaml",
    'hp_vision_agent_full': "testing_agents/ablation_config_on_mcts_agent/hp_vision_agent_full.yaml",
    'single_vision_agent': "testing_agents/ablation_config_on_mcts_agent/single_vision_agent.yaml",
    'single_vision_agent_full': "testing_agents/ablation_config_on_mcts_agent/single_vision_agent_full.yaml",
}

def make_config(type_) -> Dict:
    with open(ALL_MCTS_CONFIGS[type_]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == '_main_':
    args = get_args()