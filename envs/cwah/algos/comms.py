from autogen import AssistantAgent
from autogen import oai
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
import copy
from collections import defaultdict
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import os
import pickle
import random
import sys
import tiktoken
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'{curr_dir}')

from LLM.llm_wrapper import oai_wrapper
from agents.LLM_agent import LLM_agent
import logging

logger = logging.getLogger("__main__")
round = 1
edge_weights_time, edge_weights_token = defaultdict(int), defaultdict(int) # key: a,b stands for a->b, value: int
now = datetime.now()

class Profiler(object):
    """A profiler used to moniter the token counts."""
    _tokens_in: int = 0
    _tokens_out: int = 0
    _times_in: int = 0
    _times_out: int = 0
    
    def __init__(self, tokens_in=0, tokens_out=0, times_in=0, times_out=0) -> None:
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._times_in = times_in
        self._times_out = times_out
    
    def update_tokens(self, tokens_in=0, tokens_out=0):
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
    
    def report(self):
        return self._tokens_in, self._tokens_out, self._times_in, self._times_out

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class AssistantAgentCoT(AssistantAgent):
    """An adapted version of AssistantAgent. The received message should be a json containing 'message' and 'thought' field. The 'message' field is extracted as response."""
    
    def __init__(
        self,
        name: str,
        system_message: Optional[str] = AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, bool]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            system_message,
            llm_config,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        self.sampling_params = sampling_params
        self.profiler = Profiler()
        self.register_reply(Agent, AssistantAgentCoT.generate_oai_reply_profile, position=1)
    
    def generate_oai_reply_profile(
        self, 
        messages: Optional[List[Dict]] = None, 
        sender: Optional[Agent] = None, 
        config: Optional[Any] = None
    ) -> Tuple[bool, Union[str, Dict, None]]:
        llm_config = self.llm_config if config is None else config
        if llm_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        assert messages[-1].pop("context", None) is None
        for _ in range(3):
            try:
                response = oai_wrapper.create(messages=self._oai_system_message + messages, config_list=[llm_config], **self.sampling_params)
            except Exception as e:
                print(e)
                print("Invalid response, please try again.")
                continue
        # register the token count
        tokens_in = response["usage"]["prompt_tokens"]
        tokens_out = response["usage"]["completion_tokens"]
        self.profiler.update_tokens(tokens_in, tokens_out)
        
        return True, oai.ChatCompletion.extract_text_or_function_call(response)[0]
    
    def _prepare_chat(self, recipient: ConversableAgent, clear_history):
        super()._prepare_chat(recipient, clear_history)
        system_message = self.system_message + f" You are now having a 1 on 1 conversation with {recipient.name}"
        self.update_system_message(system_message)
        system_message = recipient.system_message + f" You are now having a 1 on 1 conversation with {self.name}"
        recipient.update_system_message(system_message)
    
    def _process_received_message(self, message, sender, silent):
        message = self._message_to_dict(message)
        original_message = copy.deepcopy(message)
        content = message["content"]
        try:
            content_json = json.loads(content)
            content = content_json.get("message", "")
            message["content"] = content
        except:
            pass
        
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(original_message, sender)

    def report(self):
        return self.profiler.report()
    
    def reset_profile(self):
        self.profiler = Profiler(0, 0, 0, 0)

    
def select_target(prompt, autogen_agent, history, cur_history, info: Dict, llm_agent: LLM_agent):
    """let the agent select who to speak to and what to ask"""
    # format prompt for selecting recipient
    dialog = format_dialog(autogen_agent.name, history, cur_history)
    message = [{"content": prompt.format(dialogue=dialog), "role": "user"}]
    # print("Prompt for selecting speaker.\n", prompt.format(dialogue=dialog))

    for _ in range(3):
        # print(message[0]["content"])
        try:
            outputs, usage = llm_agent.LLM.generator(message, llm_agent.LLM.sampling_params)
            response_json = json.loads(outputs)
            receiver, content, thoughts = response_json["receiver"], response_json["message"], response_json["thoughts"]
            break
        except Exception as e:
            print(outputs)
            print(e)
            print("retrying...")
            # update agent's config
            llm_agent.LLM.llm_config.update({"seed": llm_agent.llm_config["seed"] + 1})
            llm_agent.LLM.generator = llm_agent.LLM.lm_engine()
            # print(f"Invalid response: {response_json}, please try again")
            continue
    
    tokens_in = usage["prompt_tokens"]
    tokens_out = usage["completion_tokens"]
    info["select"]["tokens_in"] += tokens_in
    info["select"]["tokens_out"] += tokens_out
    return receiver, content, thoughts, info

def format_dialog(name, history: Dict, cur_history):
    """prepare the prompt for recipient selection"""
    # history: [[{"content": , "name": ""}, {"content": , "name": ""}], ...]
    his = history.get(name, [])
    prompt = "Previous rounds of dialogue: "
    if len(his) == 0:
        prompt += "None\n"

    for idx, messages in enumerate(his):
        prompt += f"Dialogue {idx+1}: "
        
        for message in messages:
            parts = message.split(":")
            names = parts[0].strip()
            # content = parts[1].strip()
            if name in names:
                prompt += message.replace(name, "You", 1) + "\n" #f"You: {content}\n"
            else:
                prompt += message + "\n" #f"{speaker}: {content}\n"
    prompt += "Current round of dialogue: "
    his = cur_history.get(name, [])
    if len(his) == 0:
        prompt += "None\n"
    
    for idx, message in enumerate(his):
        parts = message.split(":")
        names = parts[0].strip()
        # content = parts[1].strip()
        if name in names:
            prompt += message.replace(name, "You", 1) + "\n" #f"You: {content}\n"
        else:
            prompt += message + "\n" #f"{speaker}: {content}\n"
    # print("generated prompt: ", prompt)
    return prompt

def agent_by_names(agents, names: str) -> List[AssistantAgentCoT]:
    ret = []
    for name in names:
        for agent in agents:
            if agent.name == name:
                ret.append(agent)
                break
    return ret

def merge_history(names, previous, current):
    new = defaultdict(list)
    
    for name in names:
        prev = previous[name]
        prev.append(current[name])
        new[name] = prev
    return new

def update_info(info, sender: AssistantAgentCoT, receiver: AssistantAgentCoT, message: str):
    receiver.profiler._times_in += 1
    sender.profiler._times_out += 1
    tokens = len(message.split())
    receiver.profiler._tokens_in += tokens
    sender.profiler._tokens_out += tokens
    info["converse"][f"{sender.name} to {receiver.name}"]["times"] += 1
    info["converse"][f"{sender.name} to {receiver.name}"]["tokens"] += tokens
    tokens_in, tokens_out, times_in, times_out = sender.profiler.report()
    info["converse"][sender.name] = {"tokens_in": tokens_in, "tokens_out": tokens_out, "times_in": times_in, "times_out": times_out}
    tokens_in, tokens_out, times_in, times_out = receiver.profiler.report()
    info["converse"][receiver.name] = {"tokens_in": tokens_in, "tokens_out": tokens_out, "times_in": times_in, "times_out": times_out}
    edge_weights_token[f"{sender.name},{receiver.name}"] += num_tokens_from_string(message)
    edge_weights_time[f"{sender.name},{receiver.name}"] += 1
    return info

def visualize_comms(edges: List[Tuple], names, fig_folder_name="fig/comms"):
    global round
    # first close all the figures just in case
    plt.close('all')

    # create a graph object
    graph_time = nx.DiGraph()
    for name in names:
        graph_time.add_node(name)
    for edge in edge_weights_time:
        node1, node2 = edge.split(",")
        graph_time.add_edge(node1, node2)

    edge_colors = []
    for node1, node2 in graph_time.edges:
        edge_colors.append(edge_weights_time[f"{node1},{node2}"])

    cmap = plt.cm.plasma_r # 颜色盘设置 refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html
    pos = nx.circular_layout(graph_time)
    nodes = nx.draw_networkx_nodes(graph_time, pos, node_size=800, node_color="tab:blue")
    edges = nx.draw_networkx_edges(
        graph_time,
        pos,
        node_size=800,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
        connectionstyle="arc3, rad=0.15", # 边弯曲形状 refer to https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.ConnectionStyle.html#matplotlib.patches.ConnectionStyle
    )
    nx.draw_networkx_labels(graph_time, pos, font_size=10)

    # set alpha value for each edge 透明度设置
    for edge in edges:
        edge.set_alpha(0.9)
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.title("Communication Times Graph")

    date_hour_minute = now.strftime("%m-%d-%H-%M")

    if not os.path.exists(f"{fig_folder_name}_{date_hour_minute}"):
        os.makedirs(f"{fig_folder_name}_{date_hour_minute}")
    plt.savefig(f"{fig_folder_name}_{date_hour_minute}/times_round_{round}.png", dpi=200)

    plt.close('all')
    # create a graph object
    graph_token = nx.DiGraph()
    for name in names:
        graph_token.add_node(name)
    for edge in edge_weights_time:
        node1, node2 = edge.split(",")
        graph_token.add_edge(node1, node2)

    edge_colors = []
    for node1, node2 in graph_token.edges:
        edge_colors.append(edge_weights_token[f"{node1},{node2}"])

    cmap = plt.cm.plasma_r # 颜色盘设置 refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html
    pos = nx.circular_layout(graph_token)
    nodes = nx.draw_networkx_nodes(graph_token, pos, node_size=800, node_color="tab:blue")
    edges = nx.draw_networkx_edges(
        graph_token,
        pos,
        node_size=800,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
        connectionstyle="arc3, rad=0.15", # 边弯曲形状 refer to https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.ConnectionStyle.html#matplotlib.patches.ConnectionStyle
    )
    nx.draw_networkx_labels(graph_token, pos, font_size=10)

    # set alpha value for each edge 透明度设置
    for edge in edges:
        edge.set_alpha(0.9)
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.title("Communication Token Graph")
    plt.savefig(f"{fig_folder_name}_{date_hour_minute}/token_round_{round}.png", dpi=200)

    # save the edge list for further visualizing
    with open(f"{fig_folder_name}_{date_hour_minute}/round_{round}.pkl", "wb") as f: 
        pickle.dump(
            obj={
                "tokens": edge_weights_token,
                "times": edge_weights_time
            },
            file=f
        )
    round += 1

def communicate(names, selector_prompts, history, info: Dict, llm_agents: List[LLM_agent], converse_agents: List[AssistantAgentCoT], random_start_comm=False, log_thoughts=True, visualize=False, fig_folder_name="fig/comms"):
    """Let the agents communicate with each other. 

    Args:
        names: a list containing the names of each agent. This is used to identify the agents.
        selector_prompts: a list containing the prompts for choosing which agent to converse with.
        history: contains the conversation history. It looks like {"Agent_1": [ [{"content": , "name": ""}, {"content": , "name": ""}], ...]}
        info: A dict containing all the token usage.
            To initialize `info`, use this:
                info = {
                    "converse": {"tokens_in": 0, "tokens_out": 0},
                    "select": {"tokens_in": 0, "tokens_out": 0}
                }
        llm_agents: a list of all the agents

    Returns:
        history: renewed conversation history
    """
    
    # agents = []
    cur_history = defaultdict(list)
    llm_config_list = []
    speak_order = list(range(len(names)))
    communicate_edges = []

    if random_start_comm:
        random.shuffle(speak_order)      
    
    for agent in llm_agents:
        llm_config_list.append(agent.LLM.llm_config)
    
    for i in speak_order:
        # handle conversation one by one
        sender = converse_agents[i]
        receiver, messages, thoughts, info = select_target(selector_prompts[i], converse_agents[i], history, cur_history, info, llm_agents[i])

        logger.info(f"{names[i]}")
        logger.info(f"selected receivers: {receiver}")
        logger.info(f"messages: {messages}")
        if log_thoughts:
            logger.info(f"thoughts: {thoughts}")
        
        if (isinstance(receiver, str) and "everyone" in receiver.lower()) or (isinstance(receiver, list) and "everyone" in receiver):
            # broadcast the message
            cur_history[sender.name].append(f'{sender.name} to everyone: {messages[0]}')
            for receiver in converse_agents:
                if receiver == sender:
                    continue
                cur_history[receiver.name].append(f'{sender.name} to everyone: {messages[0]}')
                # add edges to comms graph
                communicate_edges.append((sender.name, receiver.name))
                info = update_info(info, sender, receiver, messages[0])
            continue
            
        if isinstance(receiver, str):
            receiver = [receiver]
        if isinstance(messages, str):
            messages = [messages]
        assert isinstance(receiver, list), "The selected receivers should be a list."
        assert isinstance(messages, list), "Messages to receivers should be a list."
        
        receivers = agent_by_names(converse_agents, receiver)
        # the agent decides not to send a message
        if len(receivers) == 0:
            # print("Successfully terminated conversation.")
            logger.info("Successfully terminated conversation.")
            # print("thoughts:", thoughts)
            continue
        
        # the agent decides to broadcast a message
        elif len(messages) == 1 and len(receivers) > 1: #just as  broacast
            cur_history[sender.name].append(f'{sender.name} to everyone: {messages[0]}')
            for receiver in receivers:
                if receiver == sender:
                    continue
                cur_history[receiver.name].append(f'{sender.name} to {receiver.name}: {messages[0]}')
                # add edges to comms graph
                communicate_edges.append((sender.name, receiver.name))
                info = update_info(info, sender, receiver, messages[0])
            continue
        
        elif len(messages) != len(receivers):
            receiver_len = min(len(receivers), len(messages))
            messages = messages[:receiver_len]
            receivers = receivers[:receiver_len]

        for receiver, message in zip(receivers, messages):
            if receiver == sender:
                continue
            cur_history[receiver.name].append(f'{sender.name} to {receiver.name}: {message}')
            cur_history[sender.name].append(f'{sender.name} to {receiver.name}: {message}')
            # add edges to comms graph
            communicate_edges.append((sender.name, receiver.name))
            info = update_info(info, sender, receiver, message)

    info["converse"]["overall_tokens"] = 0
    info["converse"]["overall_times"] = 0
    for name in names:
        info["converse"]["overall_tokens"] += info["converse"][name]["tokens_in"]
        info["converse"]["overall_times"] += info["converse"][name]["times_in"]

    if visualize:
        visualize_comms(communicate_edges, names, fig_folder_name)
    return merge_history(names, history, cur_history), info
