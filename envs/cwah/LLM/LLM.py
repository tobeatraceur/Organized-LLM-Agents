import random
from autogen import oai
import openai
import torch
import json
from json import JSONDecodeError
import os
import pandas as pd
from openai.error import OpenAIError
import backoff
import logging
logger = logging.getLogger("__main__")
from .llm_wrapper import oai_wrapper


class LLM:
	def __init__(self,
				 lm_id,
				 llm_config,
				 prompt_template_path,
				 args,
				 agent_id,
				 agent_names
				 ):
		self.goal_desc = None
		self.goal_location_with_r = None
		self.agent_id = agent_id
		self.agent_names = agent_names
		self.agent_name = self.agent_names[agent_id - 1]
		self.teammate_names = self.agent_names[:agent_id - 1] + self.agent_names[agent_id:]
		if self.teammate_names == []:
			self.teammate_names = ['nobody']
		self.teammate_pronoun = "she"
		self.debug = args.debug
		self.log_thoughts = args.log_thoughts
		self.goal_location = None
		self.goal_location_id = None
		self.roomname2id = {}
		self.rooms = []
		self.prompt_template_path = prompt_template_path
		self.organization_instructions = args.organization_instructions
		self.action_history_len = args.action_history_len
		self.dialogue_history_len = args.dialogue_history_len
		if self.prompt_template_path is not None:
			self.single = 'single' in self.prompt_template_path or self.teammate_names == ['nobody']
			df = pd.read_csv(self.prompt_template_path)
			self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$TEAMMATE_NAME$", ", ".join(self.teammate_names))
		
		self.lm_id = lm_id
		self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id or 'chat' in lm_id or 'human' in lm_id
		self.total_cost = 0
		self.all_actions = 0
		self.failed_actions = 0
		self.llm_config	= llm_config
		self.sampling_params = {
					"max_tokens": args.max_tokens,
					"temperature": args.t,
					"top_p": args.top_p,
					"n": args.n,
				}

		if self.lm_id == 'human':
			self.human_agent = oai_wrapper.new_human_agent()
			self.sampling_params.update({"human_agent": self.human_agent})
		self.generator = self.lm_engine()

	def lm_engine(self):

		@backoff.on_exception(backoff.expo, OpenAIError)
		def _generate(prompt, sampling_params=None):
			response = oai_wrapper.create(messages=prompt, config_list=[self.llm_config], **self.sampling_params) #, use_cache=False cache_seed=self.llm_config['seed']
			usage = response["usage"]
			response = response["choices"][0]["message"]["content"]
			return response, usage
		return _generate

	def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
		self.rooms = rooms_name
		self.roomname2id = roomname2id
		self.goal_location = goal_location
		self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
		self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)


	def goal2description(self, goals, goal_location_room):  # {predicate: count}
		map_rel_to_pred = {
			'inside': 'into',
			'on': 'onto',
		}
		s = "Find and put "
		r = None
		for predicate, vl in goals.items():
			relation, obj1, obj2 = predicate.split('_')
			count = vl
			if count == 0:
				continue
			if relation == 'holds':
				continue
			elif relation == 'sit':
				continue
			else:
				s += f"{count} {obj1}{'s' if count > 1 else ''}, "
				r = relation
		if r is None:
			return "None."

		s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
		# if type(goal_location_room) is not list:
		# 	s += f" in the {goal_location_room}."
		# else:
		# 	ss = ' or '.join([f'{room}' for room in goal_location_room])
		# 	s += f", which may be in the {ss}."
		return s, f"{map_rel_to_pred[r]} the {self.goal_location}"

	def parse_answer(self, available_actions, text):
		self.all_actions += 1
		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			# txt = text.lower()
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text or option == text[0]:
				return action
		self.failed_actions += 1
		if self.debug:
			logger.info(f"Agent_{self.agent_id} failed to generate actions: {self.failed_actions}/{self.all_actions}")
			logger.info("WARNING! Fuzzy match!")
		for i in range(len(available_actions)):
			action = available_actions[i]
			act, name, id = action.split(' ')
			option = chr(ord('A') + i)
			if f"{option} " in text or act in text or name in text or id in text:
				return action
		print("WARNING! No available action parsed!!! Random choose one")
		return random.choice(available_actions)


	def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, teammate_grabbed_objects, teammate_last_room, room_explored, steps):
		sss = {}
		for room, objs in ungrabbed_objects.items():
			cons = unchecked_containers[room]
			extra_obj = None
			if type(goal_location_room) is not list and goal_location_room == room:
				extra_obj = self.goal_location
			if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
				sss[room] = f"The {room} is unexplored. "
				continue
			s = ""
			s_obj = ""
			s_con = ""
			if extra_obj is not None:
				s_obj = f"{extra_obj}, "
			if objs is not None and len(objs) > 0:
				if len(objs) == 1:
					x = objs[0]
					s_obj += f"<{x['class_name']}> ({x['id']})"
				else:
					ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
					s_obj += ss
			elif extra_obj is not None:
				s_obj = s_obj[:-2]
			if cons is not None and len(cons) > 0:
				if len(cons) == 1:
					x = cons[0]
					s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
				else:
					ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
					s_con = f"unchecked containers " + ss
			if s_obj == "" and s_con == "":
				s += 'nothing'
				if room_explored is not None and not room_explored[room]:
					s += ' yet'
			elif s_obj != "" and s_con != "":
				s += s_obj + ', and ' + s_con
			else:
				s += s_obj + s_con
			sss[room] = s

		if len(satisfied) == 0:
			s = ""
		else:
			s = f"{'I' if self.single else 'We'}'ve already found and put "
			s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
			s += ' ' + self.goal_location_with_r + '. '

		if len(grabbed_objects) == 0:
			s += "I'm holding nothing. "
		else:
			s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
			if len(grabbed_objects) == 2:
				s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
		s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
		### teammate modeling
		if not self.single:
			for i, teammate_name in enumerate(self.teammate_names):
				ss = ""
				if len(teammate_grabbed_objects[teammate_name]) == 0:
					ss += "nothing. "
				else:
					ss += f"<{teammate_grabbed_objects[teammate_name][0]['class_name']}> ({teammate_grabbed_objects[teammate_name][0]['id']}). "
					if len(teammate_grabbed_objects[teammate_name]) == 2:
						ss = ss[:-2] + f" and <{teammate_grabbed_objects[teammate_name][1]['class_name']}> ({teammate_grabbed_objects[teammate_name][1]['id']}). "
				if teammate_last_room[i] is None:
					s += f"I don't know where {teammate_name} is. "
				elif teammate_last_room[i] == current_room['class_name']:
					s += f"I also see {teammate_name} here in the {current_room['class_name']}, {self.teammate_pronoun} is holding {ss}"
				else:
					s += f"Last time I saw {teammate_name} was in the {teammate_last_room[i]}, {self.teammate_pronoun} was holding {ss}"

		for room in self.rooms:
			if room == current_room['class_name']:
				continue
			if 'unexplored' in sss[room]:
				s += sss[room]
			else:
				s += f"I found {sss[room]} in the {room}. "

		return f"This is step {steps}. " + s


	def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
		"""
		[goexplore] <room>
		[gocheck] <container>
		[gograb] <target object>
		[goput] <goal location>
		"""
		available_plans = []
		for room in self.rooms:
			if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
				continue
			available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
		if len(grabbed_objects) < 2:
			for cl in unchecked_containers.values():
				if cl is None:
					continue
				for container in cl:
					available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
			for ol in ungrabbed_objects.values():
				if ol is None:
					continue
				for obj in ol:
					available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
		if len(grabbed_objects) > 0:
			available_plans.append(f"[goput] {self.goal_location}")

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"

		return plans, len(available_plans), available_plans


	def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, teammate_grabbed_objects, teammate_last_room, room_explored = None, steps = None):
		info = {}
		# goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
		progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, teammate_grabbed_objects, teammate_last_room, room_explored, steps)
		action_history_desc = ", ".join(action_history[-self.action_history_len:] if len(action_history) > self.action_history_len else action_history)
		dialogue_history_desc = '\n'.join(dialogue_history[-self.dialogue_history_len:] if len(dialogue_history) > self.dialogue_history_len else dialogue_history) 
		prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
		if self.organization_instructions is not None:
			prompt = prompt.replace("$ORGANIZATION_INSTRUCTIONS$", self.organization_instructions)
		prompt = prompt.replace('$PROGRESS$', progress_desc)
		prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
		message = None
		info.update({"goal": self.goal_desc,
					 "progress": progress_desc,
					 "action_history": action_history_desc,
					 "dialogue_history_desc": dialogue_history_desc})
		prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

		available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored)
		if num == 0 or (message is not None and num == 1):
			print("Warning! No available plans!")
			plan = None
			info.update({"num_available_actions": num,
					 "plan": None})
			return plan, info

		prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

		if steps == 0:
			print(f"base_prompt:\n{prompt}\n\n")
		if self.debug:
			logger.info(f"base_prompt:\n{prompt}\n\n")
		for _ in range(3):
			try:
				outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt, self.sampling_params)
				if outputs[0] != '{' and '{' in outputs:
					outputs = '{' + outputs.split('{')[1].strip()
					outputs = outputs.split('}')[0].strip() + '}'
				outputs_json = json.loads(outputs)
				output = outputs_json["action"]
				thoughts = outputs_json["thoughts"]
				break
			except (JSONDecodeError, KeyError) as e:
				print(outputs)
				print(e)
				self.llm_config.update({"seed": self.llm_config["seed"] + 1})
				self.generator = self.lm_engine()
				print("retrying...")

				continue

		# info['cot_usage'] = usage
		logger.info(f"action_output: {output}")
		if self.log_thoughts:
			logger.info(f"thoughts: {thoughts}")
		# logger.info(f"action_output: {output}, thoughts: {thoughts}\n")
		plan = self.parse_answer(available_plans_list, output)
		if self.debug:
			logger.info(f"plan:\n{plan}")
		info.update({"num_available_actions": num,
					 "prompts": prompt,
					 "outputs": outputs,
					 "plan": plan,
					 "total_cost": self.total_cost})
		return plan, info

