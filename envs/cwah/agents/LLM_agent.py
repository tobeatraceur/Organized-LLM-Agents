from LLM import *
from collections import defaultdict

class LLM_agent:
	"""
	LLM agent class
	"""
	def __init__(self, agent_id, args):
		self.debug = args.debug
		self.agent_type = 'LLM'
		self.agent_names = ["Agent_{}".format(i+1) for i in range(args.agent_num)]
		self.teammate_names = self.agent_names[:agent_id - 1] + self.agent_names[agent_id:]
		self.agent_id = agent_id
		self.teammate_agent_id = [i+1 for i in range(args.agent_num)]
		self.teammate_agent_id.remove(agent_id)
		self.lm_id = args.lm_id_list[agent_id - 1]
		self.llm_config = args.llm_config_list[agent_id - 1]
		self.prompt_template_path = args.prompt_template_path
		self.args = args
		self.LLM = LLM(self.lm_id, self.llm_config, self.prompt_template_path, self.args, self.agent_id, self.agent_names)
		self.action_history = []
		self.dialogue_history = []
		self.containers_name = []
		self.goal_objects_name = []
		self.rooms_name = []
		self.roomname2id = {}
		self.unsatisfied = {}
		self.steps = 0
		self.plan = None
		self.stuck = 0
		self.current_room = None
		self.last_room = None
		self.grabbed_objects = None
		self.teammate_grabbed_objects = defaultdict(list)
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}


	@property
	def all_relative_name(self) -> list:
		return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']
	
	def goexplore(self, LM_times):
		target_room_id = int(self.plan.split(' ')[-1][1:-1])
		if self.current_room['id'] == target_room_id:
			self.plan = None
			return None
		elif self.current_room['class_name'] != self.last_room['class_name'] and LM_times == 0: # just entered a new room, check if the agent should stop here
			self.plan = None
			return None
		return self.plan.replace('[goexplore]', '[walktowards]')
	
	
	def gocheck(self):
		assert len(self.grabbed_objects) < 2 # must have at least one free hands
		target_container_id = int(self.plan.split(' ')[-1][1:-1])
		target_container_name = self.plan.split(' ')[1]
		target_container_room = self.id_inside_room[target_container_id]
		if self.current_room['class_name'] != target_container_room:
			return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

		target_container = self.id2node[target_container_id]
		if 'OPEN' in target_container['states']:
			self.plan = None
			return None
		if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
			return self.plan.replace('[gocheck]', '[open]') # conflict will work right?
		else:
			return self.plan.replace('[gocheck]', '[walktowards]')


	def gograb(self):
		target_object_id = int(self.plan.split(' ')[-1][1:-1])
		target_object_name = self.plan.split(' ')[1]
		if target_object_id in self.grabbed_objects:
			if self.debug:
				print(f"successful grabbed!")
			self.plan = None
			return None
		assert len(self.grabbed_objects) < 2 # must have at least one free hands

		target_object_room = self.id_inside_room[target_object_id]
		if self.current_room['class_name'] != target_object_room:
			return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

		if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x_list in self.teammate_grabbed_objects.values() for x in x_list]:
			if self.debug:
				print(f"not here any more!")
			self.plan = None
			return None
		if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
			return self.plan.replace('[gograb]', '[grab]')
		else:
			return self.plan.replace('[gograb]', '[walktowards]')
	
	def goput(self):
		# if len(self.progress['goal_location_room']) > 1: # should be ruled out
		if len(self.grabbed_objects) == 0:
			self.plan = None
			return None
		if type(self.id_inside_room[self.goal_location_id]) is list:
			if len(self.id_inside_room[self.goal_location_id]) == 0:
				print(f"never find the goal location {self.goal_location}")
				self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
			target_room_name = self.id_inside_room[self.goal_location_id][0]
		else:
			target_room_name = self.id_inside_room[self.goal_location_id]

		if self.current_room['class_name'] != target_room_name:
			return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
		if self.goal_location not in self.reachable_objects:
			return f"[walktowards] {self.goal_location}"
		y = int(self.goal_location.split(' ')[-1][1:-1])
		y = self.id2node[y]
		if "CONTAINERS" in y['properties']:
			if len(self.grabbed_objects) < 2 and'CLOSED' in y['states']:
				return self.plan.replace('[goput]', '[open]')
			else:
				action = '[putin]'
		else:
			action = '[putback]'
		x = self.id2node[self.grabbed_objects[0]]
		return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


	def LLM_plan(self):
		if len(self.grabbed_objects) == 2:
			return f"[goput] {self.goal_location}", {}

		return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.teammate_grabbed_objects, [self.id_inside_room[teammate_agent_id_item] for teammate_agent_id_item in self.teammate_agent_id], None, self.steps)


	def check_progress(self, state, goal_spec):
		unsatisfied = {}
		satisfied = []
		id2node = {node['id']: node for node in state['nodes']}

		for key, value in goal_spec.items():
			elements = key.split('_')
			cnt = value[0]
			for edge in state['edges']:
				if cnt == 0:
					break
				if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and id2node[edge['from_id']]['class_name'] == elements[1]:
					satisfied.append(id2node[edge['from_id']])
					cnt -= 1
					# if self.debug:
					# 	print(satisfied)
			if cnt > 0:
				unsatisfied[key] = cnt
		return satisfied, unsatisfied


	def filter_graph(self, obs):
		relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
		relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
		new_graph = {
			"edges": [edge for edge in obs['edges'] if
					  edge['from_id'] in relative_id and edge['to_id'] in relative_id],
			"nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
		}
	
		return new_graph

	def obs_processing(self, observation, goal):
		satisfied, unsatisfied = self.check_progress(observation, goal)
		# print(f"satisfied: {satisfied}")
		if len(satisfied) > 0:
			self.unsatisfied = unsatisfied
			self.satisfied = satisfied
		obs = self.filter_graph(observation)
		self.grabbed_objects = []
		teammate_grabbed_objects = defaultdict(list)
		self.reachable_objects = []
		self.id2node = {x['id']: x for x in obs['nodes']}
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id:
				if r == 'INSIDE':
					self.current_room = self.id2node[y]
				elif r in ['HOLDS_RH', 'HOLDS_LH']:
					self.grabbed_objects.append(y)
				elif r == 'CLOSE':
					y = self.id2node[y]
					self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
			# elif x == self.teammate_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
			# 	teammate_grabbed_objects.append(self.id2node[y])
			else:
				for i, teammate_agent_id_item in enumerate(self.teammate_agent_id):
					if x == teammate_agent_id_item and r in ['HOLDS_RH', 'HOLDS_LH']:
						teammate_grabbed_objects[self.teammate_names[i]].append(self.id2node[y])

		unchecked_containers = []
		ungrabbed_objects = []
		# print(teammate_grabbed_objects)
		# print([w['id'] for w_list in teammate_grabbed_objects.values() for w in w_list])
		for x in obs['nodes']:
			if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w_list in teammate_grabbed_objects.values() for w in w_list]:
				for room, ungrabbed in self.ungrabbed_objects.items():
					if ungrabbed is None: continue
					j = None
					for i, ungrab in enumerate(ungrabbed):
						if x['id'] == ungrab['id']:
							j = i
					if j is not None:
						ungrabbed.pop(j)
				continue
			self.id_inside_room[x['id']] = self.current_room['class_name']
			if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
				unchecked_containers.append(x)
			if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w_list in teammate_grabbed_objects.values() for w in w_list]:
				ungrabbed_objects.append(x)

		if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
			self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
			if len(self.id_inside_room[self.goal_location_id]) == 1:
				self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
		self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
		self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

		info = {'graph': obs,
				"obs": {
						 "grabbed_objects": self.grabbed_objects,
						 "teammate_grabbed_objects": teammate_grabbed_objects,
						 "reachable_objects": self.reachable_objects,
						 "progress": {
								"unchecked_containers": self.unchecked_containers,
								"ungrabbed_objects": self.ungrabbed_objects,
									  },
						"satisfied": self.satisfied,
						"current_room": self.current_room['class_name'],
						},
				}
		for i, teammate_agent_id_item in enumerate(self.teammate_agent_id):
			if self.id_inside_room[teammate_agent_id_item] == self.current_room['class_name']:
				self.teammate_grabbed_objects[self.teammate_names[i]] = teammate_grabbed_objects[self.teammate_names[i]]

		return info

	def progress2text(self):
		return self.LLM.progress2text(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.satisfied, self.teammate_grabbed_objects, [self.id_inside_room[teammate_agent_id_item] for teammate_agent_id_item in self.teammate_agent_id], None, self.steps)
	
	def get_action(self, observation, goal, dialogue_history=None):
		"""
		:param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
		"nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
		"messages": [None, None]
		}
		:param goal:{predicate:[count, True, 2]}
		:return:
		"""
		#Apply autogen as an independant communication module
		if self.args.comm and dialogue_history is not None and dialogue_history != []:
			self.dialogue_history = [word for dialogue in dialogue_history for word in dialogue]

		info = self.obs_processing(observation, goal)
		action = None
		LM_times = 0
		while action is None:
			if self.plan is None:
				if LM_times > 0:
					print("Retrying LM Plan: ", info)
					self.LLM.llm_config.update({"seed": random.randint(0, 100000)})
				if LM_times > 3:
					plan = f"[wait]" #TODO just workround
					# raise Exception(f"retrying LM_plan too many times")
				plan, a_info = self.LLM_plan()
				# if self.debug:
				# 	print(LM_times, "plan: ", plan)
				if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
					print("No more things to do!")
					plan = f"[wait]"
				self.plan = plan
				a_info.update({"steps": self.steps})
				info.update({"LLM": a_info})
				LM_times += 1
			if self.plan.startswith('[goexplore]'):
				action = self.goexplore(LM_times)
			elif self.plan.startswith('[gocheck]'):
				action = self.gocheck()
			elif self.plan.startswith('[gograb]'):
				action = self.gograb()
			elif self.plan.startswith('[goput]'):
				action = self.goput()
			elif self.plan.startswith('[send_message]'):
				action = self.plan[:]
				self.plan = None
			elif self.plan.startswith('[wait]'):
				action = None
				break
			else:
				raise ValueError(f"unavailable plan {self.plan}")
			
		self.action_history.append(action if action is not None else self.plan)
			# if self.debug:
				# print("action:\n", action, "\n\n")
				# logger.info(f"action:\n{action}\n")
		self.steps += 1
		info.update({"plan": self.plan,
					 })
		if action == self.last_action and self.current_room['class_name'] == self.last_room['class_name']:
			self.stuck += 1
		else:
			self.stuck = 0
		self.last_action = action
		self.last_room = self.current_room
		if self.stuck > 20:
			print("Warning! stuck!")
			self.action_history[-1] += ' but unfinished'
			self.plan = None
			if type(self.id_inside_room[self.goal_location_id]) is list:
				target_room_name = self.id_inside_room[self.goal_location_id][0]
			else:
				target_room_name = self.id_inside_room[self.goal_location_id]
			action = f"[walktowards] {self.goal_location}"
			if self.current_room['class_name'] != target_room_name:
				action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
			self.stuck = 0
	
		return action, info

	def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
		self.steps = 0
		self.containers_name = containers_name
		self.goal_objects_name = goal_objects_name
		self.rooms_name = rooms_name
		self.roomname2id = {x['class_name']: x['id'] for x in room_info}
		self.id2node = {x['id']: x for x in obs['nodes']}
		self.stuck = 0
		self.last_room = None
		self.unsatisfied = {k: v[0] for k, v in goal.items()}
		self.satisfied = []
		self.goal_location = list(goal.keys())[0].split('_')[-1]
		self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
		self.id_inside_room = {self.goal_location_id: self.rooms_name[:]}
		for teammate_agent_id_item in self.teammate_agent_id:
			self.id_inside_room[teammate_agent_id_item] = None
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.teammate_grabbed_objects = defaultdict(list)
		if self.debug:
			print(self.agent_id)
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			# print(x, r, y)
			if x == self.agent_id and r == 'INSIDE':
				self.current_room = self.id2node[y]
				self.last_room = self.current_room
		self.plan = None
		self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
		self.dialogue_history = []
		self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)
