import pickle
import pdb
import ipdb
import sys
import os
import random
import json
import numpy as np
import copy
import argparse

curr_dir = os.path.dirname(os.path.abspath(__file__))
home_path = '../../'
sys.path.append(f'{curr_dir}/../../virtualhome')
sys.path.append(f'{curr_dir}/..')

from simulation.unity_simulator import comm_unity
from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks import Task


from utils import utils_goals

parser = argparse.ArgumentParser()
parser.add_argument('--num-per-apartment', type=int, default=1, help='Maximum #episodes/apartment')
parser.add_argument('--seed', type=int, default=10, help='Seed for the apartments')

parser.add_argument('--task', type=str, default='setup_table', help='Task name')
parser.add_argument('--apt_str', type=str, default='0,1,2,4,5', help='The apartments where we will generate the data')
parser.add_argument('--port', type=str, default='8092', help='Task name')
parser.add_argument('--display', type=int, default=0, help='Task name')
parser.add_argument('--mode', type=str, default='full', choices=['simple', 'full'], help='Task name')
parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')
parser.add_argument('--exec_file', type=str,
                    default='../../linux_exec.v2.3.0.x86_64',
                    help='Use unity editor')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed == 0:
        rand = random.Random()
    else:
        rand = random.Random(args.seed)


    with open(f'{curr_dir}/data/init_pool.json') as file:
        init_pool = json.load(file)
    # comm = comm_unity.UnityCommunication()
    if args.use_editor:
        comm = comm_unity.UnityCommunication()
    else:
        print(comm_unity)
        comm = comm_unity.UnityCommunication(port=args.port,
                                             file_name=args.exec_file,
                                             no_graphics=True,
                                             logging=False,
                                             x_display=args.display)
    comm.reset()

    ## -------------------------------------------------------------
    ## step3 load object size
    with open(f'{curr_dir}/data/class_name_size.json', 'r') as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------
    task_names = {1: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv", "setup_table_prepare_food", "setup_table_put_fridge"],
                  2: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv", "setup_table_prepare_food", "prepare_food_put_dishwasher"],
                  3: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv", "setup_table_prepare_food"],
                  4: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv", "setup_table_prepare_food"],
                  5: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge",
                      "prepare_food", "setup_table_prepare_food"],
                  6: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv", "setup_table_prepare_food"],
                  7: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food",
                      "read_book", "watch_tv", "setup_table_prepare_food"]}

    success_init_graph = []

    apartment_ids = [int(apt_id) for apt_id in args.apt_str.split(',')]
    if args.task == 'all':
        tasks = ['setup_table', 'put_fridge', 'prepare_food', 'put_dishwasher', 'read_book']
    else:
        tasks = [args.task]
    num_per_apartment = args.num_per_apartment

    for task in tasks:
        # for apartment in range(6,7):
        for apartment in apartment_ids:
            print('apartment', apartment)

            if task not in task_names[apartment + 1]: continue
            # if apartment != 4: continue
            # apartment = 3

            with open(f'{curr_dir}/data/object_info%s.json' % (apartment + 1), 'r') as file:
                obj_position = json.load(file)

            # pdb.set_trace()bathroomcounter

            # filtering out certain locations
            for obj, pos_list in obj_position.items():
                if obj in ['book', 'remotecontrol']:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['kitchencabinet', 'cabinet'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'bench', 'nightstand'] + ([] if apartment == 2 else ['kitchentable']))]
                elif obj == 'remotecontrol':
                    # TODO: we never get here
                    positions = [pos for pos in pos_list if pos[0] == 'ON' and pos[1] in \
                                 ['tvstand']]
                else:
                    positions = [pos for pos in pos_list if \
                                 pos[0] == 'INSIDE' and pos[1] in ['fridge', 'kitchencabinet', 'cabinet', 'microwave',
                                                                   'dishwasher', 'stove'] or \
                                 pos[0] == 'ON' and pos[1] in \
                                 (['cabinet', 'coffeetable', 'bench'] + ([] if apartment == 2 else ['kitchentable']))]
                obj_position[obj] = positions
            print(obj_position['cutleryfork'])

            num_test = 100 #100000
            count_success = 0
            for i in range(num_test):
                comm.reset(apartment)
                s, original_graph = comm.environment_graph()
                graph = copy.deepcopy(original_graph)

                task_name = task

                print('------------------------------------------------------------------------------')
                print('testing %d/%d: %s. apartment %d' % (i, num_test, task_name, apartment))
                print('------------------------------------------------------------------------------')

                ## -------------------------------------------------------------
                ## setup goal based on currect environment
                ## -------------------------------------------------------------
                set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, task_name, same_room=False, rand=rand)
                init_graph, env_goal, success_setup = getattr(Task, task_name)(set_init_goal, graph)
                if env_goal is None or success_setup is False:
                    pdb.set_trace()
                if success_setup:
                    # If all objects were well added
                    success, message = comm.expand_scene(init_graph, transfer_transform=False)
                    print('----------------------------------------------------------------------')
                    print('success_setup')
                    print(task_name, success, message, set_init_goal.num_other_obj)
                    print('env_goal:',env_goal)

                    if not success:
                        goal_objs = []
                        goal_names = []
                        for k, goals in env_goal.items():
                            goal_objs += [int(list(goal.keys())[0].split('_')[-1]) for goal in goals if
                                          list(goal.keys())[0].split('_')[-1] not in ['book', 'remotecontrol']]
                            goal_names += [list(goal.keys())[0].split('_')[1] for goal in goals]
                        print('message:', message)
                        obj_names = [obj.split('.')[0] for obj in message['unplaced']]
                        obj_ids = [int(obj.split('.')[1]) for obj in message['unplaced']]
                        id2node = {node['id']: node for node in init_graph['nodes']}

                        for obj_id in obj_ids:
                            print("Objects unplaced")
                            print([id2node[edge['to_id']]['class_name'] for edge in init_graph['edges'] if
                                   edge['from_id'] == obj_id])
                            ipdb.set_trace()
                        if task_name != 'read_book' and task_name != 'watch_tv':
                            intersection = set(obj_names) & set(goal_names)
                        else:
                            intersection = set(obj_ids) & set(goal_objs)

                        ## goal objects cannot be placed
                        if len(intersection) != 0:
                            success2 = False
                        else:
                            init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                            comm.reset(apartment)
                            success2, message2 = comm.expand_scene(init_graph, transfer_transform=False)
                            success = True

                    else:
                        success2 = True

                    if success2 and success:


                        success = set_init_goal.check_goal_achievable(init_graph, comm, env_goal, apartment)

                        if success:
                            init_graph0 = copy.deepcopy(init_graph)
                            comm.reset(apartment)
                            comm.expand_scene(init_graph, transfer_transform=False)
                            s, init_graph = comm.environment_graph()
                            print('final s:', s)
                            if s:
                                # for subgoal in env_goal[task_name]:
                                    # for k, v in subgoal.items():
                                for k, v in env_goal.items():
                                    elements = k.split('_')
                                    # print(elements)
                                    # pdb.set_trace()
                                    if len(elements) == 4:
                                        obj_class_name = elements[1]
                                        ids = [node['id'] for node in init_graph['nodes'] if
                                                node['class_name'] == obj_class_name]
                                        print(obj_class_name, v, ids)


                                count_success += s
                                check_result = set_init_goal.check_graph(init_graph, apartment + 1, original_graph)
                                assert check_result == True

                                success_init_graph.append({'id': count_success,
                                                           'apartment': (apartment + 1),
                                                           'task_name': task_name,
                                                           'init_graph': init_graph,
                                                           'original_graph': original_graph,
                                                           'goal': env_goal})
                        else:
                            print('failed to achieve the goal')
                            # pdb.set_trace()
                else:
                    pdb.set_trace()
                print('apartment: %d: success %d over %d (total: %d)' % (apartment, count_success, i + 1, num_test))
                if count_success >= num_per_apartment:
                    break

    
    data = success_init_graph
    env_task_set = []

    # for task in ['setup_table', 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book', 'setup_table_prepare_food', 'setup_table_put_fridge']:
    # for task in ['setup_table_prepare_food', 'setup_table_put_fridge']: #TODO for more complex tasks
    for task in tasks:    
        for task_id, problem_setup in enumerate(data):
            # pdb.set_trace()
            env_id = problem_setup['apartment'] - 1
            task_name = problem_setup['task_name']
            init_graph = problem_setup['init_graph']
            if task == 'setup_table_prepare_food' or task == 'setup_table_put_fridge'or task == 'prepare_food_put_dishwasher':
                goal = [x for x_list in problem_setup['goal'].values() for x in x_list]
            else:
                goal = problem_setup['goal'][task]
                
            print(goal)
            goals = utils_goals.convert_goal_spec(task_name, goal, init_graph,
                                                  exclude=['cutleryknife'])
            if len(list(goals.keys())[0].split('_')) == 4:
                # remove the fisrt element of the key
                updated_goals = {}
                for k, v in goals.items():
                    elements = k.split('_')[1:]
                    print(elements)
                    new_key = '{}_{}_{}'.format(elements[1], elements[0], elements[2])
                    updated_goals[new_key] = v

                goals = updated_goals

            print('env_id:', env_id)
            print('task_name:', task_name)
            print('goals:', goals)

            task_goal = {}
            for i in range(2): #TODO for more agents
                task_goal[i] = goals

            goal_class = {}
            for predicate, count in task_goal[0].items():
                elements = predicate.split('_')
                if elements[2].isdigit():
                    # ipdb.set_trace()
                    id2node = {node['id']: node for node in init_graph['nodes']}
                    new_predicate = '{}_{}_{}'.format(elements[0], elements[1], id2node[int(elements[2])]['class_name'])
                    # location_name = id2node[int(elements[2])]['class_name']
                else:
                    raise NotImplementedError
                    new_predicate = predicate
                goal_class[new_predicate] = count

            print('goal_class:', goal_class)

            env_task_set.append({'task_id': task_id, 'task_name': task_name, 'env_id': env_id, 'init_graph': init_graph,
                                 'task_goal': task_goal, 'total_goal': task_goal[0], 'goal_class': goal_class,
                                 'level': 0, 'init_rooms': rand.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)})

    pickle.dump(env_task_set, open(f'{curr_dir}/../dataset/env_task_set_{args.task}_{args.mode}_harder10.pik', 'wb'))



