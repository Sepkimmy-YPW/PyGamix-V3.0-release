import os
import sys
import re
import numpy as np
from copy import deepcopy
import random
import math
from units import TreeBasedOrigamiGraph
from utils import *
import json
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as aat
import gc

from phys_sim25 import OrigamiSimulator, VALID_BOUND, ti
import multiprocessing

from datetime import datetime

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams.update({'font.size': 28})

Version = 1.6

START = 0
END = 1
X = 0
Y = 1
VALLEY = 0
MOUNTAIN = 1
BORDER = 2

symmetric_routing = 0
double_div = 0
enable_double_P = 0

DUPLICATE_FACTOR = 1
GOOVER_UNIT_FACTOR = 1
CREASE_CROSS = 2

####################################
# origami name
origami = "robot14"
# string number
string_number = 2
# enable/disable masking the types of creases
arbitrary_crease_mode = 0
# enable only calculating the number of strategies
calculate_number = 0
# multi-process number
thread_number = 4
# reward_coeff
reward_coeff = [0.2, 0.0, 0.2] # maximal speed, minimal actuator num, minimal actuation force
# weight
reward_weight = [1.0, 0.0, 0.0]
# EA constraint
ea_constraint_initial_segment = False
ea_constraint_final_tip = True
# enable work criteria
enable_work_criteria = True
#####################################

search_mode = 0  # 0=regular, 1=fast
sim_override = {}  # CLI overrides for simulation_config

ROLLOUT_NUM = lambda x, y: y * y if search_mode else x * y
STEP_IN_INTERVAL = lambda x, y: y * y if search_mode else x * y

enable_diversity = 0
diversity_tolerance = 1e-5
enable_duplicate_id = False

disabled_panels = []
pre_calculate_condition = 1
bonus_val = 1.
fail_reward = -bonus_val
base_reward = 0.0
valid_base_reward = VALID_BOUND
encourage_reward = 1.0
episodes = 19
record_episode = 500

upper_bound = 8
MIN_A = 1
MIN_AA = 1
UNIT_PASS = 14
UNIT_NUM = 18
enable_curl_penalty = 0
GOOVER_UNIT_LOWER_BOUND = UNIT_PASS / UNIT_NUM
RATIO_LOWER_BOUND = string_number / (string_number - 1.) * (UNIT_PASS / (string_number * (upper_bound - 1)) - 1. / string_number) if string_number > 1 else 1.0
dead_maximum = (string_number * upper_bound) * (UNIT_NUM)
VALID_MINIMAL = 10
SUB_REWARD_NUM = 9

id_calculator = 'char'

fixed_action_list_initial = [
    [] for _ in range(string_number + 1)
]

additional_connection = [
    [],
]

banned_connection = [
    [],
]
# fixed_action_list_initial[1] = [2, 0, 3, 3]

seed = 3407

direct_reward = False

cut_num = 0
SIM = 1

new_method = False
initialize = 1
cut_nodes = 1

start_time = datetime.now()

candidate_methods = []
candidate_method_ids = []
candidate_method_rewards = []
candidate_method_types = []
# candidate_reward = []
# candidate_form = []
# candidate_xy_con = []
# candidate_identity = []

time = 0
step = 0
maximum_valid = SIM

BUFFER_SIZE = 512
NAME = f"SBS-{origami}-S{string_number}"
BUF_NAME = f"SBS-{origami}-S{string_number}"
file_path = ""

def createFolder():
    global NAME, BUF_NAME, file_path
    NAME = f"SBS-{origami}-S{string_number}"
    BUF_NAME = f"SBS-{origami}-S{string_number}"

    if calculate_number:
        NAME = f"CALNUM-{origami}-{SIM}sim-{string_number}string-{episodes}episodes"

    NAME += "-U" + str(upper_bound)

    NAME += "-A" + str(MIN_A)

    NAME += "-RW-" + f"{reward_weight}"
    # NAME += "-LB" + "{:.2f}".format(RATIO_LOWER_BOUND) + "-GB" + "{:.2f}".format(GOOVER_UNIT_LOWER_BOUND)

    file_path = f"./threadingResult/train-" + NAME
    try:
        os.makedirs(file_path)
    except:
        pass

def pad_and_get_mask(lists):
    lens = [len(l) for l in lists]
    max_len = max(lens)
    arr = np.zeros((len(lists), max_len), float)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(lists)
    return np.ma.array(arr, mask=~mask)

def moving_average(a, n):
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).tolist()

def plot_scores(scores, steps=None, window=100, label=None, fail_reward=0.):
    avg_scores = deepcopy(scores)
    if steps is not None:
        for i in range(len(scores)):
            avg_scores[i] = np.interp(np.arange(steps[i][-1]), steps[i], avg_scores[i])
    if len(scores) > 1:
        avg_scores = pad_and_get_mask(avg_scores)
        scores = avg_scores.mean(axis=0)
        scores_l = avg_scores.mean(axis=0) - avg_scores.std(axis=0)
        scores_h = avg_scores.mean(axis=0) + avg_scores.std(axis=0)
        idx = list(range(len(scores)))
        plt.fill_between(idx, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25)
    else:
        scores = avg_scores[0]
    plt.plot(scores, label=label)

def visualize(step, title, log_dict):
    train_window, loss_window, q_window = 1, 100, 100
    plt.figure(figsize=(19, 6))

    # plot train and eval returns
    plt.subplot(1, 3, 1)
    plt.title('Step: %s, Max Reward: %s / %s' % (step, round(log_dict['valid_reward_returns'][-1][-1], 4), round(log_dict['max_reward_returns'][-1][-1], 4)))
    plot_scores(log_dict['reward_returns'], log_dict['train_steps'], window=1, label='avg_reward', fail_reward=(fail_reward+1)*0.5)
    plot_scores(log_dict['max_reward_returns'], log_dict['train_steps'], window=1, label='maximum_reward', fail_reward=(fail_reward+1)*0.5)
    plot_scores(log_dict['tree_policy_max_reward_returns'], log_dict['train_steps'], window=1, label='maximum_reward (tree policy)', fail_reward=(fail_reward+1)*0.5)
    # plt.title('frame %s. train_return: %s' % (step, np.mean(log_dict['train_returns'][-1][-train_window:])))
    # if min([len(log_dict['eval_steps'][i]) for i in range(len(log_dict['eval_steps']))]) > 0:
    #     plot_scores(log_dict['eval_returns'], log_dict['eval_steps'], window=1, label='eval') 
    # plot_scores(log_dict['train_returns'], log_dict['train_steps'], window=100, label='train_reward')
    
    # plot_scores(log_dict['max_reward_returns'], log_dict['train_steps'], window=10, label='max_reward')
    plt.legend(loc='lower right')
    plt.xlabel('step')

    # # plot td losses
    plt.subplot(1, 3, 2)
    plt.title('Left Nodes: %s' % (log_dict['cut_number'][-1][-1]))
    plot_scores(log_dict['cut_number'], log_dict['train_steps'], window=1, label='cut_node_number')
    # plot_scores(log_dict['losses'], window=loss_window, label='loss')
    plt.xlabel('step')

    plt.subplot(1, 3, 3)
    # # plot q values
    percent = 0.0 if log_dict['number'][-1][-1] < 1 else round((log_dict['valid_number'][-1][-1] + log_dict['rollout_valid_number'][-1][-1]) / (log_dict['number'][-1][-1]) * 100.0, 4)
    plt.title('V/RV/T ' + str(log_dict['valid_number'][-1][-1]) + ' / ' + str(log_dict['rollout_valid_number'][-1][-1]) + \
        ' / ' + str(log_dict['number'][-1][-1]) + ' Percent: ' + str(percent) + '%')
    plot_scores(log_dict['valid_number'], log_dict['train_steps'], window=1, label='valid (V)')
    plot_scores(log_dict['rollout_valid_number'], log_dict['train_steps'], window=1, label='rollout_valid (RV)')
    plot_scores(log_dict['number'], log_dict['train_steps'], window=1, label='total (T)')
    
    # plot_scores(log_dict['Qs'], window=q_window, label='q_values')
    # plt.xlabel('step')
    plt.legend(loc='upper left')
    plt.xlabel('step')

    plt.suptitle(title, fontsize=18)
    plt.savefig(os.path.join(file_path, f'{NAME}.png'))
    plt.close()

class Env:
    def __init__(self, discount, upper_bound_bonus, print_result) -> None:
        self.max_edge = 4
        self.output_reward_buffer = True

        # 获取点和线段信息
        with open(f"./descriptionData/{origami}.json", 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
            
        if "diamond" in origami:
            self.simulation_config = {
                "platform_height": 100,
                "gravity_flag": 2,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "miura-ori-16" in origami:
            self.simulation_config = {
                "platform_height": 100,
                "gravity_flag": 1,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "box" in origami:
            self.simulation_config = {
                "platform_height": 1.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "resch" in origami:
            self.simulation_config = {
                "platform_height": 100,
                "gravity_flag": 2,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "tri-resch" in origami:
            self.simulation_config = {
                "platform_height": 1.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "mountain-big-fix" in origami:
            self.simulation_config = {
                "platform_height": 100,
                "gravity_flag": 1,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "test" in origami:
            self.simulation_config = {
                "platform_height": 100,
                "gravity_flag": 1,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.0,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 0,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "bird" in origami:
            self.simulation_config = {
                "platform_height": 2.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 400.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "miura" in origami:
            self.simulation_config = {
                "platform_height": 30.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 400.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "mountain" in origami:
            self.simulation_config = {
                "platform_height": 10.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 0,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 100.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        elif "robot8-big" in origami:
            self.simulation_config = {
                "platform_height": 220.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 1,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 60.0,
                "simulation_upper_time": 65.0,
                "additional_length_of_string": 0.0,
                "robot_type": 1,
                "type_of_controller": 2,
                "stroke_percent": 0.75
            }
        elif "robot14" in origami:
            self.simulation_config = {
                "platform_height": 90.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 1,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 0.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.5
            }
        elif "robot" in origami:
            self.simulation_config = {
                "platform_height": 90.,
                "gravity_flag": 6,
                "extract_mode": -1,
                "prefold": 0.0,
                "miu": 0.3,
                "control_mode": 1,
                "friction_mode": 2,
                "speed_bonus": 1.0,
                "ground_enable": 1,
                "control_mode_training_time": 15.0,
                "simulation_upper_time": 20.0,
                "additional_length_of_string": 0.0,
                "robot_type": 0,
                "type_of_controller": 1,
                "stroke_percent": 0.75
            }
        
        elif sim_override:
            # origami name unrecognized, but CLI overrides provided — build config from overrides
            self.simulation_config = {
                "platform_height": sim_override.get("platform_height", 1.0),
                "gravity_flag": sim_override.get("gravity_flag", 0),
                "extract_mode": sim_override.get("extract_mode", -1),
                "prefold": sim_override.get("prefold", 0.0),
                "miu": sim_override.get("miu", 0.3),
                "control_mode": sim_override.get("control_mode", 0),
                "friction_mode": sim_override.get("friction_mode", 2),
                "speed_bonus": sim_override.get("speed_bonus", 1.0),
                "ground_enable": sim_override.get("ground_enable", 0),
                "control_mode_training_time": sim_override.get("control_mode_training_time", 15.0),
                "simulation_upper_time": sim_override.get("simulation_upper_time", 20.0),
                "additional_length_of_string": sim_override.get("additional_length_of_string", 0.0),
                "robot_type": sim_override.get("robot_type", 0),
                "type_of_controller": sim_override.get("type_of_controller", 1),
                "stroke_percent": sim_override.get("stroke_percent", 0.75)
            }
        else:
            raise NotImplementedError(f"Simulation config for \"{origami}\" is not defined. "
                                      f"Provide sim_overrides or add a config entry for this origami.")
        # Apply any remaining CLI overrides (only matters for the elif branches above)
        if sim_override:
            self.simulation_config.update({k: v for k, v in sim_override.items()})
            
        self.kps = []
        self.lines = []
        self.units = []
        for i in range(len(input_json["kps"])):
            self.kps.append(input_json["kps"][i])
        for i in range(len(input_json["lines"])):
            self.lines.append(Crease(
                input_json["lines"][i][START], input_json["lines"][i][END], BORDER 
            ))
            self.lines[i].crease_type = input_json["line_features"][i]["type"]
            self.lines[i].level = input_json["line_features"][i]["level"]
            self.lines[i].coeff = input_json["line_features"][i]["coeff"]
            self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
            self.lines[i].hard = input_json["line_features"][i]["hard"]
            self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
            self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]
        for i in range(len(input_json["units"])):
            self.units.append(Unit())
            kps = input_json["units"][i]
            for j in range(0, -len(kps), -1):
                crease_type = BORDER
                current_kp = [kps[j][X], kps[j][Y]]
                next_kp = [kps[j - 1][X], kps[j - 1][Y]]
                for line in self.lines:
                    if (samePoint(line[START], current_kp, 3) and samePoint(line[END], next_kp, 3)) or \
                        (samePoint(line[END], current_kp, 3) and samePoint(line[START], next_kp, 3)):
                        crease_type = line.getType()
                        break
                self.units[i].addCrease(Crease(
                    current_kp, next_kp, crease_type
                ))

        try:
            self.contributions = input_json["contributions"]
            for i in range(len(self.contributions)):
                new_contribution = []
                for j in range(0, -len(self.contributions[i]), -1):
                    new_contribution.append(self.contributions[i][j])
                self.contributions[i] = new_contribution
        except:
            self.contributions = [unit.getContribution() for unit in self.units]
        
        try:
            self.initial_center_list = np.array(input_json["initial_center"])
            self.current_center_list = np.array(input_json["current_center"])
        except:
            self.initial_center_list = []
            self.current_center_list = []
            
        try:
            self.fix_id_list = input_json["fix"]
        except:
            self.fix_id_list = [-1]
        try:
            self.targets = deepcopy(input_json["crease_angle"])
        except:
            self.targets = []
            
        try:
            self.P_candidator_connections = input_json["P_candidators"]["connections"]
        except:
            self.P_candidator_connections = [-1]
            
        global enable_curl_penalty, UNIT_NUM, UNIT_PASS, MIN_A, MIN_AA, GOOVER_UNIT_LOWER_BOUND, RATIO_LOWER_BOUND, dead_maximum, upper_bound

        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)

        self.unit_number = len(self.units)
        self.best_reward = 0.0
        self.valid_best_reward = fail_reward
        self.node_num = 1

        # calculate max length of view

        self.string_number = string_number

        self.P_candidators = input_json["P_candidators"]["points"]
        
        self.P_points = [
            np.array(self.P_candidators[i]) for i in range(len(self.P_candidators))
        ]
        self.mid_x = max_x / 2.
        self.mid_y = max_y / 2.

        self.mid_point = np.array([self.mid_x, self.mid_y, 0.0])

        self.O_points = [
            np.array(self.units[i].getCenterUsingContribution(self.contributions[i])) for i in range(self.unit_number)
        ]

        self.maximum_distance_to_mid_point = max([np.linalg.norm(self.O_points[i] - self.mid_point) for i in range(len(self.O_points))])

        self.old_mean_per_string = [[] for _ in range(string_number)]
        self.old_std_per_string = [[] for _ in range(string_number)]

        self.state_number = self.string_number * (self.unit_number + 2)
        self.action_number = len(self.P_points) + self.unit_number
               
        system_type = -1
        if self.fix_id_list[0] < 0 and self.P_candidator_connections[0] < 0:
            system_type = 0
            enable_curl_penalty = 1
            MIN_A = string_number
            MIN_AA = min(2, MIN_A, len(self.P_candidators))
            print("EA System")
        elif self.fix_id_list[0] >= 0 and self.P_candidator_connections[0] < 0:
            system_type = 1
            enable_curl_penalty = 0
            MIN_A = MIN_AA = 1
            print("IA-static System")
        elif self.fix_id_list[0] < 0 and self.P_candidator_connections[0] >= 0:
            system_type = 2
            enable_curl_penalty = 0
            MIN_A = MIN_AA = 1
            print("IA-dynamic System")
        
        self.system_type = system_type
        if system_type == -1:
            raise NotImplementedError
        
        UNIT_NUM = self.unit_number
        tb = TreeBasedOrigamiGraph(self.kps, self.lines)
        tb.calculateTreeBasedGraph()
        DoF = 0
        if system_type == 0:
            DoF = 1
        for vertex in tb.vertices:
            if not vertex.is_border_node and vertex.dn >= 4:
                DoF += vertex.dn - 3
            #check short creases
            creases = [tb.lines[index] for index in vertex.connection_index]
            for crease in creases:
                if crease.getType() != BORDER and crease.getLength() <= 3.0:
                    DoF += 0.5
                
        #check individual panel
        for unit in self.units:
            crease_num = len(unit.crease)
            for i in range(crease_num):
                previous_id = (i - 1 + crease_num) % crease_num
                next_id = (i + 1) % crease_num
                if unit.crease[i].getType() != BORDER and unit.crease[previous_id].getType() == BORDER and unit.crease[next_id].getType() == BORDER:
                    DoF += 1
                    break
        
        DoF = round(DoF)
        UNIT_PASS = min(DoF, UNIT_NUM)

        if system_type == 2:
            UNIT_PASS = min(7, UNIT_PASS) #restrict the upper bound
        
        upper_bound = int(UNIT_PASS / string_number) + 3
        # upper_bound = int(UNIT_PASS / string_number) + 3
        
        if origami == '8-seg-miura':
            UNIT_PASS = 38
        
        # if system_type != 0:
        #     upper_bound -= 1
        
        if system_type == 2 or system_type == 1:
            upper_bound -= 1

        self.current_state = np.array([0. for _ in range(self.state_number)])

        self.current_stand_point_id = -1.
        self.old_stand_point_id = -1.

        self.current_trajectory = [[] for _ in range(self.string_number)]
        self.current_string_id = 1.
        self.current_valid_pass = 0
        self.temp_done = 1
        # self.done = 0
        self.reward = 0.
        
        self.previous_side = 0

        self.fail_reward = fail_reward
        self.encourage_reward = encourage_reward
        self.avg_reward = [self.fail_reward for _ in range(100)]
        self.scores = []
        self.steps = []
        
        self.valid_matrix = np.zeros((self.action_number, self.action_number), int)
        self.valid_action_num = np.zeros(self.action_number, int)

        self.node_status = np.array([0. for i in range(self.action_number)])
        self.status_number = self.action_number
        
        V = 0
        M = 0.725
        UNSURE = 0.83
        INVALID = 1
        FIX = 0.6
        try:
            # raise NotImplementedError
            final_matrix = input_json['true_valid_actions']
            if system_type == 2 and arbitrary_crease_mode:
                for i in range(len(self.P_points), self.action_number):
                    for j in range(len(self.P_points), self.action_number):
                        if abs(final_matrix[i][j]) == 1:
                            final_matrix[i][j] = 0
            self.valid_matrix = np.array(final_matrix)
            self.show_matrix = np.zeros(shape=(self.action_number, self.action_number))
            for i in range(len(self.P_points)):
                self.valid_action_num[i] = 0
                for j in range(self.action_number):
                    if self.valid_matrix[j][i] != -10:
                        self.valid_action_num[i] += 1
                    if [j, i] in additional_connection:
                        self.valid_action_num[i] += 1
                        self.valid_action_num[j] += 1
                        self.valid_matrix[i][j] = 0
                        self.valid_matrix[j][i] = 0
                    if [j, i] in banned_connection:
                        self.valid_action_num[i] -= 1
                        self.valid_action_num[j] -= 1
                        self.valid_matrix[i][j] = -10
                        self.valid_matrix[j][i] = -10
            for i in range(len(self.P_points), self.action_number):
                self.valid_action_num[i] = 1
                for j in range(self.action_number):
                    if self.valid_matrix[j][i] != -10:
                        self.valid_action_num[i] += 1
                    if [j, i] in additional_connection:
                        self.valid_action_num[i] += 1
                        self.valid_action_num[j] += 1
                        self.valid_matrix[i][j] = 0
                        self.valid_matrix[j][i] = 0
                    if [j, i] in banned_connection:
                        self.valid_action_num[i] -= 1
                        self.valid_action_num[j] -= 1
                        self.valid_matrix[i][j] = -10
                        self.valid_matrix[j][i] = -10
            
            for i in range(self.action_number):
                for j in range(self.action_number):
                    if self.valid_matrix[i][j] == 1:
                        self.show_matrix[i][j] = V
                    elif self.valid_matrix[i][j] == -1:
                        self.show_matrix[i][j] = M
                    elif self.valid_matrix[i][j] == -10:
                        self.show_matrix[i][j] = INVALID
                    elif self.valid_matrix[i][j] == -2:
                        self.show_matrix[i][j] = INVALID
                    else:
                        self.show_matrix[i][j] = UNSURE

            if print_result:
                plt.matshow(self.show_matrix, cmap="gist_ncar", vmin=0, vmax=1)
                # plt.xlabel("Unit i")
                # plt.ylabel("Unit j")
                ax = plt.gca()
                # ax.set_title('Action Matrix (Analytical)')
                ax.hlines(y=len(self.P_points)-0.5, xmin=-0.5, xmax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.vlines(x=len(self.P_points)-0.5, ymin=-0.5, ymax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.xaxis.set_label_position('top')
                plt.show()
            
            for i in range(self.action_number):
                for j in range(self.action_number):
                    if i == j and i >= len(self.P_points) and (i - len(self.P_points)) not in disabled_panels and self.valid_action_num[i] > 1:
                        self.show_matrix[i][j] = FIX
                        self.valid_matrix[i][j] = -2

            if print_result:
                plt.matshow(self.show_matrix, cmap="gist_ncar", vmin=0, vmax=1)
                # plt.xlabel("Unit i")
                # plt.ylabel("Unit j")
                ax = plt.gca()
                # ax.set_title('Action Matrix (Analytical)')
                ax.hlines(y=len(self.P_points)-0.5, xmin=-0.5, xmax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.vlines(x=len(self.P_points)-0.5, ymin=-0.5, ymax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.xaxis.set_label_position('top')
                plt.show()
            
        except:
            self.calculateValidActions()

            valid_actions_from_input = input_json["valid_actions"]
            # self.show_matrix = np.zeros(shape=(self.unit_number, self.unit_number))

            # for i in range(len(self.P_points), self.action_number):
            #     self.valid_action_num[i] = 1
            #     for j in range(len(self.P_points)):
            #         if self.valid_matrix[j][i] != -10:
            #             self.valid_action_num[i] += 1
            
            self.show_matrix = np.zeros(shape=(self.action_number, self.action_number))

            for i in range(len(self.P_points)):
                self.valid_action_num[i] = 0
            for i in range(len(self.P_points), self.action_number):
                self.valid_action_num[i] = 1
                for j in range(len(self.P_points)):
                    if self.valid_matrix[j][i] != -10:
                        self.valid_action_num[i] += 1
                    if [j, i] in additional_connection:
                        self.valid_action_num[i] += 1
                        self.valid_action_num[j] += 1
                        self.valid_matrix[i][j] = 0
                        self.valid_matrix[j][i] = 0

            self.simulator_show_matrix = np.zeros(shape=(self.action_number, self.action_number))
            
            for i in range(self.action_number):
                for j in range(self.action_number):
                    if valid_actions_from_input[i][j] == 1:
                        self.simulator_show_matrix[i][j] = V
                    elif valid_actions_from_input[i][j] == -1:
                        self.simulator_show_matrix[i][j] = M
                    else:
                        self.simulator_show_matrix[i][j] = INVALID
                    if i == j and i >= len(self.P_points):
                        if i - len(self.P_points) not in disabled_panels:
                            self.simulator_show_matrix[i][j] = INVALID
                        else:
                            self.simulator_show_matrix[i][j] = INVALID

            if print_result:
                plt.matshow(self.simulator_show_matrix, cmap="gist_ncar", vmin=0, vmax=1)
                # plt.xlabel("Unit i")
                # plt.ylabel("Unit j")
                ax = plt.gca()
                # ax.set_title('Action Matrix (SIM I)')
                ax.hlines(y=len(self.P_points)-0.5, xmin=-0.5, xmax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.vlines(x=len(self.P_points)-0.5, ymin=-0.5, ymax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.xaxis.set_label_position('top')
                plt.show()

            self.analytical_show_matrix = np.zeros(shape=(self.action_number, self.action_number))
            for i in range(self.action_number):
                for j in range(self.action_number):
                    if self.valid_matrix[i][j] == 1:
                        self.analytical_show_matrix[i][j] = V
                    elif self.valid_matrix[i][j] == -1:
                        self.analytical_show_matrix[i][j] = M
                    elif self.valid_matrix[i][j] == -10:
                        self.analytical_show_matrix[i][j] = INVALID
                    elif self.valid_matrix[i][j] == -2:
                        self.analytical_show_matrix[i][j] = INVALID
                    else:
                        self.analytical_show_matrix[i][j] = UNSURE

            if print_result:
                plt.matshow(self.analytical_show_matrix, cmap="gist_ncar", vmin=0, vmax=1)
                # plt.xlabel("Unit i")
                # plt.ylabel("Unit j")
                ax = plt.gca()
                # ax.set_title('Action Matrix (Analytical)')
                ax.hlines(y=len(self.P_points)-0.5, xmin=-0.5, xmax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.vlines(x=len(self.P_points)-0.5, ymin=-0.5, ymax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.xaxis.set_label_position('top')
                plt.show()

            for i in range(self.action_number):
                for j in range(self.action_number):
                    if (i - len(self.P_points)) in disabled_panels or (j - len(self.P_points)) in disabled_panels:
                        self.show_matrix[i][j] = INVALID
                        self.valid_matrix[i][j] = -10
                    else:
                        if abs(self.valid_matrix[i][j]) != 1:
                            if abs(valid_actions_from_input[i][j]) == 1:
                                if i < len(self.P_points) or j < len(self.P_points):
                                    self.valid_action_num[i] += 1
                                    # if (i < len(self.P_points) and self.P_points[i][Z] > 1e-5) or (j < len(self.P_points) and self.P_points[j][Z] > 1e-5):
                                    #     self.valid_matrix[i][j] = 1
                                    #     self.show_matrix[i][j] = 1.5
                                    # elif (i < len(self.P_points) and self.P_points[i][Z] < -1e-5) or (j < len(self.P_points) and self.P_points[j][Z] < -1e-5):
                                    #     self.valid_matrix[i][j] = -1
                                    #     self.show_matrix[i][j] = 0.3
                                    # else:
                                    #     self.valid_matrix[i][j] = 0
                                    #     self.show_matrix[i][j] = 0
                                    if valid_actions_from_input[i][j] == 1:
                                        self.valid_matrix[i][j] = 1
                                        self.show_matrix[i][j] = V
                                    elif valid_actions_from_input[i][j] == -1:
                                        self.valid_matrix[i][j] = -1
                                        self.show_matrix[i][j] = M
                                    else:
                                        self.valid_matrix[i][j] = 0
                                        self.show_matrix[i][j] = UNSURE  
                                else: # internal
                                    if self.valid_matrix[i][j] == 0:
                                        if valid_actions_from_input[i][j] == 1:
                                            self.show_matrix[i][j] = V
                                            self.valid_matrix[i][j] = 1
                                        elif valid_actions_from_input[i][j] == -1:
                                            self.show_matrix[i][j] = M
                                            self.valid_matrix[i][j] = -1
                                    else:
                                        self.show_matrix[i][j] = INVALID
                                        self.valid_matrix[i][j] = -10
                            else:
                                self.show_matrix[i][j] = INVALID
                                self.valid_matrix[i][j] = -10
                        else:
                            if abs(valid_actions_from_input[i][j]) == 1:
                                if (i < len(self.P_points) and self.P_points[i][Z] > 1e-5) or (j < len(self.P_points) and self.P_points[j][Z] > 1e-5):
                                    self.valid_matrix[i][j] = 1
                                elif (i < len(self.P_points) and self.P_points[i][Z] < -1e-5) or (j < len(self.P_points) and self.P_points[j][Z] < -1e-5):
                                    self.valid_matrix[i][j] = -1
                                if self.valid_matrix[i][j] == 1:
                                    self.show_matrix[i][j] = V
                                elif self.valid_matrix[i][j] == -1:
                                    self.show_matrix[i][j] = M
                                elif self.valid_matrix[i][j] == 0:
                                    if valid_actions_from_input[i][j] == 1:
                                        self.show_matrix[i][j] = V
                                        self.valid_matrix[i][j] = 1
                                    elif valid_actions_from_input[i][j] == -1:
                                        self.show_matrix[i][j] = M
                                        self.valid_matrix[i][j] = -1
                                self.valid_action_num[i] += 1
                            else:
                                self.show_matrix[i][j] = INVALID
                                self.valid_matrix[i][j] = -10
                    if i == j and i >= len(self.P_points) and (i - len(self.P_points)) not in disabled_panels:
                        self.show_matrix[i][j] = FIX
                        self.valid_matrix[i][j] = -2

            # for i in range(self.unit_number):
            #     for j in range(self.unit_number):
            #         if i in disabled_panels or j in disabled_panels:
            #             self.show_matrix[i][j] = 0.2
            #             self.valid_matrix[i + len(self.P_points)][j + len(self.P_points)] = -10
            #         else:
            #             if abs(self.valid_matrix[i + len(self.P_points)][j + len(self.P_points)]) != 1:
            #                 if valid_actions_from_input[i][j] == 1:
            #                     self.show_matrix[i][j] = 0.2
            #                 else:
            #                     self.show_matrix[i][j] = 0
            #             else:
            #                 if valid_actions_from_input[i][j] == 1:
            #                     self.show_matrix[i][j] = 1
            #                     self.valid_action_num[i + len(self.P_points)] += 1
            #                 else:
            #                     self.show_matrix[i][j] = 0.2
            #                     self.valid_matrix[i + len(self.P_points)][j + len(self.P_points)] = -10
            if print_result:
                plt.matshow(self.show_matrix, cmap="gist_ncar", vmin=0, vmax=1)
                # plt.xlabel("Unit i")
                # plt.ylabel("Unit j")
                ax = plt.gca()
                # ax.set_title('Action Matrix (Final)')
                ax.hlines(y=len(self.P_points)-0.5, xmin=-0.5, xmax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.vlines(x=len(self.P_points)-0.5, ymin=-0.5, ymax=self.action_number-0.5, colors=(0.94, 0.55, 0.0))
                ax.xaxis.set_label_position('top')
                plt.show()
                
                plt.matshow(self.valid_matrix, cmap="gist_ncar")
                plt.show()
        
        visited_id_list = [i for i in range(len(self.P_points))]
        depth = 1
        maximum_reach_number = self.action_number
        if system_type != 0:
            maximum_reach_number -= len(self.P_points)
        while len(visited_id_list) < maximum_reach_number:
            depth += 1
            new_id_list = []
            for ele in visited_id_list:
                for i in range(self.action_number):
                    if self.valid_matrix[ele][i] != -10 and i not in visited_id_list and i not in new_id_list:
                        new_id_list.append(i)
            visited_id_list += new_id_list
        upper_bound = max(upper_bound, depth) + upper_bound_bonus
        
        GOOVER_UNIT_LOWER_BOUND = (UNIT_PASS - discount) / UNIT_NUM
        RATIO_LOWER_BOUND = string_number / (string_number - 1.) * ((UNIT_PASS - discount) / (string_number * (upper_bound - 1)) - 1. / string_number) if string_number > 1 else 1.0
        dead_maximum = (string_number * upper_bound) * (UNIT_NUM)
        # dead_maximum = string_number * upper_bound
        
        self.maximum_pass_number = upper_bound #int(self.unit_number / self.string_number) + 1
        
        print(f"UNIT_PASS / UNIT_NUM : {(UNIT_PASS - discount)} / {UNIT_NUM}, upper bound: {upper_bound}, MIN_A / MIN_AA : {MIN_A} / {MIN_AA}, Curl Penalty: {enable_curl_penalty}")
                    
        self.sim_buf = []
        self.temp_sim_buf = []
        
        try:
            self.sim_buf = []
            if not calculate_number:
                path = os.path.join("./threadingResult", f"{BUF_NAME}.json")
                with open(path, 'r', encoding='utf-8') as fw:
                    self.sim_buf = json.load(fw)['buf']
                self.sim_buf = [ele for ele in self.sim_buf if ele['ra'] != 0.0]

        except:
            self.sim_buf = []
        
        if len(self.initial_center_list) != 0:
            self.maximum_ct = 0
            plt.rcParams.update({'font.size': 20})
            plt.rcParams['font.sans-serif'] = 'Arial'
            self.accurate_vector_mode = True
            fig = plt.figure()
            ax = aat.Subplot(fig, 111)
            fig.add_axes(ax)
            # ax.axis['bottom'].set_axisline_style("->", size=1.5)
            # ax.axis['left'].set_axisline_style("->", size=1.5)
            for i in range(len(self.initial_center_list)):
                length = distance(self.initial_center_list[i], self.current_center_list[i])
                if length > self.maximum_ct:
                    self.maximum_ct = length
                if length > 1.0:
                    ax.arrow(self.initial_center_list[i][X], self.initial_center_list[i][Y], \
                        self.current_center_list[i][X] - self.initial_center_list[i][X], self.current_center_list[i][Y] - self.initial_center_list[i][Y], \
                            head_width=0., head_length=0., fc=(0.5625, 0.94, 0.5625), ec=(0.5625, 0.94, 0.5625))

            for i in range(len(self.initial_center_list)):
                length = distance(self.initial_center_list[i], self.current_center_list[i])
                if length > 1.0:
                    ax.quiver(self.initial_center_list[i][X], self.initial_center_list[i][Y], \
                        self.current_center_list[i][X] - self.initial_center_list[i][X], self.current_center_list[i][Y] - self.initial_center_list[i][Y], \
                            color='black', scale=500, width=0.005)
                
                ax.scatter(self.current_center_list[i][X], self.current_center_list[i][Y], marker='o', color=(0.2157, 0.698, 0.302), s=16)
                ax.scatter(self.initial_center_list[i][X], self.initial_center_list[i][Y], marker='o', color=(0.2157, 0.698, 0.302), s=64)
            
            if print_result:    
                ax.set_aspect('equal')
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                ax.tick_params(labelsize=20, pad=15, width=1)
                plt.rcParams.update({'font.size': 28})
                plt.rcParams['font.sans-serif'] = 'Arial'
                plt.title('Displacement of threading holes', pad=18)
                
                # plt.grid(True)
                plt.show()
        else:
            self.accurate_vector_mode = False

        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.sans-serif'] = 'Arial'

        self.basic_trajectory_list = []
        self.backup_current_state_list = []
        self.backup_node_status_list = []
        self.backup_temp_done_list = []
        self.backup_current_string_id_list = []
        self.backup_current_point_num_list = []
        self.backup_valid_pass_list = []
        self.backup_old_stand_point_id_list = []
        self.backup_current_stand_point_id_list = []
        self.backup_previous_side_list = []
        
        # Create Root
        if not calculate_number:
            self.root_node = Node(0, len(self.P_points), 0, 0, 1)
        else:
            self.root_node = Node(deepcopy(self.current_state), len(self.P_points), 0, 0, 1)

        self.id = 0
        try:
            self.backup_csv = pd.read_csv(os.path.join(file_path, f'{NAME}-{GOOVER_UNIT_LOWER_BOUND}.csv'))
        except:
            self.backup_csv = None
        self.delta_time = 0
        createFolder()
        
        # self.action_number = max(max(self.valid_action_num), len(self.P_points))
    def initialize(self):
        if not calculate_number:
            self.root_node = Node(0, len(self.P_points), 0, 0, 1)
        else:
            self.root_node = Node(deepcopy(self.current_state), len(self.P_points), 0, 0, 1)

    def treePolicy(self, node: Node, scalar):
        global cut_num
        depth = 0
        while not node.done:
            
            if len(node.children) == 0:
                self.node_num += 1
                # if self.node_num % 1000 == 0:
                #     print(f"Create node with id: {self.node_num}")
                new_node = self.expand(node)
                if new_method:
                    if self.reward < 0:
                        level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                        new_node.reward = self.reward
                        # print(f"Node with level {level} is cut due to minus reward")
                    else:
                        return new_node, depth
                else:
                    return new_node, depth
                # return self.expand(node)
            # elif random.uniform(0, 1) > 1. and node.existBestChild():
            #     node, action = node.bestChild(scalar, self.best_reward / r_s)
            #     next_state, self.reward, done, truncated = self.step(action, 0, False)
            else:
                if node.done:
                    a = 1
                if not node.fullyExpanded():
                    self.node_num += 1
                    # if self.node_num % 1000 == 0:
                    #     print(f"Create node with id: {self.node_num}")
                    new_node = self.expand(node)
                    if new_method:
                        if self.reward < 0:
                            level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                            new_node.reward = self.reward
                        else:
                            return new_node, depth
                    else:
                        return new_node, depth
                else:
                    if cut_nodes:
                        if node.existBestChild((self.best_reward+1.)*0.5):
                            node, action = node.bestChild(scalar, (self.best_reward+1.)*0.5)
                            next_state, self.reward, done, truncated = self.step(action, 0, False)
                            if self.fake_step:
                                node.done = done
                                node.reward = fail_reward
                                node = node.parent
                                continue
                            else:
                                depth += 1
                        else:
                            level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                            cut_num += node.maximum_child
                            while len(node.children):
                                del(node.children[0])
                            node.children = None
                            # print(f"Node with level {level} is cut due to average reward {node.reward / (node.visits - 1)} < {self.best_reward}, cut_number: {cut_num}")
                            node.done = 1
                            
                            if depth > 0:
                                node = node.parent
                                depth -= 1
                                if node != None:
                                    self.deeppopup()
                                else:
                                    # print("Root node ends!")
                                    if node.children == None:
                                        a = 1
                                    return node, depth
                            else:
                                if node.parent != None:
                                    self.deeppopup()
                                return node, depth
                    else:
                        if new_method:
                            if node.existBestChild(0):
                                node, action = node.bestChild(scalar, 0)
                                depth += 1
                            else:                            
                                if depth > 0:
                                    node = node.parent
                                    depth -= 1
                                    if node != None:
                                        self.deeppopup()
                                    else:
                                        # print("Root node ends!")
                                        if node.children == None:
                                            a = 1
                                        return node, depth
                                else:
                                    if node.parent != None:
                                        self.deeppopup()
                                    if node.children == None:
                                        a = 1
                                    return node, depth
                        else:
                            node, action = node.bestChild(scalar, -math.inf)
                            depth += 1
                        next_state, self.reward, done, truncated = self.step(action, 0, False)
                        
        # if node.children == None:
        #     a = 1
        return node, depth
    
    def expand(self, node: Node):
        tried_children = [c.action_id for c in node.children]
        action_range = node.maximum_child
        notried_children = []
        for i in range(action_range):
            if i not in tried_children:
                notried_children.append(i)
        action = np.random.choice(np.array(notried_children), 1).item()
        next_state, self.reward, done, truncated = self.step(action, 0, False)
        if not calculate_number:
            node.addChild(
                self.current_stand_point_id, 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action, int(self.current_string_id)
            )
        else:
            node.addChild(
                deepcopy(next_state), 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action, int(self.current_string_id)
            )
        return node.children[-1]
    
    def expandAndStepInto(self, node: Node, next_state, action_id, done, string_id):
        tried_children = []
        
        for c in node.children:
            if action_id == c.action_id:
                return c
            tried_children.append(c.action_id)

        action = action_id
        if not calculate_number:
            node.addChild(
                self.current_stand_point_id, 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action, string_id
            )
        else:
            node.addChild(
                deepcopy(next_state), 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action, string_id
            )
        return node.children[-1]

    def backward(self, node, reward, num_visit):
        while node is not None:
            node.visits += num_visit
            node.reward += reward
            node = node.parent

    def calculateValidActions(self):
        for old_stand_point_id in range(self.action_number):
            for current_stand_point_id in range(old_stand_point_id, self.action_number):
                if (old_stand_point_id - len(self.P_points)) in disabled_panels or (current_stand_point_id - len(self.P_points)) in disabled_panels:
                    self.valid_matrix[old_stand_point_id][current_stand_point_id] = -10
                    self.valid_matrix[current_stand_point_id][old_stand_point_id] = -10
                    continue
                if old_stand_point_id < len(self.P_points) and current_stand_point_id < len(self.P_points):
                    self.valid_matrix[old_stand_point_id][current_stand_point_id] = -10
                    self.valid_matrix[current_stand_point_id][old_stand_point_id] = -10
                else:
                    old_stand_point = self.P_points[old_stand_point_id] if old_stand_point_id < len(self.P_points) else self.O_points[old_stand_point_id - len(self.P_points)]
                    new_stand_point = self.P_points[current_stand_point_id] if current_stand_point_id < len(self.P_points) else self.O_points[current_stand_point_id - len(self.P_points)]
                    intersection_ids, unsure_ids = self.calculateIntersectionWithCreases(old_stand_point, new_stand_point, self.lines)
                    valid = 0
                    if len(intersection_ids) > 0 and len(unsure_ids) == 0:
                        intersection_crease_type = [0, 0, 0]
                        for id in intersection_ids:
                            if self.lines[id].getLength() > 2:
                                intersection_crease_type[self.lines[id].getType()] += 1
                            else:
                                intersection_crease_type = [0, 0, 0]
                                break
                        # check valid
                        if intersection_crease_type[BORDER] > 0 and intersection_crease_type[BORDER] <= CREASE_CROSS and intersection_crease_type[MOUNTAIN] == 0 and intersection_crease_type[VALLEY] == 0:
                            valid = 1
                            side = 0
                        elif intersection_crease_type[BORDER] == 0 and intersection_crease_type[MOUNTAIN] > 0 and intersection_crease_type[MOUNTAIN] <= CREASE_CROSS and intersection_crease_type[VALLEY] == 0:
                            valid = 1
                            side = -1
                        elif intersection_crease_type[BORDER] == 0 and intersection_crease_type[MOUNTAIN] == 0 and intersection_crease_type[VALLEY] > 0 and intersection_crease_type[VALLEY] <= CREASE_CROSS:
                            valid = 1
                            side = 1
                    else:
                        if len(intersection_ids) == 0 and len(unsure_ids) >= 0:
                            valid = 1
                            side = 0
                        else:
                            valid = 0
                    if valid:
                        self.valid_matrix[old_stand_point_id][current_stand_point_id] = side
                        self.valid_matrix[current_stand_point_id][old_stand_point_id] = side
                        self.valid_action_num[old_stand_point_id] += 1
                        self.valid_action_num[current_stand_point_id] += 1
                    else:
                        self.valid_matrix[old_stand_point_id][current_stand_point_id] = -10
                        self.valid_matrix[current_stand_point_id][old_stand_point_id] = -10
                    if old_stand_point_id == current_stand_point_id:
                        self.valid_matrix[old_stand_point_id][current_stand_point_id] = -2
                        self.valid_action_num[old_stand_point_id] += 1

    def packTrajectory(self, trajectories):
        length = len(trajectories)
        dict = {
            "type": [[] for _ in range(length)],
            "id": [[] for _ in range(length)],
            "reverse": [[] for _ in range(length)]
        }
        for k in range(length):
            trajectory = trajectories[k]
            if len(trajectory) == 0:
                break
            for i in range(len(trajectory)):
                if trajectory[i][0] < len(self.P_points):
                    dict["type"][k].append("A")
                    dict["id"][k].append(trajectory[i][0])
                else:
                    dict["type"][k].append("B")
                    dict["id"][k].append(trajectory[i][0] - len(self.P_points))
            exist_non_zero_side = 0
            for i in range(len(trajectory)):
                if trajectory[i][1] != 0:
                    exist_non_zero_side = trajectory[i][1]
                    break
            if exist_non_zero_side:
                for j in range(len(trajectory)):
                    if (j - i) % 2 == 0:
                        dict["reverse"][k].append(int(trajectory[i][1]))
                    else:
                        dict["reverse"][k].append(int(-trajectory[i][1]))
                dict["reverse"][k][0] = int(dict["reverse"][k][1])
            else:
                for j in range(len(trajectory)):
                    if j % 2 == 0:
                        dict["reverse"][k].append(-1)
                    else:
                        dict["reverse"][k].append(1)
                dict["reverse"][k][0] = 1
        return dict
    
    def unpackTrajectory(self, dict):
        length = len(dict["type"])
        trajectories = []

        for k in range(length):
            types   = dict["type"][k]
            ids     = dict["id"][k]
            reverse = dict["reverse"][k]

            # 空轨迹（对应正向 break 的情况）
            if len(types) == 0:
                trajectories.append([])
                continue

            trajectory = []
            for i in range(len(types)):
                # 还原 node_index
                if types[i] == "A":
                    node_index = ids[i]
                    side = 0
                else:  # "B"
                    node_index = ids[i] + len(self.P_points)

                    # 还原 side（直接使用 reverse 值）
                    side = reverse[i]

                trajectory.append([node_index, side])

            trajectories.append(trajectory)

        return trajectories
    
    def calculateIntersectionWithCreases(self, P_choice, O_choice, creases):
        ids = []
        unsure_ids = []
        for i in range(len(creases)):
            # if creases[i].getLength() < 4:
            #     continue
            crease_start_point = creases[i][START]
            crease_end_point = creases[i][END]
            vec1_2D = np.array([crease_end_point[X] - crease_start_point[X], crease_end_point[Y] - crease_start_point[Y]])
            vec1_relevant1_2D = np.array([P_choice[X] - crease_start_point[X], P_choice[Y] - crease_start_point[Y]])
            vec1_relevant2_2D = np.array([O_choice[X] - crease_start_point[X], O_choice[Y] - crease_start_point[Y]])
            result1 = np.cross(vec1_2D, vec1_relevant1_2D).item() * np.cross(vec1_2D, vec1_relevant2_2D).item()
            vec2_2D = np.array([O_choice[X] - P_choice[X], O_choice[Y] - P_choice[Y]])
            vec2_relevant1_2D = np.array([crease_start_point[X] - P_choice[X], crease_start_point[Y] - P_choice[Y]])
            vec2_relevant2_2D = np.array([crease_end_point[X] - P_choice[X], crease_end_point[Y] - P_choice[Y]])
            result2 = np.cross(vec2_2D, vec2_relevant1_2D).item() * np.cross(vec2_2D, vec2_relevant2_2D).item()
            if result1 < -1e-3 and result2 < -1e-3:
                ids.append(i)
            elif (abs(result1) < 1e-3 and result2 < 0) or (abs(result2) < 1e-3 and result1 < 0):
                unsure_ids.append(i)
        return ids, unsure_ids

    def reset(self):
        self.current_state = np.array([0. for _ in range(self.state_number)])
        self.current_stand_point_id = -1
        self.old_stand_point_id = -1.
        self.current_valid_pass = 0
        self.current_trajectory = [[] for _ in range(self.string_number)]
        self.new_trajectory = [[] for _ in range(self.string_number)]
        # self.avg_reward = [self.fail_reward for _ in range(100)]

        self.node_status = np.array([0 for _ in range(self.status_number)])

        self.current_string_id = 1.
        self.current_point_num = 0
        self.previous_side = 0
        self.temp_done = 1
        self.done = 0
        self.reward = 0.

        self.basic_trajectory_list = []
        self.backup_current_state_list = []
        self.backup_node_status_list = []
        self.backup_temp_done_list = []
        self.backup_current_string_id_list = []
        self.backup_current_point_num_list = []
        self.backup_valid_pass_list = []
        self.backup_old_stand_point_id_list = []
        self.backup_current_stand_point_id_list = []
        self.backup_previous_side_list = []
        self.backup_id_list = []

        # self.current_state[0] = 6.
        # self.current_stand_point_id = 5
        # self.old_stand_point_id = 5
        # self.current_trajectory[0].append((5, 0))
        # self.node_status[5] = 1
        # self.current_point_num += 1
        # self.temp_done = 0
        self.deepbackup()
        return np.append(self.current_state, self.current_string_id), 0
    
    def sample(self, precise=False, invalid_list=[]):
        if not precise:
            return np.random.choice(self.action_number, 1).item()
        else:
            if self.temp_done:
                return np.random.choice(len(self.P_points), 1).item()
            else:
                valid_number = self.valid_action_num[self.current_stand_point_id]
                prob = np.zeros(valid_number)
                per_prob = 1. / (valid_number - len(invalid_list)) if valid_number > len(invalid_list) else 0.
                for i in range(valid_number):
                    if i in invalid_list:
                        prob[i] = 0.
                    else:
                        prob[i] = per_prob
                return np.random.choice(valid_number, 1, p=prob).item()

    def getRange(self):
        if self.temp_done:
            return len(self.P_points)
        else:
            return self.valid_action_num[self.current_stand_point_id]
        
    def equalMethod(self, method1, method2):
        equal = False
        for id1 in method1:
            equal = False
            for id2 in method2:
                if id1 == id2:  
                    equal = True
                    break
            if not equal:
                break
        equal1 = equal
        for id2 in method2:
            equal = False
            for id1 in method1:
                if id1 == id2:  
                    equal = True
                    break
            if not equal:
                break
        return equal and equal1
    
    # def existInBuffer(self, method):
    #     for k in range(len(self.existing_id_reward)):
    #         if self.equalMethod(method["id"], self.existing_id_reward[k]["id"]):
    #             return k
    #     return -1

    def output(self, episode, method, reward):
        if self.output_reward_buffer and not calculate_number:
            self.scores.append(reward)
            self.steps.append(episode)
            total_string = deepcopy(method)
            total_string["score"] = reward

            try:
                with open(os.path.join(file_path, "result_step_" + str(episode) + "_score_" + str(round(reward, 2))) + ".json", 'w', encoding="utf-8") as f:
                    json.dump(total_string, f, indent=4)
            except:
                pass
            
            score_list = {
                "score": self.scores,
                "steps": self.steps
            }

            try:
                with open(os.path.join(file_path, "score.json"), 'w', encoding="utf-8") as f:
                    json.dump(score_list, f, indent=4)
            except:
                pass

    # def getSimReward(self, method):
    #     ori_sim = OrigamiSimulator(use_gui=False)

    #     ori_sim.string_total_information = methodToTotalInformation(method, self.P_points, self.O_points)
    #     ori_sim.pref_pack = {
    #         "tsa_resolution": len(self.P_points),
    #         "tsa_radius": self.panel_size
    #     }

    #     ori_sim.startOnlyTSA(self.units, self.max_size, self.total_bias, self.max_edge)
    #     ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
    #     ori_sim.initializeRunning()
        
    #     step = 1
    #     while 1:
    #         ori_sim.step()
    #         if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 1000 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
    #             break
    #         if step % 225 == 0:
    #             if abs(ori_sim.recorded_folding_percent[-1]) < 1e-2:
    #                 ori_sim.can_rotate = False
    #                 break
    #         step += 1
            
    #     if not ori_sim.can_rotate:
    #         print("Simulation done, fail reward")
    #         reward = self.fail_reward
    #     else:
    #         folding_percent = ori_sim.recorded_folding_percent[-1]
    #         folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])

    #         value = folding_speed / (0.9 - folding_percent)
    #         print(f"Simulation done, reward: {value}")
            
    #         reward = value
    #     return reward
    
    # def getReward(self, trajectory, episode, bonus = 1.):
    #     reward = 0.0
    #     method = self.packTrajectory(trajectory)
    #     # print(method)

    #     index = self.existInBuffer(method)
    #     if index >= 0:
    #         # print("existing reward")
    #         reward = self.existing_id_reward[index]["reward"]
    #         self.output(episode, method, reward)
    #         return reward
    
    #     # reward = self.getSimReward(method)
    #     reward = bonus_val * sum([len(trajectory[i]) for i in range(len(trajectory))])
    #     self.output(episode, method, reward)
        
    #     if self.output_reward_buffer:
    #         self.existing_id_reward.append({
    #             "id": deepcopy(method["id"]),
    #             "reward": reward
    #         })
    #         try:
    #             with open(os.path.join(file_path, "simulation_buffer.json"), 'w', encoding="utf-8") as f:
    #                 json.dump({
    #                     "id_reward": self.existing_id_reward
    #                 }, f, indent=4)
    #         except:
    #             pass

    #     return reward

    def getStateIndex(self):
        return self.current_point_num + int(self.current_string_id - 1) * (self.unit_number + 2)
    
    def beginRouting(self):
        self.temp_done = 0
        self.current_valid_pass = 0

    def updateStateAndTrajectory(self, action, dir, origin_action):
        '''
        action: 选择点的id\n
        dir: 当前选择点的id和上一个id连接后形成的绳段在折纸平面上的方向，0表示未知，一般是action是外部点的时候，因为绳段还没形成（没有上一个id），-1表示在下方，1表示在上方\n
        origin_action: 已弃用
        '''
        self.current_stand_point_id = action
        self.current_state[self.getStateIndex()] = action + 1

        if dir != -2:
            self.current_point_num += 1
            if self.node_status[action] != -1:
                self.node_status[action] += 1
            self.current_trajectory[(int(self.current_string_id) - 1)].append((action, dir))
        else:
            self.node_status[action] = -1
        
    def getActionIndex(self, action_id):
        if self.current_stand_point_id == -1:
            return action_id, 0
        pointer = -1
        valid_vector = self.valid_matrix[self.current_stand_point_id]
        for i in range(self.status_number):
            if valid_vector[i] != -10:
                pointer += 1
                if pointer == action_id:
                    return i, valid_vector[i]
        return -1, -10

    def endString(self):
        self.temp_done = 1
        self.current_string_id += 1.
        self.current_point_num = 0

    def backup(self):
        self.basic_trajectory = deepcopy(self.current_trajectory)
        self.backup_current_state = deepcopy(self.current_state)
        self.backup_node_status = deepcopy(self.node_status)
        self.backup_temp_done = self.temp_done
        self.backup_current_string_id = self.current_string_id
        self.backup_current_point_num = self.current_point_num
        self.backup_valid_pass = self.current_valid_pass
        self.backup_old_stand_point_id = self.old_stand_point_id
        self.backup_current_stand_point_id = self.current_stand_point_id
        self.backup_previous_side = self.previous_side
        self.backup_id = self.id

        # self.all_backup_list = deepcopy([
        #     self.basic_trajectory_list,
        #     self.backup_current_state_list,
        #     self.backup_node_status_list,
        #     self.backup_temp_done_list,
        #     self.backup_current_string_id_list,
        #     self.backup_current_point_num_list,
        #     self.backup_valid_pass_list,
        #     self.backup_old_stand_point_id_list,
        #     self.backup_current_stand_point_id_list,
        #     self.backup_previous_side_list,
        #     self.backup_id_list
        # ])
    
    def deepbackup(self):
        self.basic_trajectory_list.append(deepcopy(self.current_trajectory))
        self.backup_current_state_list.append(deepcopy(self.current_state))
        self.backup_node_status_list.append(deepcopy(self.node_status))
        self.backup_temp_done_list.append(self.temp_done)
        self.backup_current_string_id_list.append(self.current_string_id)
        self.backup_current_point_num_list.append(self.current_point_num)
        self.backup_valid_pass_list.append(self.current_valid_pass)
        self.backup_old_stand_point_id_list.append(self.old_stand_point_id)
        self.backup_current_stand_point_id_list.append(self.current_stand_point_id)
        self.backup_previous_side_list.append(self.previous_side)
        self.backup_id_list.append(self.id)
    
    def deeppopup(self):
        del(self.basic_trajectory_list[-1])
        del(self.backup_current_state_list[-1])
        del(self.backup_node_status_list[-1])
        del(self.backup_temp_done_list[-1])
        del(self.backup_valid_pass_list[-1])
        del(self.backup_current_string_id_list[-1])
        del(self.backup_current_point_num_list[-1])
        del(self.backup_old_stand_point_id_list[-1])
        del(self.backup_current_stand_point_id_list[-1])
        del(self.backup_previous_side_list[-1])
        del(self.backup_id_list[-1])
        self.current_trajectory = deepcopy(self.basic_trajectory_list[-1])
        self.current_state = deepcopy(self.backup_current_state_list[-1])
        self.node_status = deepcopy(self.backup_node_status_list[-1])
        self.temp_done = self.backup_temp_done_list[-1]
        self.current_valid_pass = self.backup_valid_pass_list[-1]
        self.current_string_id = self.backup_current_string_id_list[-1]
        self.current_point_num = self.backup_current_point_num_list[-1]
        self.old_stand_point_id = self.backup_old_stand_point_id_list[-1]
        self.current_stand_point_id = self.backup_current_stand_point_id_list[-1]
        self.previous_side = self.backup_previous_side_list[-1]
        self.id = self.backup_id_list[-1]


    
    def takeFixedActions(self, fixed_action_list):
        while 1:
            old_string_id = self.current_string_id
            for action_id in fixed_action_list[int(self.current_string_id)]:
                next_state, reward, done, _ = self.step(action_id, 0, False)
                if not self.fake_step:
                    env.root_node = self.expandAndStepInto(env.root_node, next_state, action_id, done, int(env.current_string_id))
            if self.current_string_id <= old_string_id:
                break

    def popup(self):
        self.current_trajectory = deepcopy(self.basic_trajectory)
        self.current_state = deepcopy(self.backup_current_state)
        self.node_status = deepcopy(self.backup_node_status)
        self.temp_done = self.backup_temp_done
        self.current_valid_pass = self.backup_valid_pass
        self.current_string_id = self.backup_current_string_id
        self.current_point_num = self.backup_current_point_num
        self.old_stand_point_id = self.backup_old_stand_point_id
        self.current_stand_point_id = self.backup_current_stand_point_id
        self.previous_side = self.backup_previous_side
        self.id = self.backup_id

    def rollout(self, number, front_node):
        self.backup()

        methods = []
        invalid_list = []
        
        ensure_valid = 0
        i = 0

        backup_number = number

        while i < number:
            self.fake_step = 0
            need_to_check_front_node = True
            while 1:
                if not self.fake_step:
                    if self.temp_done:
                        choose_time = np.zeros(len(self.P_points))
                        impossible_action = []
                    else:
                        choose_time = np.zeros(self.valid_action_num[self.current_stand_point_id])
                        impossible_action = []
                
                if self.valid_action_num[self.current_stand_point_id] == 0:
                    invalid_list.append(0)
                    break

                action = self.sample(True, impossible_action)

                if need_to_check_front_node:
                    exist, child = front_node.existChildWithAction(action)
                    if exist:
                        node_done = child.done
                    else:
                        need_to_check_front_node = False

                _, reward, done, _ = self.step(action, 0, False)

                if need_to_check_front_node and node_done != done: 
                    self.deeppopup()
                    self.fake_step = 1
                    choose_time[action] = self.fail_reward
                    impossible_action.append(action)
                else:
                    choose_time[action] = reward
                    impossible_action.append(action)
                    if need_to_check_front_node:
                        front_node = child

                if (done and reward > 0) or (choose_time < 0).all():
                    if (choose_time < 0).all():
                        invalid_list.append(0)
                    else:
                        invalid_list.append(1)
                    break
            
            methods.append(deepcopy(self.current_trajectory))
            
            reward_list = self.getReward_Initial([methods[i]], [invalid_list[i]])
            if reward_list[0] == base_reward + 1.0:
                ensure_valid += 1
            
            if i == number - 1 and ensure_valid < thread_number:
                number += 1
            
            if number >= thread_number * backup_number or ensure_valid == thread_number:
                self.popup()
                break

            self.popup()
            i += 1
        
        return methods, invalid_list, number

    def getStringNumber(self, trajectories):
        string_number_correct = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            string_number_correct_temp = 1.
            for trajectory in strings:
                if len(trajectory) == 0:
                    string_number_correct_temp = 0.
                    break
            string_number_correct[h] = string_number_correct_temp
        return string_number_correct

    def calculateConflictRouting(self, methods):
        valid = [True for _ in range(len(methods))]
        for h in range(len(methods)):
            strings = methods[h]
            for i in range(len(strings)):
                line_use = [0 for _ in range(len(self.lines))]
                for j in range(len(strings[i]) - 1):          
                    current_point_id = strings[i][j][0]
                    next_point_id = strings[i][j + 1][0]
                    # if current_point_id == 2 and next_point_id == 6 and string_length == 2:
                    #     a = 1
                    previous_center = self.P_points[current_point_id] if current_point_id < len(self.P_points) else self.O_points[current_point_id - len(self.P_points)]
                    current_center = self.P_points[next_point_id] if next_point_id < len(self.P_points) else self.O_points[next_point_id - len(self.P_points)]
                    
                    for kk in range(len(self.lines)):
                        kp1 = self.lines[kk][START]
                        kp2 = self.lines[kk][END]
                        v1 = [kp2[X] - previous_center[X], kp2[Y] - previous_center[Y]]
                        v2 = [kp1[X] - previous_center[X], kp1[Y] - previous_center[Y]]
                        v3 = [kp2[X] - current_center[X], kp2[Y] - current_center[Y]]
                        v4 = [kp1[X] - current_center[X], kp1[Y] - current_center[Y]]
                        val = (v1[X] * v2[Y] - v2[X] * v1[Y]) * (v3[X] * v4[Y] - v4[X] * v3[Y])
                        val2 = (v1[X] * v3[Y] - v3[X] * v1[Y]) * (v2[X] * v4[Y] - v4[X] * v2[Y])
                        
                        if val < 1e-3 and val2 < 1e-3:
                            if line_use[kk] == 0:
                                line_use[kk] = strings[i][j + 1][1]
                            else:
                                if line_use[kk] != strings[i][j + 1][1]:
                                    #up/down routing
                                    valid[h] = False
                                    break
                    
                    if not valid[h]:
                        break
                if not valid[h]:
                    break    
        return valid

    def getXYDiversion(self, method):
        x_contribution = 0.0
        y_contribution = 0.0
        trajectory_number = len(method["id"])
        
        for i in range(trajectory_number):
            # string_length = len(method["id"][i])
            if method["type"][i][0] == 'A':
                current_point_id = method["id"][i][0]
                next_point_id = method["id"][i][1]
                current_point = self.P_points[current_point_id]
                next_point = self.O_points[next_point_id]
                v = current_point - next_point
                x_contribution += v[X] / np.linalg.norm(v)
                y_contribution += v[Y] / np.linalg.norm(v)
            if method["type"][i][-1] == 'A':
                current_point_id = method["id"][i][-2]
                next_point_id = method["id"][i][-1]
                current_point = self.O_points[current_point_id]
                next_point = self.P_points[next_point_id]
                v = next_point - current_point
                x_contribution += v[X] / np.linalg.norm(v)
                y_contribution += v[Y] / np.linalg.norm(v)
        return x_contribution, y_contribution
    
    def getCurl(self, method):
        curl = 0.0
        x_contribution = 0.0
        y_contribution = 0.0
        trajectory_number = len(method["id"])
        
        for i in range(trajectory_number):
            # string_length = len(method["id"][i])
            if method["type"][i][0] == 'A':
                current_point_id = method["id"][i][0]
                next_point_id = method["id"][i][1]
                current_point = self.P_points[current_point_id]
                next_point = self.O_points[next_point_id]
                v = current_point - next_point
                v2 = self.mid_point - next_point
                curl += (v[X] * v2[Y] - v2[X] * v[Y]) / np.linalg.norm(v)

                x_contribution += v[X] / np.linalg.norm(v)
                y_contribution += v[Y] / np.linalg.norm(v)
            if method["type"][i][-1] == 'A':
                current_point_id = method["id"][i][-2]
                next_point_id = method["id"][i][-1]
                current_point = self.O_points[current_point_id]
                next_point = self.P_points[next_point_id]
                v = next_point - current_point
                v2 = self.mid_point - current_point
                curl += (v[X] * v2[Y] - v2[X] * v[Y]) / np.linalg.norm(v)

                x_contribution += v[X] / np.linalg.norm(v)
                y_contribution += v[Y] / np.linalg.norm(v)
        return abs(curl) / self.maximum_distance_to_mid_point, x_contribution, y_contribution
    
    def getDiversion(self, trajectories):
        div = [0. for _ in range(len(trajectories))]
        if double_div:
            double_div_list = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            # external_div = []
            for trajectory in strings:
                string_length = len(trajectory)
                div_one_string = 0.0
                double_div_one_string = 0.0
                for i in range(string_length - 1):
                    current_point_id = trajectory[i][0]
                    next_point_id = trajectory[i + 1][0]
                    # if current_point_id == 2 and next_point_id == 6 and string_length == 2:
                    #     a = 1
                    current_point = self.P_points[current_point_id] if current_point_id < len(self.P_points) else self.O_points[current_point_id - len(self.P_points)]
                    next_point = self.P_points[next_point_id] if next_point_id < len(self.P_points) else self.O_points[next_point_id - len(self.P_points)]
                                        
                    if i != 0:
                        if self.accurate_vector_mode:
                            v1 = self.initial_center_list[current_point_id - len(self.P_points)] - self.current_center_list[current_point_id - len(self.P_points)]
                            v2 = next_point - current_point
                            div_one_string += v1.dot(v2) / np.linalg.norm(v2) / self.maximum_ct
                            if double_div:
                                double_current_point = self.P_points[current_point_id] if current_point_id < len(self.P_points) else current_point - v1
                                double_next_point = self.P_points[next_point_id] if next_point_id < len(self.P_points) else next_point - v1
                                v2 = double_next_point - double_current_point
                                double_div_one_string += v1.dot(v2) / np.linalg.norm(v2) / self.maximum_ct
                        else:
                            v1 = np.array([current_point[X] - self.mid_x, current_point[Y] - self.mid_y])
                            v2 = np.array([next_point[X] - current_point[X], next_point[Y] - current_point[Y]])
                            div_one_string += v1.dot(v2) / np.linalg.norm(v2) / np.linalg.norm(v1)
                            # if i == 1 or (i == string_length - 2 and next_point_id <= len(self.P_points)):
                            #     external_div.append([div_one_string[X], div_one_string[Y]])
                            # div[h] += 1 if v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2) >= 0 else -1
                    if i != string_length - 2 or (i == string_length - 2 and next_point_id >= len(self.P_points)):
                        if self.accurate_vector_mode:
                            v1 = self.initial_center_list[next_point_id - len(self.P_points)] - self.current_center_list[next_point_id - len(self.P_points)]
                            v2 = -(next_point - current_point)
                            div_one_string += v1.dot(v2) / np.linalg.norm(v2) / self.maximum_ct
                            if double_div:
                                double_current_point = self.P_points[current_point_id] if current_point_id < len(self.P_points) else current_point - v1
                                double_next_point = self.P_points[next_point_id] if next_point_id < len(self.P_points) else next_point - v1
                                v2 = -(double_next_point - double_current_point)
                                double_div_one_string += v1.dot(v2) / np.linalg.norm(v2) / self.maximum_ct
                        else:
                            v1 = np.array([next_point[X] - self.mid_x, next_point[Y] - self.mid_y])
                            v2 = -np.array([next_point[X] - current_point[X], next_point[Y] - current_point[Y]])
                            div_one_string += v1.dot(v2) / np.linalg.norm(v2) / np.linalg.norm(v1)
                            # if i == 0 or (i == string_length - 2 and next_point_id > len(self.P_points)):
                            #     external_div.append([div_one_string[X], div_one_string[Y]])

                            # div[h] += 1 if v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2) >= 0 else -1
                if div_one_string > 0:
                    div[h] = div_one_string
                    break
                if double_div and double_div_one_string > 0:
                    double_div_list[h] = double_div_one_string
                    break
                else:
                    div[h] += div_one_string
                    if double_div:
                        double_div_list[h] += double_div_one_string
        if double_div:
            div = [max(div[i], double_div_list[i]) for i in range(len(trajectories))]
        return div

    def getReplicatedPoint(self, trajectories):
        ratio = [0. for _ in range(len(trajectories))]
        go_over_all_units = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            duplicated_number = [0 for _ in range(len(self.P_points) + self.unit_number)]
            total_point_number = 0
            for trajectory in strings:
                string_length = len(trajectory)
                total_point_number += string_length
                if trajectory[0][0] < len(self.P_points):
                    total_point_number -= 1
                if trajectory[-1][0] < len(self.P_points):
                    total_point_number -= 1
                for i in range(string_length):
                    duplicated_number[trajectory[i][0]] += 1
            coef = sum([1 if duplicated_number[i] > 0 else 0 for i in range(len(self.P_points), len(duplicated_number))])
            go_over_all_units[h] = (coef / (self.unit_number)) ** GOOVER_UNIT_FACTOR
            x = coef / total_point_number
            if self.string_number == 1:
                ratio[h] = 1
            else:
                ratio[h] = (self.string_number / (self.string_number - 1.) * (x - 1. / self.string_number)) ** DUPLICATE_FACTOR
        
        return ratio, go_over_all_units

    def getValidCross(self, trajectories):
        ratio = [0 for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            duplicated_number = [0 for _ in range(len(self.P_points) + self.unit_number)]
            total_point_number = 0
            for trajectory in strings:
                string_length = len(trajectory)
                total_point_number += string_length
                if trajectory[0][0] < len(self.P_points):
                    total_point_number -= 1
                if trajectory[-1][0] < len(self.P_points):
                    total_point_number -= 1
                for i in range(string_length):
                    duplicated_number[trajectory[i][0]] += 1
                    
            # TODO: 交叉判定
            twice_ids = []
            problem_pairs = [
                [] for _ in range(len(strings))
            ]
            for i in range(len(duplicated_number)):
                if duplicated_number[i] >= 2:
                    twice_ids.append(i)
            
            #找出可能存在交叠的线段组合    
            for i in range(len(strings)):
                string = strings[i]
                for j in range(len(string) - 1):
                    if string[j][0] in twice_ids and string[j + 1][0] in twice_ids:
                        problem_pair = [string[j][0], string[j + 1][0], -1, -1]
                        if j >= 1:
                            problem_pair[2] = string[j - 1][0]
                        if j <= len(string) - 3:
                            problem_pair[3] = string[j + 2][0]
                        problem_pairs[i].append(problem_pair)
            
            #遍历判断交叠合法性
            success = True
            for i in range(len(strings)):
                for j in range(len(problem_pairs[i])):
                    pair1 = problem_pairs[i][j]
                    for k in range(i + 1, len(strings)):
                        for l in range(len(problem_pairs[k])):
                            pair2 = problem_pairs[k][l]
                            
                            matched = False
                            if (pair1[1] == pair2[1] and pair1[0] == pair2[0]):
                                previous_id1 = pair1[2]
                                next_id1 = pair1[3]
                                previous_id2 = pair2[2]
                                next_id2 = pair2[3]
                                string_id1 = pair1[0]
                                string_id2 = pair1[1]
                                matched = True
                            elif (pair1[1] == pair2[0] and pair1[0] == pair2[1]):
                                previous_id1 = pair1[2]
                                next_id1 = pair1[3]
                                previous_id2 = pair2[3]
                                next_id2 = pair2[2]
                                string_id1 = pair1[0]
                                string_id2 = pair1[1]
                                matched = True
                            if matched:
                                real_point_string1 = np.array(self.P_points[string_id1] if string_id1 < len(self.P_points) else self.initial_center_list[string_id1 - len(self.P_points)])
                                real_point_string2 = np.array(self.P_points[string_id2] if string_id2 < len(self.P_points) else self.initial_center_list[string_id2 - len(self.P_points)])
                                vec = real_point_string2 - real_point_string1
                                #判据：两个绳段的前后id均在绳的一侧，但两个绳段不能同时在同一侧
                                if previous_id1 != -1:
                                    real_point_previous1 = np.array(self.P_points[previous_id1] if previous_id1 < len(self.P_points) else self.initial_center_list[previous_id1 - len(self.P_points)])
                                    real_point_previous2 = np.array(self.P_points[previous_id2] if previous_id2 < len(self.P_points) else self.initial_center_list[previous_id2 - len(self.P_points)])
                                    vec1 = (real_point_previous1 - real_point_string1) / np.linalg.norm(real_point_previous1 - real_point_string1)
                                    val_vec1 = vec1 - vec1.dot(vec) / np.linalg.norm(vec) ** 2 * vec
                                    vec2 = (real_point_previous2 - real_point_string1) / np.linalg.norm(real_point_previous2 - real_point_string1)
                                    val_vec2 = vec2 - vec2.dot(vec) / np.linalg.norm(vec) ** 2 * vec
                                    val = val_vec1.dot(val_vec2)
                                    if val > 0.1:
                                        success = False
                                        break
                                if next_id1 != -1:
                                    real_point_next1 = np.array(self.P_points[next_id1] if next_id1 < len(self.P_points) else self.initial_center_list[next_id1 - len(self.P_points)])
                                    real_point_next2 = np.array(self.P_points[next_id2] if next_id2 < len(self.P_points) else self.initial_center_list[next_id2 - len(self.P_points)])
                                    vec3 = (real_point_next1 - real_point_string2) / np.linalg.norm(real_point_next1 - real_point_string2)
                                    val_vec3 = vec3 - vec3.dot(vec) / np.linalg.norm(vec) ** 2 * vec
                                    vec4 = (real_point_next2 - real_point_string2) / np.linalg.norm(real_point_next2 - real_point_string2)
                                    val_vec4 = vec4 - vec4.dot(vec) / np.linalg.norm(vec) ** 2 * vec
                                    val2 = val_vec3.dot(val_vec4)
                                    if val2 > 0.1:
                                        success = False
                                        break
                                if previous_id1 != -1 and next_id1 != -1:
                                    val3 = np.cross(val_vec1 - vec, val_vec2 - vec).dot(np.cross(val_vec3 + vec, val_vec4 + vec))
                                    if val3 > 0:
                                        success = False
                                        break
                        if not success:
                            break
                    if not success:
                        break
                if not success:
                    break
            if not success:
                ratio[h] = 0
            else:
                ratio[h] = 1
        return ratio
                      
                    
    def getAList(self, trajectories):
        As = [0 for _ in range(len(trajectories))]
        AAs = [0 for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            duplicated_number = [0 for _ in range(len(self.P_points))]
            a_number = 0

            for k in range(len(strings)):
                trajectory = strings[k]
                if len(trajectory) > 0:
                    if trajectory[0][0] < len(self.P_points):
                        duplicated_number[trajectory[0][0]] = 1
                        a_number += 1
                    if trajectory[-1][0] < len(self.P_points):
                        duplicated_number[trajectory[-1][0]] = 1
                        a_number += 1
                    if self.system_type == 0:
                        # if trajectory[1][1] != 1 or trajectory[-1][1] != -1:
                        if (ea_constraint_initial_segment and trajectory[1][1] != 1) or \
                            (ea_constraint_final_tip and trajectory[-1][1] != -1):
                            a_number = 0
                            if trajectory[0][0] < len(self.P_points):
                                duplicated_number[trajectory[0][0]] = 0
                            if trajectory[-1][0] < len(self.P_points):
                                duplicated_number[trajectory[-1][0]] = 0
            
            As[h] = sum(duplicated_number)
            AAs[h] = a_number
        return [1. if As[h] >= MIN_A else 0 for h in range(len(trajectories))], [1. if AAs[h] >= MIN_AA else 0 for h in range(len(trajectories))]

    def getEqualLength(self, trajectories):
        ratio = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            string_length = np.array([len(trajectory) for trajectory in strings])
            std = string_length.std()
            # std = string_length.max() - string_length.min()
            # ratio[h] = 1. - std / len(self.O_points)
            ratio[h] = 1. - 0.7071 * std / (0.5 * self.unit_number + 1)
        return ratio

    def getRewardForOneControlChain(self, chain):
        reward = (-np.array(self.getDiversion([[chain]])))[0]
        return reward
    
    def getMeanAndStdForOneControlChain(self, chain):
        array = np.array([chain[i][0] for i in range(len(chain))])
        return array.mean(), array.std()
    
    def calculateBufferId(self, method, consider_A=True):
        start = 0
        if not consider_A:
            start = 1
        if id_calculator == 'char':
            id = ''
            for i in range(len(method)):
                for j in range(start, len(method[i])):
                    id += f'{method[i][j][0]}'
        else:
            id = 0
            interval = self.action_number
            for i in range(len(method)):
                for j in range(start, len(method[i])):
                    id += 2 ** (i * interval + method[i][j][0])
        return id

    def getReward_Initial(self, methods, invalid_list):
        string_number_correct_list = self.getStringNumber(methods)
        true_reward_list = np.zeros_like(string_number_correct_list)
        ratio_list_total = np.zeros_like(string_number_correct_list)
        go_over_all_units_list_total = np.zeros_like(string_number_correct_list)
        As_list_total = np.zeros_like(string_number_correct_list)
        AAs_list_total = np.zeros_like(string_number_correct_list)
        up_down_pass_list_total = np.zeros_like(string_number_correct_list)
        
        for i in range(len(methods)):
            if not string_number_correct_list[i]:
                true_reward_list[i] = fail_reward
            elif invalid_list[i] < 1.:
                true_reward_list[i] = fail_reward
            else:
                reward_list = -np.array(self.getDiversion([methods[i]]))
                if self.system_type == 2:
                    reward_list = [1.0]
                ratio_list, go_over_all_units_list = np.array(self.getReplicatedPoint([methods[i]]))
                up_down_pass_list = self.calculateConflictRouting([methods[i]])
                if self.string_number > 1:
                    ret1, ret2 = self.getAList([methods[i]])
                    As_list = np.array(ret1)
                    AAs_list = np.array(ret2)
                else:
                    As_list = AAs_list = np.array([1])

                if enable_work_criteria and reward_list[0] < 0:
                    true_reward_list[i] = fail_reward# + go_over_all_units_list[0]
                else:
                    curl_penalty = x_penalty = y_penalty = 0.0
                    if enable_curl_penalty:
                        curl_penalty, x_penalty, y_penalty = self.getCurl(self.packTrajectory(methods[i]))
                    true_reward_list[i] = (reward_list[0] * (1 + go_over_all_units_list[0] - GOOVER_UNIT_LOWER_BOUND)) - curl_penalty - math.sqrt(x_penalty ** 2 + y_penalty ** 2)
                    ratio_list_total[i] = ratio_list[0]
                    go_over_all_units_list_total[i] = go_over_all_units_list[0]
                    As_list_total[i] = As_list[0]
                    AAs_list_total[i] = AAs_list[0]
                    up_down_pass_list_total[i] = up_down_pass_list[0]
                    if ratio_list_total[i] < RATIO_LOWER_BOUND ** DUPLICATE_FACTOR or \
                        As_list_total[i] == 0 or AAs_list_total[i] == 0 or \
                        go_over_all_units_list_total[i] < GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR or \
                        not up_down_pass_list_total[i]:
                        true_reward_list[i] = min(-1.0 + go_over_all_units_list_total[i] / (GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR), 0.0)
                        true_reward_list[i] += base_reward
                    else:
                        ratio_intersect = self.getValidCross([methods[i]])
                        if ratio_intersect[0]:
                            true_reward_list[i] = base_reward + 1.
                        else:
                            true_reward_list[i] = min(-1.0 + go_over_all_units_list_total[i] / (GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR), 0.0)
                            true_reward_list[i] += base_reward
        
        return true_reward_list
        
    def getRewardList(self, methods, origami, invalid_list, step):
        string_number_correct_list = self.getStringNumber(methods)
        true_reward_list = np.zeros_like(string_number_correct_list)
        true_reward_list1 = np.zeros_like(string_number_correct_list)
        true_reward_list2 = np.zeros_like(string_number_correct_list)
        ratio_list_total = np.zeros_like(string_number_correct_list)
        go_over_all_units_list_total = np.zeros_like(string_number_correct_list)
        As_list_total = np.zeros_like(string_number_correct_list)
        AAs_list_total = np.zeros_like(string_number_correct_list)
        up_down_pass_list_total = np.zeros_like(string_number_correct_list)
        
        valid_methods = []
        valid_id_list = []
        valid_buf_id_list = []
        
        simulated_number = 0

        ref_id_list = [-1 for _ in range(len(methods))]
        
        for i in range(len(methods)):
            if not string_number_correct_list[i]:
                true_reward_list[i] = fail_reward
            elif invalid_list[i] < 1.:
                true_reward_list[i] = fail_reward
            else:
                reward_list = -np.array(self.getDiversion([methods[i]]))
                if self.system_type == 2:
                    reward_list = [1.0]
                ratio_list, go_over_all_units_list = np.array(self.getReplicatedPoint([methods[i]]))
                up_down_pass_list = self.calculateConflictRouting([methods[i]])
                if self.string_number > 1:
                    ret1, ret2 = self.getAList([methods[i]])
                    As_list = np.array(ret1)
                    AAs_list = np.array(ret2)
                else:
                    As_list = AAs_list = np.array([1])
                # equal_length_list = np.array(self.getEqualLength([methods[i]]))

                if enable_work_criteria and reward_list[0] < 0:
                    true_reward_list[i] = fail_reward# + go_over_all_units_list[0]
                else:
                    # bonus = 0 #0.125 * go_over_all_units_list[0] * self.unit_number * ratio_list[0] * equal_length_list[0]
                    curl_penalty = x_penalty = y_penalty = 0.0
                    if enable_curl_penalty:
                        curl_penalty, x_penalty, y_penalty = self.getCurl(self.packTrajectory(methods[i]))
                    # if bonus > reward_list[0]:
                    #     bonus = reward_list[0]
                    true_reward_list[i] = (reward_list[0] * (1 + go_over_all_units_list[0] - GOOVER_UNIT_LOWER_BOUND)) - curl_penalty - math.sqrt(x_penalty ** 2 + y_penalty ** 2)
                    ratio_list_total[i] = ratio_list[0]
                    go_over_all_units_list_total[i] = go_over_all_units_list[0]
                    As_list_total[i] = As_list[0]
                    AAs_list_total[i] = AAs_list[0]
                    up_down_pass_list_total[i] = up_down_pass_list[0]
                    if ratio_list_total[i] < RATIO_LOWER_BOUND ** DUPLICATE_FACTOR or \
                        As_list_total[i] == 0 or AAs_list_total[i] == 0 or \
                        go_over_all_units_list_total[i] < GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR or \
                        not up_down_pass_list_total[i]:
                        true_reward_list[i] = min(-1.0 + go_over_all_units_list_total[i] / (GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR), 0.0)
                        true_reward_list[i] += base_reward
                    else:
                        id = self.calculateBufferId(methods[i])
                        if id in valid_buf_id_list:
                            ref_id_list[i] = valid_buf_id_list.index(id)
                        else:
                            ratio_intersect = self.getValidCross([methods[i]])
                            if ratio_intersect[0]:
                                direct_reward = -10
                                if len(self.sim_buf):
                                    for ele in self.sim_buf:
                                        if ele["id"] == id:
                                            direct_reward = ele["reward"]
                                            r_speed = ele["reward_s"]
                                            r_force = ele["reward_f"]
                                            rf = ele["rf"]
                                            rm = ele["rm"]
                                            rs = ele["rs"]
                                            ra = ele["ra"]
                                            rh = ele["rh"]
                                            rforce = ele["rforce"]
                                            rforce2 = ele["rforce2"]
                                            true_reward_list[i] = direct_reward
                                            true_reward_list1[i] = r_speed
                                            true_reward_list2[i] = r_force
                                            exist_in_temp_buf = False
                                            for temp_ele in self.temp_sim_buf:
                                                if temp_ele["id"] == id:
                                                    exist_in_temp_buf = True
                                                    break
                                            if not exist_in_temp_buf:
                                                simulated_number += 1
                                                self.temp_sim_buf.append({
                                                    "id": id,
                                                    "reward": direct_reward,
                                                    "reward_s": r_speed,
                                                    "reward_f": r_force,
                                                    "rf": rf,
                                                    "rm": rm,
                                                    "rs": rs,
                                                    "ra": ra,
                                                    "rh": rh,
                                                    "rforce": rforce,
                                                    "rforce2": rforce2
                                                })
                                                print(id)
                                            break
                                
                                # if direct_reward > 1.:
                                #     ratio_intersect = self.getValidCross([methods[i]])
                                if direct_reward == -10:
                                    valid_methods.append(self.packTrajectory(methods[i]))
                                    valid_id_list.append(i)
                                    valid_buf_id_list.append(id)
                            else:
                                # ratio_intersect = self.getValidCross([methods[i]])
                                true_reward_list[i] = min(-1.0 + go_over_all_units_list_total[i] / (GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR), 0.0)
                                true_reward_list[i] += base_reward
                # true_reward_list = (reward_list * ratio_list * As_list * np_invalid_list).tolist()
                # true_reward_list = (reward_list).tolist()
        
        if len(valid_methods):
            # 进行有效方法的处理
            simulated_number += len(valid_methods)
            print(str(len(valid_methods)) + ' / ' + str(len(methods)) + " cases need to be simulated.")
            # print(valid_buf_id_list)
            if calculate_number:
                for i in range(len(valid_id_list)):
                    id = valid_id_list[i]
                    simulation_reward = 1 + math.log10(step) * 0.1
                    simulation_reward += base_reward
                    true_reward_list[id] = simulation_reward
                    true_reward_list1[id] = simulation_reward
                    true_reward_list2[id] = simulation_reward
            else: 
                height = self.simulation_config["platform_height"]
                extract = self.simulation_config["extract_mode"]
                prefold = self.simulation_config["prefold"]
                miu = self.simulation_config["miu"]
                ctm = self.simulation_config["control_mode"]
                fm = self.simulation_config["friction_mode"]
                gravity_flag = self.simulation_config["gravity_flag"]
                speed_bonus = self.simulation_config["speed_bonus"]
                ground_enable = self.simulation_config["ground_enable"]
                control_mode_training_time = self.simulation_config["control_mode_training_time"]
                simulation_upper_time = self.simulation_config["simulation_upper_time"]
                additional_length = self.simulation_config["additional_length_of_string"]
                robot_type = self.simulation_config["robot_type"]
                type_of_controller = self.simulation_config["type_of_controller"]
                stroke_percent = self.simulation_config["stroke_percent"]
                simulation_reward_list = self.getRewardList_Direct(valid_methods, 
                                                                   origami, buf_id=0, sim_time=0, extract=extract, 
                                                                   height=height, string_number=self.string_number, 
                                                                   gravity_flag=gravity_flag, prefold=prefold, 
                                                                   miu=miu, ctm=ctm, noise=None, fm=fm, 
                                                                   speed_bonus=speed_bonus, ground_enable=ground_enable, 
                                                                   control_mode_training_time=control_mode_training_time, 
                                                                   simulation_upper_time=simulation_upper_time, 
                                                                   additional_length=additional_length, 
                                                                   robot_type=robot_type, 
                                                                   type_of_controller=type_of_controller,
                                                                   stroke_percent=stroke_percent)
                for i in range(len(valid_id_list)):
                    id = valid_id_list[i]
                    simulation_reward = reward_weight[0] * simulation_reward_list[i] + reward_weight[2] * simulation_reward_list[i + len(valid_methods)]
                    simulation_reward += base_reward
                    true_reward_list[id] = simulation_reward
                    true_reward_list1[id] = simulation_reward_list[i]
                    true_reward_list2[id] = simulation_reward_list[i + len(valid_methods)]
                    # input_sim_buffer['buf'][j]["reward_without_actuator"] = simulation_reward_list[i + len(valid_methods)]
                    info = {
                        "id": valid_buf_id_list[i],
                        "reward": simulation_reward,
                        "reward_s": simulation_reward_list[i],
                        "reward_f": simulation_reward_list[i + len(valid_methods)],
                        'rf': simulation_reward_list[i + 2*len(valid_methods)],
                        'rm': simulation_reward_list[i + 3*len(valid_methods)],
                        'rs': simulation_reward_list[i + 4*len(valid_methods)],
                        'ra': simulation_reward_list[i + 5*len(valid_methods)],
                        'rh': simulation_reward_list[i + 6*len(valid_methods)],
                        'rforce': simulation_reward_list[i + 7*len(valid_methods)],
                        'rforce2': simulation_reward_list[i + 8*len(valid_methods)],
                    }
                    self.sim_buf.append(info)
                    self.temp_sim_buf.append(info)
                    
                with open(os.path.join("./threadingResult", f"{BUF_NAME}.json"), 'w', encoding="utf-8") as f:
                    json.dump({
                        "buf": self.sim_buf
                    }, f, indent=4)
                
        for i in range(len(ref_id_list)):
            if ref_id_list[i] != -1:
                true_reward_list[i] = true_reward_list[valid_id_list[ref_id_list[i]]]
                true_reward_list1[i] = true_reward_list1[valid_id_list[ref_id_list[i]]]
                true_reward_list2[i] = true_reward_list2[valid_id_list[ref_id_list[i]]]

        return true_reward_list, true_reward_list1, true_reward_list2, ratio_list_total, go_over_all_units_list_total, As_list_total, AAs_list_total, simulated_number
        
    def getRewardList_Direct(self, methods, origami, buf_id=0, sim_time=0, extract=-1, height=10, string_number=2, gravity_flag=0, prefold=0.00, miu=0.5, ctm=0, noise=0, fm=2, speed_bonus=1.0, ground_enable=0, control_mode_training_time=15.0, simulation_upper_time=20.0, additional_length=100.0, robot_type=0, type_of_controller=1, stroke_percent=0.75):
        valid_number = len(methods)

        if valid_number >= 1: 
            initial_fitness_list = [0. for _ in range(SUB_REWARD_NUM * valid_number)]

            mlist = multiprocessing.Manager().list(initial_fitness_list)

            p_list = []

            pointer = 0
            previous_lower = 0

            while pointer < thread_number:
                interval = int((valid_number - 1) / thread_number) + 1
                normal_cases = thread_number * interval
                
                lower = pointer * interval
                upper = (pointer + 1) * interval

                if valid_number < normal_cases:
                    interval = int(valid_number / thread_number)
                    delta = valid_number - interval * thread_number
                    if pointer < delta:
                        interval += 1
                    lower = previous_lower
                    upper = lower + interval
                    previous_lower = upper
                    if upper > valid_number:
                        upper = valid_number
                all_method = methods[lower: upper]
                
                if lower < upper:
                    print(f"Allocating range {lower} to {upper} in process {pointer}. Total number: {upper - lower}")
                    p = multiprocessing.Process(target=workerMultisim, args=(
                            mlist, origami, all_method, valid_number, self.contributions,
                            self.P_points, self.P_candidator_connections, self.max_edge, self.units, self.max_size, self.total_bias, 
                            buf_id + sim_time * thread_number * interval, interval, extract, lower, height, string_number, self.fix_id_list, gravity_flag, prefold, miu, self.targets, 
                            ctm, noise, fm, speed_bonus,
                            ground_enable, control_mode_training_time, simulation_upper_time, additional_length, robot_type, type_of_controller, stroke_percent
                        )
                    )
                    p_list.append(p)
                pointer += 1
            
            process_id = 0

            total_process_number = thread_number

            current_process_number = 0

            while process_id < len(p_list):
                while current_process_number < total_process_number:
                    p_list[process_id].start()
                    current_process_number += 1
                    process_id += 1
                    if current_process_number == total_process_number or process_id == len(p_list):
                        break

                while current_process_number > 0:
                    p_list[process_id - current_process_number].join()
                    current_process_number -= 1

            reward_list = list(mlist)

        return reward_list
    
    def getTrajectoryLength(self):
        return sum([len(self.current_trajectory[i]) for i in range(self.string_number)])
    
    def step(self, action, step, rollout=True):
        # 清空fake_step，后续根据action能否被执行，决定是否是假动作
        self.fake_step = 0
        
        # 判断是否是绳索还未选点的状态，即根节点此时需要选择外界驱动点
        if self.temp_done:
            # 如果根节点此时需要选择外界驱动点，那么 action的id 和 action实际对应的点的id 是相同的，都必须在P_points范围内
            if action < len(self.P_points):
                # 不用管，因为现在symmetric_routing始终置False
                if symmetric_routing and string_number == 2 and self.current_string_id == 2:
                    previous_action = self.current_trajectory[0][0][0]
                    p1 = self.P_points[previous_action]
                    p2 = self.P_points[action]
                    mid = (p1 + p2) * 0.5
                    dis = distance(mid, [self.mid_x, self.mid_y])
                    if dis > 1:
                        self.fake_step = 1
                        next_state = np.append(self.current_state, self.current_string_id)
                        next_state[self.getStateIndex()] = action + 1
                        return next_state, self.fail_reward, 1, 0
                
                #开始穿线，此时一定不会fake_step。故关闭temp_done，并重置current_valid_pass=0，只有线路通过了至少2个内部点，current_valid_pass才会被置为1
                self.beginRouting()
                #更新状态和当前轨迹（穿线进度）
                self.updateStateAndTrajectory(action, 0, action)
                
                if rollout:
                    # 不用管，因为没有采用此方法
                    methods = self.rollout(8)
                    reward_list = self.getRewardList(methods, step)
                    reward = sum(reward_list) / 8.
                else:
                    # 计算当前轨迹的长度作为奖励（实际上只要是正的就行）
                    reward = self.getTrajectoryLength()
                # 存储当前状态和轨迹，供后续回溯使用
                if not rollout:
                    self.deepbackup()
                # 返回 下一状态（在updateStateAndTrajectory中修改后的当前状态），奖励，done标志（0表示未完成，1表示完成），truncated表示是否被截断（0表示未截断）（但已弃用truncated）
                return np.append(self.current_state, self.current_string_id), reward, 0, 0
            else:
                # 如果action的id大于P_points的长度，说明是选择了O_points中的点，这是无效的选择（但这一步一般不会触发这个）
                self.fake_step = 1
                next_state = np.append(self.current_state, self.current_string_id)
                next_state[self.getStateIndex()] = action + 1
                return next_state, self.fail_reward, 1, 0
        else:
            #走到此处表明当前根节点不是需要选择外界驱动点，而是需要选择内部点
            
            #存一下当前的stand_point_id，即当前agent在id为几的点上
            self.old_stand_point_id = self.current_stand_point_id
            #通过action的id，算出这个action_id实际对应的点的id（例如连接矩阵为[-10, -10, 1, -10, -1]，则action_id=0对应的实际点的id是2，action_id=1对应的实际点的id是4），以及该点的side（连接类型）（-1表示下方，1表示上方，0表示未知）
            true_action_index, side = self.getActionIndex(action)
            if self.current_stand_point_id != -1 and side == 0:
                side = -self.previous_side

            # if self.previous_side != 0:
            #     has_solution = False
            #     for ele in self.valid_matrix[self.old_stand_point_id]:
            #         if ele == -self.previous_side:
            #             has_solution = True
            #             break
            #     if has_solution and side != ele:
            #         self.fake_step = 1
            #         next_state = np.append(self.current_state, self.current_string_id)
            #         next_state[self.getStateIndex()] = action + 1
            #         return next_state, self.fail_reward, 1, 0
            
            # 判定绳索选择当前action之后，是否会自我交叉，通过按序遍历current_state中当前绳索的id，检查该绳索是不是之前已经经过了此板
            id_start = int((self.current_string_id - 1) * (self.unit_number + 2))
            id_end = int(self.current_string_id * (self.unit_number + 2))
            for i in range(id_start, id_end):
                id = self.current_state[i]
                
                # 如果id为0，说明到达绳索末端
                if id == 0.0:
                    break
                
                if id > len(self.P_points) and int(id) == true_action_index + 1 and self.current_state[i + 1] != 0.0:
                    # 如果当前action_id对应的点的id和current_state中当前绳索的id相同，且下一个点不是0（即不是末端），说明当前action_id选择的点已经被当前绳索经过了
                    # 此时fake_step置1，表示当前action_id是无效的选择
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
            
            # 如果当前action_id对应的点的id为-1，说明当前action_id选择的是无效点，保证安全性（一般也不会触发，如果在选action的时候就确保选对的话）。
            if true_action_index == -1:
                self.fake_step = 1
                next_state = np.append(self.current_state, self.current_string_id)
                next_state[self.getStateIndex()] = true_action_index + 1
                return next_state, self.fail_reward, 1, 0
            
            # 接下来，choose P or O is both correct, but needs validation
            
            # 判断绳索是否经过了别的绳索的固定点，如果是，则直接强行固定该绳索的绳端在该点处，结束该绳索的搜索
            # node_status指出了当前所有点被绳索穿过的情况，node_status[i] = N 表示id为i的点被N条绳索穿过，node_status[i] = -1 表示至少有一根绳子将其末端固定在id为i的点。
            if self.node_status[true_action_index] == -1.:
                # 这时需要强行固定末端点，先判断一下最后一个绳段的情况，can't be the same side
                if side + self.previous_side != 0 and self.current_valid_pass:
                    #表明当前绳段和上一绳段均位于折纸同侧，这不合理
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                
                current_length = len(self.current_trajectory[int(self.current_string_id) - 1])
                if current_length >= self.maximum_pass_number: #绳索经过的点数已经到达上限，不能再经过当前点了
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                # return np.append(self.current_state, self.current_string_id), self.fail_reward * (sum([len(trajectory) for trajectory in self.current_trajectory]) + 1), 1, 0
                
                # 解析式reward的判断标准是所有单条绳索均对折纸的折叠做正功，如果当前绳索做负功，则直接返回失败，fake_step=1表示不可step进来，后续绳索可以不用搜索。
                if pre_calculate_condition:
                    # 得到单条绳索的解析reward
                    if self.system_type == 0:
                        reward_one_string = self.getRewardForOneControlChain(self.current_trajectory[int(self.current_string_id) - 1])
                        if enable_work_criteria and reward_one_string <= 0:
                            # 如果reward小于等于0，说明当前绳索的穿线方式不合理，不能通过
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0

                    # 这里我们用一个二进制id来唯一表示一根绳索（唯一标识码），考虑到一种情况是，如果搜2根线，那么对于agent而言，[0, 2, 3]+[1, 3, 4] 和 [1, 3, 4]+[0, 2, 3]是等价的，
                    # 这就会导致穿线方案冗余，因此我们这里规定假设搜索N条绳，第K条绳的唯一标识码需要大于第K-1条的标识码（即把排列数变为组合数），可以减少A(N,N)种穿线方案，A为排列数。
                    if 1:
                        id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                        if int(self.current_string_id) == 1:
                            self.id = id
                        else:
                            #如果当前绳索的唯一标识码小于等于之前的标识码，说明当前穿线方案是冗余的，一定能通过其他方式plan出来等价的方案
                            if id <= self.id:
                                self.fake_step = 1
                                next_state = np.append(self.current_state, self.current_string_id)
                                next_state[self.getStateIndex()] = true_action_index + 1
                                return next_state, self.fail_reward, 1, 0
                            else:
                                self.id = id
                    else: # 弃用
                        # pass
                        id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                        same = False
                        for i in range(int(self.current_string_id) - 1):
                            id_i = sum([2 ** self.current_trajectory[i][j][0] for j in range(len(self.current_trajectory[i]))])
                            if id in self.old_mean_per_string[i] and id < id_i:
                                same = True
                                break
                        if same:
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0
                        else:
                            if id not in self.old_mean_per_string[int(self.current_string_id) - 1]:
                                self.old_mean_per_string[int(self.current_string_id) - 1].append(id)

                # 更新状态和当前轨迹（穿线进度） 
                self.updateStateAndTrajectory(true_action_index, side, action)
                # 当前绳索结束穿线
                if self.current_string_id < self.string_number: #如果当前绳的id<总绳数-1，表明还有绳索没有规划
                    # 置temp_done=1，开启新一条绳索，下一次搜索的时候，从选择外界驱动点开始。
                    self.endString()
                    if rollout: #弃用
                        methods = self.rollout(8)
                        reward_list = self.getRewardList(methods, step)
                        reward = sum(reward_list) / 8.
                    else: #返回一个reward（正的就行）
                        reward = self.getTrajectoryLength()
                    if not rollout:
                        self.deepbackup()
                        
                    # 有时为了简化搜索，用户可以根据经验指定每根线的一些初始动作，但一般为空，可不管
                    self.takeFixedActions(fixed_action_list_initial)
                    
                    return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else: #如果当前绳的id=总绳数-1，表明所有绳索都规划完毕，穿线方案已形成
                    # 彻底结束，done=1
                    self.done = 1
                    # simulator here
                    if direct_reward: #原本的思路是，形成穿线方案后，丢到仿真器仿真评估结果，但已经弃用
                        self.reward = self.getReward(self.current_trajectory, step)
                    else: #返回一个reward（正的就行）
                        self.reward = 1.
                    # 存一个轨迹（目前没有啥用）
                    self.new_trajectory = deepcopy(self.current_trajectory)
                    if not rollout:
                        self.deepbackup()
                    return np.append(self.current_state, self.current_string_id), self.reward, 1, 0
            
            # 如果agent在id号点上选择了自己，则表明绳索将绳端固定在该点处
            elif self.old_stand_point_id == true_action_index:
                # 当前被固定的点不能被其他绳索穿过
                passed = False
                for string in self.current_trajectory:
                    for index in range(len(string)):
                        if string[index][0] == true_action_index and index != len(string) - 1:
                            passed = True
                            break
                    if passed:
                        break
                if passed: #如果当前被固定的点被其他绳索穿过，则动作无效
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                
                # 道理同前，第1740行，解析式reward的判断标准是所有单条绳索均对折纸的折叠做正功，如果当前绳索做负功，则直接返回失败，fake_step=1表示不可step进来，后续绳索可以不用搜索。
                if pre_calculate_condition:
                    if self.system_type == 0:
                        reward_one_string = self.getRewardForOneControlChain(self.current_trajectory[int(self.current_string_id) - 1])
                        if enable_work_criteria and reward_one_string <= 0:
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0
                    # else:
                    if 1:
                        id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                        # if id in [156250240, 46448644]:
                        #     a = 1
                        if int(self.current_string_id) == 1:
                            self.id = id
                        else:
                            if id <= self.id:
                                self.fake_step = 1
                                next_state = np.append(self.current_state, self.current_string_id)
                                next_state[self.getStateIndex()] = true_action_index + 1
                                return next_state, self.fail_reward, 1, 0
                            else:
                                self.id = id
                    else: #弃用
                        # pass
                        # mean, std = self.getMeanAndStdForOneControlChain(self.current_trajectory[int(self.current_string_id) - 1])
                        id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                        same = False
                        for i in range(int(self.current_string_id) - 1):
                            id_i = sum([2 ** self.current_trajectory[i][j][0] for j in range(len(self.current_trajectory[i]))])
                            if id in self.old_mean_per_string[i] and id < id_i:
                                same = True
                                break
                        if same:
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0
                        else:
                            if id not in self.old_mean_per_string[int(self.current_string_id) - 1]:
                                self.old_mean_per_string[int(self.current_string_id) - 1].append(id)
                    
                # 道理同前，第1784行
                self.updateStateAndTrajectory(true_action_index, side, action)
                # end one string
                if self.current_string_id < self.string_number: 
                    self.endString()
                    if rollout:
                        methods = self.rollout(8)
                        reward_list = self.getRewardList(methods, step)
                        reward = sum(reward_list) / 8.
                    else:
                        reward = self.getTrajectoryLength()
                    if not rollout:
                        self.deepbackup()
                    self.takeFixedActions(fixed_action_list_initial)
                    return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else:
                    self.done = 1
                    # simulator here
                    if direct_reward:
                        self.reward = self.getReward(self.current_trajectory, step)
                    else:
                        self.reward = 1.
                    self.new_trajectory = deepcopy(self.current_trajectory)
                    if not rollout:
                        self.deepbackup()
                    return np.append(self.current_state, self.current_string_id), self.reward, 1, 0
            else: #走到这里，表明是选择了一个id不等于当前id的其他点，接下来就需要判断合理性（主要是看side是否与前面绳段相反）
                # 同样判断一下绳索穿过点的个数是否超过上限
                current_length = len(self.current_trajectory[int(self.current_string_id) - 1])
                if current_length >= self.maximum_pass_number:
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                
                # 判断绳索是否至少经过了2个内部点，只有这样，current_valid_pass才会被置为1，表明随时可以通过 固定绳端/再选一个外界驱动点 来终止单绳的搜索，但是现在已经不允许绳子将两个绳端都固定在外界驱动点，所以实际上current_valid_pass是没啥用的
                if self.current_valid_pass: # 绳索至少经过了2个内部点
                    if true_action_index < len(self.P_points):
                        # 再选择外界点，终止该绳的搜索，形成A, B, B, A形式的绳索，但是现在不允许enable_double_P，相当于弃用
                        if enable_double_P: #一定是False
                            if side != 0 and side + self.previous_side != 0:
                                self.fake_step = 1
                                next_state = np.append(self.current_state, self.current_string_id)
                                next_state[self.getStateIndex()] = true_action_index + 1
                                return next_state, self.fail_reward, 1, 0
                            if pre_calculate_condition:
                                if self.system_type == 0:
                                    reward_one_string = self.getRewardForOneControlChain(self.current_trajectory[int(self.current_string_id) - 1])
                                    if enable_work_criteria and reward_one_string <= 0:
                                        self.fake_step = 1
                                        next_state = np.append(self.current_state, self.current_string_id)
                                        next_state[self.getStateIndex()] = true_action_index + 1
                                        return next_state, self.fail_reward, 1, 0

                                if 1:
                                    id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                                    if int(self.current_string_id) == 1:
                                        self.id = id
                                    else:
                                        if id <= self.id:
                                            self.fake_step = 1
                                            next_state = np.append(self.current_state, self.current_string_id)
                                            next_state[self.getStateIndex()] = true_action_index + 1
                                            return next_state, self.fail_reward, 1, 0
                                        else:
                                            self.id = id
                                else:
                                    id = sum([2 ** self.current_trajectory[int(self.current_string_id) - 1][i][0] for i in range(len(self.current_trajectory[int(self.current_string_id) - 1]))])
                                    same = False
                                    for i in range(int(self.current_string_id) - 1):
                                        id_i = sum([2 ** self.current_trajectory[i][j][0] for j in range(len(self.current_trajectory[i]))])
                                        if id in self.old_mean_per_string[i] and id < id_i:
                                            same = True
                                            break
                                    if same:
                                        self.fake_step = 1
                                        next_state = np.append(self.current_state, self.current_string_id)
                                        next_state[self.getStateIndex()] = true_action_index + 1
                                        return next_state, self.fail_reward, 1, 0
                                    else:
                                        if id not in self.old_mean_per_string[int(self.current_string_id) - 1]:
                                            self.old_mean_per_string[int(self.current_string_id) - 1].append(id)

                            self.updateStateAndTrajectory(true_action_index, side, action)
                            if self.current_string_id < self.string_number:
                                self.endString()
                                if rollout:
                                    methods = self.rollout(8)
                                    reward_list = self.getRewardList(methods, step)
                                    reward = sum(reward_list) / 8.
                                else:
                                    reward = self.getTrajectoryLength()
                                if not rollout:
                                    self.deepbackup()
                                self.takeFixedActions(fixed_action_list_initial)
                                return np.append(self.current_state, self.current_string_id), reward, 0, 0
                            else:
                                self.done = 1
                                # simulator here
                                if direct_reward:
                                    self.reward = self.getReward(self.current_trajectory, step)
                                else:
                                    self.reward = 1.
                                self.new_trajectory = deepcopy(self.current_trajectory)
                                if not rollout:
                                    self.deepbackup()
                                return np.append(self.current_state, self.current_string_id), self.reward, 1, 0   
                        else: #不能构成ABBA形式的穿线
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0
                    elif side + self.previous_side != 0: #当前绳段和前一条绳段在折纸同一侧（（1，1）或（-1，-1）形式的绳索），不能构成合法绳索
                        self.fake_step = 1
                        next_state = np.append(self.current_state, self.current_string_id)
                        next_state[self.getStateIndex()] = true_action_index + 1
                        return next_state, self.fail_reward, 1, 0
                    else: #当前绳段和前一条绳段在折纸不同侧（（1，-1）或（-1，1）形式的绳索），可以构成合法绳索
                        self.previous_side = side #记录当前绳段的side
                        self.updateStateAndTrajectory(true_action_index, side, action)
                        if rollout: #弃用
                            methods = self.rollout(8)
                            reward_list = self.getRewardList(methods, step)
                            reward = sum(reward_list) / 8.
                        else: # 计算当前轨迹的长度作为奖励（实际上只要是正的就行）
                            reward = self.getTrajectoryLength()
                        if not rollout:
                            self.deepbackup()
                    
                        return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else: # 绳索没有经过2个内部点，不能通过 固定绳端/再选一个外界驱动点 来终止单绳的搜索 （实际上current_valid_pass是没啥用的）
                    if true_action_index < len(self.P_points): # 绳索没有经过2个内部点，不能选择外部驱动点来终止单绳的搜索，必须要选内部点
                        self.fake_step = 1
                        next_state = np.append(self.current_state, self.current_string_id)
                        next_state[self.getStateIndex()] = true_action_index + 1
                        return next_state, self.fail_reward, 1, 0
                        # else:
                        #     self.updateStateAndTrajectory(true_action_index, 0, action)
                        #     if rollout:
                        #         methods = self.rollout(8)
                        #         reward_list = self.getRewardList(methods, step)
                        #         reward = sum(reward_list) / 8.
                        #     else:
                        #         reward = self.getTrajectoryLength()
                        #     if not rollout:
                        #         self.deepbackup()
                        #     return np.append(self.current_state, self.current_string_id), reward, 0, 0
                    # if side == 0:
                    #     # no valid pass
                    #     if true_action_index < len(self.P_points):
                    #         self.fake_step = 1
                    #         next_state = np.append(self.current_state, self.current_string_id)
                    #         next_state[self.getStateIndex()] = true_action_index + 1
                    #         return next_state, self.fail_reward, 1, 0
                    #     else:
                    #         self.updateStateAndTrajectory(true_action_index, 0, action)
                    #         if rollout:
                    #             methods = self.rollout(8)
                    #             reward_list = self.getRewardList(methods, step)
                    #             reward = sum(reward_list) / 8.
                    #         else:
                    #             reward = self.getTrajectoryLength()
                    #         if not rollout:
                    #             self.deepbackup()
                    #         return np.append(self.current_state, self.current_string_id), reward, 0, 0
                    
                    else: # 绳索没有经过2个内部点，必须要选内部点
                        self.updateStateAndTrajectory(true_action_index, side, action)
                        if side != 0: #如果绳段的方向明确（要么1，要么-1），则认为当前绳段是有效的，给出valid_pass=1
                            self.current_valid_pass = 1
                            self.previous_side = side
                        if rollout:
                            methods = self.rollout(8)
                            reward_list = self.getRewardList(methods, step)
                            reward = sum(reward_list) / 8.
                        else:
                            reward = self.getTrajectoryLength()
                        if not rollout:
                            self.deepbackup()
                        return np.append(self.current_state, self.current_string_id), reward, 0, 0

    def outputGraphAndCSV(self, log_dict):
        try:
            visualize(step, "String-routing Path Search", log_dict)
            data = [
                [
                    log_dict['train_steps'][-1][i], 
                    log_dict['reward_returns'][-1][i], 
                    log_dict['max_reward_returns'][-1][i], 
                    log_dict['tree_policy_max_reward_returns'][-1][i],
                    log_dict["cut_number"][-1][i], 
                    log_dict["valid_number"][-1][i], 
                    log_dict["rollout_valid_number"][-1][i], 
                    log_dict["number"][-1][i],
                    log_dict["simulated_number"][-1][i],
                    log_dict["time"][-1][i]
                ] 
                for i in range(len(log_dict['train_steps'][-1]))
            ]
            
            if self.backup_csv is not None:
                for i in range(len(log_dict['train_steps'][-1])):
                    buffer_list = self.backup_csv['step'].to_list()
                    if i < len(buffer_list):
                        true_time = self.backup_csv['time'].tolist()[i]
                        self.delta_time = true_time - data[i][-1]
                        data[i][-1] = true_time
                        
                    else: #new
                        data[i][-1] += self.delta_time

            if self.backup_csv is None or (len(log_dict['train_steps'][-1]) > len(buffer_list)):
                test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "tree_policy_maximum_reward", "cut", "valid", "rollout_valid", "total", "simulated_number", "time"], data=data)
                test.to_csv(os.path.join(file_path, f'{NAME}-{GOOVER_UNIT_LOWER_BOUND}.csv'))
        except:
            plt.close()

# def eval(env, agent, episode):
#     state, _ = env.reset()
#     done, truncated = False, False

#     while not (done or truncated):
#         state, reward, done, truncated = env.step(agent.get_action(state).item(), episode)
   
#     if reward > 0:
#         return env.getReward(env.new_trajectory, episode)
#     else:
#         return reward

def get_epsilon(step, eps_min, eps_max, eps_steps, warmup_steps):
    if step < warmup_steps:
        return eps_max
    elif step > eps_steps:
        return eps_min
    else:
        return eps_max - (eps_max - eps_min) / (eps_steps - warmup_steps) * (step - warmup_steps)

def outputAllMethod(time):
    try:
        if not calculate_number:
            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
            buf = 0
            while buf < len(sorted_method):
                added_buffer = BUFFER_SIZE if buf + BUFFER_SIZE < len(sorted_method) else len(sorted_method) - buf
                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                    json.dump({
                        "method": sorted_method[buf: buf + added_buffer],
                        "number": added_buffer,
                        "time": (datetime.now() - start_time).total_seconds()
                    }, f, indent=4)
                buf += added_buffer
        print(f"{len(candidate_methods)} methods become candidators")
    except:
        pass
    
def outputBestTrajectory(env, best_trajectory, best_reward):
    try:
        env.output(step, best_trajectory, best_reward)
        print(best_trajectory)
    except:
        pass
    try:
        if not calculate_number:
            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
            with open(os.path.join(file_path, "result_train_time_" + str(time) + "_candidate.json"), 'w', encoding="utf-8") as f:
                json.dump({
                    "method": sorted_method,
                    "number": len(sorted_method)
                }, f, indent=4)
        print(f"{len(candidate_methods)} methods become candidators")
    except:
        pass

def setSeeds(seed):
    random.seed(seed + time)
    np.random.seed(seed + time)
    os.environ["PYTHONHASHSEED"] = str(seed + time) 

def setup():
    global cut_num, roller
    cut_num = 0
    roller = 0

def train(env: Env, origami):
    global cut_num, roller, step, candidate_methods, time
    cut_num = 0
    roller = 0
    step = 0
    time = 0
    candidate_methods.clear()

    log_dict = {
        "train_steps": [],
        "train_returns": [],
        "tree_policy_max_reward_returns": [],
        "reward_returns": [],
        "valid_reward_returns": [],
        "max_reward_returns": [],
        "cut_number": [],
        "valid_number": [],
        "number": [],
        "rollout_valid_number": [],
        "simulated_number": [],
        "time": []
    }
    
    while time < 1:
        time += 1
        env.initialize()
        setSeeds(seed + time)
        setup()

        step_info = []

        log_dict["train_steps"].append([0])
        log_dict["train_returns"].append([(fail_reward+1)*0.5])
        log_dict["tree_policy_max_reward_returns"].append([(fail_reward+1)*0.5])
        log_dict["reward_returns"].append([(fail_reward+1)*0.5])
        log_dict["max_reward_returns"].append([(fail_reward+1)*0.5])
        log_dict["valid_reward_returns"].append([(fail_reward+1)*0.5])
        log_dict["cut_number"].append([0])
        log_dict["valid_number"].append([0])
        log_dict["number"].append([0])
        log_dict["rollout_valid_number"].append([0])
        log_dict["simulated_number"].append([0])
        log_dict["time"].append([0])

        best_reward = env.fail_reward
        tree_policy_best_reward = env.fail_reward
        previous_best = best_reward
        best_trajectory = None

        bonus = 1.
        valid_method_num = 0
        backup_valid_method_num = 0
        rollout_valid_method_num = 0
        backup_rollout_valid_method_num = 0
        backup_simulated_num = -1
        backup_num = -1
        method_num = 0
        simulated_num = 0
        step = 0
        epsilon_bonus = 1
        dead_count = 0
        eps = 1.
        reward = 0

        CLEAR_STEP = 10
        
        rollout_early_stop_enable = 0

        if 0:
            pass
            # while 1:
            #     step += 1
            #     env.reset()
            #     current_node = env.root_node

            #     front_node, _ = env.treePolicy(current_node, eps * bonus)
            #     level = [len(env.current_trajectory[i]) for i in range(env.string_number)]

            #     if front_node == None or (current_node.parent == None and current_node.children == None):
            #         log_dict["train_steps"][-1].append(step)
            #         log_dict["train_returns"][-1].append((reward+1)*0.5)
            #         log_dict["tree_policy_max_reward_returns"][-1].append((tree_policy_best_reward+1)*0.5)
            #         log_dict["reward_returns"][-1].append((sum(env.avg_reward) / 100.+1)*0.5)
            #         log_dict["max_reward_returns"][-1].append((best_reward+1)*0.5)
            #         log_dict["valid_reward_returns"][-1].append((env.valid_best_reward+1)*0.5)
            #         log_dict["cut_number"][-1].append(step - cut_num)
            #         log_dict["valid_number"][-1].append(valid_method_num)
            #         log_dict["rollout_valid_number"][-1].append(rollout_valid_method_num)
            #         log_dict["number"][-1].append(method_num)
            #         log_dict["simulated_number"][-1].append(simulated_num)
            #         log_dict["time"][-1].append((datetime.now() - start_time).total_seconds())
            #         print(f"Root node has done, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num} / {method_num}")
            #         env.outputGraphAndCSV(log_dict)
            #         try:
            #             env.output(step, best_trajectory, best_reward)
            #             # print(best_trajectory)
            #         except:
            #             pass
            #         return log_dict

            #     if not front_node.done:
            #         rollout_number = front_node.maximum_child ** 2
            #         methods, invalid_list, rollout_number = env.rollout(rollout_number, front_node)
            #         reward_list, ratio_list, goover_unit_list, As_list, AAs_list, sub_simulated_num = env.getRewardList(methods, origami, invalid_list, step)
            #         simulated_num += sub_simulated_num
            #         total_reward = sum(reward_list)
            #         reward = total_reward / rollout_number

            #         for i in range(rollout_number):
            #             if reward_list[i] > best_reward:
            #                 best_reward = reward_list[i]
            #                 env.valid_best_reward = best_reward
            #                 best_trajectory = env.packTrajectory(methods[i])

            #         env.avg_reward[roller % 100] = (reward+1.)*0.5
            #         roller += 1
            #         env.backward(front_node, (reward+1.)*0.5, 1)

            #     else:
            #         if env.reward < 0:
            #             reward = env.reward
            #         else:
            #             if front_node.visits != 1:
            #                 reward = front_node.reward / (front_node.visits - 1)
            #             else:
            #                 method_num += 1
            #                 reward_list, ratio_list, goover_unit_list, As_list, AAs_list, sub_simulated_num = env.getRewardList([deepcopy(env.current_trajectory)], origami, [1], step)
            #                 simulated_num += sub_simulated_num
            #                 reward = reward_list[0]
            #                 env.avg_reward[roller % 100] = reward
            #                 roller += 1
            #                 if reward > best_reward:
            #                     best_reward = reward
            #                     env.valid_best_reward = reward
            #                     best_trajectory = env.packTrajectory(env.new_trajectory)
            #                 if reward > 0:
            #                     valid_method_num += 1
            
            #         env.backward(front_node, (reward+1.)*0.5, 1)

            #     if step % record_episode == 0:
            #         log_dict["train_steps"][-1].append(step)
            #         log_dict["train_returns"][-1].append((reward+1)*0.5)
            #         log_dict["tree_policy_max_reward_returns"][-1].append((tree_policy_best_reward+1)*0.5)
            #         log_dict["reward_returns"][-1].append((sum(env.avg_reward) / 100.+1)*0.5)
            #         log_dict["max_reward_returns"][-1].append((best_reward+1)*0.5)
            #         log_dict["valid_reward_returns"][-1].append((env.valid_best_reward+1)*0.5)
            #         log_dict["cut_number"][-1].append(step - cut_num)
            #         log_dict["valid_number"][-1].append(valid_method_num)
            #         log_dict["rollout_valid_number"][-1].append(rollout_valid_method_num)
            #         log_dict["number"][-1].append(method_num)
            #         log_dict["simulated_number"][-1].append(simulated_num)
            #         log_dict["time"][-1].append((datetime.now() - start_time).total_seconds())

            #     env.best_reward = best_reward
            #     bonus = best_reward if best_reward > 1. else 1.

            #     if step % record_episode == 0:
            #         print(f"Step: {step}, Best reward: {round(best_reward, 2)}, Bonus/epsilon: {round(bonus, 2)}/{round(eps, 2)}, Level: {level}, Valid method number: {valid_method_num} / {method_num}")
            #         if best_reward > base_reward:
            #             print("Best: " + str(best_trajectory["id"]))
            #             if len(env.new_trajectory[0]):
            #                 print("Current: " + str(env.packTrajectory(env.new_trajectory)["id"]))
            #             else:
            #                 print("Current: No new trajectory")
            #         env.outputGraphAndCSV(log_dict)
        else:
            trial = 0
            while 1:
                # if step > 0 and simulated_num == 0:
                #     print("Constraints are too strict! Reset UNIT_PASS. ")
                #     break
                step_info.append(step)
                with open(os.path.join(file_path, f"step_info"), 'w', encoding="utf-8") as f:
                    json.dump({"step": step_info}, f, indent=4)
                if (rollout_valid_method_num + valid_method_num >= VALID_MINIMAL):
                    trial += 1
                if trial >= 100000 and not calculate_number:
                    break
                error_flag = 0
                fixed_action_list = deepcopy(fixed_action_list_initial)
                # if trial >= 1:
                #     temp_pointer = env.root_node
                #     total_pointers.append(temp_pointer)
                while env.root_node.parent != None:
                    env.root_node = env.root_node.parent
                    
                if initialize:
                    print("initialize tree")
                    # env.root_node.initializeTree(env.best_reward)
                    sub_cut_num = env.root_node.initializeTree2((env.best_reward+1.)*0.5)
                    cut_num += sub_cut_num
                # for pointer in total_pointers:
                #     pointer.reward = fail_reward
                
                # if best_reward > previous_best or epsilon_bonus > upper_bonus:
                #     print("initialize tree")
                #     env.root_node.initializeTree(env.best_reward)
                #     epsilon_bonus = 1
                #     previous_best = best_reward
                    # rare_rate = (max(method_num - valid_method_num, valid_method_num) + 1) / (valid_method_num + 1)
                if best_reward > previous_best:
                    previous_best = best_reward
                
                depth = 0
                depth_left = string_number * (upper_bound + 1) - depth
                valid_child_num = env.root_node.maximum_child

                # if rollout_valid_method_num <= backup_rollout_valid_method_num and rollout_valid_method_num < valid_method_num and best_reward <= previous_best:
                if cut_nodes:
                    if rollout_valid_method_num < valid_method_num and rollout_valid_method_num + valid_method_num >= VALID_MINIMAL:
                        rollout_early_stop_enable += 1
                    else:
                        rollout_early_stop_enable = 0
                else:
                    if (log_dict['simulated_number'][-1][-1] == backup_simulated_num) and \
                        (log_dict['number'][-1][-1] == backup_num):
                        rollout_early_stop_enable += 1
                    else:
                        rollout_early_stop_enable = 0

                if (dead_count >= dead_maximum or (rollout_early_stop_enable >= CLEAR_STEP) and rollout_valid_method_num + valid_method_num >= VALID_MINIMAL):
                    # if eps <= 1e-5:
                    if dead_count >= dead_maximum:
                        print("No improvement, early stop.")
                    else:
                        print("Tree policy dominated, early stop.")
                    env.outputGraphAndCSV(log_dict)
        
                    if not cut_nodes or (cut_nodes and (best_reward > base_reward + valid_base_reward)):
                        return log_dict

                    # elif eps - 2 / dead_maximum <= 1e-5:
                    #     eps -= 4 / dead_maximum**2
                    # else:
                    #     eps -= 2 / dead_maximum
                    #     print("No improvement, Decreasing Epsilon")
                                
                backup_valid_method_num = valid_method_num
                backup_rollout_valid_method_num = rollout_valid_method_num
                backup_simulated_num = log_dict['simulated_number'][-1][-1]
                backup_num = log_dict['number'][-1][-1]
                last_step = step
                
                while 1:
                    step += 1
                    env.reset()
                    while env.root_node.parent != None:
                        env.root_node = env.root_node.parent
                    env.takeFixedActions(fixed_action_list)
                    current_string_id = int(env.current_string_id)
                    current_node = env.root_node

                    if current_node.parent == None and (current_node.children == None or (current_node.fullyExpanded() and not current_node.existBestChild((env.best_reward+1.)*0.5))):
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append((reward+1)*0.5)
                        log_dict["tree_policy_max_reward_returns"][-1].append((tree_policy_best_reward+1)*0.5)
                        log_dict["reward_returns"][-1].append((sum(env.avg_reward) / 100.+1)*0.5)
                        log_dict["max_reward_returns"][-1].append((best_reward+1)*0.5)
                        log_dict["valid_reward_returns"][-1].append((env.valid_best_reward+1)*0.5)
                        log_dict["cut_number"][-1].append(step - cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["rollout_valid_number"][-1].append(rollout_valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        log_dict["simulated_number"][-1].append(simulated_num)
                        log_dict["time"][-1].append((datetime.now() - start_time).total_seconds())
                        print(f"Root node ends, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num} / {method_num}")

                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        outputAllMethod(time)
                        return log_dict
                    
                    if current_node.done or (current_node.fullyExpanded() and not current_node.existBestChild((env.best_reward+1.)*0.5)):
                        print(f"Trial {trial} ends, Step: {step}, Cut: {cut_num}, V / RV / T: {valid_method_num} / {rollout_valid_method_num} / {method_num}")
                        if best_reward <= previous_best and previous_best > base_reward:
                            epsilon_bonus += 1
                            dead_count += 1
                        else:
                            epsilon_bonus = 1
                            dead_count = 0
                        outputAllMethod(time)
                        break

                    front_node, depth_use = env.treePolicy(current_node, eps)
                    level = [len(env.current_trajectory[i]) for i in range(env.string_number)]

                    if front_node.parent == None and (front_node.children == None or (front_node.fullyExpanded() and not front_node.existBestChild((env.best_reward+1.)*0.5))):
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append((reward+1)*0.5)
                        log_dict["tree_policy_max_reward_returns"][-1].append((tree_policy_best_reward+1)*0.5)
                        log_dict["reward_returns"][-1].append((sum(env.avg_reward) / 100.+1)*0.5)
                        log_dict["max_reward_returns"][-1].append((best_reward+1)*0.5)
                        log_dict["valid_reward_returns"][-1].append((env.valid_best_reward+1)*0.5)
                        log_dict["cut_number"][-1].append(step - cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["rollout_valid_number"][-1].append(rollout_valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        log_dict["simulated_number"][-1].append(simulated_num)
                        log_dict["time"][-1].append((datetime.now() - start_time).total_seconds())
                        print(f"Root node ends, Step: {step}, Cut: {cut_num}, Valid / Total: {valid_method_num} / {method_num}")
                        env.outputGraphAndCSV(log_dict)
                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        outputAllMethod(time)
                        return log_dict
                    
                    if front_node.children == None or (front_node.fullyExpanded() and not front_node.existBestChild((env.best_reward+1.)*0.5)):
                        print(f"Trial {trial} ends, Step: {step}, Cut: {cut_num}, V / RV / T: {valid_method_num} / {rollout_valid_method_num} / {method_num}")
                        if best_reward <= previous_best and previous_best > base_reward:
                            epsilon_bonus += 1
                            dead_count += 1
                        else:
                            epsilon_bonus = 1
                            dead_count = 0
                        outputAllMethod(time)
                        break
                    
                    new_best = False
                    # print(f"step: {step}, depth use: {depth_use}")
                    if not front_node.done:
                        rollout_number = int(ROLLOUT_NUM(depth_left - 1 - depth_use, valid_child_num))
                        # rollout_number = (front_node.maximum_child) ** 2
                        if valid_child_num < 2 or rollout_number <= 0:
                            rollout_number = 0
                        methods, invalid_list, rollout_number = env.rollout(rollout_number, front_node)
                        # length = [sum([len(method) for method in methods[i]]) for i in range(len(methods))]
                        reward_list, r1_list, r2_list, ratio_list, go_over_all_units_list, As_list, AAs_list, sub_simulated_num = env.getRewardList(methods, origami, invalid_list, step)
                        simulated_num += sub_simulated_num
                        total_reward = 0.0

                        rollout_number = min(rollout_number, len(reward_list))

                        if rollout_number > 0:    
                            total_reward = (np.mean(reward_list) + 1) * 0.5    
                            env.backward(front_node, total_reward, 1)
                            env.avg_reward[roller % 100] = np.mean(reward_list)
                            roller += 1
                            
                        for i in range(rollout_number):
                            reward = reward_list[i]
   
                            if reward > best_reward:
                                best_reward = reward
                                new_best = True
                                if reward > 0:
                                    best_trajectory = env.packTrajectory(methods[i])
                                    try:
                                        env.output(step, best_trajectory, best_reward)
                                    except:
                                        pass
                            
                            if reward > base_reward + valid_base_reward:
                                # if ratio_list[i] < RATIO_LOWER_BOUND ** DUPLICATE_FACTOR or As_list[i] == 0 or AAs_list[i] == 0 or go_over_all_units_list[i] < GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR:
                                #     continue
                                if reward > env.valid_best_reward:
                                    env.valid_best_reward = reward
                                    
                                equal = False
                                current_method = env.packTrajectory(methods[i])
                                x_con, y_con = env.getXYDiversion(current_method)

                                id = env.calculateBufferId(methods[i], False)

                                # index = int(reward_list[i] * 2)
                                # for j in range(len(candidate_reward[index])):
                                #     if abs(reward_list[i] - candidate_reward[index][j]) < diversity_tolerance:
                                #         if abs(abs(x_con) - abs(candidate_xy_con[index][j][X])) < diversity_tolerance and abs(abs(y_con) - abs(candidate_xy_con[index][j][Y])) < diversity_tolerance:
                                #             equal = True
                                #             break
                                
                                if enable_diversity:
                                    for j in range(len(candidate_methods)):
                                        if abs(reward_list[i] - candidate_methods[j]["score"]) < diversity_tolerance:
                                            if (abs(abs(x_con) - abs(candidate_methods[j]["xy_con"][X])) < diversity_tolerance and abs(abs(y_con) - abs(candidate_methods[j]["xy_con"][Y])) < diversity_tolerance)\
                                                and 1:#not (abs(x_con - candidate_methods[j]["xy_con"][X]) < diversity_tolerance and abs(y_con - candidate_methods[j]["xy_con"][Y]) < diversity_tolerance):
                                                equal = True
                                                break
                                
                                if not equal:
                                    add_new = True
                                    if not enable_duplicate_id:
                                        if id in candidate_method_ids:
                                            add_new = False
                                            idx = candidate_method_ids.index(id)
                                            if candidate_method_rewards[idx] < reward_list[i]:
                                                candidate_method_rewards[idx] = reward_list[i]
                                                if candidate_method_types[idx]:
                                                    valid_method_num -= 1
                                                    rollout_valid_method_num += 1
                                                    candidate_method_types[idx] = False
                                                if not calculate_number:
                                                    candidate_methods[idx]['method'] = current_method
                                                    candidate_methods[idx]['score'] = reward_list[i]
                                                    candidate_methods[idx]['reward_speed'] = r1_list[i]
                                                    candidate_methods[idx]['reward_force'] = r2_list[i]
                                                    candidate_methods[idx]['form'] = "rollout"
                                                    candidate_methods[idx]['xy_con'] = (x_con, y_con)
                                    if add_new:
                                        rollout_valid_method_num += 1
                                        info = 0
                                        if not calculate_number:
                                            info = {
                                                "method": current_method,
                                                "score": reward_list[i],
                                                "reward_speed": r1_list[i],
                                                "reward_force": r2_list[i],
                                                "form": "rollout",
                                                "id": id,
                                                "xy_con": (x_con, y_con)
                                            }
                                        candidate_methods.append(info)      
                                        candidate_method_ids.append(id)
                                        candidate_method_rewards.append(reward_list[i])   
                                        candidate_method_types.append(False)
                                        outputAllMethod(time)
                                    # candidate_reward[index].append(reward_list[i])
                                    # candidate_xy_con[index].append((x_con, y_con))
                                    # candidate_form[index].append(0)

                    else:
                        if env.reward < 0:
                            # length = [sum([len(method) for method in env.current_trajectory]) for i in range(len(env.current_trajectory))]
                            reward = env.reward
                            env.avg_reward[roller % 100] = reward
                            roller += 1
                        else:
                            if front_node.visits != 1:
                                reward = front_node.reward / (front_node.visits - 1) * 2. - 1.
                                env.avg_reward[roller % 100] = reward
                                roller += 1
                            else:
                                method_num += 1
                                reward_list, r1_list, r2_list, ratio_list, go_over_all_units_list, As_list, AAs_list, sub_simulated_num = env.getRewardList([deepcopy(env.current_trajectory)], origami, [1], step)
                                simulated_num += sub_simulated_num
                                reward = reward_list[0]

                                env.avg_reward[roller % 100] = reward
                                roller += 1
                                if reward > tree_policy_best_reward:
                                    tree_policy_best_reward = reward
                                if reward > best_reward:
                                    best_reward = reward
                                    best_trajectory = env.packTrajectory(env.new_trajectory)
                                    try:
                                        env.output(step, best_trajectory, best_reward)
                                        new_best = True
                                        # print(f"Best: {best_trajectory}")
                                    except:
                                        pass
                                    
                                if reward > base_reward + valid_base_reward:
                                    # if not (ratio_list[0] < RATIO_LOWER_BOUND ** DUPLICATE_FACTOR or As_list[0] == 0 or AAs_list[0] == 0 or go_over_all_units_list[0] < GOOVER_UNIT_LOWER_BOUND ** GOOVER_UNIT_FACTOR):
                                    if reward > env.valid_best_reward:
                                        env.valid_best_reward = reward
                                    equal = False
                                    current_method = env.packTrajectory(env.current_trajectory)
                                    # identity = env.calculateIdentity()
                                    x_con, y_con = env.getXYDiversion(current_method)
                                    
                                    id = env.calculateBufferId(env.current_trajectory, False)
                                    # id_for_each_string = [0 for _ in range(string_number)]
                                    # cal_times = [0 for _ in range(env.unit_number)]
                                    # index = 0
                                    # for string in env.current_trajectory:
                                    #     for node in string:
                                    #         cal_index = node[0] - len(env.P_points)
                                    #         if cal_index >= 0:
                                    #             id += 2 ** (cal_times[cal_index] * env.unit_number + cal_index)
                                    #             id_for_each_string[index] += 2 ** cal_index
                                    #             cal_times[cal_index] += 1
                                    #     index += 1

                                    # index = int(reward * 2)
                                    # for i in range(len(candidate_reward[index])):
                                    #     if abs(reward - candidate_reward[index][i]) < diversity_tolerance:
                                    #         if abs(abs(x_con) - abs(candidate_xy_con[index][i][X])) < diversity_tolerance and abs(abs(y_con) - abs(candidate_xy_con[index][i][Y])) < diversity_tolerance:
                                    #             equal = True
                                    #             form = candidate_form[index][i]
                                    #             break
                                    
                                    if enable_diversity:
                                        equal_index = 0
                                        for j in range(len(candidate_methods)):
                                            if abs(reward - candidate_methods[j]["score"]) < diversity_tolerance:
                                                if abs(abs(x_con) - abs(candidate_methods[j]["xy_con"][X])) < diversity_tolerance and abs(abs(y_con) - abs(candidate_methods[j]["xy_con"][Y])) < diversity_tolerance\
                                                    and 1:#not (abs(x_con - candidate_methods[j]["xy_con"][X]) < diversity_tolerance and abs(y_con - candidate_methods[j]["xy_con"][Y]) < diversity_tolerance):
                                                    equal = True
                                                    form = candidate_methods[j]["form"]
                                                    equal_index = j
                                                    break
                                            
                                    if not equal:
                                        add_new = True
                                        if not enable_duplicate_id:
                                            if id in candidate_method_ids:
                                                idx = candidate_method_ids.index(id)
                                                add_new = False

                                                if candidate_method_rewards[idx] < reward:
                                                    candidate_method_rewards[idx] = reward
                                                    if not candidate_method_types[idx]:
                                                        valid_method_num += 1
                                                        rollout_valid_method_num -= 1
                                                        candidate_method_types[idx] = True

                                                    if not calculate_number:
                                                        candidate_methods[idx]['method'] = current_method
                                                        candidate_methods[idx]['score'] = reward
                                                        candidate_methods[idx]['reward_speed'] = r1_list[0]
                                                        candidate_methods[idx]['reward_force'] = r2_list[0]
                                                        candidate_methods[idx]['form'] = "tree"
                                                        candidate_methods[idx]['xy_con'] = (x_con, y_con)

                                                else:
                                                    if not candidate_method_types[idx]:
                                                        valid_method_num += 1
                                                        rollout_valid_method_num -= 1
                                                        candidate_method_types[idx] = True
                                                    
                                                    if not calculate_number:
                                                        if candidate_methods[idx]["form"] == "rollout":
                                                            candidate_methods[idx]["form"] = "tree"

                                        if add_new:
                                            valid_method_num += 1
                                            info = 0
                                            if not calculate_number:
                                                info = {
                                                    "method": current_method,
                                                    "score": reward,
                                                    "reward_speed": r1_list[0],
                                                    "reward_force": r2_list[0],
                                                    "id": id,
                                                    "xy_con": (x_con, y_con)
                                                }
                                            candidate_methods.append(info)
                                            candidate_method_ids.append(id)
                                            candidate_method_rewards.append(reward)   
                                            candidate_method_types.append(True)
                                            outputAllMethod(time)
                                        # candidate_reward[index].append(reward)
                                        # candidate_xy_con[index].append((x_con, y_con))
                                        # candidate_form[index].append(1)
                                    else:
                                        if form == "rollout":
                                            valid_method_num += 1
                                            rollout_valid_method_num -= 1
                                            candidate_methods[equal_index]["form"] = "tree"

                        backward_reward = (reward + 1.) * 0.5
    
                        env.backward(front_node, backward_reward, 1)

                    env.best_reward = best_reward     

                    output_result = False
                    if simulated_num > log_dict["simulated_number"][-1][-1]:
                        output_result = True
                        
                    if (step - last_step == int(STEP_IN_INTERVAL(depth_left, valid_child_num)) or new_best or output_result):
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append((reward+1)*0.5)
                        log_dict["tree_policy_max_reward_returns"][-1].append((tree_policy_best_reward+1)*0.5)
                        log_dict["reward_returns"][-1].append((sum(env.avg_reward) / 100.+1)*0.5)
                        log_dict["valid_reward_returns"][-1].append((env.valid_best_reward+1)*0.5)
                        log_dict["max_reward_returns"][-1].append((best_reward+1)*0.5)
                        log_dict["cut_number"][-1].append(step - cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["rollout_valid_number"][-1].append(rollout_valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        log_dict["simulated_number"][-1].append(simulated_num)
                        log_dict["time"][-1].append((datetime.now() - start_time).total_seconds())
                        if new_best:
                            try:
                                env.output(step, best_trajectory, best_reward)
                                if not calculate_number:
                                    print(best_trajectory)
                            except:
                                pass
                        print(f"Step: {step}, Best: {round(env.valid_best_reward, 2)} / {round(best_reward, 2)}, Dead count: {dead_count} / {dead_maximum}, Level: {level}")

                    if (step % record_episode == 0) or new_best or output_result: 
                        env.outputGraphAndCSV(log_dict)
                        
                    if step - last_step == int(STEP_IN_INTERVAL(depth_left, valid_child_num)) or valid_child_num < 2:
                        step_delta = step - last_step
                        last_step = step
                        while 1:
                            depth += 1
                            depth_left = string_number * (upper_bound + 1) - depth
                            string_id = env.root_node.string_id
                            try:
                                env.root_node, action_id = env.root_node.bestChild(0, fail_reward)
                            except:
                                error_flag = 1
                                break
                            fixed_action_list[string_id].append(action_id)
                            print(f"Sub-trial: {trial}-{string_id}-{len(fixed_action_list[string_id])} ends using {step_delta} steps, Choose action: {action_id}, Depth: {depth}")
                            # print(f"Current fixed policy list: {fixed_action_list}")
                            
                            if env.root_node.done:
                                break
                            
                            valid_child_num = env.root_node.existOnlyOneValid(fail_reward)
                            if valid_child_num > 1:
                                break

                    if error_flag:
                        break
                        
    return log_dict

def workerMultisim(mlist, origami, 
                   methods, 
                   valid_number, 
                   cons, p_points, 
                   p_connections, 
                   max_edge, units, max_size, total_bias, buf_id, interval, extract, lower, 
                   height, string_number, fix_id_list, gravity_flag, prefold, miu, targets, ctm, noise_seed, fm, speed_bonus,
                   ground_enable, control_mode_training_time, simulation_upper_time, additional_length, robot_type, type_of_controller, stroke_percent):
    
    if gravity_flag == 0:
        standard_g = [0., 0., 0.]
    elif gravity_flag == 1:
        standard_g = [9810., 0., 0.]
    elif gravity_flag == 2:
        standard_g = [0., 9810., 0.]
    elif gravity_flag == 3:
        standard_g = [0., 0., 9810.]
    elif gravity_flag == 4:
        standard_g = [-9810., 0., 0.]
    elif gravity_flag == 5:
        standard_g = [0., -9810., 0.]
    elif gravity_flag == 6:
        standard_g = [0., 0., -9810.]

    print(stroke_percent)
    if len(methods):
        ori_sim = OrigamiSimulator(use_gui=0, fast_simulation=1, origami_name=origami, g=standard_g, strict=extract, h=height, ground_miu=miu,
                                control_mode=ctm, friction_mode=fm, speed_bonus=speed_bonus, check_connection_matrix=False, const_stiff_of_crease=False,
                                    a_speed=reward_coeff[0], a_act=reward_coeff[1], a_force=reward_coeff[2], robot_type=robot_type,
                                    default_ground=ground_enable, simulation_upper_time=simulation_upper_time,
                                    control_mode_training_time=control_mode_training_time, additional_length=additional_length,
                                    type_of_controller=type_of_controller, stroke_percent=stroke_percent)
        
        ori_sim.targets = targets

        ori_sim.prefold = prefold

        ori_sim.start(origami, 4, ori_sim.TSA_SIM, 0, string_number)
        
        noise = None
        
        for index in range(len(methods)):
            ori_sim.ID = buf_id + lower + index
            ori_sim.method = methods[index]
            
            ori_sim.initializeRunning(extract, noise)

            ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
            
            while 1:
                ori_sim.step()
                if ori_sim.stop():
                    break
                
            value, value_f, rf, rm, rs, actuator_bonus, rh, rforce, rforce2 = ori_sim.reward()

            mlist[lower + index] = value
            mlist[lower + index + valid_number] = value_f
            mlist[lower + index + valid_number * 2] = rf
            mlist[lower + index + valid_number * 3] = rm
            mlist[lower + index + valid_number * 4] = rs
            mlist[lower + index + valid_number * 5] = actuator_bonus
            mlist[lower + index + valid_number * 6] = rh
            mlist[lower + index + valid_number * 7] = rforce
            mlist[lower + index + valid_number * 8] = rforce2
            
            print("No. " + str(ori_sim.ID) + ", Value: " + str(round(value, 3)))
        
        gc.collect()

        ti.reset()


if __name__ == "__main__":
    normal = 1
    if "-o" in sys.argv:
        index = sys.argv.index("-o") + 1
        origami = sys.argv[index]
        
    if "-s" in sys.argv:
        index = sys.argv.index("-s") + 1
        string_number_text = sys.argv[index]
        numbers = re.findall(r"\d+", string_number_text)
        if len(numbers) == 0:
            print("Wrong string number!")
            normal = 0
        else:
            string_number = int(numbers[0])
        fixed_action_list_initial = [
            [] for _ in range(string_number + 1)
        ]

    if "-t" in sys.argv:
        index = sys.argv.index("-t") + 1
        str_thread_number = sys.argv[index]
        numbers = re.findall(r"\d+", str_thread_number)
        if len(numbers) == 0:
            print("Wrong thread number!")
            normal = 0
        else:
            thread_number = int(numbers[0])
            if thread_number > 16:
                thread_number = 16
    
    print_result = 0
    if "-print" in sys.argv:
        index = sys.argv.index("-print") + 1
        str_print_number = sys.argv[index]
        numbers = re.findall(r"\d+", str_print_number)
        if len(numbers) == 0:
            print("Wrong print number!")
            normal = 0
        else:
            print_result = int(numbers[0])
            
    if "-arbi" in sys.argv:
        index = sys.argv.index("-arbi") + 1
        str_arbi_number = sys.argv[index]
        numbers = re.findall(r"\d+", str_arbi_number)
        if len(numbers) == 0:
            print("Wrong settings!")
            normal = 0
        else:
            arbitrary_crease_mode = int(numbers[0])

    if "-ea-init" in sys.argv:
        index = sys.argv.index("-ea-init") + 1
        str_ea_init = sys.argv[index]
        numbers = re.findall(r"\d+", str_ea_init)
        if len(numbers) > 0:
            ea_constraint_initial_segment = bool(int(numbers[0]))

    if "-ea-final" in sys.argv:
        index = sys.argv.index("-ea-final") + 1
        str_ea_final = sys.argv[index]
        numbers = re.findall(r"\d+", str_ea_final)
        if len(numbers) > 0:
            ea_constraint_final_tip = bool(int(numbers[0]))

    if "-work" in sys.argv:
        index = sys.argv.index("-work") + 1
        str_work = sys.argv[index]
        numbers = re.findall(r"\d+", str_work)
        if len(numbers) > 0:
            enable_work_criteria = bool(int(numbers[0]))

    if "-search-mode" in sys.argv:
        index = sys.argv.index("-search-mode") + 1
        str_search = sys.argv[index]
        numbers = re.findall(r"\d+", str_search)
        if len(numbers) > 0:
            search_mode = int(numbers[0])

    # --- Simulation config CLI overrides ---
    if "-height" in sys.argv:
        index = sys.argv.index("-height") + 1
        sim_override["platform_height"] = float(re.findall(r"[\d.]+", sys.argv[index])[0])
    if "-ctm" in sys.argv:
        index = sys.argv.index("-ctm") + 1
        sim_override["control_mode"] = int(re.findall(r"\d+", sys.argv[index])[0])
    if "-ground" in sys.argv:
        index = sys.argv.index("-ground") + 1
        sim_override["ground_enable"] = int(re.findall(r"\d+", sys.argv[index])[0])
    if "-gravity" in sys.argv:
        index = sys.argv.index("-gravity") + 1
        sim_override["gravity_flag"] = int(re.findall(r"\d+", sys.argv[index])[0])
    if "-sim-time" in sys.argv:
        index = sys.argv.index("-sim-time") + 1
        sim_override["control_mode_training_time"] = float(re.findall(r"[\d.]+", sys.argv[index])[0])
    if "-sim-upper" in sys.argv:
        index = sys.argv.index("-sim-upper") + 1
        sim_override["simulation_upper_time"] = float(re.findall(r"[\d.]+", sys.argv[index])[0])
    if "-add-len" in sys.argv:
        index = sys.argv.index("-add-len") + 1
        sim_override["additional_length_of_string"] = float(re.findall(r"[\d.]+", sys.argv[index])[0])
    if "-robot-type" in sys.argv:
        index = sys.argv.index("-robot-type") + 1
        sim_override["robot_type"] = int(re.findall(r"\d+", sys.argv[index])[0])
    if "-miu" in sys.argv:
        index = sys.argv.index("-miu") + 1
        sim_override["miu"] = float(re.findall(r"[\d.]+", sys.argv[index])[0])
    if "-ctrl-type" in sys.argv:
        index = sys.argv.index("-ctrl-type") + 1
        sim_override["type_of_controller"] = int(re.findall(r"\d+", sys.argv[index])[0])
         
    print(f"Origami: {origami}, String Number: {string_number}, Threads: {thread_number}")    
    
    i = 0
    j = 0
    env = Env(discount=i, upper_bound_bonus=j, print_result=1)
    maximum_discount = max(0, UNIT_PASS - 2)
    maximum_bonus = 2
    # env.getRewardList(methods=[
    #     [
    #         [(0, 1), (5, 1), (3, -1)],
    #         [(2, 1), (4, 1), (6, -1)],
    #     ]
    # ],
    # origami=origami,invalid_list=[1],step=0)
    del(env)
    while 1:
        env = Env(discount=i, upper_bound_bonus=j, print_result=print_result)
        log_dict = train(env, origami)
        env.outputGraphAndCSV(log_dict)
        if log_dict["valid_number"][-1][-1] + log_dict["rollout_valid_number"][-1][-1] == 0:
            if i < maximum_discount and log_dict["simulated_number"][-1][-1] == 0:
                best_pass = UNIT_PASS - round((env.best_reward + 1.) * UNIT_PASS)
                if best_pass >= 1:
                    i += best_pass
                else:
                    i += 1
                del(env)
            elif j < maximum_bonus and log_dict["simulated_number"][-1][-1] > 0:
                j += 1
                del(env)
            else:
                i = 0
                j = 0
                del(env)
                string_number += 1
                if string_number > 6:
                    break
                fixed_action_list_initial = [
                    [] for _ in range(string_number + 1)
                ]
        else:
            break
    