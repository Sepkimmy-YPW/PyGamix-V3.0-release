import os
import sys
import re
import numpy as np
from copy import deepcopy
# import random
# import math
# import matplotlib.pyplot as plt
# from units import UnitPackParserReverse
# import dxfgrabber
from phys_sim25 import OrigamiSimulator
from utils import *
import json
import pandas as pd
import multiprocessing

Version = 1.1

START = 0
END = 1
X = 0
Y = 1
VALLEY = 0
MOUNTAIN = 1
BORDER = 2

MIN_A = 4
LB = 0.0
GB = 0.0

bonus_val = 1.
fail_reward = -bonus_val
encourage_reward = 1.0
episodes = 4001

seed = 3407

string_number = 2
direct_reward = False
height = 30
gravity_flag = 0
prefold = 0.00
miu = 0.6

cut_num = 0
SIM = 1
origami = "mountain15"
# reward_coeff
reward_coeff = [0.2, 0.0, 0.0] # maximal speed, minimal actuator num, minimal mises stress
new_method = False
calculate_number = False
initialize = False
cut_nodes = True

candidate_methods = []
best_method_list = []
time = 1
step = 0
maximum_valid = SIM

upper_bound = 5

maximum_thread = 1
simulation_case = 8

FOLDING_MAXIMUM = 163.0 / 180.0 * 0.95

BUFFER_INIT = 512

best_solution_upperbound = 1.0
extract_simulation_speed = 1.0

extract_mode = 0
control_mode = 0
friction_mode = 0
restart = 0

def workerMultisim(mlist, origami,
                   methods, 
                   valid_number, 
                   cons, p_points, 
                   p_connections, 
                   max_edge, units, max_size, total_bias, buf_id, interval, extract, lower, 
                   height, string_number, fix_id_list, gravity_flag, prefold, miu, targets, ctm, noise_seed, fm, final_file_path):
    
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
      
    ori_sim = OrigamiSimulator(use_gui=False, origami_name=origami, g=standard_g, strict=extract, h=height, ground_miu=miu, \
                               control_mode=ctm, friction_mode=fm, check_connection_matrix=False, const_stiff_of_crease=False)

    if origami == 'bird3':
        ori_sim.additional_length_of_string = 400.0
    elif origami == 'robot-24-11':
        ori_sim.additional_length_of_string = 0.0
    else:
        ori_sim.additional_length_of_string = 100.0
    
    ori_sim.targets = targets

    ori_sim.prefold = prefold

    ori_sim.start(origami, 4, ori_sim.TSA_SIM, 0, string_number)
    
    noise = None
    
    if extract == 2:
        noise_dict = None
        try:
            with open(os.path.join(final_file_path, f"noise_3407.json"), 'r', encoding='utf-8') as fw:
                noise_dict = json.load(fw)
            print(f"Seed dict {noise_seed} exists!")
        except:
            noise_dict = None
            
        np.random.seed(noise_seed)
        # noise = np.random.normal(loc=0, scale=0.01, size=ori_sim.bending_pairs_num)
        noise = np.random.uniform(low=-0.02, high=0.02, size=ori_sim.bending_pairs_num)
        
        if noise_dict != None:
            total_num = noise_dict["total"]
            positive_noise_list = noise_dict["+1"]
            zero_noise_list = noise_dict["0"]
            negative_noise_list = noise_dict["-1"]
            for i in range(total_num):
                if i in positive_noise_list:
                    noise[i] = abs(noise[i])
                elif i in zero_noise_list:
                    noise[i] = 0.0
                elif i in negative_noise_list:
                    noise[i] = -abs(noise[i])
                
        print(f"Seed: {noise_seed}, Noise: {(noise * 180.0).tolist()}")
    
    for index in range(len(methods)):
        ori_sim.ID = buf_id + lower + index
        print(f"EXMODE: {extract}")
        ori_sim.method = methods[index]
        
        ori_sim.initializeRunning(extract, noise)

        if extract != 3:
            ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
        else:
            ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step * extract_simulation_speed
        
        while 1:
            ori_sim.step()
            if ori_sim.stop():
                break
            
        value, value_without_actuator, rf, rm, rs, actuator_bonus = ori_sim.reward()

        mlist[lower + index] = value
        mlist[lower + index + valid_number] = value_without_actuator
        mlist[lower + index + valid_number * 2] = rf
        mlist[lower + index + valid_number * 3] = rm
        mlist[lower + index + valid_number * 4] = rs
        mlist[lower + index + valid_number * 5] = actuator_bonus

        left = 0
        for i in range(valid_number):
            if mlist[i + valid_number * 5] == 0.0:
                left += 1
        
        print("No. " + str(ori_sim.ID) + ", Value: " + str(round(value, 3)) + ", with sub-value: " + str(round(rf, 3)) + '/' + str(round(rm, 3)) + '/' + str(round(rs, 3)) + '/' + str(round(actuator_bonus, 3)) + ', ' + str(left) + " left.")

class Env:
    def __init__(self) -> None:
        self.max_edge = 4
        self.output_reward_buffer = True

        # 获取点和线段信息
        with open(f"./descriptionData/{origami}.json", 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
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
            self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
            self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]
        for i in range(len(input_json["units"])):
            self.units.append(Unit())
            kps = input_json["units"][i]
            for j in range(0, -len(kps), -1):
                crease_type = BORDER
                hard = False
                current_kp = [kps[j][X], kps[j][Y]]
                next_kp = [kps[j - 1][X], kps[j - 1][Y]]
                for line in self.lines:
                    if (distance(line[START], current_kp) < 1e-3 and distance(line[END], next_kp) < 1e-3) or \
                        (distance(line[END], current_kp) < 1e-3 and distance(line[START], next_kp) < 1e-3):
                        crease_type = line.getType()
                        hard = line.hard
                        folding_angle_upper_bound = line.folding_angle_upper_bound
                        folding_angle_lower_bound = line.folding_angle_lower_bound
                        break
                self.units[i].addCrease(Crease(
                    current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                ))
        try:
            self.contributions = deepcopy(input_json["contributions"])
            for i in range(len(self.contributions)):
                new_contribution = []
                for j in range(0, -len(self.contributions[i]), -1):
                    new_contribution.append(self.contributions[i][j])
                self.contributions[i] = new_contribution
        except:
            self.contributions = [[]]
        
        try:
            self.fix_id = deepcopy(input_json["fix"])
        except:
            self.fix_id = [-1]
        try:
            self.targets = deepcopy(input_json["crease_angle"])
        except:
            self.targets = []
            
        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)

        self.unit_number = len(self.units)
        self.best_reward = 0.0
        self.node_num = 1

        self.string_number = string_number

        self.P_candidators = input_json["P_candidators"]["points"]
        self.P_candidators_connections = input_json["P_candidators"]["connections"]

        self.P_points = [
            np.array(self.P_candidators[i]) for i in range(len(self.P_candidators))
        ]

        self.O_points = [
            np.array(self.units[i].getCenter()) for i in range(self.unit_number)
        ]

    def getRewardList(self, methods, origami, buf_id, sim_time, extract=False, height=10, string_number=2, gravity_flag=0, prefold=0.00, miu=0.6, ctm=0, noise=0, fm=0, final_file_path=""):
        valid_number = len(methods)

        if valid_number >= 1: 
            print("There are " + str(valid_number) + " valid cases, using multi-process technology")
            initial_fitness_list = [0. for _ in range(6 * valid_number)]

            mlist = multiprocessing.Manager().list(initial_fitness_list)

            p_list = []

            pointer = 0
            previous_lower = 0

            while pointer < maximum_thread:
                interval = simulation_case
                normal_cases = maximum_thread * simulation_case
                
                lower = pointer * interval
                upper = (pointer + 1) * interval

                if valid_number < normal_cases:
                    interval = int(valid_number / maximum_thread)
                    delta = valid_number - interval * maximum_thread
                    if pointer < delta:
                        interval += 1
                    lower = previous_lower
                    upper = lower + interval
                    previous_lower = upper
                    if upper > valid_number:
                        upper = valid_number
                all_method = methods[lower: upper]
                print(f"Allocating range {lower} to {upper} in process {pointer}. Total number: {upper - lower}")
                p = multiprocessing.Process(target=workerMultisim, args=(
                        mlist, origami, all_method, valid_number, self.contributions,
                        self.P_points, self.P_candidators_connections, self.max_edge, self.units, self.max_size, self.total_bias, 
                        buf_id + sim_time * maximum_thread * simulation_case, interval, extract, lower, height, string_number, self.fix_id, gravity_flag, prefold, miu, self.targets, 
                        ctm, noise, fm, final_file_path
                    )
                )
                p_list.append(p)
                pointer += 1
            
            process_id = 0

            total_process_number = maximum_thread

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
    
def train(env: Env, origami, buf=0, extract=0, height=10, string_number=2, gravity_flag=0, prefold=0.00, miu=0.6, ctm=0, fm=0):
    if extract in [-1, 0, 1]:
        try:
            path = os.path.join(final_file_path, f"result_{time}_buf_{buf}_true_reward.json")
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
        except:
            path = os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json")
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
    else:
        try:
            path = os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json")
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
        except:
            try:
                path = os.path.join(final_file_path, f"result_{time}_buf_{buf}_true_reward.json")
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
            except:
                path = os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json")
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
    try:
        path = os.path.join(final_file_path, f"sim_buffer.json")
        with open(path, 'r', encoding='utf-8') as fw:
            input_sim_buffer = json.load(fw)
        i = 0
        print("Deleting zero reward in buffer, current buffer size: " + str(len(input_sim_buffer["buf"])))
        while i < len(input_sim_buffer["buf"]):
            if input_sim_buffer["buf"][i]["reward"] == 0.0 and input_sim_buffer["buf"][i]["ra"] == 0.0:
                del(input_sim_buffer["buf"][i])
                i -= 1
            i += 1
        print("Current buffer size: " + str(len(input_sim_buffer["buf"])))
    except:
        input_sim_buffer = {
            "buf": []
        }

    total_number = len(input_json["method"])
    methods = input_json["method"]
    # maximum_thread = maximum_thread
    sub_methods = []
    sub_ids = []
    best_reward = fail_reward
    sim_step = []
    tension_scores = []
    reward_scores = []
    ids = []
    id_list = []
    buffer = []

    sim_time = 0
    
    noise = 3407 + buf

    print(f"----Training using candidate: {final_file_path}. \nTotal number: {total_number}----")
    
    for step in range(0, total_number):
        sim_step.append(step)
        tension_scores.append(methods[step]["score"])
        ids.append(methods[step]["id"])
        id_list.append(methods[step]["id_list"])
        if extract <= 0:
            try:
                find = False
                if len(input_sim_buffer['buf']):
                    current_id = ids[step]
                    current_id_list = id_list[step]
                    for i in range(len(input_sim_buffer["buf"])):
                        if input_sim_buffer["buf"][i]["id"] == current_id and input_sim_buffer["buf"][i]["id_list"] == current_id_list:
                            reward = input_sim_buffer["buf"][i]["reward"]
                            reward_without_actuator = input_sim_buffer['buf'][i]["reward_without_actuator"]
                            rf = input_sim_buffer["buf"][i]["rf"]
                            rm = input_sim_buffer["buf"][i]["rm"]
                            rs = input_sim_buffer["buf"][i]["rs"]
                            ra = input_sim_buffer["buf"][i]["ra"]
                            phase = input_sim_buffer["buf"][i]["phase"]
                            recal = False
                            
                            input_sim_buffer["buf"][i]["reward"] = reward
                            find = True
                            print(f"Reward is already in buffer. Step: {step + buf}, Reward: {round(reward, 3)} / {round(best_reward, 3)}, Sub-reward: {round(rf, 3)} / {round(rm, 3)} / {round(rs, 3)} / {round(ra, 3)}" + ("/Recal/" if recal else ""))
                            break
                if not find:
                    reward = methods[step]["reward"]
                    reward_without_actuator = methods[step]["reward_without_actuator"]
                    rf = methods[step]["rf"]
                    rm = methods[step]["rm"]
                    rs = methods[step]["rs"]
                    ra = methods[step]["ra"]
                    phase = methods[step]["phase"]
                    recal = False
                    
                    print(f"Reward is already in history. Step: {step + buf}, Reward: {round(reward, 3)} / {round(best_reward, 3)}, Sub-reward: {round(rf, 3)} / {round(rm, 3)} / {round(rs, 3)} / {round(ra, 3)}" + ("/Recal/" if recal else ""))
                
                if (reward == 0.0 and ra == 0.0) or (reward == 1.0 and phase == -1 and extract == 0):
                    raise RuntimeError
                
                reward_scores.append(reward)
                if reward >= best_reward:
                    best_reward = reward
                    best_trajectory = methods[step]["method"]
                    output_trajectory = deepcopy(best_trajectory)
                    output_trajectory["location"] = [[] for _ in range(env.string_number)]
                    for i in range(len(best_trajectory['id'])):
                        for j in range(len(best_trajectory['id'][i])):
                            if best_trajectory['type'][i][j] == 'A':
                                output_trajectory['location'][i].append(env.P_points[best_trajectory['id'][i][j]].tolist())
                            else:
                                output_trajectory['location'][i].append(env.O_points[best_trajectory['id'][i][j]].tolist())
                    # with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_buf_{buf}_best.json"), 'w', encoding="utf-8") as f:
                    #     json.dump(output_trajectory, f, indent=4)
                if reward >= best_solution_upperbound:
                    best_method_list.append(methods[step])
                    with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "method": best_method_list,
                            "number": len(best_method_list),
                            "control_mode": control_mode,
                            "phase": max(extract, phase)
                        }, f, indent=4)  
                if len(input_sim_buffer['buf']) == 0:
                    buffer.append({
                        "id": ids[step],
                        "id_list": id_list[step],
                        "reward": reward,
                        "reward_without_actuator": reward_without_actuator,
                        "rf": rf,
                        "rm": rm,
                        "rs": rs,
                        "ra": ra,
                        "phase": max(extract, phase)
                    })
                    with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "buf": buffer,
                            "number": len(buffer)
                        }, f, indent=4)
                else:
                    if not find:
                        input_sim_buffer['buf'].append({
                            "id": ids[step], 
                            "id_list": id_list[step],
                            "reward": reward,
                            "reward_without_actuator": reward_without_actuator,
                            "rf": rf,
                            "rm": rm,
                            "rs": rs,
                            "ra": ra,
                            "phase": max(extract, phase)
                        })
                        input_sim_buffer['number'] = len(input_sim_buffer['buf'])
                        with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                            json.dump(input_sim_buffer, f, indent=4)
                    if find:
                        methods[step]["reward"] = reward
                        methods[step]['reward_without_actuator'] = reward_without_actuator
                        methods[step]["rf"] = rf
                        methods[step]["rm"] = rm
                        methods[step]["rs"] = rs
                        methods[step]["ra"] = ra
                        methods[step]["phase"] = max(extract, phase)
                        with open(os.path.join(final_file_path, f"result_{time}_buf_{buf}_true_reward.json"), 'w', encoding="utf-8") as f:
                            json.dump({
                                "method": methods,
                                "number": total_number,
                                "simulated_number": step,
                                "control_mode": control_mode,
                                "phase": max(extract, phase)
                            }, f, indent=4) 
                # print(best_trajectory)
                data = [[sim_step[i], tension_scores[i], reward_scores[i]] for i in range(len(sim_step))]
                test = pd.DataFrame(columns=["step", "tension_score", "true_reward"], data=data)
                test.to_csv(os.path.join(final_file_path, f'result_train_time_{time}_candidate_buf_{buf}.csv'))
                sim_time = int((step + 1) / (maximum_thread * simulation_case))
            except:
                # print(env.P_points)
                reward_scores.append(-1.0)
                if len(sub_methods) < maximum_thread * simulation_case and step < total_number:
                    sub_methods.append(methods[step]["method"])
                    sub_ids.append(step)
                    level = [len(ele) for ele in methods[step]["method"]["id"]]
                    # print(f"Add id {step} with tension score {tension_scores[-1]} and level {level}")
                    if len(sub_methods) < maximum_thread * simulation_case and step < total_number - 1:
                        continue
                    else:
                        print(f"Step: {step}, Simulation Sub-ID list: {sub_ids}")
                        reward_list = env.getRewardList(sub_methods, origami, buf, sim_time, extract, height, string_number, gravity_flag, prefold, miu, ctm, noise, fm, final_file_path)
                        sim_time += 1
                        for i in range(len(sub_methods)):
                            methods[sub_ids[i]]["reward"] = reward_list[i]
                            methods[sub_ids[i]]['reward_without_actuator'] = reward_list[i + len(sub_methods)]
                            methods[sub_ids[i]]["rf"] = reward_list[i + 2*len(sub_methods)]
                            methods[sub_ids[i]]["rm"] = reward_list[i + 3*len(sub_methods)]
                            methods[sub_ids[i]]["rs"] = reward_list[i + 4*len(sub_methods)]
                            methods[sub_ids[i]]["ra"] = reward_list[i + 5*len(sub_methods)]
                            methods[sub_ids[i]]["phase"] = extract
                            reward_scores[sub_ids[i]] = reward_list[i]
                            if reward_list[i] >= best_reward:
                                best_reward = reward_list[i]
                                best_trajectory = methods[sub_ids[i]]["method"]
                                output_trajectory = deepcopy(best_trajectory)
                                output_trajectory["location"] = [[] for _ in range(env.string_number)]
                                # print(best_trajectory)
                                for j in range(len(best_trajectory['id'])):
                                    for k in range(len(best_trajectory['id'][j])):
                                        if best_trajectory['type'][j][k] == 'A':
                                            output_trajectory['location'][j].append(env.P_points[best_trajectory['id'][j][k]])
                                        else:
                                            output_trajectory['location'][j].append(env.O_points[best_trajectory['id'][j][k]])
                                # with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_buf_{buf}_best.json"), 'w', encoding="utf-8") as f:
                                #     json.dump(best_trajectory, f, indent=4)
                            if reward_list[i] >= best_solution_upperbound:
                                best_method_list.append(methods[sub_ids[i]])
                                with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "method": best_method_list,
                                        "number": len(best_method_list),
                                        "control_mode": control_mode,
                                        "phase": extract
                                    }, f, indent=4)  
                            if len(input_sim_buffer['buf']) == 0:
                                buffer.append({
                                    "id": ids[sub_ids[i]],
                                    "id_list": id_list[sub_ids[i]],
                                    "reward": reward_list[i],
                                    "reward_without_actuator": reward_list[i + len(sub_methods)],
                                    "rf": reward_list[i + 2*len(sub_methods)],
                                    "rm": reward_list[i + 3*len(sub_methods)],
                                    "rs": reward_list[i + 4*len(sub_methods)],
                                    "ra": reward_list[i + 5*len(sub_methods)],
                                    "phase": extract
                                })
                                with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "buf": buffer,
                                        "number": len(buffer)
                                    }, f, indent=4)
                            else:
                                find_exist = False
                                for j in range(len(input_sim_buffer["buf"])):
                                    if input_sim_buffer["buf"][j]["id"] == ids[sub_ids[i]] and input_sim_buffer["buf"][j]["id_list"] == id_list[sub_ids[i]]:
                                        input_sim_buffer["buf"][j]["reward"] = reward_list[i]
                                        input_sim_buffer['buf'][j]["reward_without_actuator"] = reward_list[i + len(sub_methods)]
                                        input_sim_buffer["buf"][j]["rf"] = reward_list[i + 2*len(sub_methods)]
                                        input_sim_buffer["buf"][j]["rm"] = reward_list[i + 3*len(sub_methods)]
                                        input_sim_buffer["buf"][j]["rs"] = reward_list[i + 4*len(sub_methods)]
                                        input_sim_buffer["buf"][j]["ra"] = reward_list[i + 5*len(sub_methods)]
                                        input_sim_buffer["buf"][j]["phase"] = extract
                                        find_exist = True
                                if not find_exist:
                                    input_sim_buffer['buf'].append({
                                        "id": ids[sub_ids[i]], 
                                        "id_list": id_list[sub_ids[i]],
                                        "reward": reward_list[i],
                                        "reward_without_actuator": reward_list[i + len(sub_methods)],
                                        "rf": reward_list[i + 2*len(sub_methods)],
                                        "rm": reward_list[i + 3*len(sub_methods)],
                                        "rs": reward_list[i + 4*len(sub_methods)],
                                        "ra": reward_list[i + 5*len(sub_methods)],
                                        "phase": extract
                                    })
                                input_sim_buffer['number'] = len(input_sim_buffer['buf'])
                                with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                                    json.dump(input_sim_buffer, f, indent=4)

                        print(f"Step: {step}, Best reward: {best_reward}")
                        # print(best_trajectory)   
                        
                        with open(os.path.join(final_file_path, f"result_{time}_buf_{buf}_true_reward.json"), 'w', encoding="utf-8") as f:
                            json.dump({
                                "method": methods,
                                "number": total_number,
                                "simulated_number": step,
                                "control_mode": control_mode,
                                "phase": extract
                            }, f, indent=4)    
                        sub_methods.clear()
                        sub_ids.clear()

                        data = [[sim_step[i], tension_scores[i], reward_scores[i]] for i in range(len(sim_step))]
                        test = pd.DataFrame(columns=["step", "tension_score", "true_reward"], data=data)
                        test.to_csv(os.path.join(final_file_path, f'result_train_time_{time}_candidate_buf_{buf}.csv'))
        elif extract == 2:
            if len(sub_methods) < maximum_thread * simulation_case and step < total_number:
                sub_methods.append(methods[step]["method"])
                sub_ids.append(step)
                level = [len(ele) for ele in methods[step]["method"]["id"]]
                print(f"Add id {step} with tension score {tension_scores[-1]} and level {level}")
                if len(sub_methods) < maximum_thread * simulation_case and step < total_number - 1:
                    continue
                else:
                    print(f"Sub list: {sub_ids}")
                    reward_list = env.getRewardList(sub_methods, origami, buf, sim_time, extract, height, string_number, gravity_flag, prefold, miu, ctm, noise, fm, final_file_path)
                    sim_time += 1
                    for i in range(len(sub_methods)):
                        # methods[sub_ids[i]]["reward"] = reward_list[i]
                        hard_sim_reward_list = []
                        hard_sim_reward_list_without_actuator = []
                        hard_sim_rf_list = []
                        hard_sim_rm_list = []
                        hard_sim_rs_list = []
                        hard_sim_ra_list = []
                        success = 0
                        fail = 0
                        success_rate = 0.0
                        soft_success = 0
                        soft_fail = 0
                        soft_success_rate = 0.0
                        if not restart or (restart and buf != 0):
                            try:
                                hard_sim_reward_list = methods[sub_ids[i]]["hard_sim_result"]
                                hard_sim_reward_list_without_actuator = methods[sub_ids[i]]["hard_sim_result_without_actuator"]
                                hard_sim_rf_list = methods[sub_ids[i]]["hard_sim_rf_list"]
                                hard_sim_rm_list = methods[sub_ids[i]]["hard_sim_rm_list"]
                                hard_sim_rs_list = methods[sub_ids[i]]["hard_sim_rs_list"]
                                hard_sim_ra_list = methods[sub_ids[i]]["hard_sim_ra_list"]
                                success = methods[sub_ids[i]]["success"]
                                fail = methods[sub_ids[i]]["fail"]
                                success_rate = methods[sub_ids[i]]["success_rate"]
                                soft_success = methods[sub_ids[i]]["soft_success"]
                                soft_fail = methods[sub_ids[i]]["soft_fail"]
                                soft_success_rate = methods[sub_ids[i]]["soft_success_rate"]
                            except:
                                pass
                        
                        hard_sim_reward_list.append(reward_list[i])
                        hard_sim_reward_list_without_actuator.append(reward_list[i + len(sub_methods)])
                        hard_sim_rf_list.append(reward_list[i + 2*len(sub_methods)])
                        hard_sim_rm_list.append(reward_list[i + 3*len(sub_methods)])
                        hard_sim_rs_list.append(reward_list[i + 4*len(sub_methods)])
                        hard_sim_ra_list.append(reward_list[i + 5*len(sub_methods)])
                            
                        if reward_list[i] >= best_solution_upperbound:
                            success += 1
                        else:
                            fail += 1
                        if reward_list[i] >= best_solution_upperbound * 0.8:
                            soft_success += 1
                        else:
                            soft_fail += 1
                        success_rate = success / (success + fail)
                        soft_success_rate = soft_success / (soft_success + soft_fail)
                        methods[sub_ids[i]]["hard_sim_result"] = hard_sim_reward_list
                        methods[sub_ids[i]]["hard_sim_result_without_actuator"] = hard_sim_reward_list_without_actuator
                        methods[sub_ids[i]]["hard_sim_rf_list"] = hard_sim_rf_list
                        methods[sub_ids[i]]["hard_sim_rm_list"] = hard_sim_rm_list
                        methods[sub_ids[i]]["hard_sim_rs_list"] = hard_sim_rs_list
                        methods[sub_ids[i]]["hard_sim_ra_list"] = hard_sim_ra_list
                        methods[sub_ids[i]]["success"] = success
                        methods[sub_ids[i]]["fail"] = fail
                        methods[sub_ids[i]]["success_rate"] = success_rate
                        methods[sub_ids[i]]["soft_success"] = soft_success
                        methods[sub_ids[i]]["soft_fail"] = soft_fail
                        methods[sub_ids[i]]["soft_success_rate"] = soft_success_rate
                        methods[sub_ids[i]]["phase"] = extract
                    
                    sorted_method = sorted(methods, key=lambda x: x['success_rate'] + x['soft_success_rate'], reverse=True)
                    with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "method": sorted_method,
                            "number": total_number,
                            "simulated_number": step,
                            "simulation_speed_of_string_contraction": extract_simulation_speed,
                            "control_mode": control_mode,
                            "phase": extract
                        }, f, indent=4)    
                    sub_methods.clear()
                    sub_ids.clear()
    # left
    if len(sub_methods):
        print(f"Sub list: {sub_ids}")
        if extract == 0:
            reward_list = env.getRewardList(sub_methods, origami, buf, sim_time, extract, height, string_number, gravity_flag, prefold, miu, ctm, noise, fm, final_file_path)
            sim_time += 1
            for i in range(len(sub_methods)):
                methods[sub_ids[i]]["reward"] = reward_list[i]
                methods[sub_ids[i]]['reward_without_actuator'] = reward_list[i + len(sub_methods)]
                methods[sub_ids[i]]["rf"] = reward_list[i + 2*len(sub_methods)]
                methods[sub_ids[i]]["rm"] = reward_list[i + 3*len(sub_methods)]
                methods[sub_ids[i]]["rs"] = reward_list[i + 4*len(sub_methods)]
                methods[sub_ids[i]]["ra"] = reward_list[i + 5*len(sub_methods)]
                methods[sub_ids[i]]["phase"] = extract
                reward_scores[sub_ids[i]] = reward_list[i]
                if reward_list[i] >= best_reward:
                    best_reward = reward_list[i]
                    best_trajectory = methods[sub_ids[i]]["method"]
                    output_trajectory = deepcopy(best_trajectory)
                    output_trajectory["location"] = [[] for _ in range(env.string_number)]
                    # print(best_trajectory)
                    for j in range(len(best_trajectory['id'])):
                        for k in range(len(best_trajectory['id'][j])):
                            if best_trajectory['type'][j][k] == 'A':
                                output_trajectory['location'][j].append(env.P_points[best_trajectory['id'][j][k]])
                            else:
                                output_trajectory['location'][j].append(env.O_points[best_trajectory['id'][j][k]])
                    # with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_buf_{buf}_best.json"), 'w', encoding="utf-8") as f:
                    #     json.dump(best_trajectory, f, indent=4)
                if reward_list[i] >= best_solution_upperbound:
                    best_method_list.append(methods[sub_ids[i]])
                    with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "method": best_method_list,
                            "number": len(best_method_list),
                            "control_mode": control_mode,
                            "phase": extract
                        }, f, indent=4)  
                if len(input_sim_buffer['buf']) == 0:
                    buffer.append({
                        "id": ids[sub_ids[i]],
                        "id_list": id_list[sub_ids[i]],
                        "reward": reward_list[i],
                        "reward_without_actuator": reward_list[i + len(sub_methods)],
                        "rf": reward_list[i + 2*len(sub_methods)],
                        "rm": reward_list[i + 3*len(sub_methods)],
                        "rs": reward_list[i + 4*len(sub_methods)],
                        "ra": reward_list[i + 5*len(sub_methods)],
                        "phase": extract
                    })
                    with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "buf": buffer,
                            "number": len(buffer)
                        }, f, indent=4)
                else:
                    find_exist = False
                    for j in range(len(input_sim_buffer["buf"])):
                        if input_sim_buffer["buf"][j]["id"] == ids[sub_ids[i]] and input_sim_buffer["buf"][j]["id_list"] == id_list[sub_ids[i]]:
                            input_sim_buffer["buf"][j]["reward"] = reward_list[i]
                            input_sim_buffer['buf'][j]["reward_without_actuator"] = reward_list[i + len(sub_methods)]
                            input_sim_buffer["buf"][j]["rf"] = reward_list[i + 2*len(sub_methods)]
                            input_sim_buffer["buf"][j]["rm"] = reward_list[i + 3*len(sub_methods)]
                            input_sim_buffer["buf"][j]["rs"] = reward_list[i + 4*len(sub_methods)]
                            input_sim_buffer["buf"][j]["ra"] = reward_list[i + 5*len(sub_methods)]
                            input_sim_buffer["buf"][j]["phase"] = extract
                            find_exist = True
                    if not find_exist:
                        input_sim_buffer['buf'].append({
                            "id": ids[sub_ids[i]], 
                            "id_list": id_list[sub_ids[i]],
                            "reward": reward_list[i],
                            "reward_without_actuator": reward_list[i + len(sub_methods)],
                            "rf": reward_list[i + 2*len(sub_methods)],
                            "rm": reward_list[i + 3*len(sub_methods)],
                            "rs": reward_list[i + 4*len(sub_methods)],
                            "ra": reward_list[i + 5*len(sub_methods)],
                            "phase": extract
                        })
                    input_sim_buffer['number'] = len(input_sim_buffer['buf'])
                    with open(os.path.join(final_file_path, f"sim_buffer.json"), 'w', encoding="utf-8") as f:
                        json.dump(input_sim_buffer, f, indent=4)

            print(f"Step: {step}, Best reward: {best_reward}")
            # print(best_trajectory)   
            
            with open(os.path.join(final_file_path, f"result_{time}_buf_{buf}_true_reward.json"), 'w', encoding="utf-8") as f:
                json.dump({
                    "method": methods,
                    "number": total_number,
                    "simulated_number": step,
                    "control_mode": control_mode,
                    "phase": extract
                }, f, indent=4)    
            sub_methods.clear()
            sub_ids.clear()

            data = [[sim_step[i], tension_scores[i], reward_scores[i]] for i in range(len(sim_step))]
            test = pd.DataFrame(columns=["step", "tension_score", "true_reward"], data=data)
            test.to_csv(os.path.join(final_file_path, f'result_train_time_{time}_candidate_buf_{buf}.csv'))
        elif extract == 2:
            reward_list = env.getRewardList(sub_methods, origami, buf, sim_time, extract, height, string_number, gravity_flag, prefold, miu, ctm, noise, fm, final_file_path)
            sim_time += 1
            for i in range(len(sub_methods)):
                # methods[sub_ids[i]]["reward"] = reward_list[i]
                hard_sim_reward_list = []
                hard_sim_reward_list_without_actuator = []
                hard_sim_rf_list = []
                hard_sim_rm_list = []
                hard_sim_rs_list = []
                hard_sim_ra_list = []
                success = 0
                fail = 0
                success_rate = 0.0
                soft_success = 0
                soft_fail = 0
                soft_success_rate = 0.0
                
                if not restart or (restart and buf != 0):
                    try:
                        hard_sim_reward_list = methods[sub_ids[i]]["hard_sim_result"]
                        hard_sim_reward_list_without_actuator = methods[sub_ids[i]]["hard_sim_result_without_actuator"]
                        hard_sim_rf_list = methods[sub_ids[i]]["hard_sim_rf_list"]
                        hard_sim_rm_list = methods[sub_ids[i]]["hard_sim_rm_list"]
                        hard_sim_rs_list = methods[sub_ids[i]]["hard_sim_rs_list"]
                        hard_sim_ra_list = methods[sub_ids[i]]["hard_sim_ra_list"]
                        success = methods[sub_ids[i]]["success"]
                        fail = methods[sub_ids[i]]["fail"]
                        success_rate = methods[sub_ids[i]]["success_rate"]
                        soft_success = methods[sub_ids[i]]["soft_success"]
                        soft_fail = methods[sub_ids[i]]["soft_fail"]
                        soft_success_rate = methods[sub_ids[i]]["soft_success_rate"]
                    except:
                        pass
                
                hard_sim_reward_list.append(reward_list[i])
                hard_sim_reward_list_without_actuator.append(reward_list[i + len(sub_methods)])
                hard_sim_rf_list.append(reward_list[i + 2*len(sub_methods)])
                hard_sim_rm_list.append(reward_list[i + 3*len(sub_methods)])
                hard_sim_rs_list.append(reward_list[i + 4*len(sub_methods)])
                hard_sim_ra_list.append(reward_list[i + 5*len(sub_methods)])
                    
                if reward_list[i] >= best_solution_upperbound:
                    success += 1
                else:
                    fail += 1
                if reward_list[i] >= best_solution_upperbound * 0.8:
                    soft_success += 1
                else:
                    soft_fail += 1
                success_rate = success / (success + fail)
                soft_success_rate = soft_success / (soft_success + soft_fail)
                methods[sub_ids[i]]["hard_sim_result"] = hard_sim_reward_list
                methods[sub_ids[i]]["hard_sim_result_without_actuator"] = hard_sim_reward_list_without_actuator
                methods[sub_ids[i]]["hard_sim_rf_list"] = hard_sim_rf_list
                methods[sub_ids[i]]["hard_sim_rm_list"] = hard_sim_rm_list
                methods[sub_ids[i]]["hard_sim_rs_list"] = hard_sim_rs_list
                methods[sub_ids[i]]["hard_sim_ra_list"] = hard_sim_ra_list
                methods[sub_ids[i]]["success"] = success
                methods[sub_ids[i]]["fail"] = fail
                methods[sub_ids[i]]["success_rate"] = success_rate
                methods[sub_ids[i]]["soft_success"] = soft_success
                methods[sub_ids[i]]["soft_fail"] = soft_fail
                methods[sub_ids[i]]["soft_success_rate"] = soft_success_rate
                methods[sub_ids[i]]["phase"] = extract
            
            sorted_method = sorted(methods, key=lambda x: x['success_rate'] + x['soft_success_rate'], reverse=True)
            with open(os.path.join(final_file_path, f"result_train_time_{time}_candidate_extract.json"), 'w', encoding="utf-8") as f:
                json.dump({
                    "method": sorted_method,
                    "number": total_number,
                    "simulated_number": step,
                    "simulation_speed_of_string_contraction": extract_simulation_speed,
                    "control_mode": control_mode,
                    "phase": extract
                }, f, indent=4)    
            sub_methods.clear()
            sub_ids.clear()
    
    

if __name__ == "__main__":
    normal = 1

    if len(sys.argv) <= 1:
        print("Please input the buffer you want to train!")
        normal = 0
    else:
        if "-o" in sys.argv:
            index = sys.argv.index("-o") + 1
            origami = sys.argv[index]
            
        if "-ex" in sys.argv:
            index = sys.argv.index("-ex") + 1
            extract_number = sys.argv[index]
            numbers = re.findall(r"\d+", extract_number)
            if len(numbers) == 0:
                print("Wrong ex number!")
                normal = 0
            else:
                extract_mode = int(numbers[0])
                if extract_mode == 4:
                    extract_mode = -1
            
        if "-e" in sys.argv:
            index = sys.argv.index("-e") + 1
            episodes_number = sys.argv[index]
            numbers = re.findall(r"\d+", episodes_number)
            if len(numbers) == 0:
                print("Wrong episodes number!")
                normal = 0
            else:
                episodes = int(numbers[0])

        if "-u" in sys.argv:
            index = sys.argv.index("-u") + 1
            str_upper_number = sys.argv[index]
            numbers = re.findall(r"\d+", str_upper_number)
            if len(numbers) == 0:
                print("Wrong upper_bound number!")
                normal = 0
            else:
                upper_bound = int(numbers[0])
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
                    print("Thread number is too large!")
                    normal = 0
                else:
                    maximum_thread = thread_number
        if "-s" in sys.argv:
            index = sys.argv.index("-s") + 1
            str_case_number = sys.argv[index]
            numbers = re.findall(r"\d+", str_case_number)
            if len(numbers) == 0:
                print("Wrong case number!")
                normal = 0
            else:
                case_number = int(numbers[0])
                if case_number > 128:
                    print("Case number is too large!")
                    normal = 0
                else:
                    simulation_case = case_number
        if "-b" in sys.argv:
            index = sys.argv.index("-b") + 1
            buf_id = sys.argv[index]
            numbers = re.findall(r"\d+", buf_id)
            if len(numbers) == 0:
                print("Wrong buffer ID_low!")
                normal = 0
            else:
                buf_low = int(numbers[0])
            
            index = sys.argv.index("-b") + 2
            if index < len(sys.argv):
                buf_id = sys.argv[index]
                numbers = re.findall(r"\d+", buf_id)
                if len(numbers) == 0:
                    print("Wrong buffer ID_high!")
                    buf_high = buf_low
                else:
                    buf_high = int(numbers[0])
            else:
                buf_high = buf_low
        
        if "-a" in sys.argv:
            index = sys.argv.index("-a") + 1
            str_A_number = sys.argv[index]
            numbers = re.findall(r"\d+", str_A_number)
            if len(numbers) == 0:
                print("Wrong A number!")
                normal = 0
            else:
                MIN_A = int(numbers[0])
        
        if "-str" in sys.argv:
            index = sys.argv.index("-str") + 1
            str_number = sys.argv[index]
            numbers = re.findall(r"\d+", str_number)
            if len(numbers) == 0:
                print("Wrong str number!")
                normal = 0
            else:
                string_number = int(numbers[0])
        
        if "-h" in sys.argv:
            index = sys.argv.index("-h") + 1
            str_height = sys.argv[index]
            numbers = re.findall(r"\d+", str_height)
            if len(numbers) == 0:
                print("Wrong height!")
                normal = 0
            else:
                height = int(numbers[0])
        
        if "-lb" in sys.argv:
            index = sys.argv.index("-lb") + 1
            str_lb = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_lb)
            if len(numbers) == 0:
                print("Wrong lb!")
                normal = 0
            else:
                LB = float(numbers[0])
        
        if "-gb" in sys.argv:
            index = sys.argv.index("-gb") + 1
            str_gb = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_gb)
            if len(numbers) == 0:
                print("Wrong gb!")
                normal = 0
            else:
                GB = float(numbers[0])
        
        if "-sim" in sys.argv:
            index = sys.argv.index("-sim") + 1
            str_sim = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_sim)
            if len(numbers) == 0:
                print("Wrong SIM!")
                normal = 0
            else:
                SIM = int(numbers[0])
        
        if "-g" in sys.argv:
            index = sys.argv.index("-g") + 1
            str_sim = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_sim)
            if len(numbers) == 0:
                print("Wrong gravity!")
                normal = 0
            else:
                gravity_flag = int(numbers[0])
            
        if "-pf" in sys.argv:
            index = sys.argv.index("-pf") + 1
            str_sim = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_sim)
            if len(numbers) == 0:
                print("Wrong prefold!")
                normal = 0
            else:
                prefold = int(numbers[0]) / 100.0
        
        if "-miu" in sys.argv:
            index = sys.argv.index("-miu") + 1
            str_miu = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_miu)
            if len(numbers) == 0:
                print("Wrong miu!")
                normal = 0
            else:
                miu = float(numbers[0])
        
        if '-ctm' in sys.argv:
            index = sys.argv.index("-ctm") + 1
            str_ctm = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_ctm)
            if len(numbers) == 0:
                print("Wrong ctm!")
                normal = 0
            else:
                control_mode = int(numbers[0])
        
        if '-fm' in sys.argv:
            index = sys.argv.index("-fm") + 1
            str_friction = sys.argv[index]
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str_friction)
            if len(numbers) == 0:
                print("Wrong friction!")
                normal = 0
            else:
                friction_mode = int(numbers[0])

    NAME = f"SBS-{origami}-{SIM}sim-{string_number}string-{episodes}episodes"

    if calculate_number:
        NAME = f"CALNUM-{origami}-{SIM}sim-{string_number}string-{episodes}episodes"

    if initialize:
        NAME += "-initialize"

    if new_method:
        NAME += "-precut"

    if cut_nodes:
        NAME += "-cutnodes"

    NAME += "-uplimit_" + str(upper_bound)

    NAME += "-A" + str(MIN_A)
    
    NAME += "-LB" + "{:.2f}".format(LB) + "-GB" + "{:.2f}".format(GB)

    file_path = f"./threadingResult/train-" + NAME

    if not normal:
        from gui.buffer_reader import *
        
        file_selected = False
        
        root = tk.Tk()
        root.withdraw()
        
        folder_path = filedialog.askdirectory(title="Please select a folder")
        if not folder_path:
            print("Unknown folder path.")
            os._exit(0)
        
        folder_name = os.path.basename(folder_path)
        file_path = f"./threadingResult/{folder_name}"
        
        pattern = r'^train-SBS-([\w-]+)-(\d+)sim-(\d+)string-(\d+)episodes-cutnodes-uplimit_(\d+)-A(\d+)-LB(\d+\.\d{2})-GB(\d+\.\d{2})$'
        pattern = r'^train-SBS-([\w-]+)-S(\d+)-U(\d+)-A(\d+)-LB(\d+\.\d{2})-GB(\d+\.\d{2})$'
        match = re.match(pattern, folder_name)
        
        if not match:
            print("Incorrect buffer.")
            os._exit(0)
        
        # origami = match.group(1)
        # SIM = int(match.group(2))
        # string_number = int(match.group(3))
        # episodes = int(match.group(4))
        # upper_bound = int(match.group(5))
        # MIN_A = int(match.group(6))
        # LB = float(match.group(7))
        # GB = float(match.group(8))
        
        origami = match.group(1)
        string_number = int(match.group(2))
        upper_bound = int(match.group(3))
        MIN_A = int(match.group(4))
        LB = float(match.group(5))
        GB = float(match.group(6))
        
        root.destroy()
        
        app = FolderParserApp(file_path)
        
        try:
            height = app.height
            gravity_flag = app.gravity_flag
            prefold = 0.0
            miu = app.miu
            control_mode = app.control_mode 
            friction_mode = 2 if app.friction_mode else 0
            buf_low = app.buf_low
            buf_high = app.buf_high
            maximum_thread = app.thread_number
            simulation_case = app.batch_number
            extract_mode = app.extract_mode
            normal = 1
        except:
            print("Incorrect parameter setting.")
            os._exit(0)
            
    env = Env()

    final_file_path = os.path.join(file_path, f"train_data_{origami}_h{height}_g{gravity_flag}_miu{miu}_ctm{control_mode}_fm{friction_mode}")

    try:
        os.makedirs(final_file_path)
        s = {
            "origami_name": origami,
            "string_number": string_number,
            "upper_bound": upper_bound,
            "external_nodes_number": MIN_A,
            "platform_height": height,
            "gravity_flag": gravity_flag,
            "duplicate_ratio": LB,
            "global_pass_ratio": GB,
            "training_thread_number": maximum_thread,
            "simulation_case": simulation_case,
            "buffer_start_id": buf_low,
            "buffer_end_id": buf_high,
            "extract_mode": extract_mode,
            "prefold": prefold,
            "miu": miu,
            "control_mode": control_mode,
            "friction_mode": friction_mode,
        }
        with open(os.path.join(final_file_path, f"settings.json"), 'w', encoding="utf-8") as f:
            json.dump(s, f, indent=4)
    except:
        pass

    if normal or extract_mode:
        for i in range(buf_low, buf_high):
            train(env, origami, int(i * BUFFER_INIT), extract_mode, height, string_number, gravity_flag, prefold, miu, control_mode, friction_mode)
