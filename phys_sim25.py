import taichi as ti
import taichi.math as tm

from ori_sim_sys import *
from desc import *
from units import *
import json
import os
import sys
import argparse
import time

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams.update({'font.size': 12})

FOLDING_ERROR_MINIMUM = 5.0 / 180.0 * np.pi
FOLDING_ERROR_CANDIDATE_RATIO = 0.1
CREASE_CROSS = 2
STROKE_PERCENT = 0.5
VALID_BOUND = 0.2
INITIAL_RATIO = 0.05

NORMAL = 0
LARGE = 1
LARGE_THICK = 2
NORMAL_2 = 3

#----折纸信息初始化开始----#
data_type = ti.f64
ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, verbose=False, cpu_max_num_threads=1)

@ti.data_oriented
class OrigamiSimulator:
    """
    折纸物理仿真器主类，基于Taichi实现高性能物理仿真。
    Origami physics simulator main class, implemented with Taichi for high-performance physics simulation.
    
    该类实现了折纸结构的完整物理仿真，包括折痕弯曲、面内拉伸、绳索驱动、碰撞检测等功能。
    This class implements complete physics simulation for origami structures, including crease bending,
    in-plane stretching, string actuation, collision detection, etc.
    
    :param use_gui: 是否启用GUI界面 / Whether to enable GUI interface
    :param debug_mode: 是否启用调试模式 / Whether to enable debug mode
    :param origami_name: 折纸模型名称 / Origami model name
    :param fast_simulation: 是否启用快速仿真模式 / Whether to enable fast simulation mode
    :param mode: 仿真模式，可选'fast'或'accurate' / Simulation mode, 'fast' or 'accurate'
    :param g: 重力加速度向量 [x, y, z] / Gravitational acceleration vector [x, y, z]
    :param default_ground: 默认地面高度 / Default ground height
    :param control_mode: 控制模式，0为普通折叠，1为控制模式 / Control mode, 0 for normal folding, 1 for control mode
    :param ground_miu: 地面摩擦系数 / Ground friction coefficient
    :param h: 折纸初始高度 / Initial origami height
    :param strict: 严格模式级别 / Strict mode level
    :param strict_speed: 严格模式速度系数 / Strict mode speed factor
    :param friction_mode: 摩擦模式 / Friction mode
    :param const_stiff_of_crease: 是否使用恒定折痕刚度 / Whether to use constant crease stiffness
    :param speed_bonus: 速度奖励系数 / Speed bonus coefficient
    :param check_connection_matrix: 是否检查连接矩阵 / Whether to check connection matrix
    :param additional_length: 绳索附加长度 / Additional string length
    :param facet_mode: 面片模式 / Facet mode
    :param fillet_hole: 圆孔倒角模式 / Fillet hole mode
    :param simulation_upper_time: 仿真时间上限 / Simulation upper time limit
    :param control_mode_training_time: 控制模式训练时间 / Control mode training time
    :param mean_stress_baseline: 平均应力基线 / Mean stress baseline
    :param a_speed: 速度奖励权重 / Speed reward weight
    :param a_act: 动作奖励权重 / Action reward weight
    :param a_force: 力奖励权重 / Force reward weight
    :param robot_type: 机器人类型 / Robot type
    """
    def __init__(self, use_gui=True, debug_mode=False, origami_name="default", fast_simulation=False, mode='fast', g=[0,0,-9810.0], default_ground=1,\
        control_mode=0, ground_miu=0.1, h=100., strict=0, strict_speed=1.0, friction_mode=0, const_stiff_of_crease=False, speed_bonus=1.0, check_connection_matrix=True, additional_length=100.0, facet_mode=1, fillet_hole=1, \
            simulation_upper_time=20., control_mode_training_time=15., mean_stress_baseline=1., a_speed=0.2, a_act=0.0, a_force=0.2, robot_type=NORMAL, type_of_controller=1, tracking_camera=True, stroke_percent=STROKE_PERCENT) -> None:

        self.origami_name = origami_name
        self.mode = mode
        self.control_mode = control_mode
        self.speed_bonus = speed_bonus
        self.check_connection_matrix = check_connection_matrix

        LUMPED_MASS = 0
        CONS_MASS = 1

        self.robot_type = robot_type
        self.tracking_camera = tracking_camera

        self.mass_mode = CONS_MASS

        self.tension_bonus = 1.

        self.stroke_percent = stroke_percent

        #reward_coeff
        self.a_s = a_speed
        self.a_a = a_act
        self.a_f = a_force
        
        self.use_gui = use_gui
        self.debug_mode = debug_mode
        self.fast_simulation_mode = fast_simulation
        
        self.facet_mode = facet_mode

        self.rot_z = 10.444 * 3.1415926 / 180. * 0.

        self.ground_d = 100.

        self.control_mode_simulation_time = control_mode_training_time
        self.simulation_upper_time = simulation_upper_time
        self.mean_stress_baseline = mean_stress_baseline #MPa

        if self.robot_type in [LARGE, LARGE_THICK]:
            self.control_mode_simulation_time = 60
            self.simulation_upper_time = 65

        # whether to put the target angles to the barrier
        self.ref_crease_angle_mode = REF_TARGET_ANGLE
        
        self.noise_dict = {
                "+1": [
                    2, 1,
                ],
                "0": [
                    4, 8, 6, 10, 14, 17, 20, 16, 19, 22
                ],
                "-1":[
                    0, 5, 9, 12, 3, 7, 11, 13, 15, 18, 21, 23
                ],
                "total": 24
            }
        
        if use_gui:
            self.window = ti.ui.Window("Origami Simulation", (1600, 900), vsync=True)
            self.gui = self.window.get_gui()
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((1., 1., 1.))
            self.scene = self.window.get_scene()
            self.camera = ti.ui.Camera()
       
        if not self.fast_simulation_mode and use_gui:
            self.time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            try:
                if self.control_mode:
                    os.makedirs(f"./physResult/{self.origami_name}-{self.time}-{self.mode}/phase1")
                    os.makedirs(f"./physResult/{self.origami_name}-{self.time}-{self.mode}/phase2")
                    os.makedirs(f"./physResult/{self.origami_name}-{self.time}-{self.mode}/phase3")
                else:
                    os.makedirs(f"./physResult/{self.origami_name}-{self.time}-{self.mode}")
            except:
                pass
        
        self.dxfg = DxfDirectGrabber()

        print(f"Origami simulator with G: {g}, ctm: {control_mode}, h: {h}, strict: {strict}, fm: {friction_mode}, csoc: {const_stiff_of_crease}")
        # Simulation Scenary
        self.FOLD_SIM = 0
        self.TSA_SIM = 1
        self.MAXIMUM_FIX_PANEL = 1

        if type_of_controller == 1:
            self.controller_side_length = 59
            self.controller_rotation = math.pi / 4.
            self.controller_mass = 122.23e-6
            self.accessory_mass = 12.805e-6
            self.controller_centroid_bias = [-4., -4., 38.5]
            controller_speed = 0.19177
            self.actuation_discount = True
            self.control_mode_max_force = 20.1978
        elif type_of_controller == 2:
            self.controller_side_length = 59
            self.controller_rotation = math.pi / 4.
            self.controller_mass = 348.39e-6
            self.accessory_mass = 12.805e-6
            self.controller_centroid_bias = [0., 5.8, 54.]
            controller_speed = 0.2776
            self.actuation_discount = False
            self.control_mode_max_force = 35.

        self.controller_z_warning = 10.
        self.motor_speed = 0.0694
        if control_mode:
            self.motor_speed = controller_speed

        # String information
        self.string_total_information = []
        
        self.connected_unit_pairs = []

        # Preference pack
        self.pref_pack = None

        self.input_json = None

        self.ITER = 10

        self.BIAS = 0.15

        self.print = 0

        self.ID = 0

        self.d = 0.3

        self.d_coeff = 0.18 / self.d
        
        self.ground_miu = ground_miu

        self.string_contract_ratio_initial = INITIAL_RATIO
        
        self.strict = strict
        self.strict_speed = strict_speed
        
        self.allow_initialize = False
        self.robot_stand = True
        
        self.static_friction = 0.00

        self.sparse_solver = ti.linalg.SparseSolver(data_type, "LDLT")
        rot_x = tm.pi / 2
        rot_y = 0.0
        rot_z = 0.0

        self.R = tm.mat3([
            [tm.cos(rot_z), -tm.sin(rot_z), 0.],
            [tm.sin(rot_z), tm.cos(rot_z), 0.],
            [0., 0., 1.]
        ]) @ tm.mat3([
            [tm.cos(rot_y), 0., tm.sin(rot_y)],
            [0., 1., 0.], 
            [-tm.sin(rot_y), 0., tm.cos(rot_y)]
        ]) @ tm.mat3([
            [1., 0., 0.], 
            [0., tm.cos(rot_x), -tm.sin(rot_x)],
            [0., tm.sin(rot_x), tm.cos(rot_x)]
        ])

        self.default_ground = default_ground
        self.standard_g = g
        self.prefold = 0.00
        self.maximum_tension = 25.0 * self.tension_bonus

        self.double_side_folding_enable = 1
        
        self.origami_thickness = 1.32
        if self.robot_type == LARGE:
            self.origami_thickness = 10.0
        elif self.robot_type == LARGE_THICK:
            self.origami_thickness = 10.0
        elif self.robot_type == NORMAL_2:
            self.origami_thickness = 4.12

        self.crease_thickness = 0.12
        
        self.panel_bias = 4.0
        if self.robot_type == LARGE:
            self.panel_bias = 5.0
        elif self.robot_type == NORMAL_2:
            self.panel_bias = 2.0

        self.panel_half_height = (self.origami_thickness - self.crease_thickness) * 0.5
        self.kappa = math.atan(self.panel_half_height / self.panel_bias)
        
        self.folding_angle_maximum_ratio = (1. - 2. * self.kappa / math.pi)

        if self.robot_type == LARGE_THICK:
            self.folding_angle_maximum_ratio = 1.
            self.double_side_folding_enable = 0
        
        self.facet_bending_bonus = 250 * (self.origami_thickness) ** 3
        if self.robot_type in [LARGE, LARGE_THICK]:
            self.facet_bending_bonus = 5. / 12. * (self.origami_thickness) ** 3
        self.h = h
        
        self.string_thickness = 1.0
        self.fillet_hole = fillet_hole
        
        #折纸参数
        self.d_hole = 3.7 + self.origami_thickness - self.crease_thickness #折纸上所打通孔的直径
        self.h_hole = self.origami_thickness #通孔高度
        if self.robot_type == LARGE_THICK:
            self.h_hole = 2.0
    
        if self.string_thickness >= 3.7:
            raise RuntimeError

        if self.fillet_hole:
            self.beta = math.asin((self.h_hole + self.string_thickness) / (self.d_hole))
        else:
            self.beta = math.atan((self.h_hole + self.string_thickness) / (self.d_hole - self.string_thickness))
        
        self.targets = []

        self.control_steps = 0
        
        self.multiple_origami = 1
        
        self.friction_mode = friction_mode
        
        self.intersection_threshold = 1.0
        
        self.connection_matrix_interval = 12
        
        self.connection_matrix_step = 180.0 / self.connection_matrix_interval
        
        self.const_stiff_of_crease = const_stiff_of_crease
        
        self.DEAD_MAXIMUM = 240
        
        self.additional_length_of_string = additional_length
        
        self.maximum_epsilon_of_string = 0.057343
        # self.maximum_epsilon_of_string = 0.05
        
        self.gamma_bound_0_degree = [0.409889, 0.6272695]
        self.gamma_bound_45_degree = [0.513037, 0.673403]
        self.gamma_bound_90_degree = [0.447521, 0.6841345]
        self.gamma_bound_135_degree = [0.569443, 0.7351448]
        self.gamma_bound_180_degree = [0.802567, 0.861414]

        # no friction settings
        # self.gamma_bound_0_degree = [1, 1]
        # self.gamma_bound_45_degree = [1, 1]
        # self.gamma_bound_90_degree = [1, 1]
        # self.gamma_bound_135_degree = [1, 1]
        # self.gamma_bound_180_degree = [1, 1]

        self.key = ''
    
    ################################# USER DEFINED #################################
    @ti.kernel
    def getActuationForce(self, force: data_type) -> data_type:
        """
        计算驱动力，根据力的大小进行非线性映射。
        Calculate actuation force with nonlinear mapping based on force magnitude.
        
        :param force: 输入力大小 / Input force magnitude
        :return: 映射后的驱动力 / Mapped actuation force
        """
        actuation_force = force
        if not self.control_mode:
            # actuation_force = force / (-3.32194 / (force + 8.28297) + 0.81321)     
            actuation_force = force / (-2.22809 / (force + 5.1903) + 1.00049)
        return actuation_force
    
    @ti.func
    def ks(self, epsilon):
        """
        计算绳索的拉伸刚度系数。
        Calculate the stretching stiffness coefficient of the string.
        
        :param epsilon: 绳索应变 / String strain
        :return: 拉伸刚度系数 / Stretching stiffness coefficient
        """
        # return 500.0
        return (156810.524 * tm.clamp(epsilon, 0.0, self.maximum_epsilon_of_string) ** 2.50317 + 313.80523)
    
    @ti.func
    def kb(self, theta): #deg
        """
        计算折痕的弯曲刚度系数（基于角度，单位：度）。
        Calculate the bending stiffness coefficient of the crease (based on angle in degrees).
        
        :param theta: 折痕角度（度）/ Crease angle (degrees)
        :return: 弯曲刚度系数 / Bending stiffness coefficient
        """
        # return 0.2
        return_value = 0.0
        if theta < 60.:
            return_value = 0.0000170361 * (theta - 60.) ** 2 + 0.085341
        elif theta < 100.:
            return_value = -0.000020471 * (theta - 60.) ** 2 + 0.085341
        else:
            return_value = -0.000254165 * (theta - 100.) + 0.0521554
        return return_value

    @ti.func
    def getBarrierBonus(self, theta, theta_ref):
        """
        计算碰撞屏障奖励系数，用于防止折痕过度折叠。
        Calculate collision barrier bonus coefficient to prevent excessive crease folding.
        
        :param theta: 当前折痕角度 / Current crease angle
        :param theta_ref: 参考目标角度 / Reference target angle
        :return: 屏障奖励系数 / Barrier bonus coefficient
        """
        theta_ref = min(35. / 36. * tm.pi, theta_ref)
        val = self.collision_indice
        if theta > theta_ref:
            val = abs((tm.pi - theta_ref) / (tm.pi - theta)) * self.collision_indice
        if theta > tm.pi or val > 1e-1:
            val = 1e-1
        return val
    
    @ti.func
    def kb_damping(self, theta, vel): #deg
        """
        计算折痕的弯曲阻尼系数（基于角度和角速度）。
        Calculate the bending damping coefficient of the crease (based on angle and angular velocity).
        
        :param theta: 折痕角度（度）/ Crease angle (degrees)
        :param vel: 角速度 / Angular velocity
        :return: 弯曲阻尼系数 / Bending damping coefficient
        """
        return_value = 0.00663
        return return_value
        # return_value = 0.0
        # if vel > 0.052905:
        #     return_value = 5.35238 - 0.283167 / vel
        # if theta < 15.0:
        #     return_value = theta / 15.0 * return_value
        # return return_value * 0.01
    ################################# USER DEFINED #################################
          
    # start part 1
    def commonStart_1(self, unit_edge_max):
        """
        仿真初始化第一阶段：构建折纸系统并添加单元。
        Simulation initialization phase 1: Build origami system and add units.
        
        :param unit_edge_max: 单元最大边数 / Maximum number of edges per unit
        """
        # Calculate TSA information 
        self.unit_edge_max = unit_edge_max
                    
        # 构造折纸系统
        density = 1.24e-9
        if self.robot_type in [LARGE, LARGE_THICK]:
            density = 0.08e-9
        self.ori_sim = OrigamiSimulationSystem(unit_edge_max, material_density=density)
        for ele in self.units:
            self.ori_sim.addUnit(ele, ele.special, self.origami_thickness)
        self.ori_sim.mesh() #构造三角剖分

        self.unit_edge_max = self.ori_sim.unit_edge_max
        self.ori_sim.fillBlankIndices() # fill all blank indice with -1
    
    def commonStart_2(self, unit_edge_max, constraint_number=-1):
        """
        仿真初始化第二阶段：初始化物理场和约束条件。
        Simulation initialization phase 2: Initialize physics fields and constraints.
        
        :param unit_edge_max: 单元最大边数 / Maximum number of edges per unit
        :param constraint_number: 约束数量，-1表示自动计算 / Number of constraints, -1 for auto calculation
        """
        ori_sim = self.ori_sim

        new_lines = ori_sim.getNewLines()  
        new_line_indices = ori_sim.getNewLineIndices()
        self.kps = ori_sim.kps                                           # all keypoints of origami
        self.creases = new_lines                                         # all creases of origami
        self.tri_indices = ori_sim.tri_indices                           # all triangle indices of origami
        self.kp_num = len(ori_sim.kps)                                   # total number of keypoints
        self.indices_num = len(ori_sim.tri_indices)                      # total number of triangles indices
        self.div_indices_num = int(self.indices_num / 3)                 # total_number of triangles
        self.unit_indices_num = len(ori_sim.indices)                     # total number of units
        self.line_total_indice_num = len(new_line_indices)               # total number of lines
        self.bending_pairs_num = len(ori_sim.bending_pairs)              # total number of bending pairs
        self.crease_pairs_num = len(ori_sim.crease_pairs)                # total number of crease pairs
        self.facet_bending_pairs_num = len(ori_sim.facet_bending_pairs)  # total number of facet bending pairs
        self.facet_crease_pairs_num = len(ori_sim.facet_crease_pairs)    # total number of facet crease pairs
        self.mass_list = ori_sim.mass_list                               # mass list of kps
        self.consistent_mass_list = ori_sim.consistent_mass_list         # consistent mass

        self.gravitational_acc = tm.vec3(self.standard_g)  #全局重力场
        #折纸初始高度

        self.origami_z_bias = self.h
        self.minimum_z = self.origami_z_bias
        # if self.control_mode == 0:
        #     ################################# USER DEFINED #################################
        #     self.base_decrease_step = self.motor_speed * self.speed_bonus
        #     # self.base_decrease_step = 10.0 / 240.0
        #     ################################# USER DEFINED #################################
        # else:
        #     self.base_decrease_step = self.motor_speed

        # self.string_length_decrease_step = self.base_decrease_step
        # if self.strict == 3:
        #     self.string_length_decrease_step = self.base_decrease_step * self.strict_speed

        self.controller_k = 1e3
        self.thick_panel_k = 1e3
        self.bending_k = self.ori_sim.bending_k
        
        self.bending_param = ti.field(data_type, shape=1)
        self.bending_k_list = ti.field(data_type, shape=self.bending_pairs_num)
        self.bending_k_damping_list = ti.field(data_type, shape=self.bending_pairs_num)
        self.barrier_coeff_list = ti.field(data_type, shape=self.bending_pairs_num)
        self.facet_bending_param = ti.field(data_type, shape=1)
        self.controller_param = ti.field(data_type, shape=1)
        self.const_stiff_of_crease_param = ti.field(bool, shape=1)
        self.k_bending_damping_param = ti.field(data_type, shape=1)
        
        self.facet_k = self.bending_k

        self.collision_indice = 1e-3
        if self.robot_type in [LARGE, LARGE_THICK, NORMAL_2]:
            self.collision_indice = 1e-1
        self.collision_d = 1e-4

        self.ground_barrier = 1.
        self.ground_collision_indice = 1e-3
        self.ground_barrier_energy_maximum = -self.ground_collision_indice * self.ground_barrier ** 2 * tm.log(self.collision_d / self.ground_barrier)
        self.ground_force_maximum = self.ground_collision_indice * (2 * self.ground_barrier * tm.log(self.collision_d / self.ground_barrier) + self.ground_barrier ** 2 / self.collision_d)
        self.df_ground_force_maximum = self.ground_collision_indice * (4. * self.ground_barrier / self.collision_d + 2 * tm.log(self.collision_d / self.ground_barrier) - self.ground_barrier ** 2 / self.collision_d ** 2)

        self.velocity_barrier = 1e-1
        
        self.penetration_barrier = 1e-3

        self.enable_add_folding_angle = 0. #启用折角增加的仿真模式
        self.enable_tsa_rotate = 0 #启用TSA驱动的仿真模式

        self.dt_bonus = ti.field(data_type, shape=1)

        self.folding_angle = 0.0 #当前的目标折叠角度

        self.string_width = 0.

        #折纸最大折叠能量
        self.total_energy = ti.field(data_type, shape=1)

        #最大末端作用力
        self.end_force = ti.field(data_type, shape=1)
        
        #面折痕仿真模式
        self.facet_mode_flag = ti.field(bool, shape=1)
        
        self.overall_string_maximum_force = ti.field(data_type, shape=1)

        #----define parameters for taichi----#
        self.masses = ti.field(dtype=data_type, shape=self.kp_num) # 质量信息
        self.mass_matrix = ti.field(dtype=data_type, shape=(3*self.kp_num, 3*self.kp_num))
        self.mass_matrix_inv = ti.field(dtype=data_type, shape=(3*self.kp_num, 3*self.kp_num))

        self.original_vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 原始点坐标
        
        self.vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 点坐标
        self.vertices_color = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 点坐标颜色
        
        self.ground_vertices = ti.Vector.field(3, dtype=data_type, shape=16) # 点坐标
        self.ground_indices = ti.field(int, shape=54) # 点索引
        self.ground_vertices_color = ti.Vector.field(3, dtype=data_type, shape=16)
                
        self.ground_line_vertex = ti.Vector.field(3, dtype=data_type, shape=288) #地面线段顶点位置，用于渲染
        self.ground_fix_line_vertex = ti.Vector.field(3, dtype=data_type, shape=4) #地面线段顶点位置，用于渲染

        self.unit_indices = ti.Vector.field(unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息
        self.unit_kp_num_list = ti.field(dtype=int, shape=self.unit_indices_num)
        self.unit_contributions = ti.Vector.field(unit_edge_max, dtype=data_type, shape=self.unit_indices_num) # 每个单元的贡献度
        self.fix_id_list = ti.field(dtype=int, shape=self.MAXIMUM_FIX_PANEL)
        self.kp_add_height = ti.field(dtype=bool, shape=self.kp_num) # 关键点的高度
        self.additional_height = ti.field(dtype=data_type, shape=1)
        self.ground_attachment = ti.field(dtype=bool, shape=1)
        self.ground_miu_param = ti.field(dtype=data_type, shape=1)
        self.maximum_recorded_trajectory_point = 720
        self.offset_x = 0.
        self.offset_y = 0.
        self.current_center_trajectory_step = 0
        self.center_trajectory = ti.Vector.field(3, dtype=data_type, shape=self.maximum_recorded_trajectory_point)
        self.fae_information = ti.field(data_type, shape=8)
        self.target_angles = ti.field(dtype=data_type, shape=self.crease_pairs_num)

        self.sequence_level = ti.field(int, shape=2) # max, min
        self.folding_micro_step = ti.field(data_type, shape=1) # step calculated by sequence_level max and min

        self.unit_center_initial_point = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的初始中心点位置
        self.unit_center = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的中心点位置
        self.indices = ti.field(int, shape=self.indices_num) #三角面索引信息

        self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
        self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息
        self.previous_dir = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.line_pairs = ti.field(dtype=int, shape=(self.line_total_indice_num, 2)) #线段索引信息，用于初始化渲染

        self.folding_angle_upper_bound = ti.field(dtype=data_type, shape=self.bending_pairs_num) #折痕折角上限，正值或0
        self.folding_angle_lower_bound = ti.field(dtype=data_type, shape=self.bending_pairs_num) #折痕折角下限，负值或0
        
        self.enable_equivalent_torque = ti.field(dtype=bool, shape=self.bending_pairs_num) #是否启用等效扭矩
        self.backup_enable_equivalent_torque = ti.field(dtype=bool, shape=self.bending_pairs_num) #是否启用等效扭矩, 备份

        self.crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.backup_crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.crease_velocity = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.backup_crease_velocity = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        # self.energy_buffer = ti.field(dtype=data_type, shape=self.linear_search_step)

        self.crease_folding_angle = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕折角
        self.crease_folding_accumulate = ti.field(dtype=data_type, shape=self.crease_pairs_num) #补偿折角
        self.bending_pairs_area = ti.field(dtype=data_type, shape=(self.bending_pairs_num, 2)) #弯曲对面积信息
        self.crease_initial_length = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕长度

        self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
        self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
        self.crease_coeff = ti.field(dtype=data_type, shape=self.crease_pairs_num)

        self.maximum_level_number = 8
        self.recover_level_need = ti.field(dtype=bool, shape=(self.crease_pairs_num, self.maximum_level_number))
        self.recover_level = ti.field(dtype=int, shape=(self.crease_pairs_num, self.maximum_level_number))
        self.recover_angle = ti.field(dtype=float, shape=(self.crease_pairs_num, self.maximum_level_number))

        # self.connection_matrix = ti.field(dtype=data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的连接矩阵

        self.line_color = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段颜色，用于渲染
        self.line_vertex = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段顶点位置，用于渲染
        self.line_connection_unit = ti.Vector.field(2, dtype=int, shape=self.line_total_indice_num) #线段颜色，用于渲染

        self.P_number = len(self.P_candidate_connection)
        self.border_vertex = ti.Vector.field(3, dtype=data_type, shape=self.P_number + 1)
        #----TSA constraints----#
        self.thick_panel_additional_connection_id = ti.Vector.field(2, dtype=int, shape=self.unit_indices_num * self.unit_edge_max)
        self.thick_panel_additional_connection_id.fill(-1)
        
        self.dfdx_thick_panel_connection = ti.Matrix.field(3, 3, dtype=data_type, shape=(self.unit_indices_num * self.unit_edge_max, self.unit_edge_max, self.unit_edge_max))
        
        try:
            self.constraint_number = len(self.method["id"]) #约束的数量
        except:
            self.constraint_number = 0
            pass
            
        if constraint_number != -1:
            self.constraint_number = constraint_number
        
        self.max_control_length = 2 + self.unit_indices_num
        self.random_folding_target_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        
        #根据约束数量，确定约束起始点和终止点位置， 若初始点一致，则识别为TSA
        if self.constraint_number == 0:
            self.string_params = ti.field(dtype=data_type, shape=1)
            self.string_params_bonus = ti.field(dtype=data_type, shape=1)
            self.string_params_clip = ti.field(dtype=data_type, shape=1)
            self.enable_plasticity = ti.field(dtype=bool, shape=1)
            
            self.equivalent_internal_point_id = ti.field(dtype=int, shape=1)
            
            self.epsilon_string = ti.field(data_type, shape=1)
            self.tolerance = ti.field(data_type, shape=1)
            
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_start_point_candidate_id = ti.field(dtype=int, shape=1)
            self.equivalent_torque_index = ti.field(dtype=data_type, shape=(1, self.bending_pairs_num, self.max_control_length))
            self.equivalent_torque_influence_id = ti.field(dtype=int, shape=(1, self.bending_pairs_num))
            self.equivalent_torque_coeff = ti.field(dtype=data_type, shape=(1, self.bending_pairs_num, self.max_control_length))
            
            self.string_number_each = ti.field(dtype=int, shape=1)
            self.string_length_decrease = ti.field(dtype=data_type, shape=1) #当前绳子的减少长度
            self.target_string_length_decrease = ti.field(dtype=data_type, shape=(1, 1)) #摩擦影响下的目标收缩
            self.backup_delta_length = ti.field(dtype=data_type, shape=1)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=1)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_end_point_candidate_id = ti.field(dtype=int, shape=1)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            #---#
            self.unit_control = ti.field(dtype=int, shape=(1, 1))

            self.initial_length_per_string = ti.field(dtype=data_type, shape=(1, 1))
            self.current_length_per_string = ti.field(dtype=data_type, shape=(1, 1))
            self.delta_length_per_string = ti.field(dtype=data_type, shape=(1, 1))
            
            self.string_force_initial_discount = ti.field(dtype=data_type, shape=(1, 1))
            self.string_force_current_discount = ti.field(dtype=data_type, shape=(1, 1))
            
            self.string_force_discount_sum = ti.field(dtype=data_type, shape=1)

            self.intersection_points = ti.Vector.field(3, dtype=data_type, shape=1)
            self.intersection_points2 = ti.Vector.field(3, dtype=data_type, shape=1)
            self.intersection_infos = ti.Vector.field(3, dtype=data_type, shape=1)
            self.intersection_infos2 = ti.Vector.field(3, dtype=data_type, shape=1)

            self.intersection_flag = ti.field(dtype=int, shape=(1, 1))
            self.intersection_flag2 = ti.field(dtype=int, shape=(1, 1))

            self.intersection_flag2_initial = ti.field(dtype=int, shape=1)
            self.intersection_points2_initial = ti.Vector.field(3, dtype=data_type, shape=1)
            self.intersection_infos2_initial = ti.Vector.field(3, dtype=data_type, shape=1)

            #穿线方向信息，选中某ID时：-1表示下方绳索，1表示上方绳索, -2表示下方绳索，但在下一个板的上方，2表示上方绳索，但在下一个板的下方
            self.hole_dir = ti.field(dtype=data_type, shape=(1, 1))
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=1)
            self.constraint_length = ti.field(data_type, shape=1)
            self.backup_constraint_length = ti.field(data_type, shape=1)
            self.points = ti.Vector.field(3, dtype=data_type, shape=1)

            self.string_force_each = ti.field(data_type, shape=1)
            self.backup_string_force_each = ti.field(data_type, shape=1)
            self.max_force = ti.field(data_type, shape=1)

            self.error_status_buffer = ti.field(bool, shape=1)
        else:
            self.string_params = ti.field(dtype=data_type, shape=self.constraint_number)
            self.string_params_bonus = ti.field(dtype=data_type, shape=self.constraint_number)
            self.string_params_clip = ti.field(dtype=data_type, shape=self.constraint_number)
            self.enable_plasticity = ti.field(dtype=bool, shape=self.constraint_number)
            
            self.equivalent_internal_point_id = ti.field(dtype=int, shape=self.constraint_number)
            
            self.epsilon_string = ti.field(data_type, shape=self.constraint_number)
            self.tolerance = ti.field(data_type, shape=self.constraint_number)
        
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.constraint_start_point_candidate_id = ti.field(dtype=int, shape=self.constraint_number)
            self.equivalent_torque_index = ti.field(dtype=data_type, shape=(self.constraint_number, self.bending_pairs_num, self.max_control_length))
            self.equivalent_torque_influence_id = ti.field(dtype=int, shape=(self.constraint_number, self.bending_pairs_num))
            self.equivalent_torque_coeff = ti.field(dtype=data_type, shape=(self.constraint_number, self.bending_pairs_num, self.max_control_length))
            
            self.string_number_each = ti.field(dtype=int, shape=self.constraint_number)
            self.string_length_decrease = ti.field(dtype=data_type, shape=self.constraint_number) #当前绳子的减少长度
            self.target_string_length_decrease = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length)) #摩擦影响下的目标收缩
            self.backup_delta_length = ti.field(dtype=data_type, shape=self.constraint_number)
            
            self.constraint_end_point_existence = ti.field(dtype=bool, shape=self.constraint_number)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.constraint_end_point_candidate_id = ti.field(dtype=int, shape=self.constraint_number)

            # 要控制的点的索引信息，至多控制单元数目个数的点, TYPE B的点
            # try:
            #     self.max_control_length = max([len(ele) for ele in self.method["id"]])
            # except:
            #     pass
            self.unit_control = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))
            self.initial_length_per_string = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            self.current_length_per_string = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            self.delta_length_per_string = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            
            self.string_force_initial_discount = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            self.string_force_current_discount = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            
            self.string_force_discount_sum = ti.field(dtype=data_type, shape=self.constraint_number)
            
            self.intersection_points = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number * self.max_control_length)
            self.intersection_points2 = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number * self.max_control_length)
            self.intersection_infos = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number * self.max_control_length)
            self.intersection_infos2 = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number * self.max_control_length)

            self.intersection_flag2 = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))
            self.intersection_flag = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))

            self.intersection_flag2_initial = ti.field(dtype=int, shape=self.constraint_number)
            self.intersection_points2_initial = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.intersection_infos2_initial = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)

            #穿线方向信息，选中某ID时：-1表示下方绳索，1表示上方绳索, -2表示下方绳索，但在下一个板的上方，2表示上方绳索，但在下一个板的下方
            self.hole_dir = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))

            self.tsa_string_number = (self.unit_indices_num + 2) * self.constraint_number
            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=int(self.tsa_string_number * 4))
            
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            #---#
            
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=self.constraint_number)
            self.constraint_length = ti.field(data_type, shape=self.constraint_number)
            self.backup_constraint_length = ti.field(data_type, shape=self.constraint_number)
            self.points = ti.Vector.field(3, dtype=data_type, shape=1)
            self.string_force_each = ti.field(data_type, shape=self.constraint_number)
            self.backup_string_force_each = ti.field(data_type, shape=self.constraint_number)
            self.max_force = ti.field(data_type, shape=self.constraint_number)

            self.error_status_buffer = ti.field(bool, shape=self.constraint_number)

        if self.P_number == 0:
            self.dfdx_connection = ti.Matrix.field(3, 3, dtype=data_type, shape=(1, 4, self.unit_edge_max))
            self.constraint_start_point_candidate = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_start_point_candidate_connection = ti.field(dtype=int, shape=1)
            self.loc_of_unit = ti.field(dtype=int, shape=1)
            self.connection_number = ti.field(dtype=int, shape=1)

            self.constraint_height = ti.field(dtype=data_type, shape=1)
        else:
            self.dfdx_connection = ti.Matrix.field(3, 3, dtype=data_type, shape=(self.P_number, 4, self.unit_edge_max))
            self.constraint_start_point_candidate = ti.Vector.field(3, dtype=data_type, shape=self.P_number)
            self.constraint_start_point_candidate_connection = ti.field(dtype=int, shape=self.P_number)
            self.loc_of_unit = ti.field(dtype=int, shape=self.P_number)
            self.connection_number = ti.field(dtype=int, shape=1)
            
            self.constraint_height = ti.field(dtype=data_type, shape=self.P_number)
            
        # 受力顶点，渲染时用
        self.force_vertex = ti.Vector.field(3, dtype=data_type, shape=2*self.kp_num)

        # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
        if self.facet_bending_pairs_num > 0:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_bending_pairs_num, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_crease_pairs_num, 2))
            self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(self.facet_bending_pairs_num, 2)) #弯曲对面积信息
            self.facet_crease_initial_length = ti.field(dtype=data_type, shape=self.facet_bending_pairs_num) #折痕长度
            self.facet_bending_pairs_distance = ti.field(dtype=data_type, shape=self.facet_bending_pairs_num) #折痕有效弯曲长度

        else:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(1, 2)) #弯曲对面积信息
            self.facet_crease_initial_length = ti.field(dtype=data_type, shape=1) #折痕长度
            self.facet_bending_pairs_distance = ti.field(dtype=data_type, shape=1) #折痕有效弯曲长度
        
        #----simulator information----#
        self.x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的位置
        self.v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的速度
        self.dv = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的加速度
        self.force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的力
        self.record_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #跟踪的某一类型的力
        self.mises = ti.field(dtype=data_type, shape=self.kp_num)
        self.print_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #打印的某一类型的力
        self.u = ti.Vector.field(2, dtype=data_type, shape=self.kp_num)
        self.field_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的场力
        self.stress = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的应力(mises)
        self.cauchy_stress_tensor = ti.Matrix.field(3, 3, dtype=data_type, shape=self.kp_num)
        self.cauchy_area = ti.field(dtype=data_type, shape=self.kp_num)

        self.backup_ground_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的历史摩擦

        self.percent = ti.field(data_type, shape=1)
        self.equal_arm_distance = ti.field(data_type, shape=1)
        self.friction_mode_enable = ti.field(dtype=int, shape=1)

        self.intersection_penalty = ti.field(dtype=int, shape=1)
        self.neglect_initial_penetration = ti.field(dtype=bool, shape=1)
        self.neglect_count = ti.field(dtype=int, shape=1)
        
        # self.backup_hole_friction_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的孔内摩擦力
        # self.hole_friction_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的孔内摩擦力

        if self.constraint_number:
            self.dldx_force = ti.Vector.field(3, dtype=data_type, shape=(self.constraint_number, self.kp_num)) #string
            # self.dldx_force_single = ti.Vector.field(3, dtype=data_type, shape=(self.constraint_number, self.max_control_length, self.kp_num)) #string
            # self.ddl_ddx = ti.Matrix.field(3 * self.unit_edge_max, 3 * self.unit_edge_max, data_type, shape=self.unit_indices_num)
            # self.dldx_friction_force = ti.Vector.field(3, dtype=data_type, shape=(self.constraint_number, self.kp_num)) #string

            self.ta_vector = ti.Vector.field(self.unit_edge_max, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.tb_vector = ti.Vector.field(self.unit_edge_max, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.unit_id_list = ti.field(dtype=int, shape=self.max_control_length * self.constraint_number)
            # self.unit_id_list2 = ti.field(dtype=int, shape=self.max_control_length * self.constraint_number)
            self.need_ta = ti.field(dtype=bool, shape=self.max_control_length * self.constraint_number)
            self.need_tb = ti.field(dtype=bool, shape=self.max_control_length * self.constraint_number)
            self.dndc = ti.Matrix.field(3, 3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.dndc2 = ti.Matrix.field(3, 3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.dnda = ti.Matrix.field(3, 3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.dndb = ti.Matrix.field(3, 3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            
            self.history_maximum_force = ti.field(dtype=data_type, shape=self.constraint_number)

        else:
            self.dldx_force = ti.Vector.field(3, dtype=data_type, shape=(1, 1)) #string
            # self.dldx_force_single = ti.Vector.field(3, dtype=data_type, shape=(1, 1, 1)) #string
            # self.ddl_ddx = ti.Matrix.field(3 * self.unit_edge_max, 3 * self.unit_edge_max, data_type, shape=1)
            # self.dldx_friction_force = ti.Vector.field(3, dtype=data_type, shape=(1, 1)) #string

            self.ta_vector = ti.Vector.field(self.unit_edge_max, dtype=data_type, shape=1)
            self.tb_vector = ti.Vector.field(self.unit_edge_max, dtype=data_type, shape=1)
            self.unit_id_list = ti.field(dtype=int, shape=1)
            # self.unit_id_list2 = ti.field(dtype=int, shape=1)
            self.need_ta = ti.field(dtype=bool, shape=1)
            self.need_tb = ti.field(dtype=bool, shape=1)
            self.dndc = ti.Matrix.field(3, 3, dtype=data_type, shape=1)
            self.dndc2 = ti.Matrix.field(3, 3, dtype=data_type, shape=1)
            self.dnda = ti.Matrix.field(3, 3, dtype=data_type, shape=1)
            self.dndb = ti.Matrix.field(3, 3, dtype=data_type, shape=1)
            
            self.history_maximum_force = ti.field(dtype=data_type, shape=1)

        self.ddl_ddx_num = ti.field(dtype=int, shape=1)
        self.system_type = ti.field(dtype=int, shape=1)

        self.stvk_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.bending_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.facet_bending_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.bending_damping_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.facet_bending_damping_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        # self.viscosity_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.string_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.ground_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.df_ground_force = ti.field(dtype=data_type, shape=self.kp_num)
        self.ground_friction = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)

        self.ddn1_ddx = ti.Matrix.field(3, 3, dtype=data_type, shape=(3, 3))
        self.ddn2_ddx = ti.Matrix.field(3, 3, dtype=data_type, shape=(3, 3))
        self.total_dvdx_field = ti.Vector.field(3, dtype=data_type, shape=9)

        self.connection_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)

        self.fix_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)

        self.back_up_x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.back_up_v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.back_up_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number + 1)

        self.start_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number + 1)
        self.start_x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的位置

        self.error_status = ti.field(bool, shape=1)
        
        self.epsilon_v = ti.field(data_type, shape=1)
        
    
    def commonStart_3(self):
        """
        仿真初始化第三阶段：设置时间步长和仿真参数。
        Simulation initialization phase 3: Set time step and simulation parameters.
        """
        self.current_t = 0.0
        self.dt = 1. / 240.
        self.half_dt = self.dt * 0.5

        self.folding_step = tm.pi * 18.0 / 18.0
        self.folding_max = tm.pi * 18.0 / 18.0

        self.half_max_size = self.max_size * 0.5

        self.folding_angle_reach_pi = ti.field(dtype=bool, shape=1)

        self.past_move_indice = 0.0

        self.stable_state = 0

        self.I2 = ti.Matrix.field(2, 2, dtype=data_type, shape=())
        self.I3 = ti.Matrix.field(3, 3, dtype=data_type, shape=())

        self.dm = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        self.dndx1 = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        self.dndx2 = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        self.dm2 = ti.Vector.field(3, dtype=data_type, shape=self.div_indices_num)
        self.f = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        self.piola_temp = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        self.piola = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        
        self.A = ti.field(dtype=data_type, shape=self.div_indices_num)
        self.A_total = ti.field(dtype=data_type, shape=1)

        self.mu = 3. / 2.6

        self.landa = 3. * 0.3 / (1 + 0.3) / (1 - 0.6)

        self.lame_k = 1000.

        self.dx_tolerance = 1e-6 * self.kp_num

        # derivative
        self.dDs = ti.Matrix.field(3, 3, dtype=data_type, shape=(3, 3))
        self.b = ti.field(data_type, shape=3 * self.kp_num)
        self.M_inv_f_ext = ti.field(data_type, shape=3 * self.kp_num)
        self.barrier_k = ti.field(data_type, shape=self.crease_pairs_num)
        self.K_element = ti.Matrix.field(9, 9, data_type, shape=self.div_indices_num)
        self.K_element_bending = ti.Matrix.field(12, 12, data_type, shape=self.bending_pairs_num+self.facet_bending_pairs_num)
        self.K_element_bending_damping = ti.Matrix.field(12, 12, data_type, shape=self.bending_pairs_num+self.facet_bending_pairs_num)

        self.triplets = ti.Vector.field(3, dtype=int, shape=self.div_indices_num)
        self.triplets_bending = ti.Vector.field(4, dtype=int, shape=self.bending_pairs_num+self.facet_bending_pairs_num)

        if self.constraint_number == 0 or self.P_number == 0:
            self.outer_P = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_Q = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_PI = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_QI = ti.Vector.field(3, dtype=data_type, shape=1)
            self.exist_grad_number = ti.field(dtype=int, shape=1)
        else:
            self.outer_P = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_Q = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_PI = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_QI = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.exist_grad_number = ti.field(dtype=int, shape=1)
        
        # trips = 108 * self.unit_edge_max + 15 * self.kp_num + 81 * self.div_indices_num + 288 * (self.bending_pairs_num + self.facet_bending_pairs_num) + (self.constraint_number * (self.max_control_length + 1)) * 9 * self.kp_num ** 2 + 576 * self.P_number + 36 * self.unit_indices_num * self.unit_edge_max ** 3
        trips = 9 * self.kp_num ** 2

        self.AK = ti.linalg.SparseMatrixBuilder(3 * self.kp_num, 3 * self.kp_num, max_num_triplets=trips, dtype=data_type)

        self.AK_field = ti.field(data_type, shape=(3 * self.kp_num, 3 * self.kp_num))

        self.AM = ti.linalg.SparseMatrix(3 * self.kp_num, 3 * self.kp_num, dtype=data_type)

        self.u0 = ti.field(data_type, shape=3 * self.kp_num) # solution

        self.lames_bonus = ti.field(data_type, shape=2) # mu, landa

        # flat the line_indices
        for i in range(len(self.ori_sim.line_indices)):
            self.ori_sim.line_indices[i] = [self.ori_sim.line_indices[i][0][START], self.ori_sim.line_indices[i][0][END], self.ori_sim.line_indices[i][1], self.ori_sim.line_indices[i][2], self.ori_sim.line_indices[i][3]]
      
    def biasKp(self, kp, bias):
        """
        对关键点坐标施加偏移量。
        Apply bias to keypoint coordinates.
        
        :param kp: 原始关键点坐标 / Original keypoint coordinates
        :param bias: 偏移量 [x, y, z] / Bias [x, y, z]
        :return: 偏移后的关键点坐标 / Biased keypoint coordinates
        """
        new_kp = deepcopy(kp)
        for i in range(min(len(new_kp), len(bias))):
            new_kp[i] += bias[i]
        return new_kp
    
    def biasId(self, id, bias):
        """
        对ID施加偏移量。
        Apply bias to ID.
        
        :param id: 原始ID / Original ID
        :param bias: 偏移量 / Bias
        :return: 偏移后的ID / Biased ID
        """
        id += bias
        return id
      
    def pointInList(self, kp, tolerance=2):
        """
        检查关键点是否已存在于列表中。
        Check if a keypoint already exists in the list.
        
        :param kp: 待检查的关键点 / Keypoint to check
        :param tolerance: 距离容差 / Distance tolerance
        :return: 已存在点的索引，若不存在则返回-1 / Index of existing point, -1 if not found
        """
        for i in range(len(self.kps)):
            if distance3D(kp, self.kps[i]) < tolerance:
                return i
        return -1      

    def reconstructRoutingUsingMap(self, method, map):
        new_method = {
            "type": [],
            "id": [],
            "reverse": []
        }
        types = method["type"]
        string_number = len(types)
        ids = method["id"]
        directions = method["reverse"]
        for i in range(string_number):
            current_type_list = types[i]
            current_id_list = ids[i]
            current_direction_list = directions[i]
            new_current_type_list = []
            new_id_list = []
            new_direction_list = []
            length = len(current_type_list)
            for j in range(length):
                current_type = current_type_list[j]
                current_id = current_id_list[j]
                current_direction = current_direction_list[j]
                unit_id_map = map[current_id]
                if current_type == 'A':
                    new_current_type_list.append(current_type)
                    new_id_list.append(current_id)
                    new_direction_list.append(current_direction)
                if current_type == 'B':
                    min_z_axis = self.units[unit_id_map[0]].crease[0][START][Z]
                    min_choosed_id = unit_id_map[0]
                    max_z_axis = self.units[unit_id_map[0]].crease[0][START][Z]
                    max_choosed_id = unit_id_map[0]

                    for unit_id in unit_id_map:
                        new_z_axis = self.units[unit_id].crease[0][START][Z]
                        if new_z_axis < min_z_axis:
                            min_z_axis = new_z_axis
                            min_choosed_id = unit_id
                        if new_z_axis > max_z_axis:
                            max_z_axis = new_z_axis
                            max_choosed_id = unit_id

                    if len(unit_id_map) > 1:
                        if current_direction == -1:
                            new_current_type_list.append(current_type)
                            new_id_list.append(min_choosed_id)
                            new_direction_list.append(current_direction)
                            new_current_type_list.append(current_type)
                            new_id_list.append(max_choosed_id)
                            new_direction_list.append(-1)
                        elif current_direction == 1:
                            new_current_type_list.append(current_type)
                            new_id_list.append(max_choosed_id)
                            new_direction_list.append(current_direction)
                            new_current_type_list.append(current_type)
                            new_id_list.append(min_choosed_id)
                            new_direction_list.append(1)
                    else:
                        new_current_type_list.append(current_type)
                        new_id_list.append(unit_id_map[0])
                        new_direction_list.append(current_direction)

            new_method["type"].append(new_current_type_list)
            new_method["id"].append(new_id_list)
            new_method["reverse"].append(new_direction_list)
        return new_method
    
    # def startOnlyTSA(self, units, max_size, total_bias, unit_edge_max, connected_pairs=None): #外面定义好
    #     self.units = units
    #     self.max_size = max_size
    #     self.total_bias = total_bias
    #     # self.contributions = [[]]
    #     for i in range(len(self.P_candidate)):
    #         if self.P_candidate_connection[i] >= 0:
    #             # deal with mountain
    #             self.units.append(Unit())
    #             self.units[-1].special = True
    #             self.units[-1].special_mass = self.accessory_mass

    #             relative_length = 0.7071 * self.controller_side_length

    #             kps = [
    #                 [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],
    #                 [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],
    #                 [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],      
    #                 [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],                                                             
    #             ]
    #             for j in range(0, -len(kps), -1):
    #                 crease_type = BORDER
    #                 current_kp = kps[j]
    #                 next_kp = kps[j - 1]

    #                 self.units[-1].addCrease(Crease(
    #                     current_kp, next_kp, crease_type
    #                 ))
    #             if self.mode == self.TSA_SIM:
    #                 #deal with valley
    #                 z_bias = 10.0
    #                 x_bias = -4.0
    #                 y_bias = -4.0
    #                 self.P_candidate.append([self.P_candidate[i][X], self.P_candidate[i][Y], self.P_candidate[i][Z] + z_bias])
    #                 self.P_candidate_connection.append(self.P_candidate_connection[i])
    #                 self.units.append(Unit())
    #                 self.units[-1].special = True
    #                 self.units[-1].special_mass = self.accessory_mass

    #                 relative_length = 0.7071 * self.controller_side_length

    #                 kps = [
    #                     [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
    #                     [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
    #                     [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],      
    #                     [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],                                                             
    #                 ]
    #                 for j in range(0, -len(kps), -1):
    #                     crease_type = BORDER
    #                     current_kp = kps[j]
    #                     next_kp = kps[j - 1]

    #                     self.units[-1].addCrease(Crease(
    #                         current_kp, next_kp, crease_type
    #                     ))
    #                 # weight
    #                 z_bias = 38.5
    #                 self.P_candidate.append([self.P_candidate[i][X] + x_bias, self.P_candidate[i][Y] + y_bias, self.P_candidate[i][Z] + z_bias])
    #                 self.P_candidate_connection.append(self.P_candidate_connection[i])
    #                 self.units.append(Unit())
    #                 self.units[-1].special = True
    #                 self.units[-1].special_mass = self.controller_mass

    #                 relative_length = 0.7071 * self.controller_side_length

    #                 kps = [
    #                     [self.P_candidate[i][X] + x_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
    #                     [self.P_candidate[i][X] + x_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
    #                     [self.P_candidate[i][X] + x_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],      
    #                     [self.P_candidate[i][X] + x_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],                                                             
    #                 ]
    #                 for j in range(0, -len(kps), -1):
    #                     crease_type = BORDER
    #                     current_kp = kps[j]
    #                     next_kp = kps[j - 1]

    #                     self.units[-1].addCrease(Crease(
    #                         current_kp, next_kp, crease_type
    #                     ))
                
    #     self.commonStart_1(unit_edge_max)
    #     self.sim_mode = self.TSA_SIM
    #     self.sequence_level_initial = 0
    #     self.commonStart_2(unit_edge_max)
    #     self.gravitational_acc = tm.vec3(self.standard_g) 
    #     self.commonStart_3()
    
    def start(self, filepath, unit_edge_max, sim_type, thick_mode=False, additional_constraint_number=-1):
        """
        启动仿真，加载折纸模型并初始化所有物理参数。
        Start simulation, load origami model and initialize all physics parameters.
        
        :param filepath: 折纸模型文件路径 / Origami model file path
        :param unit_edge_max: 单元最大边数 / Maximum number of edges per unit
        :param sim_type: 仿真类型（FOLD_SIM或TSA_SIM）/ Simulation type (FOLD_SIM or TSA_SIM)
        :param thick_mode: 是否启用厚板模式 / Whether to enable thick panel mode
        :param additional_constraint_number: 附加约束数量 / Additional constraint number
        """
        # 存储厚板模式标志 / Store thick mode flag
        self.thick_mode_flag = thick_mode
        
        # 获取点和线段信息
        if sim_type == self.FOLD_SIM and (not self.check_connection_matrix or thick_mode):
            self.folding_angle_maximum_ratio = 1.
            
        with open("./descriptionData/" + filepath + ".json", 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
        self.input_json = input_json
        self.kps = []
        self.lines = []
        self.units = []
        
        self.contributions = []
        
        if not thick_mode:
            for i in range(len(input_json["kps"])):
                self.kps.append(input_json["kps"][i])
                
            for i in range(len(input_json["lines"])):
                self.lines.append(Crease(
                    input_json["lines"][i][START], input_json["lines"][i][END], BORDER 
                ))
                self.lines[i].crease_type = input_json["line_features"][i]["type"]
                self.lines[i].level = input_json["line_features"][i]["level"]
                self.lines[i].coeff = input_json["line_features"][i]["coeff"]
                try:
                    self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
                    if type(self.lines[i].recover_level) != list:
                        self.lines[i].recover_level = []
                except:
                    self.lines[i].recover_level = []
                try:
                    self.lines[i].recover_angle = input_json["line_features"][i]["recover_angle"]
                except:
                    self.lines[i].recover_angle = []
                try:
                    self.lines[i].thick_panel_height = input_json["line_features"][i]["thick_panel_height"]
                except:
                    self.lines[i].thick_panel_height = 0.0
                self.lines[i].hard = input_json["line_features"][i]["hard"]
                self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
                self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]
            for i in range(len(input_json["units"])):
                self.units.append(Unit())
                kps = deepcopy(input_json["units"][i])
                for j in range(0, -len(kps), -1):
                    crease_type = BORDER
                    hard = False
                    current_kp = deepcopy(kps[j])
                    next_kp = deepcopy(kps[j - 1])
                    for line in self.lines:
                        if (distance3D(line[START], current_kp) < 1e-3 and distance3D(line[END], next_kp) < 1e-3) or \
                            (distance3D(line[END], current_kp) < 1e-3 and distance3D(line[START], next_kp) < 1e-3):
                            crease_type = line.getType()
                            hard = line.hard
                            folding_angle_upper_bound = line.folding_angle_upper_bound
                            folding_angle_lower_bound = line.folding_angle_lower_bound
                            break
                    self.units[i].addCrease(Crease(
                        current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                    ))
            
                try:
                    contribution_for_unit = deepcopy(input_json["contributions"][i])
                    new_contribution = []
                    for j in range(0, -len(contribution_for_unit), -1):
                        new_contribution.append(contribution_for_unit[j])
                    self.contributions.append(new_contribution)
                except:
                    pass
            
        else:
            unit_mapping = []
            for i in range(len(input_json["lines"])):
                line_start = deepcopy(input_json["lines"][i][START])

                add_height = 10.0 if input_json["line_features"][i]["type"] == 0 else -10.0
                try:
                    add_height = input_json["line_features"][i]["thick_panel_height"]
                except:
                    pass
                if abs(add_height) < 1e-2:
                    add_height = 10.0 if input_json["line_features"][i]["type"] == 0 else -10.0

                if len(line_start) == 2:
                    line_start.append(add_height)
                else:
                    line_start[Z] = add_height
                line_end = deepcopy(input_json["lines"][i][END])
                if len(line_end) == 2:
                    line_end.append(add_height)
                else:
                    line_end[Z] = add_height
                
                self.lines.append(Crease(
                    line_start, line_end, BORDER 
                ))
                self.lines[i].crease_type = input_json["line_features"][i]["type"]
                self.lines[i].level = input_json["line_features"][i]["level"]
                self.lines[i].coeff = input_json["line_features"][i]["coeff"]
                try:
                    self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
                    if type(self.lines[i].recover_level) != list:
                        self.lines[i].recover_level = []
                except:
                    self.lines[i].recover_level = []
                try:
                    self.lines[i].recover_angle = input_json["line_features"][i]["recover_angle"]
                except:
                    self.lines[i].recover_angle = []
                try:
                    self.lines[i].thick_panel_height = add_height
                except:
                    self.lines[i].thick_panel_height = 10.0 if self.lines[i].crease_type == 0 else -10.0
                self.lines[i].hard = input_json["line_features"][i]["hard"]
                self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
                self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]

            for i in range(len(input_json["units"])):
                kps = deepcopy(input_json["units"][i])
                # check different height
                height_parameters = []
                for j in range(0, -len(kps), -1):
                    crease_type = BORDER
                    hard = False
                    current_kp = deepcopy(kps[j])
                    next_kp = deepcopy(kps[j - 1])
                    for line in self.lines:
                        if (distance(line[START], current_kp) < 1e-3 and distance(line[END], next_kp) < 1e-3) or \
                            (distance(line[END], current_kp) < 1e-3 and distance(line[START], next_kp) < 1e-3):
                                if (line.getType() == VALLEY or line.getType() == MOUNTAIN) and line.thick_panel_height not in height_parameters:
                                    height_parameters.append(line.thick_panel_height)
                                break
                
                unit_mapping.append([len(self.units) + k for k in range(len(height_parameters))])
                if i == 0 and len(height_parameters) >= 2:
                    self.connected_unit_pairs += [[0, x] for x in range(1, len(height_parameters))]
                elif i > 0 and len(height_parameters) >= 2:
                    self.connected_unit_pairs += [[len(self.units), x + len(self.units)] for x in range(1, len(height_parameters))]
                
                for height in height_parameters:
                    self.units.append(Unit())
                    for j in range(0, -len(kps), -1):
                        crease_type = BORDER
                        hard = False
                        current_kp = deepcopy(kps[j])
                        current_kp[Z] = height
                        next_kp = deepcopy(kps[j - 1])
                        next_kp[Z] = height
                        folding_angle_upper_bound = math.pi
                        folding_angle_lower_bound = -math.pi
                        level = 0
                        coeff = 1.
                        rec_level = []
                        rec_angle = []
                        
                        for line in self.lines:
                            if (distance3D(line[START], current_kp) < 1e-3 and distance3D(line[END], next_kp) < 1e-3) or \
                                (distance3D(line[END], current_kp) < 1e-3 and distance3D(line[START], next_kp) < 1e-3):
                                    current_kp[Z] = line.thick_panel_height
                                    next_kp[Z] = line.thick_panel_height
                                    crease_type = line.getType()
                                    hard = line.hard
                                    folding_angle_upper_bound = line.folding_angle_upper_bound
                                    folding_angle_lower_bound = line.folding_angle_lower_bound
                                    level = line.level
                                    coeff = line.coeff
                                    rec_level = line.recover_level
                                    rec_angle = line.recover_angle
                                    break
                                
                        if self.pointInList(current_kp, 1e-3) == -1:
                            self.kps.append(current_kp)
                        if self.pointInList(next_kp, 1e-3) == -1:
                            self.kps.append(next_kp) 
                             
                        new_crease = Crease(
                            current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                        )
                        new_crease.level = level
                        new_crease.coeff = coeff
                        new_crease.recover_level = rec_level
                        new_crease.recover_angle = rec_angle
                        self.units[-1].addCrease(new_crease)
                    
                    try:
                        contribution_for_unit = deepcopy(input_json["contributions"][i])
                        new_contribution = []
                        for j in range(0, -len(contribution_for_unit), -1):
                            new_contribution.append(contribution_for_unit[j])
                        self.contributions.append(new_contribution)
                    except:
                        pass
                        
        if len(self.contributions) == 0:
            self.contributions.append([])
        try:
            self.method = deepcopy(input_json["strings"])
            if thick_mode:
                self.method = self.reconstructRoutingUsingMap(self.method, unit_mapping)
        except:
            self.method = None
        try:
            self.P_candidate = input_json["P_candidators"]["points"]
            self.P_candidate_connection = input_json["P_candidators"]["connections"]
            true_P_number = len(self.P_candidate)
            for i in range(true_P_number):
                if self.P_candidate_connection[i] >= 0:
                    # deal with mountain
                    self.units.append(Unit())
                    self.units[-1].special = True
                    self.units[-1].special_mass = self.accessory_mass

                    relative_length = 0.7071 * self.controller_side_length

                    kps = [
                        [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],
                        [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],
                        [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],      
                        [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],                                                             
                    ]
                    for j in range(0, len(kps), 1):
                        crease_type = BORDER
                        current_kp = kps[j]
                        next_kp = kps[(j + 1) % len(kps)]

                        self.units[-1].addCrease(Crease(
                            current_kp, next_kp, crease_type
                        ))
                    if sim_type == self.TSA_SIM:
                        #deal with valley
                        z_bias = 10.0
                        x_bias = 0.0
                        y_bias = 0.0
                        self.P_candidate.append([self.P_candidate[i][X] + x_bias, self.P_candidate[i][Y] + y_bias, self.P_candidate[i][Z] + z_bias])
                        self.P_candidate_connection.append(self.P_candidate_connection[i])
                        self.units.append(Unit())
                        self.units[-1].special = True
                        self.units[-1].special_mass = self.accessory_mass

                        relative_length = 0.7071 * self.controller_side_length

                        kps = [
                            [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                            [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                            [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],      
                            [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],                                                             
                        ]
                        for j in range(0, len(kps), 1):
                            crease_type = BORDER
                            current_kp = kps[j]
                            next_kp = kps[(j + 1) % len(kps)]

                            self.units[-1].addCrease(Crease(
                                current_kp, next_kp, crease_type
                            ))
                        # weight
                        z_bias = self.controller_centroid_bias[Z]
                        x_bias = self.controller_centroid_bias[X]
                        y_bias = self.controller_centroid_bias[Y]

                        # New: split into half
                        h_half = z_bias - 10.0
                        bottom_z_bias = h_half * (1. - 1. / math.sqrt(3)) + 10.0
                        upper_z_bias = h_half * (1. + 1. / math.sqrt(3)) + 10.0
                        m_half = 0.5 * self.controller_mass

                        # bottom
                        self.P_candidate.append([self.P_candidate[i][X] + x_bias, self.P_candidate[i][Y] + y_bias, self.P_candidate[i][Z] + bottom_z_bias])
                        self.P_candidate_connection.append(self.P_candidate_connection[i])
                        self.units.append(Unit())
                        self.units[-1].special = True
                        self.units[-1].special_mass = m_half

                        relative_length = 0.7071 * self.controller_side_length

                        kps = [
                            [self.P_candidate[i][X] + x_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + bottom_z_bias],
                            [self.P_candidate[i][X] + x_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + bottom_z_bias],
                            [self.P_candidate[i][X] + x_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + bottom_z_bias],      
                            [self.P_candidate[i][X] + x_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + bottom_z_bias],                                                             
                        ]
                        for j in range(0, len(kps), 1):
                            crease_type = BORDER
                            current_kp = kps[j]
                            next_kp = kps[(j + 1) % len(kps)]

                            self.units[-1].addCrease(Crease(
                                current_kp, next_kp, crease_type
                            ))
                    
                        # upper
                        self.P_candidate.append([self.P_candidate[i][X] + x_bias, self.P_candidate[i][Y] + y_bias, self.P_candidate[i][Z] + upper_z_bias])
                        self.P_candidate_connection.append(self.P_candidate_connection[i])
                        self.units.append(Unit())
                        self.units[-1].special = True
                        self.units[-1].special_mass = m_half

                        relative_length = 0.7071 * self.controller_side_length

                        kps = [
                            [self.P_candidate[i][X] + x_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + upper_z_bias],
                            [self.P_candidate[i][X] + x_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + upper_z_bias],
                            [self.P_candidate[i][X] + x_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + upper_z_bias],      
                            [self.P_candidate[i][X] + x_bias + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + y_bias - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + upper_z_bias],                                                             
                        ]
                        for j in range(0, len(kps), 1):
                            crease_type = BORDER
                            current_kp = kps[j]
                            next_kp = kps[(j + 1) % len(kps)]

                            self.units[-1].addCrease(Crease(
                                current_kp, next_kp, crease_type
                            ))
        except:
            self.P_candidate = [[]]
            self.P_candidate_connection = []
        try:
            self.fix_id = deepcopy(input_json["fix"])
            if len(self.connected_unit_pairs):
                new_fix_id = []
                for ele in self.fix_id:
                    true_unit_id_list = unit_mapping[ele]
                    new_fix_id += true_unit_id_list
                self.fix_id = new_fix_id
        except:
            self.fix_id = [-1]
        try:
            self.targets = deepcopy(input_json["crease_angle"])
        except:
            self.targets = []
            
        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)
        
        self.kp_number_each = len(self.kps)
        self.line_number_each = len(self.lines)
        self.unit_number_each = len(self.units)
        self.P_number_each = len(self.P_candidate)
        
        if self.multiple_origami > 1:
            raw = int(self.multiple_origami ** 0.5)
            
            for total in range(1, self.multiple_origami):
                x_bias = (total % raw) * max_x * 1.2
                y_bias = (total // raw) * max_y * 1.2
                bias = [x_bias, y_bias, 0]
                
                for i in range(self.kp_number_each):
                    self.kps.append(self.biasKp(input_json["kps"][i], bias))
                    
                for i in range(self.line_number_each):
                    self.lines.append(Crease(
                        self.biasKp(input_json["lines"][i][START], bias), self.biasKp(input_json["lines"][i][END], bias), BORDER 
                    ))
                    index = i + self.line_number_each * total
                    self.lines[index].crease_type = input_json["line_features"][i]["type"]
                    self.lines[index].level = input_json["line_features"][i]["level"]
                    self.lines[index].coeff = input_json["line_features"][i]["coeff"]
                    try:
                        self.lines[index].recover_level = input_json["line_features"][i]["recover_level"]
                        if type(self.lines[index].recover_level) != list:
                            self.lines[index].recover_level = []
                    except:
                        self.lines[index].recover_level = []
                    try:
                        self.lines[index].recover_angle = input_json["line_features"][i]["recover_angle"]
                    except:
                        self.lines[index].recover_angle = []
                    self.lines[index].hard = input_json["line_features"][i]["hard"]
                    self.lines[index].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
                    self.lines[index].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]
                
                for i in range(self.unit_number_each):
                    self.units.append(Unit())
                    kps = deepcopy(input_json["units"][i])
                    for kp_id in range(len(kps)):
                        kps[kp_id] = self.biasKp(kps[kp_id], bias)
                        
                    for j in range(0, -len(kps), -1):
                        crease_type = BORDER
                        hard = False
                        current_kp = deepcopy(kps[j])
                        next_kp = deepcopy(kps[j - 1])
                        for line_id in range(self.line_number_each * total, self.line_number_each * (total + 1)):
                            line = self.lines[line_id]
                            if (distance3D(line[START], current_kp) < 1e-3 and distance3D(line[END], next_kp) < 1e-3) or \
                                (distance3D(line[END], current_kp) < 1e-3 and distance3D(line[START], next_kp) < 1e-3):
                                crease_type = line.getType()
                                hard = line.hard
                                folding_angle_upper_bound = line.folding_angle_upper_bound
                                folding_angle_lower_bound = line.folding_angle_lower_bound
                                break
                        self.units[i + total * self.unit_number_each].addCrease(Crease(
                            current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                        ))
                
                try:
                    new_method = deepcopy(input_json["strings"])
                    strings_id = deepcopy(new_method["id"])
                    for string_number in range(len(strings_id)):
                        string_len = len(strings_id[string_number])
                        for string_id in range(string_len):
                            if new_method['type'][string_number][string_id] == 'A':
                                new_method['id'][string_number][string_id] = self.biasId(new_method['id'][string_number][string_id], self.P_number_each * total)
                            else:
                                new_method['id'][string_number][string_id] = self.biasId(new_method['id'][string_number][string_id], self.unit_number_each * total)
                        self.method["type"].append(new_method["type"][string_number])
                        self.method["id"].append(new_method["id"][string_number])
                        self.method["reverse"].append(new_method["reverse"][string_number])
                except:
                    new_method = None
                try:
                    for i in range(true_P_number):
                        self.P_candidate.append(self.biasKp(input_json["P_candidators"]["points"][i], bias))
                        self.P_candidate_connection.append(-1 if input_json["P_candidators"]["connections"][i] == -1 else \
                            self.biasId(input_json["P_candidators"]["connections"][i], self.unit_number_each * total))
                        
                    for i in range(total * self.P_number_each, total * self.P_number_each + true_P_number):
                        if self.P_candidate_connection[i] >= 0:
                            # deal with mountain
                            self.units.append(Unit())
                            self.units[-1].special = True
                            self.units[-1].special_mass = self.accessory_mass

                            relative_length = 0.7071 * self.controller_side_length

                            kps = [
                                [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],
                                [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],
                                [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z]],      
                                [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z]],                                                             
                            ]
                            for j in range(0, -len(kps), -1):
                                crease_type = BORDER
                                current_kp = kps[j]
                                next_kp = kps[j - 1]

                                self.units[-1].addCrease(Crease(
                                    current_kp, next_kp, crease_type
                                ))
                            if sim_type == self.TSA_SIM:
                                #deal with valley
                                z_bias = 10.0
                                self.P_candidate.append([self.P_candidate[i][X], self.P_candidate[i][Y], self.P_candidate[i][Z] + z_bias])
                                self.P_candidate_connection.append(self.P_candidate_connection[i])
                                self.units.append(Unit())
                                self.units[-1].special = True
                                self.units[-1].special_mass = self.accessory_mass

                                relative_length = 0.7071 * self.controller_side_length

                                kps = [
                                    [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                                    [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                                    [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],      
                                    [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],                                                             
                                ]
                                for j in range(0, -len(kps), -1):
                                    crease_type = BORDER
                                    current_kp = kps[j]
                                    next_kp = kps[j - 1]

                                    self.units[-1].addCrease(Crease(
                                        current_kp, next_kp, crease_type
                                    ))
                                # weight
                                z_bias = self.controller_centroid_bias[Z]
                                x_bias = self.controller_centroid_bias[X]
                                y_bias = self.controller_centroid_bias[Y]
                                self.P_candidate.append([self.P_candidate[i][X], self.P_candidate[i][Y], self.P_candidate[i][Z] + z_bias])
                                self.P_candidate_connection.append(self.P_candidate_connection[i])
                                self.units.append(Unit())
                                self.units[-1].special = True
                                self.units[-1].special_mass = self.controller_mass

                                relative_length = 0.7071 * self.controller_side_length

                                kps = [
                                    [self.P_candidate[i][X] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                                    [self.P_candidate[i][X] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] + relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],
                                    [self.P_candidate[i][X] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Z] + z_bias],      
                                    [self.P_candidate[i][X] + relative_length * math.sin(self.controller_rotation), self.P_candidate[i][Y] - relative_length * math.cos(self.controller_rotation), self.P_candidate[i][Z] + z_bias],                                                             
                                ]
                                for j in range(0, -len(kps), -1):
                                    crease_type = BORDER
                                    current_kp = kps[j]
                                    next_kp = kps[j - 1]

                                    self.units[-1].addCrease(Crease(
                                        current_kp, next_kp, crease_type
                                    ))
                except:
                    pass
                try:
                    new_contributions_for_new_unit = deepcopy(input_json["contributions"])
                    for i in range(len(new_contributions_for_new_unit)):
                        new_contribution = []
                        for j in range(0, -len(new_contributions_for_new_unit[i]), -1):
                            new_contribution.append(new_contributions_for_new_unit[i][j])
                        self.contributions.append(new_contribution)
                except:
                    pass
                try:
                    new_fix_id = deepcopy(input_json["fix"])
                    for i in range(len(new_fix_id)):
                        new_fix_id[i] = self.biasId(new_fix_id[i], self.unit_number_each * total)
                    self.fix_id += new_fix_id
                except:
                    pass
                try:
                    new_targets = deepcopy(input_json["crease_angle"])
                    self.targets += new_targets
                except:
                    pass
        
        self.commonStart_1(unit_edge_max)
    
        self.commonStart_2(self.unit_edge_max, additional_constraint_number)
        
        self.commonStart_3()
        self.sim_mode = sim_type

    @ti.kernel
    def fill_line_vertex(self):
        """
        填充线段顶点位置，为谷折线添加偏移以可视化厚度效果。
        Fill line vertex positions, add offset for valley lines to visualize thickness effect.
        """
        for i in ti.ndrange(self.line_total_indice_num):
            indice1 = self.line_pairs[i, 0]
            indice2 = self.line_pairs[i, 1]
            line_type = self.line_color[2 * i]
            bias = tm.vec3([0., 0., 0.])
            if line_type[1] == 0.17: # valley
                unit_connection_1 = self.line_connection_unit[i][0]
                unit_connection_2 = self.line_connection_unit[i][1]
                normal_1 = self.calculateNormalVectorWithUnitId(self.unit_indices[unit_connection_1])
                normal_2 = self.calculateNormalVectorWithUnitId(self.unit_indices[unit_connection_2])
                bias = self.BIAS * (normal_1 + normal_2) * 2. / (tm.max((normal_1 + normal_2).norm(), 1.0)) ** 2
            self.line_vertex[2 * i] = self.vertices[indice1] + bias
            self.line_vertex[2 * i + 1] = self.vertices[indice2] + bias
    
    @ti.func
    def getBendingEnergy(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, a1=1., a2=1., L=1., id=-1, d=1., tsa_mode=False):
        """
        计算折痕的弯曲能量。
        Calculate the bending energy of a crease.
        
        :param cs: 折痕起点坐标 / Crease start point coordinates
        :param ce: 折痕终点坐标 / Crease end point coordinates
        :param p1: 第一面板点坐标 / First panel point coordinates
        :param p2: 第二面板点坐标 / Second panel point coordinates
        :param k: 弯曲刚度系数 / Bending stiffness coefficient
        :param theta: 目标折叠角度 / Target folding angle
        :param crease_type: 折痕类型（山折/谷折）/ Crease type (mountain/valley)
        :param debug: 是否启用调试模式 / Whether to enable debug mode
        :param enable_dynamic_change: 是否启用动态变化 / Whether to enable dynamic change
        :param a1: 第一面板面积 / First panel area
        :param a2: 第二面板面积 / Second panel area
        :param L: 折痕长度 / Crease length
        :param id: 折痕ID / Crease ID
        :param d: 厚度参数 / Thickness parameter
        :param tsa_mode: 是否为TSA模式 / Whether in TSA mode
        :return: 弯曲能量 / Bending energy
        """
        barrier_left = tm.pi / 36.
        collision_indice = self.collision_indice

        folding_angle_upper_bound = tm.pi
        upper_barrier = folding_angle_upper_bound - barrier_left
        folding_angle_lower_bound = -tm.pi
        lower_barrier = folding_angle_lower_bound + barrier_left

        if id != -1:
            folding_angle_upper_bound = self.folding_angle_upper_bound[id]
            upper_barrier = folding_angle_upper_bound - barrier_left
            folding_angle_lower_bound = self.folding_angle_lower_bound[id]
            lower_barrier = folding_angle_lower_bound + barrier_left
            collision_indice = self.barrier_coeff_list[id]

        upper_barrier_energy_maximum = -collision_indice * barrier_left ** 2 * tm.log(self.collision_d / barrier_left) #positive
        upper_barrier_maximum = collision_indice * (2 * barrier_left * tm.log(self.collision_d / barrier_left) - barrier_left ** 2 / self.collision_d) #negative
        upper_barrier_df_maximum = 2 * collision_indice * (2 * barrier_left / self.collision_d + barrier_left ** 2 / (2 * (self.collision_d ** 2)) - tm.log(self.collision_d / barrier_left)) #positive

        lower_barrier_energy_maximum = upper_barrier_energy_maximum #positive
        lower_barrier_maximum = -upper_barrier_maximum #positive
        lower_barrier_df_maximum = upper_barrier_df_maximum #positive

        xc = ce - cs

        f11 = p1 - cs
        f22 = p2 - cs
        n1 = xc.cross(f11)
        n2 = f22.cross(xc)

        energy = 0.0

        n1_norm = n1.norm()
        n2_norm = n2.norm()

        multi_n1_n2 = n1_norm * n2_norm

        dir = n1.cross(n2).dot(xc)
        val = n1.dot(n2)

        norm_val = val / multi_n1_n2

        current_theta = 0.0
        if norm_val >= 1.0:
            val = multi_n1_n2
            norm_val = 1.0
            current_theta = 0.0
        elif norm_val <= -1.0:
            val = -multi_n1_n2
            norm_val = -1.0
            current_theta = tm.pi
        else:
            current_theta = tm.acos(norm_val)
            
        # if id != -1 and not self.backup_enable_equivalent_torque[id] and tsa_mode:
        #     theta = 0.0
    
        # 求折叠角
        n_value = 0.
        backup_n_value = 0.
        signed_current_theta = 0.
       
        # 求折叠角
        if dir >= 0.:
            if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5: #180~270
                signed_current_theta = 2. * tm.pi - current_theta
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
                energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5
            else: #-180~0
                signed_current_theta = -current_theta        
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                t11 = signed_current_theta - lower_barrier
                t22 = folding_angle_lower_bound - self.collision_d - signed_current_theta
                if t11 <= 0 and t11 >= -barrier_left:
                    n_value += collision_indice * (2 * t11 * tm.log(t22 / -(barrier_left)) - t11 ** 2 / t22)
                    energy += -collision_indice * t11 ** 2 * tm.log(t22 / -(barrier_left))
                elif t11 < -barrier_left:
                    n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
                    energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
        else:
            if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5: #-270~-180
                signed_current_theta = current_theta - 2. * tm.pi
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
                energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
            else: #0~180
                signed_current_theta = current_theta
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                t11 = signed_current_theta - upper_barrier
                t22 = folding_angle_upper_bound + self.collision_d - signed_current_theta
                if t11 >= 0 and t11 <= barrier_left:
                    n_value += collision_indice * (2 * t11 * tm.log(t22 / barrier_left) - t11 ** 2 / t22)
                    energy += -collision_indice * t11 ** 2 * tm.log(t22 / barrier_left)
                elif t11 > barrier_left:
                    n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
                    energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5

        energy += 0.5 * k * L * backup_n_value ** 2

        return energy
    
    @ti.func
    def getSkewMatrix(self, x):
        """
        计算向量的斜对称矩阵（叉积矩阵）。
        Calculate the skew-symmetric matrix (cross product matrix) of a vector.
        
        :param x: 输入向量 / Input vector
        :return: 3x3斜对称矩阵 / 3x3 skew-symmetric matrix
        """
        return ti.Matrix.cols([[0., x[Z], -x[Y]], [-x[Z], 0., x[X]], [x[Y], -x[X], 0.]])
    
    @ti.func
    def getDthetaDx(self, x0, x1, x2, x3, theta):
        """
        计算折痕角度对四个顶点位置的梯度。
        Calculate the gradient of crease angle with respect to four vertex positions.
        
        :param x0: 折痕起点 / Crease start point
        :param x1: 第一面板点 / First panel point
        :param x2: 折痕终点 / Crease end point
        :param x3: 第二面板点 / Second panel point
        :param theta: 当前折痕角度 / Current crease angle
        :return: 四个梯度矩阵 (dtheta/dx0, dtheta/dx1, dtheta/dx2, dtheta/dx3) / Four gradient matrices
        """
        s1 = x1 - x0
        cr = x2 - x0
        s2 = x3 - x0

        e = -cr / cr.norm() # valley crease is positive

        v1 = cr.cross(s2)
        v2 = s1.cross(cr)

        v1_norm = v1.norm()
        v2_norm = v2.norm()

        n1 = v1 / v1_norm
        n2 = v2 / v2_norm

        proj_v1 = self.I3[None] - n1.outer_product(n1)
        proj_v2 = self.I3[None] - n2.outer_product(n2)

        dv1dx0 = self.getSkewMatrix(x3 - x2)
        # dv1dx1 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        dv1dx2 = self.getSkewMatrix(x0 - x3)
        dv1dx3 = self.getSkewMatrix(x2 - x0)

        dv2dx0 = self.getSkewMatrix(x2 - x1)
        dv2dx1 = self.getSkewMatrix(x0 - x2)
        dv2dx2 = self.getSkewMatrix(x1 - x0)
        # dv2dx3 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

        v1_skew = self.getSkewMatrix(v1)
        v2_skew = self.getSkewMatrix(v2)

        cos_term_x0 = (dv2dx0 @ proj_v2 @ v1_skew - dv1dx0 @ proj_v1 @ v2_skew) @ e
        cos_term_x1 = (dv2dx1 @ proj_v2 @ v1_skew) @ e
        cos_term_x2 = (dv2dx2 @ proj_v2 @ v1_skew - dv1dx2 @ proj_v1 @ v2_skew) @ e
        cos_term_x3 = (-dv1dx3 @ proj_v1 @ v2_skew) @ e

        sin_term_x0 = dv1dx0 @ proj_v1 @ v2 + dv2dx0 @ proj_v2 @ v1
        sin_term_x1 = dv2dx1 @ proj_v2 @ v1
        sin_term_x2 = dv1dx2 @ proj_v1 @ v2 + dv2dx2 @ proj_v2 @ v1
        sin_term_x3 = dv1dx3 @ proj_v1 @ v2

        k = 1. / (v1_norm * v2_norm)

        dthetadx0 = k * (tm.cos(theta) * cos_term_x0 + tm.sin(theta) * sin_term_x0)
        dthetadx1 = k * (tm.cos(theta) * cos_term_x1 + tm.sin(theta) * sin_term_x1)
        dthetadx2 = k * (tm.cos(theta) * cos_term_x2 + tm.sin(theta) * sin_term_x2)
        dthetadx3 = k * (tm.cos(theta) * cos_term_x3 + tm.sin(theta) * sin_term_x3)

        return dthetadx0, dthetadx1, dthetadx2, dthetadx3
    
    @ti.func
    def getBendingForce(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, a1=1., a2=1., L=1., id=-1, d=1., tsa_mode=False):
        """
        计算折痕的弯曲力。
        Calculate the bending force of a crease.
        
        :param cs: 折痕起点坐标 / Crease start point coordinates
        :param ce: 折痕终点坐标 / Crease end point coordinates
        :param p1: 第一面板点坐标 / First panel point coordinates
        :param p2: 第二面板点坐标 / Second panel point coordinates
        :param k: 弯曲刚度系数 / Bending stiffness coefficient
        :param theta: 目标折叠角度 / Target folding angle
        :param crease_type: 折痕类型（山折/谷折）/ Crease type (mountain/valley)
        :param debug: 是否启用调试模式 / Whether to enable debug mode
        :param enable_dynamic_change: 是否启用动态变化 / Whether to enable dynamic change
        :param a1: 第一面板面积 / First panel area
        :param a2: 第二面板面积 / Second panel area
        :param L: 折痕长度 / Crease length
        :param id: 折痕ID / Crease ID
        :param d: 厚度参数 / Thickness parameter
        :param tsa_mode: 是否为TSA模式 / Whether in TSA mode
        :return: 四个顶点的力向量 (f_cs, f_ce, f_p1, f_p2) / Force vectors for four vertices
        """
        # 求折痕的信息
        barrier_left = tm.pi / 36.
        collision_indice = self.collision_indice

        folding_angle_upper_bound = tm.pi
        upper_barrier = folding_angle_upper_bound - barrier_left
        folding_angle_lower_bound = -tm.pi
        lower_barrier = folding_angle_lower_bound + barrier_left

        if id != -1:
            folding_angle_upper_bound = self.folding_angle_upper_bound[id]
            upper_barrier = folding_angle_upper_bound - barrier_left
            folding_angle_lower_bound = self.folding_angle_lower_bound[id]
            lower_barrier = folding_angle_lower_bound + barrier_left
            collision_indice = self.barrier_coeff_list[id]

        upper_barrier_energy_maximum = -collision_indice * barrier_left ** 2 * tm.log(self.collision_d / barrier_left) #positive
        upper_barrier_maximum = collision_indice * (2 * barrier_left * tm.log(self.collision_d / barrier_left) - barrier_left ** 2 / self.collision_d) #negative
        upper_barrier_df_maximum = 2 * collision_indice * (2 * barrier_left / self.collision_d + barrier_left ** 2 / (2 * (self.collision_d ** 2)) - tm.log(self.collision_d / barrier_left)) #positive

        lower_barrier_energy_maximum = upper_barrier_energy_maximum #positive
        lower_barrier_maximum = -upper_barrier_maximum #positive
        lower_barrier_df_maximum = upper_barrier_df_maximum #positive

        xc = ce - cs

        # xc_norm = xc.norm()

        energy = 0.0

        # 求单元法向量
        f11 = p1 - cs
        f22 = p2 - cs
        n1 = xc.cross(f11)
        n2 = f22.cross(xc)
        # self.n1[0] = n1
        # self.n2[0] = n2

        n1_norm = n1.norm()
        n2_norm = n2.norm()

        multi_n1_n2 = n1_norm * n2_norm

        dir = n1.cross(n2).dot(xc)

        val = n1.dot(n2)

        norm_val = val / multi_n1_n2

        current_theta = 0.0
        if norm_val >= 1.0:
            val = multi_n1_n2
            norm_val = 1.0
            current_theta = 0.0
        elif norm_val <= -1.0:
            val = -multi_n1_n2
            norm_val = -1.0
            current_theta = tm.pi
        else:
            current_theta = tm.acos(norm_val)
        
        # if id != -1:
        #     if crease_type == VALLEY:
        #         if (dir < 0 or (dir >= 0 and id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5)) and abs(current_theta) > 2. * self.beta and tsa_mode:
        #             # theta = 0.0
        #             self.enable_equivalent_torque[id] = False
        #         else:
        #             self.enable_equivalent_torque[id] = True
        #     else:
        #         if (dir >= 0 or (dir < 0 and id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5)) and abs(current_theta) > 2. * self.beta and tsa_mode:
        #             # theta = 0.0
        #             self.enable_equivalent_torque[id] = False
        #         else:
        #             self.enable_equivalent_torque[id] = True
        
        # if tsa_mode and id != -1 and not self.backup_enable_equivalent_torque[id]:
        #     theta = 0.0

        n_value = 0.
        backup_n_value = 0.
        signed_current_theta = 0.
       
        # 求折叠角
        if dir >= 0.: #mountain
            if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5: #180~270
                if crease_type == VALLEY:
                    self.crease_angle[id] = 1.
                else:
                    self.crease_angle[id] = -1.
                signed_current_theta = 2. * tm.pi - current_theta
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
                energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5
                self.barrier_k[id] = upper_barrier_df_maximum
            else: #-180~0
                if id != -1:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = -1.
                    else:
                        self.crease_angle[id] = 1.
                signed_current_theta = -current_theta        
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                t11 = signed_current_theta - lower_barrier
                t22 = folding_angle_lower_bound - self.collision_d - signed_current_theta
                if id != -1:
                    self.barrier_k[id] = 0.
                    self.previous_dir[id] = dir 
                if t11 <= 0 and t11 >= -barrier_left:
                    n_value += collision_indice * (2 * t11 * tm.log(t22 / -(barrier_left)) - t11 ** 2 / t22)
                    energy += -collision_indice * t11 ** 2 * tm.log(t22 / -(barrier_left))
                    if id != -1:
                        self.barrier_k[id] = collision_indice * 2. * (t11 ** 2 / (2. * t22 ** 2) + 2. * t11 / t22 - tm.log(t22 / -(barrier_left)))
                elif t11 < -barrier_left:
                    n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
                    energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
                    if id != -1:
                        self.barrier_k[id] = lower_barrier_df_maximum
        else:
            if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5: #-270~-180
                if crease_type == VALLEY:
                    self.crease_angle[id] = -1.
                else:
                    self.crease_angle[id] = 1.
                signed_current_theta = current_theta - 2. * tm.pi
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
                energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
                self.barrier_k[id] = lower_barrier_df_maximum
            else: #0~180
                if id != -1:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = 1.
                    else:
                        self.crease_angle[id] = -1.
                signed_current_theta = current_theta
                n_value = theta - signed_current_theta
                backup_n_value = n_value
                t11 = signed_current_theta - upper_barrier
                t22 = folding_angle_upper_bound + self.collision_d - signed_current_theta
                if id != -1:  
                    self.barrier_k[id] = 0.0
                    self.previous_dir[id] = dir 
                if t11 >= 0 and t11 <= barrier_left:
                    n_value += collision_indice * (2 * t11 * tm.log(t22 / barrier_left) - t11 ** 2 / t22)
                    energy += -collision_indice * t11 ** 2 * tm.log(t22 / barrier_left)
                    if id != -1: 
                        self.barrier_k[id] = 2. * collision_indice * (t11 ** 2 / (2. * t22 ** 2) + 2. * t11 / t22 - tm.log(t22 / barrier_left))
                elif t11 > barrier_left:
                    n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
                    energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5
                    if id != -1: 
                        self.barrier_k[id] = upper_barrier_df_maximum
        
        dqdx0, dqdx1, dqdx2, dqdx3 = self.getDthetaDx(cs, p2, ce, p1, signed_current_theta)
        
        # 求折叠角与目标之差
        if abs(current_theta) >= tm.pi * self.folding_angle_maximum_ratio - barrier_left:
            self.folding_angle_reach_pi[0] = True
            
        # 计算折痕等效弯曲系数
        k_crease = k * L

        #计算力
        force = (k_crease * backup_n_value + n_value - backup_n_value)
        
        csf = force * dqdx0
        rpf2 = force * dqdx1
        cef = force * dqdx2
        rpf1 = force * dqdx3

        #计算能量
        energy += 0.5 * k_crease * backup_n_value ** 2

        return csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, abs(signed_current_theta / tm.pi), dir

    @ti.func
    def get_position_with_index(self, index: int):
        """
        根据索引获取点的位置。
        Get the position of a point by index.
        
        :param index: 点的索引 / Point index
        :return: 点的位置向量 / Point position vector
        """
        return self.x[index]

    @ti.func
    def get_velocity_with_index(self, index: int):
        """
        根据索引获取点的速度。
        Get the velocity of a point by index.
        
        :param index: 点的索引 / Point index
        :return: 点的速度向量 / Point velocity vector
        """
        return self.v[index]
    
    # @ti.func
    # def getViscousity(self, posvel, other_posvel, i, j, viscosity):
    #     ret = tm.vec3([0., 0., 0.])
    #     if abs(self.connection_matrix[i, j] - self.ori_sim.spring_k) < 1e-5:
    #         direction = other_posvel - posvel
    #         ret = viscosity * direction
    #     return ret

    @ti.func
    def calculateCenterPoint3DWithUnitId(self, unit_kps, unit_id):
        """
        根据单元ID计算单元的三维中心点位置（基于当前位置）。
        Calculate the 3D center point of a unit by unit ID (based on current positions).
        
        :param unit_kps: 单元关键点索引列表 / Unit keypoint indices list
        :param unit_id: 单元ID / Unit ID
        :return: 单元中心点坐标 / Unit center point coordinates
        """
        center_accumulate = tm.vec3([0., 0., 0.])
        for i in ti.ndrange(self.unit_edge_max):
            if unit_kps[i] != -1:
                center_accumulate += self.get_position_with_index(unit_kps[i]) * self.unit_contributions[unit_id][i]
        return center_accumulate
    
    @ti.func
    def calculateCenterPoint3DWithVerticeUnitId(self, unit_kps, unit_id):
        """
        根据单元ID计算单元的三维中心点位置（基于原始顶点位置）。
        Calculate the 3D center point of a unit by unit ID (based on original vertex positions).
        
        :param unit_kps: 单元关键点索引列表 / Unit keypoint indices list
        :param unit_id: 单元ID / Unit ID
        :return: 单元中心点坐标 / Unit center point coordinates
        """
        center_accumulate = tm.vec3([0., 0., 0.])
        for i in ti.ndrange(self.unit_edge_max):
            if unit_kps[i] != -1:
                center_accumulate += self.original_vertices[unit_kps[i]] * self.unit_contributions[unit_id][i]
        return center_accumulate

    @ti.func
    def calculatedDs(self, x0, x1, x2, n, A):
        basic = 1. / A * (self.I3[None] - n.outer_product(n) / (A ** 2))
        dndx0 = basic @ self.getSkewMatrix(x2 - x1)
        self.dDs[0, 0][0, 2] = dndx0[0, 0]
        self.dDs[0, 0][1, 2] = dndx0[1, 0]
        self.dDs[0, 0][2, 2] = dndx0[2, 0]

        self.dDs[0, 1][0, 2] = dndx0[0, 1]
        self.dDs[0, 1][1, 2] = dndx0[1, 1]
        self.dDs[0, 1][2, 2] = dndx0[2, 1]
        
        self.dDs[0, 2][0, 2] = dndx0[0, 2]
        self.dDs[0, 2][1, 2] = dndx0[1, 2]
        self.dDs[0, 2][2, 2] = dndx0[2, 2]

        dndx1 = basic @ self.getSkewMatrix(x0 - x2)
        self.dDs[1, 0][0, 2] = dndx1[0, 0]
        self.dDs[1, 0][1, 2] = dndx1[1, 0]
        self.dDs[1, 0][2, 2] = dndx1[2, 0]

        self.dDs[1, 1][0, 2] = dndx1[0, 1]
        self.dDs[1, 1][1, 2] = dndx1[1, 1]
        self.dDs[1, 1][2, 2] = dndx1[2, 1]

        self.dDs[1, 2][0, 2] = dndx1[0, 2]
        self.dDs[1, 2][1, 2] = dndx1[1, 2]
        self.dDs[1, 2][2, 2] = dndx1[2, 2]

        dndx2 = basic @ self.getSkewMatrix(x1 - x0)
        self.dDs[2, 0][0, 2] = dndx2[0, 0]
        self.dDs[2, 0][1, 2] = dndx2[1, 0]
        self.dDs[2, 0][2, 2] = dndx2[2, 0]
        
        self.dDs[2, 1][0, 2] = dndx2[0, 1]
        self.dDs[2, 1][1, 2] = dndx2[1, 1]
        self.dDs[2, 1][2, 2] = dndx2[2, 1]
        
        self.dDs[2, 2][0, 2] = dndx2[0, 2]
        self.dDs[2, 2][1, 2] = dndx2[1, 2]
        self.dDs[2, 2][2, 2] = dndx2[2, 2]

        return dndx1, dndx2

    @ti.func
    def get_ddv_ddx(self, j, k, i):
        ddn1 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        ddn2 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        if j == 0:
            if k == 0:
                ddn1 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            elif k == 1:
                ddn1 = ti.Matrix.rows([[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
            else:
                ddn1 = ti.Matrix.rows([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
        elif j == 1:
            if k == 0:
                ddn1 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
            elif k == 1:
                ddn1 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]])
            else:
                ddn1 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]])
        elif j == 2:
            if k == 0:
                ddn1 = ti.Matrix.rows([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            elif k == 1:
                ddn1 = ti.Matrix.rows([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            else:
                ddn1 = ti.Matrix.rows([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
                ddn2 = ti.Matrix.rows([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

        return tm.vec3([ddn1[i, 0], ddn1[i, 1], ddn1[i, 2]]), tm.vec3([ddn2[i, 0], ddn2[i, 1], ddn2[i, 2]])
    
    @ti.func
    def fill_ddn_ddx(self, x0, x1, x2, v, v_norm):
        dvdx0 = self.getSkewMatrix(x2 - x1)
        dvdx1 = self.getSkewMatrix(x0 - x2)
        dvdx2 = self.getSkewMatrix(x1 - x0)

        self.total_dvdx_field[0] = [dvdx0[0, X], dvdx0[1, X], dvdx0[2, X]]
        self.total_dvdx_field[1] = [dvdx0[0, Y], dvdx0[1, Y], dvdx0[2, Y]]
        self.total_dvdx_field[2] = [dvdx0[0, Z], dvdx0[1, Z], dvdx0[2, Z]]
        self.total_dvdx_field[3] = [dvdx1[0, X], dvdx1[1, X], dvdx1[2, X]]
        self.total_dvdx_field[4] = [dvdx1[0, Y], dvdx1[1, Y], dvdx1[2, Y]]
        self.total_dvdx_field[5] = [dvdx1[0, Z], dvdx1[1, Z], dvdx1[2, Z]]
        self.total_dvdx_field[6] = [dvdx2[0, X], dvdx2[1, X], dvdx2[2, X]]
        self.total_dvdx_field[7] = [dvdx2[0, Y], dvdx2[1, Y], dvdx2[2, Y]]
        self.total_dvdx_field[8] = [dvdx2[0, Z], dvdx2[1, Z], dvdx2[2, Z]]

        for i, j, k in ti.ndrange(3, 3, 3):
            index = 3 * i + j
            a_n1 = v.dot(self.total_dvdx_field[3 + k]) / v_norm ** 3
            a_n2 = v.dot(self.total_dvdx_field[6 + k]) / v_norm ** 3
            ddv1, ddv2 = self.get_ddv_ddx(i, j, k)
            da_n1dx = self.total_dvdx_field[index].dot(self.total_dvdx_field[3 + k]) / v_norm ** 3 + v.dot(ddv1) / v_norm ** 3 - 3. * v.dot(self.total_dvdx_field[3 + k]) / v_norm ** 5 * (self.total_dvdx_field[index].dot(v))
            da_n2dx = self.total_dvdx_field[index].dot(self.total_dvdx_field[6 + k]) / v_norm ** 3 + v.dot(ddv2) / v_norm ** 3 - 3. * v.dot(self.total_dvdx_field[6 + k]) / v_norm ** 5 * (self.total_dvdx_field[index].dot(v))
            ddn1_ddx = -1. / v_norm ** 3 * (self.total_dvdx_field[index].dot(v)) * self.total_dvdx_field[3 + k] + 1. / v_norm * ddv1 - a_n1 * self.total_dvdx_field[index] - da_n1dx * v
            ddn2_ddx = -1. / v_norm ** 3 * (self.total_dvdx_field[index].dot(v)) * self.total_dvdx_field[6 + k] + 1. / v_norm * ddv2 - a_n2 * self.total_dvdx_field[index] - da_n2dx * v
            self.ddn1_ddx[i, j][0, k] = ddn1_ddx[0]
            self.ddn1_ddx[i, j][1, k] = ddn1_ddx[1]
            self.ddn1_ddx[i, j][2, k] = ddn1_ddx[2]
            self.ddn2_ddx[i, j][0, k] = ddn2_ddx[0]
            self.ddn2_ddx[i, j][1, k] = ddn2_ddx[1]
            self.ddn2_ddx[i, j][2, k] = ddn2_ddx[2]


    @ti.func
    def calculateKpNumWithUnitId(self, unit_kps):
        """
        计算单元的有效关键点数量。
        Calculate the number of valid keypoints for a unit.
        
        :param unit_kps: 单元关键点索引列表 / Unit keypoint indices list
        :return: 有效关键点数量 / Number of valid keypoints
        """
        kp_len = 0
        for i in ti.ndrange(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        return kp_len
    
    @ti.func
    def calculateNormalVectorWithUnitId(self, unit_kps):
        """
        计算单元的法向量（基于当前位置）。
        Calculate the normal vector of a unit (based on current positions).
        
        :param unit_kps: 单元关键点索引列表 / Unit keypoint indices list
        :return: 单位法向量 / Unit normal vector
        """
        n = tm.vec3([0.0, 0.0, 0.0])
        kp_len = 0
        for i in ti.ndrange(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        for i in ti.ndrange(kp_len):
            v1 = self.get_position_with_index(unit_kps[(i + 1) % kp_len]) - self.get_position_with_index(unit_kps[i])
            v2 = self.get_position_with_index(unit_kps[(i - 1 + kp_len) % kp_len]) - self.get_position_with_index(unit_kps[i])
            c_n = tm.normalize(v1.cross(v2))
            area = ti.Matrix.cols([v1, v2, c_n]).determinant()
            n += area * c_n
            # n += v1.cross(v2)
        return tm.normalize(n)

    @ti.kernel
    def calculateExtendedVector(self, unit_id: int, length: data_type) -> ti.math.vec3:
        nm = self.calculateNormalVectorWithUnitId(self.unit_indices[unit_id])
        return nm * length
    
    @ti.func
    def coplanar(self, v1_start, v1_end, v2_start, v2_end, id):
        """
        判断两条线段是否共面。
        Check if two line segments are coplanar.
        
        :param v1_start: 第一条线段起点 / First line segment start point
        :param v1_end: 第一条线段终点 / First line segment end point
        :param v2_start: 第二条线段起点 / Second line segment start point
        :param v2_end: 第二条线段终点 / Second line segment end point
        :param id: 线段ID（用于调试）/ Line ID (for debugging)
        :return: 是否共面 / Whether coplanar
        """
        value = False
        v1 = v1_end - v1_start
        v2 = v2_end - v2_start
        v1_start_to_v2_end = v2_end - v1_start
        n = tm.cross(v1, v1_start_to_v2_end)
        n /= n.norm(1e-6)
        v2 /= v2.norm(1e-6)
        if abs(tm.dot(n, v2)) < 1e-2:
            value = True
        return value

    @ti.func
    def rapidRepel(self, v1_start, v1_end, v2_start, v2_end):
        """
        快速排斥检测，检查两条线段的包围盒是否相交。
        Rapid rejection test, check if bounding boxes of two line segments intersect.
        
        :param v1_start: 第一条线段起点 / First line segment start point
        :param v1_end: 第一条线段终点 / First line segment end point
        :param v2_start: 第二条线段起点 / Second line segment start point
        :param v2_end: 第二条线段终点 / Second line segment end point
        :return: 包围盒是否相交 / Whether bounding boxes intersect
        """
        value = False
        if (    max(v1_start[0], v1_end[0]) >= min(v2_start[0], v2_end[0])
            and min(v1_start[0], v1_end[0]) <= max(v2_start[0], v2_end[0])
            and max(v1_start[1], v1_end[1]) >= min(v2_start[1], v2_end[1])
            and min(v1_start[1], v1_end[1]) <= max(v2_start[1], v2_end[1])
            and max(v1_start[2], v1_end[2]) >= min(v2_start[2], v2_end[2])
            and min(v1_start[2], v1_end[2]) <= max(v2_start[2], v2_end[2])
        ):
            value = True

        return value

    @ti.func
    def straddle(self, v1_start, v1_end, v2_start, v2_end):
        """
        跨立检测，判断一条线段是否跨立另一条线段所在平面。
        Straddle test, check if one line segment straddles the plane of another.
        
        :param v1_start: 第一条线段起点 / First line segment start point
        :param v1_end: 第一条线段终点 / First line segment end point
        :param v2_start: 第二条线段起点 / Second line segment start point
        :param v2_end: 第二条线段终点 / Second line segment end point
        :return: 是否跨立 / Whether straddling
        """
        value = False
        V1 = v1_end - v1_start
        V2 = v2_end - v2_start
        tempV1 = v2_end - v1_start
        tempV2 = v2_start - v1_start
        N1 = tm.cross(tempV1, V1)
        N2 = tm.cross(V1, tempV2)
        res1 = tm.dot(N1, N2)

        tempV3 = v1_end - v2_start
        tempV4 = v1_start - v2_start
        N3 = tm.cross(tempV3, V2)
        N4 = tm.cross(V2, tempV4)
        res2 = tm.dot(N3, N4)

        if res1 > 0 and res2 > 0:
            value = True
        return value

    @ti.func
    def intersection3D(self, v1_start, v1_end, v2_start, v2_end, id):
        """
        检测三维空间中两条线段是否相交。
        Detect if two line segments intersect in 3D space.
        
        :param v1_start: 第一条线段起点 / First line segment start point
        :param v1_end: 第一条线段终点 / First line segment end point
        :param v2_start: 第二条线段起点 / Second line segment start point
        :param v2_end: 第二条线段终点 / Second line segment end point
        :param id: 线段ID / Line ID
        :return: 是否相交 / Whether intersecting
        """
        value = False
        if self.coplanar(v1_start, v1_end, v2_start, v2_end, id):
            if self.rapidRepel(v1_start, v1_end, v2_start, v2_end):
                if self.straddle(v1_start, v1_end, v2_start, v2_end):
                    value = True
        return value
    
    @ti.func
    def intersection3D_panel_segment(self, x1, x2, x3, p1, p2, epsilon=0.05):
        """
        检测线段是否与三角形面片相交（Möller–Trumbore算法）。
        Detect if a line segment intersects with a triangle face (Möller–Trumbore algorithm).
        
        :param x1: 三角形顶点1 / Triangle vertex 1
        :param x2: 三角形顶点2 / Triangle vertex 2
        :param x3: 三角形顶点3 / Triangle vertex 3
        :param p1: 线段起点 / Line segment start point
        :param p2: 线段终点 / Line segment end point
        :param epsilon: 数值容差 / Numerical tolerance
        :return: 是否相交 / Whether intersecting
        """ 
        ret = True
        # 提取点坐标
        D = p1
        E = p2
        A = x1
        B = x2
        C = x3
        
        # 1. 计算三角形边向量与线段方向向量
        edge1 = B - A
        edge2 = C - A
        direction = E - D  # 线段方向向量（避免与Python内置函数dir冲突）
        orig = D           # 线段起点（作为射线原点）
        
        # 2. 计算行列式（判断是否与平面平行/共面）
        pvec = tm.cross(direction, edge2)
        det = tm.dot(edge1, pvec)

        u = v = t = 0.
        
        # 若行列式接近0，说明线段与三角形平面平行或共面：直接不相交
        if abs(det) < epsilon:
            ret = False
        
        else:
            inv_det = 1.0 / det
            tvec = orig - A  # 线段起点到三角形顶点A的向量
            
            # 3. 计算重心坐标 u：严格在 (0, 1) 内（排除三角形AB/AC边）
            u = tm.dot(tvec, pvec) * inv_det
            if u <= epsilon or u >= 1.0 - epsilon:
                ret = False
            else:
                # 4. 计算重心坐标 v：严格大于0且u+v严格小于1（排除三角形AB/BC边和顶点）
                qvec = tm.cross(tvec, edge1)
                v = tm.dot(direction, qvec) * inv_det
                if v <= epsilon or (u + v) >= 1.0 - epsilon:
                    ret = False
                else:
                    # 5. 计算参数 t：严格在 (0, 1) 内（排除线段的两个端点）
                    t = tm.dot(edge2, qvec) * inv_det
                    if t <= 0.0 or t >= 1.0:
                        ret = False
        
        # 所有条件满足：线段严格穿过三角形内部
        # if ret:
        #     print(u, v, t)
        return ret

    @ti.func
    def calculateUnitCenter(self):
        """
        计算所有单元的中心点位置。
        Calculate the center positions of all units.
        """
        for unit_id in ti.ndrange(self.unit_indices_num):
            self.unit_center[unit_id] = self.calculateCenterPoint3DWithUnitId(self.unit_indices[unit_id], unit_id)    
            # print(f"{unit_id}, {self.unit_center[unit_id]}") 
    
    @ti.func
    def intersect4p(self, i1, c1, c2, fd, dir):
        """
        计算点与折痕的相交参数（用于面片碰撞检测）。
        Calculate intersection parameter between a point and a crease (for panel collision detection).
        
        :param i1: 检测点坐标 / Detection point coordinates
        :param c1: 折痕起点 / Crease start point
        :param c2: 折痕终点 / Crease end point
        :param fd: 力方向 / Force direction
        :param dir: 方向系数 / Direction coefficient
        :return: 相交参数t，-1表示不相交 / Intersection parameter t, -1 means no intersection
        """
        t = -1.0
        fd = dir * fd
        nm = tm.normalize(tm.cross(c1 - i1, c2 - i1))
        
        fd_nm = fd.dot(nm) * nm
        
        ffd = fd - fd_nm

        # preminary criteria
        o1_t = i1 + ffd / ffd.norm(1e-6) * self.max_size
        flag1_t = (i1 - c1).cross(o1_t - c1)
        flag2_t = (i1 - c2).cross(o1_t - c2)
        flag3_t = (i1 - c2).cross(i1 - c1)
        flag4_t = (o1_t - c2).cross(o1_t - c1)
        if flag1_t.dot(flag2_t) < 0 and flag3_t.dot(flag4_t) < 0:
            crease_dir = tm.normalize(c2 - c1)
            v_c2 = c2 - i1
            proj = v_c2.dot(crease_dir) * crease_dir
            v_vertical = v_c2 - proj

            diff = ffd - v_vertical
            coeff = v_vertical.norm() / (v_vertical.norm() + fd_nm.norm())
            real_facet_force_dir_norm = v_vertical + coeff * diff

            facet_force_dir_norm = real_facet_force_dir_norm / real_facet_force_dir_norm.norm()
            o1 = i1 + facet_force_dir_norm * self.max_size
            flag1 = (i1 - c1).cross(o1 - c1)
            flag2 = (i1 - c2).cross(o1 - c2)
            flag3 = (i1 - c2).cross(i1 - c1)
            flag4 = (o1 - c2).cross(o1 - c1)
            # facet_force_dir_norm = ffd / ffd.norm()
            if flag1.dot(flag2) < 0 and flag3.dot(flag4) < 0:
                v1 = c2 - i1
                v2 = c1 - i1
                vertical1 = (v1 - v1.dot(facet_force_dir_norm) * facet_force_dir_norm)
                vertical2 = (v2 - v2.dot(facet_force_dir_norm) * facet_force_dir_norm)
                if vertical1.dot(vertical2) < 0:
                    # careful
                    out_o1 = i1 + fd
                    crease_dir = tm.normalize(c2 - c1)
                    v_c2_i1 = c2 - i1
                    v_c2_o1 = c2 - out_o1

                    v_vertical_i1 = v_c2_i1 - v_c2_i1.dot(crease_dir) * crease_dir
                    v_vertical_o1 = v_c2_o1 - v_c2_o1.dot(crease_dir) * crease_dir

                    diff = fd + v_vertical_o1 - v_vertical_i1
                    coeff = v_vertical_i1.norm() / (v_vertical_i1.norm() + v_vertical_o1.norm())
                    real_facet_force_dir_norm = v_vertical_i1 + coeff * diff

                    facet_force_dir_norm = real_facet_force_dir_norm / real_facet_force_dir_norm.norm()
                    o1 = i1 + facet_force_dir_norm * self.max_size
                    flag1 = (i1 - c1).cross(o1 - c1)
                    flag2 = (i1 - c2).cross(o1 - c2)
                    flag3 = (i1 - c2).cross(i1 - c1)
                    flag4 = (o1 - c2).cross(o1 - c1)
                    # facet_force_dir_norm = ffd / ffd.norm()
                    if flag1.dot(flag2) < 0 and flag3.dot(flag4) < 0:
                        v1 = c2 - i1
                        v2 = c1 - i1
                        vertical1 = (v1 - v1.dot(facet_force_dir_norm) * facet_force_dir_norm)
                        vertical2 = (v2 - v2.dot(facet_force_dir_norm) * facet_force_dir_norm)
                        if vertical1.dot(vertical2) < 0:
                            t = vertical2.norm() / (vertical1.norm() + vertical2.norm())
                    # t = vertical2.norm() / (vertical1.norm() + vertical2.norm())
        return t

    @ti.func
    def calculateOtherPanelIntersection(self, s, e, kp_id, before_kp_id, kp_num, before_kp_num):
        """
        计算线段与其他面片的相交情况。
        Calculate intersections between a line segment and other panels.
        
        :param s: 线段起点 / Line segment start point
        :param e: 线段终点 / Line segment end point
        :param kp_id: 当前关键点ID列表 / Current keypoint ID list
        :param before_kp_id: 之前的关键点ID列表 / Previous keypoint ID list
        :param kp_num: 当前关键点数量 / Current keypoint number
        :param before_kp_num: 之前的关键点数量 / Previous keypoint number
        :return: 相交数量 / Number of intersections
        """
        intersect_num = 0

        for i in ti.ndrange(self.div_indices_num - 2 * self.connection_number[0]):
            i0 = self.indices[3 * i]
            i1 = self.indices[3 * i + 1]
            i2 = self.indices[3 * i + 2]

            self_panel = False
            for j in ti.ndrange(kp_num):
                if kp_id[j] == i0:
                    self_panel = True
                    break

            if self_panel:
                self_panel = False
                for j in ti.ndrange(kp_num):
                    if kp_id[j] == i1:
                        self_panel = True
                        break

                if self_panel:
                    self_panel = False
                    for j in ti.ndrange(kp_num):
                        if kp_id[j] == i2:
                            self_panel = True
                            break
            
            before_self_panel = False
            for j in ti.ndrange(before_kp_num):
                if before_kp_id[j] == i0:
                    before_self_panel = True
                    break

            if before_self_panel:
                before_self_panel = False
                for j in ti.ndrange(before_kp_num):
                    if before_kp_id[j] == i1:
                        before_self_panel = True
                        break

                if before_self_panel:
                    before_self_panel = False
                    for j in ti.ndrange(before_kp_num):
                        if before_kp_id[j] == i2:
                            before_self_panel = True
                            break

            if (not self_panel) and (not before_self_panel):
                x0 = self.get_position_with_index(i0)
                x1 = self.get_position_with_index(i1)
                x2 = self.get_position_with_index(i2)

                intersect_with_others = self.intersection3D_panel_segment(x0, x1, x2, s, e, 0.05)
                if intersect_with_others:
                    baseline_height = 0.0
                    if self.system_type[0] == 1 or self.system_type[0] == 3:
                        baseline_height = self.ground_barrier * 5
                    else:
                        baseline_height = self.origami_z_bias + self.ground_barrier * 5
                    if (x0[Z] + x1[Z] + x2[Z]) * 0.333 > baseline_height:
                        intersect_num += 1
                        # print(s, e, i0, i1, i2, kp_num)
                        # print(kp_id, before_kp_id)
                        # self_panel = False
                        # for j in ti.ndrange(kp_num):
                        #     if kp_id[j] == i0:
                        #         self_panel = True
                        #         break

                        # if self_panel:
                        #     self_panel = False
                        #     for j in ti.ndrange(kp_num):
                        #         print(kp_id[j], i1)
                        #         if kp_id[j] == i1:
                        #             self_panel = True
                        #             break

                        #     if self_panel:
                        #         self_panel = False
                        #         for j in ti.ndrange(kp_num):
                        #             if kp_id[j] == i2:
                        #                 self_panel = True
                        #                 break

        return intersect_num

    @ti.func
    def calculateOtherIntersection(self, kp_id, unit_kp_num, s, e, additional_kp_id, additional_kp_num):
        """
        计算线段与其他折痕的相交情况。
        Calculate intersections between a line segment and other creases.
        
        :param kp_id: 当前单元关键点ID列表 / Current unit keypoint ID list
        :param unit_kp_num: 当前单元关键点数量 / Current unit keypoint number
        :param s: 线段起点 / Line segment start point
        :param e: 线段终点 / Line segment end point
        :param additional_kp_id: 附加关键点ID列表 / Additional keypoint ID list
        :param additional_kp_num: 附加关键点数量 / Additional keypoint number
        :return: 相交数量 / Number of intersections
        """
        intersect_with_others_number = 0
        del_number = 4 * self.connection_number[0]
        for ii in ti.ndrange(self.line_total_indice_num - del_number):
            id1 = self.line_pairs[ii, 0]
            id2 = self.line_pairs[ii, 1]
            if id1 > id2:
                id1 = self.line_pairs[ii, 1]
                id2 = self.line_pairs[ii, 0]
            self_unit = False
            for l in ti.ndrange(unit_kp_num):
                cur_x_id = kp_id[l]
                next_x_id = kp_id[(l + 1) % unit_kp_num]
                if cur_x_id > next_x_id:
                    cur_x_id = kp_id[(l + 1) % unit_kp_num]
                    next_x_id = kp_id[l]
                if (id1 == cur_x_id and id2 == next_x_id):
                    self_unit = True
                    break
            for l in ti.ndrange(additional_kp_num):
                cur_x_id = additional_kp_id[l]
                next_x_id = additional_kp_id[(l + 1) % additional_kp_num]
                if cur_x_id > next_x_id:
                    cur_x_id = additional_kp_id[(l + 1) % additional_kp_num]
                    next_x_id = additional_kp_id[l]
                if (id1 == cur_x_id and id2 == next_x_id):
                    self_unit = True
                    break
            intersect_with_others = False
            if not self_unit:
                intersect_with_others = self.intersection3D(s, e, self.x[id1], self.x[id2], ii)
            if intersect_with_others:
                baseline_height = 0.0
                if self.system_type[0] == 1 or self.system_type[0] == 3:
                    baseline_height = self.ground_barrier * 5
                else:
                    baseline_height = self.origami_z_bias + self.ground_barrier * 5
                if (self.x[id1][Z] + self.x[id2][Z]) * 0.5 > baseline_height:
                    intersect_with_others_number += 1
        return intersect_with_others_number
    
    @ti.func
    def calculateEquivalentArm(self, angle):
        """
        计算等效力臂距离（用于绳索驱动扭矩计算）。
        Calculate equivalent arm distance (for string-driven torque calculation).
        
        :param angle: 绳索角度 / String angle
        :return: 等效力臂距离 / Equivalent arm distance
        """
        equivalent_arm_dis = 0.
        if not self.fillet_hole:
            equivalent_arm_dis = ((self.h_hole + self.string_thickness) * tm.cos(angle) - (self.d_hole - self.string_thickness) * tm.sin(angle)) * 0.5
        else:
            equivalent_arm_dis = ((self.h_hole + self.string_thickness) - (self.d_hole) * tm.sin(angle)) * 0.5
        return equivalent_arm_dis

    @ti.func
    def calculateInitialTorque_FieldForce(self, angle, kp_id, center, force_dir, force, start_index, step):
        """
        计算初始扭矩和场力（用于绳索驱动）。
        Calculate initial torque and field force (for string actuation).
        
        :param angle: 绳索角度 / String angle
        :param kp_id: 关键点ID列表 / Keypoint ID list
        :param center: 中心点坐标 / Center point coordinates
        :param force_dir: 力的方向 / Force direction
        :param force: 力的大小 / Force magnitude
        :param start_index: 起始索引 / Start index
        :param step: 步长 / Step size
        """
        if angle < self.beta:
            equivalent_arm_dis = self.calculateEquivalentArm(angle)
            moment = force * equivalent_arm_dis
            nm = self.calculateNormalVectorWithUnitId(kp_id)
            area = 0.
        
            for i in ti.ndrange((start_index, start_index + step)):
                i0 = self.indices[3 * i]
                i1 = self.indices[3 * i + 1]
                i2 = self.indices[3 * i + 2]

                x1 = self.get_position_with_index(i0)
                x2 = self.get_position_with_index(i1)
                x3 = self.get_position_with_index(i2)

                sub_area = ti.Matrix.cols([x2 - x1, x3 - x1, tm.normalize((x2 - x1).cross(x3 - x1))]).determinant()
                if sub_area > 1:
                    area += sub_area

            # for i, j, k in ti.ndrange(kp_num - 2, kp_num - 1, kp_num):
            #     if i < j and j < k:
            #         x1 = self.get_position_with_index(kp_id[i])
            #         x2 = self.get_position_with_index(kp_id[j])
            #         x3 = self.get_position_with_index(kp_id[k])
            #         sub_area = ti.Matrix.cols([x2 - x1, x3 - x1, tm.normalize((x2 - x1).cross(x3 - x1))]).determinant()
            #         if sub_area > 1:
            #             area += sub_area

            # for i, j, k in ti.ndrange(kp_num - 2, kp_num - 1, kp_num):
            #     if i < j and j < k:
            #         x1 = self.get_position_with_index(kp_id[i])
            #         x2 = self.get_position_with_index(kp_id[j])
            #         x3 = self.get_position_with_index(kp_id[k])

            for i in ti.ndrange((start_index, start_index + step)):
                i0 = self.indices[3 * i]
                i1 = self.indices[3 * i + 1]
                i2 = self.indices[3 * i + 2]

                x1 = self.get_position_with_index(i0)
                x2 = self.get_position_with_index(i1)
                x3 = self.get_position_with_index(i2)

                v1 = (x1 - center)
                v2 = (x2 - center)
                v3 = (x3 - center)
                a = ti.Matrix.cols([x2 - x1, x3 - x1, tm.normalize((x2 - x1).cross(x3 - x1))]).determinant()
                if a > 1:
                    norm_force_dir = force_dir / force_dir.norm()
                    arm1 = v1.dot(norm_force_dir)
                    arm1_vertical = (v1 - arm1 * norm_force_dir).norm() * tm.sign(((v1 - arm1 * norm_force_dir).cross(norm_force_dir)).dot(nm))
                    arm2 = v2.dot(norm_force_dir)
                    arm2_vertical = (v2 - arm2 * norm_force_dir).norm() * tm.sign(((v2 - arm2 * norm_force_dir).cross(norm_force_dir)).dot(nm))
                    arm3 = v3.dot(norm_force_dir)
                    arm3_vertical = (v3 - arm3 * norm_force_dir).norm() * tm.sign(((v3 - arm3 * norm_force_dir).cross(norm_force_dir)).dot(nm))
                    m = moment * a / area
                    b = tm.vec3([m, 0., 0.])
                    A = ti.Matrix.rows([[arm1, arm2, arm3], [arm1_vertical, arm2_vertical, arm3_vertical], [1., 1., 1.]])
                    A_inv = A.inverse()
                    res = A_inv @ b
                    self.field_force[i0] += res[0] * nm
                    self.field_force[i1] += res[1] * nm
                    self.field_force[i2] += res[2] * nm
                    # ti.atomic_add(self.field_force[i0], res[0] * nm)
                    # ti.atomic_add(self.field_force[i1], res[1] * nm)
                    # ti.atomic_add(self.field_force[i2], res[2] * nm)

    @ti.kernel
    def calculateDiscount(self, mode: int):
        """
        计算绳索摩擦力折扣系数并施加场力。
        Calculate string friction discount coefficient and apply field force.
        
        :param mode: 模式标志（0或1）/ Mode flag (0 or 1)
        """
        self.field_force.fill(0.)
        for i in ti.ndrange(self.constraint_number):
            basic_factor = 1.0
            current_force = self.string_force_each[i]
            sum_coeff = 0.0
            initial_length = 0.0
            
            if self.equivalent_internal_point_id[i] >= 0:
                start_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.equivalent_internal_point_id[i]], self.equivalent_internal_point_id[i])
                
                previous_force = self.backup_string_force_each[i] * self.string_force_current_discount[i, 0]

                before_kp_id = self.unit_indices[self.equivalent_internal_point_id[i]]
                kp_id = self.unit_indices[self.unit_control[i, 0]]
                    
                hole_direction = self.hole_dir[i, 0]
                before_direction = -hole_direction
                
                before_nm = self.calculateNormalVectorWithUnitId(before_kp_id)
                nm = self.calculateNormalVectorWithUnitId(kp_id)
                
                previous_angle = tm.pi * 0.5
                current_angle = 0.0

                before_force_dir = before_nm * hole_direction
                    
                end_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, 0]], self.unit_control[i, 0])

                if self.intersection_flag[i, 0]: # 当前绳和当前板有穿透
                    self.intersection_penalty[0] += 1
                    end_point = self.intersection_points[i * self.max_control_length]
                if self.intersection_flag2_initial[i]:
                    self.intersection_penalty[0] += 1
                    end_point = self.intersection_points2_initial[i]

                force_dir = end_point - start_point

                end_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.equivalent_internal_point_id[i]], self.equivalent_internal_point_id[i])
                if self.intersection_flag2_initial[i]:
                    end_point = self.intersection_points2_initial[i]
                if self.intersection_flag[i, 0]: # 当前绳和当前板有穿透
                    end_point = self.intersection_points[i * self.max_control_length]
                
                next_force_dir = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, 0]], self.unit_control[i, 0]) - end_point
                    
                direction = before_force_dir.dot(before_nm) * before_direction
                direction2 = force_dir.dot(before_nm) * before_direction
                direction3 = next_force_dir.dot(nm) * hole_direction

                if mode and (direction > 2. or direction2 > 2. or direction3 > 2.):
                    self.error_status[0] = True
                    # print(f"{0}, {direction}, {direction2}, {direction3}")

                angle = tm.pi

                lower_bound = self.gamma_bound_180_degree[0]
                upper_bound = self.gamma_bound_180_degree[1]
                        
                discount_factor = lower_bound + tm.clamp(current_force / self.maximum_tension, 0., 1.) * (upper_bound - lower_bound)       
                #后面绳与前面板夹角
                current_angle = tm.pi * 0.5 - tm.acos(abs(tm.clamp(-force_dir.dot(before_nm) * before_direction / force_dir.norm(1e-6), -1.0, 1.0)))
                if force_dir.norm() < self.origami_thickness:
                    current_angle = tm.pi * 0.5
                #前面绳与前面板夹角
                previous_angle = tm.pi * 0.5
                
                base = self.beta

                dif = abs(current_angle - previous_angle) * 0.5 / base
                discount_factor = discount_factor ** dif

                basic_factor *= discount_factor
                basic_factor -= self.static_friction
                if basic_factor < 0.:
                    basic_factor = 0.

                current_force = self.string_force_each[i] * basic_factor
                
                # print(f"{i}, 0")
                start_tri_index = 0
                for kk in ti.ndrange(self.equivalent_internal_point_id[i]):
                    start_tri_index += self.unit_kp_num_list[kk] - 2
                self.calculateInitialTorque_FieldForce(current_angle, before_kp_id, start_point, force_dir - force_dir.dot(before_nm) * before_nm, -current_force * hole_direction, start_tri_index, self.unit_kp_num_list[self.equivalent_internal_point_id[i]] - 2)

                final_basic_factor = basic_factor

                # if self.delta_length_per_string[i, 0] >= 0:
                # #     if previous_force > self.string_force_each[i] * basic_factor and previous_force <= self.string_force_each[i]:
                # #         basic_factor = previous_force / self.string_force_each[i]
                # #         current_force = self.string_force_each[i] * basic_factor
                # #         final_basic_factor = basic_factor
                # #     elif previous_force > self.string_force_each[i]:
                # #         basic_factor = 1.
                # #         current_force = self.string_force_each[i]
                # #         final_basic_factor = 1.
                # # else:
                #     if previous_force > self.string_force_each[i] * basic_factor and previous_force <= self.string_force_each[i] / basic_factor:
                #         basic_factor = previous_force / self.string_force_each[i]
                #         current_force = self.string_force_each[i] * basic_factor
                #         final_basic_factor = basic_factor
                #     elif previous_force > self.string_force_each[i] / basic_factor:
                #         basic_factor = 1. / basic_factor
                #         current_force = self.string_force_each[i] * basic_factor
                #         final_basic_factor = basic_factor
                
                self.string_force_current_discount[i, 0] = final_basic_factor

                # print(i, 0, final_basic_factor)
                
                for j in ti.ndrange(self.bending_pairs_num):
                    if (self.equivalent_torque_influence_id[i, j]) & 1:
                        self.equivalent_torque_index[i, j, 1] = final_basic_factor * self.equivalent_torque_coeff[i, j, 1]
                
                sum_coeff += final_basic_factor

                initial_length += self.initial_length_per_string[i, 0] * self.string_force_current_discount[i, 0]

                if self.unit_control[i, 1] == -1:
                    final_angle = tm.pi * 0.5 - tm.acos(abs(tm.clamp(-next_force_dir.dot(nm) * hole_direction / next_force_dir.norm(1e-6), -1.0, 1.0)))
                    if next_force_dir.norm() < self.origami_thickness:
                        final_angle = tm.pi * 0.5

                    start_tri_index = 0
                    for kk in ti.ndrange(self.unit_control[i, 0]):
                        start_tri_index += self.unit_kp_num_list[kk] - 2
                    self.calculateInitialTorque_FieldForce(final_angle, kp_id, self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, 0]], self.unit_control[i, 0]), \
                                                            -next_force_dir + next_force_dir.dot(nm) * nm, -current_force * hole_direction, start_tri_index, self.unit_kp_num_list[self.unit_control[i, 0]] - 2)
                
                # if i == 0:
                #     print(i, 0, self.string_force_current_discount[i, 0])
            
            delta_length = self.constraint_length[i] - (self.constraint_initial_length[i] - self.string_length_decrease[i])
            
            for k in ti.ndrange(self.max_control_length):
                if self.unit_control[i, k] != -1 and k > 0:
                    start_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k - 1]], self.unit_control[i, k - 1])

                    previous_force = self.backup_string_force_each[i] * self.string_force_current_discount[i, k]

                    before_kp_id = self.unit_indices[self.unit_control[i, k - 1]]
                    kp_id = self.unit_indices[self.unit_control[i, k]]
                    
                    before_direction = self.hole_dir[i, k - 1]
                    hole_direction = self.hole_dir[i, k]
                    
                    before_nm = self.calculateNormalVectorWithUnitId(before_kp_id)
                    nm = self.calculateNormalVectorWithUnitId(kp_id)
                    
                    previous_angle = 0.0
                    current_angle = 0.0
                    
                    # before
                    previous_point = self.constraint_start_point[i]
                    if k > 1:
                        previous_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k - 2]], self.unit_control[i, k - 2])
                        
                    if self.intersection_flag[i, k - 1]: # 前面板和前面绳有穿透
                        previous_point = self.intersection_points[k - 1 + i * self.max_control_length]

                    if k > 1:
                        if self.intersection_flag2[i, k - 2]:
                            previous_point = self.intersection_points2[k - 2 + i * self.max_control_length]
                    else:
                        if self.intersection_flag2_initial[i]:
                            previous_point = self.intersection_points2_initial[i]
                    
                    before_force_dir = start_point - previous_point

                    # cur
                    end_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k]], self.unit_control[i, k])
                    if self.intersection_flag[i, k]: # 当前绳和当前板有穿透
                        self.intersection_penalty[0] += 1
                        end_point = self.intersection_points[k + i * self.max_control_length]
                    if self.intersection_flag2[i, k - 1]: # 当前绳和前面板有穿透
                        self.intersection_penalty[0] += 1
                        end_point = self.intersection_points2[k - 1 + i * self.max_control_length]

                    force_dir = end_point - start_point

                    # next
                    end_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k - 1]], self.unit_control[i, k - 1])
                    if self.intersection_flag2[i, k - 1]: # 当前绳和前面板有穿透
                        end_point = self.intersection_points2[k - 1 + i * self.max_control_length]
                    if self.intersection_flag[i, k]: # 当前绳和当前板有穿透
                        end_point = self.intersection_points[k + i * self.max_control_length]

                    next_force_dir = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k]], self.unit_control[i, k]) - end_point 
                    
                    direction = before_force_dir.dot(before_nm) * before_direction
                    direction2 = force_dir.dot(before_nm) * before_direction
                    direction3 = next_force_dir.dot(nm) * hole_direction

                    if mode and (direction > 2. or direction2 > 2. or direction3 > 2.):
                        self.error_status[0] = True
                        # print(f"{k}, {direction}, {direction2}, {direction3}")
                    
                    in_plane_before_force_dir = before_force_dir - before_force_dir.dot(before_nm) * before_nm
                    in_plane_force_dir = force_dir - force_dir.dot(before_nm) * before_nm

                    angle = tm.acos(tm.clamp(-in_plane_before_force_dir.dot(in_plane_force_dir) / (in_plane_before_force_dir.norm(1e-6) * in_plane_force_dir.norm(1e-6)), -1.0, 1.0))
                    discount_factor = 1.0
                    # print(i, k, angle)
                    
                    lower_bound = 0.0
                    upper_bound = 0.0
                    if angle < np.pi * 0.25:
                        lower_bound = self.gamma_bound_0_degree[0] + (angle / (np.pi * 0.25)) * (self.gamma_bound_45_degree[0] - self.gamma_bound_0_degree[0])
                        upper_bound = self.gamma_bound_0_degree[1] + (angle / (np.pi * 0.25)) * (self.gamma_bound_45_degree[1] - self.gamma_bound_0_degree[1])
                    elif angle < np.pi * 0.5:
                        lower_bound = self.gamma_bound_45_degree[0] + ((angle - np.pi * 0.25) / (np.pi * 0.25)) * (self.gamma_bound_90_degree[0] - self.gamma_bound_45_degree[0])
                        upper_bound = self.gamma_bound_45_degree[1] + ((angle - np.pi * 0.25) / (np.pi * 0.25)) * (self.gamma_bound_90_degree[1] - self.gamma_bound_45_degree[1])
                    elif angle < np.pi * 0.75:
                        lower_bound = self.gamma_bound_90_degree[0] + ((angle - np.pi * 0.5) / (np.pi * 0.25)) * (self.gamma_bound_135_degree[0] - self.gamma_bound_90_degree[0])
                        upper_bound = self.gamma_bound_90_degree[1] + ((angle - np.pi * 0.5) / (np.pi * 0.25)) * (self.gamma_bound_135_degree[1] - self.gamma_bound_90_degree[1])
                    else:
                        lower_bound = self.gamma_bound_135_degree[0] + ((angle - np.pi * 0.75) / (np.pi * 0.25)) * (self.gamma_bound_180_degree[0] - self.gamma_bound_135_degree[0])
                        upper_bound = self.gamma_bound_135_degree[1] + ((angle - np.pi * 0.75) / (np.pi * 0.25)) * (self.gamma_bound_180_degree[1] - self.gamma_bound_135_degree[1])
                         
                    discount_factor = lower_bound + tm.clamp(current_force / self.maximum_tension, 0., 1.) * (upper_bound - lower_bound)       
                    #后面绳与前面板夹角
                    current_angle = tm.pi * 0.5 - tm.acos(abs(tm.clamp(-force_dir.dot(before_nm) * before_direction / force_dir.norm(1e-6), -1.0, 1.0)))
                    if force_dir.norm() < self.origami_thickness:
                        current_angle = tm.pi * 0.5
                    #前面绳与前面板夹角
                    previous_angle = tm.pi * 0.5 - tm.acos(abs(tm.clamp(-before_force_dir.dot(before_nm) * before_direction / before_force_dir.norm(1e-6), -1.0, 1.0)))
                    if before_force_dir.norm() < self.origami_thickness:
                        previous_angle = tm.pi * 0.5
                    # print(f"{i}, {k}, 1")
                    start_tri_index = 0
                    for kk in ti.ndrange(self.unit_control[i, k - 1]):
                        start_tri_index += self.unit_kp_num_list[kk] - 2
                    self.calculateInitialTorque_FieldForce(previous_angle, before_kp_id, start_point, -before_force_dir + before_force_dir.dot(before_nm) * before_nm, -current_force * before_direction, start_tri_index, self.unit_kp_num_list[self.unit_control[i, k - 1]] - 2)
                    
                    # previous_ratio = previous_angle / tm.pi * 0.5
                    # current_ratio = current_angle / tm.pi * 0.5
                    base = tm.clamp(tm.pi * 0.5 - ((lower_bound + upper_bound) - (self.gamma_bound_0_degree[0] + self.gamma_bound_0_degree[1])) / \
                        ((self.gamma_bound_180_degree[0] + self.gamma_bound_180_degree[1]) - (self.gamma_bound_0_degree[0] + self.gamma_bound_0_degree[1])) * (tm.pi * 0.5 - self.beta), self.beta, tm.pi * 0.5)

                    if previous_angle < base and current_angle < base:
                        previous_ratio = tm.clamp(previous_angle / base, 0.0, 1.0)
                        current_ratio = tm.clamp(current_angle / base, 0.0, 1.0)

                        avg_ratio = (previous_ratio + current_ratio) * 0.5

                        discount_factor = discount_factor ** (1.0 - avg_ratio)
                    else:
                        dif = abs(current_angle - previous_angle) * 0.5 / base
                        discount_factor = discount_factor ** dif

                    # print(avg_ratio, discount_factor)
                    basic_factor *= discount_factor
                    basic_factor -= self.static_friction
                    if basic_factor < 0.:
                        basic_factor = 0.
                    current_force = self.string_force_each[i] * basic_factor

                    # print(f"{i}, {k}, 2")
                    start_tri_index = 0
                    for kk in ti.ndrange(self.unit_control[i, k - 1]):
                        start_tri_index += self.unit_kp_num_list[kk] - 2
                    self.calculateInitialTorque_FieldForce(current_angle, before_kp_id, start_point, force_dir - force_dir.dot(before_nm) * before_nm, -current_force * hole_direction, start_tri_index, self.unit_kp_num_list[self.unit_control[i, k - 1]] - 2)

                    final_basic_factor = basic_factor

                    # if self.delta_length_per_string[i, k] >= 0:
                    # #     if previous_force > self.string_force_each[i] * basic_factor and previous_force <= self.string_force_each[i] * self.string_force_current_discount[i, k - 1]:
                    # #         basic_factor = previous_force / self.string_force_each[i]
                    # #         current_force = self.string_force_each[i] * basic_factor
                    # #         final_basic_factor = basic_factor
                    # #     elif previous_force > self.string_force_each[i] * self.string_force_current_discount[i, k - 1]:
                    # #         basic_factor = self.string_force_current_discount[i, k - 1]
                    # #         current_force = self.string_force_each[i] * basic_factor
                    # #         final_basic_factor = basic_factor
                    # # else:
                    #     if previous_force > self.string_force_each[i] * basic_factor and previous_force <= self.string_force_each[i] * self.string_force_current_discount[i, k - 1] ** 2 / basic_factor:
                    #         basic_factor = previous_force / self.string_force_each[i]
                    #         current_force = self.string_force_each[i] * basic_factor
                    #         final_basic_factor = basic_factor
                    #     elif previous_force > self.string_force_each[i] * self.string_force_current_discount[i, k - 1] ** 2 / basic_factor:
                    #         basic_factor = self.string_force_current_discount[i, k - 1] ** 2 / basic_factor
                    #         current_force = self.string_force_each[i] * basic_factor
                    #         final_basic_factor = basic_factor
                    
                    self.string_force_current_discount[i, k] = final_basic_factor
                    # print(i, k, final_basic_factor)
                    
                    for j in ti.ndrange(self.bending_pairs_num):
                        if (self.equivalent_torque_influence_id[i, j] >> (k - 1)) & 1:
                            self.equivalent_torque_index[i, j, k] = final_basic_factor * self.equivalent_torque_coeff[i, j, k]
                    
                    sum_coeff += final_basic_factor

                    initial_length += self.initial_length_per_string[i, k] * self.string_force_current_discount[i, k]

                    if self.unit_control[i, k + 1] == -1:
                        final_angle = tm.pi * 0.5 - tm.acos(abs(tm.clamp(-next_force_dir.dot(nm) * hole_direction / next_force_dir.norm(1e-6), -1.0, 1.0)))
                        if next_force_dir.norm() < self.origami_thickness:
                            final_angle = tm.pi * 0.5
                        # print(f"{i}, {k}, 3")
                        start_tri_index = 0
                        for kk in ti.ndrange(self.unit_control[i, k]):
                            start_tri_index += self.unit_kp_num_list[kk] - 2
                        self.calculateInitialTorque_FieldForce(final_angle, kp_id, self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k]], self.unit_control[i, k]), \
                                                               -next_force_dir + next_force_dir.dot(nm) * nm, -current_force * hole_direction, start_tri_index, self.unit_kp_num_list[self.unit_control[i, k]] - 2)
            # method 1
            # self.target_string_length_decrease[i, 0] = (initial_length - delta_length * basic_factor) 
            # self.string_params_bonus[i] /= basic_factor
            # method 2
            self.target_string_length_decrease[i, 0] = (initial_length - delta_length)
            self.backup_delta_length[i] = delta_length
            # print(initial_length, delta_length)
            # self.string_params_bonus[i] /= basic_factor
            # for k in ti.ndrange(self.max_control_length):
            #     if self.unit_control[i, k] != -1:
            #         self.target_string_length_decrease[i, k] = delta_length * self.string_force_current_discount[i, k] / sum_coeff
            #     self.string_force_discount_sum[i] = sum_coeff
                        
    @ti.func
    def calculateStringLength(self, i, line_search=False, use_history_info=False):
        """
        计算绳索约束的长度，考虑绳索与面片的相交情况。
        Calculate the length of string constraint, considering intersections between string and panels.
        
        :param i: 约束索引 / Constraint index
        :param line_search: 是否进行线搜索 / Whether to perform line search
        :param use_history_info: 是否使用历史信息 / Whether to use history information
        """
        self.constraint_length[i] = 0.0
        start_point = self.constraint_start_point[i]
        current_id = 0
        for k in ti.ndrange(self.max_control_length):
            if self.unit_control[i, k] != -1:
                start_point = self.constraint_start_point[i]
                if k > 0:
                    start_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, k - 1]], self.unit_control[i, k - 1])
                current_id = k

                if k == 0:
                    additional_id = -1
                    if self.equivalent_internal_point_id[i] >= 0:
                        additional_id = self.equivalent_internal_point_id[i]
                    else:
                        candidate_id = self.constraint_start_point_candidate_id[i]
                        unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                        if unit_connection_id >= 0:
                            additional_id = self.loc_of_unit[candidate_id]

                    if additional_id >= 0:
                        start_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[additional_id], additional_id)

                    kp_id = self.unit_indices[self.unit_control[i, k]]
                    end_point = self.calculateCenterPoint3DWithUnitId(kp_id, self.unit_control[i, k])

                    force_dir_1 = end_point - start_point
                    if additional_id >= 0:
                        if self.intersection_flag2_initial[i] and (not self.intersection_flag[i, k] or \
                                                                (self.intersection_flag[i, k] and \
                                                                (self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][1])] != \
                                                                    self.unit_indices[additional_id][int(self.intersection_infos2_initial[i][2])] or \
                                                                    self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][2])] != \
                                                                    self.unit_indices[additional_id][int(self.intersection_infos2_initial[i][1])]))): 
                            t = self.intersection_infos2_initial[i][0]
                            l1 = int(self.intersection_infos2_initial[i][1])
                            l2 = int(self.intersection_infos2_initial[i][2])
                            cur_x = self.get_position_with_index(self.unit_indices[additional_id][l1])
                            next_x = self.get_position_with_index(self.unit_indices[additional_id][l2])
                            intersect = cur_x + (next_x - cur_x) * t
                            force_dir_1 = end_point - intersect
                    
                    force_dir_2 = end_point - start_point
                    if self.intersection_flag[i, k] and (not self.intersection_flag2_initial[i] or \
                                                         (self.intersection_flag2_initial[i] and \
                                                          (self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][1])] != \
                                                            self.unit_indices[additional_id][int(self.intersection_infos2_initial[i][2])] or \
                                                            self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][2])] != \
                                                            self.unit_indices[additional_id][int(self.intersection_infos2_initial[i][1])]))):
                        t = self.intersection_infos[k + i * self.max_control_length][0]
                        l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                        l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                        cur_x = self.get_position_with_index(kp_id[l1])
                        next_x = self.get_position_with_index(kp_id[l2])
                        intersect = cur_x + (next_x - cur_x) * t
                        force_dir_2 = intersect - start_point

                    unit_kp_num = self.unit_kp_num_list[self.unit_control[i, k]]
                    hole_direction = self.hole_dir[i, k]
                    nm = self.calculateNormalVectorWithUnitId(kp_id)
                    penetration = force_dir_1.dot(nm) * hole_direction / force_dir_1.norm(1e-6)
                    length = 0.0
                    self.current_length_per_string[i, k] = 0.0
                    intersect = end_point
                    intersect2 = start_point
                    # if 0: #penetration
                    if force_dir_1.norm() < self.origami_thickness:
                        if self.intersection_flag[i, k] and penetration <= 0:
                            penetration = 0.01
                        elif not self.intersection_flag[i, k] and penetration > 0:
                            penetration = -0.01
                    # print(f"{i}, {k}, {penetration}, {force_dir.norm(1e-6)}, {self.intersection_flag[i, k]}")
                    if not use_history_info:
                        if penetration > 0: #penetration
                            find = 0
                            for l in ti.ndrange(unit_kp_num):
                                cur_x = self.get_position_with_index(kp_id[l])
                                next_x = self.get_position_with_index(kp_id[(l + 1) % unit_kp_num])
                                if (not self.intersection_flag[i, k]) or \
                                    (self.intersection_flag[i, k] and self.intersection_infos[k + i * self.max_control_length][0] >= 1e-2 and \
                                    self.intersection_infos[k + i * self.max_control_length][0] <= 0.99 and \
                                    l == self.intersection_infos[k + i * self.max_control_length][1]):
                                        t = self.intersect4p(end_point, cur_x, next_x, force_dir_1, -1.)
                                        if t < -1e-2 and self.intersection_flag[i, k]:
                                            t = self.intersect4p(end_point, cur_x, next_x, force_dir_1, 1.)
                                        if t > -1e-2 and t < 1.01:
                                            intersect = cur_x + (next_x - cur_x) * t
                                            self.intersection_points[k + i * self.max_control_length] = intersect
                                            self.intersection_infos[k + i * self.max_control_length] = ti.Vector([t, l, (l + 1) % unit_kp_num])
                                            if not line_search:
                                                self.intersection_flag[i, k] = 1
                                            length = (intersect - end_point).norm()
                                            self.current_length_per_string[i, k] += length
                                            self.constraint_length[i] += length
                                            find = 1
                                            break
                            if not find:
                                # if not line_search:
                                if self.intersection_flag[i, k]:
                                    t = self.intersection_infos[k + i * self.max_control_length][0]
                                    l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                                    l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                                    cur_x = self.get_position_with_index(kp_id[l1])
                                    next_x = self.get_position_with_index(kp_id[l2])
                                    intersect = cur_x + (next_x - cur_x) * t
                                    self.intersection_points[k + i * self.max_control_length] = intersect

                                    length = (intersect - end_point).norm()
                                    self.current_length_per_string[i, k] += length
                                    self.constraint_length[i] += length
                                else:
                                    penetration = 0
                    else:
                        if self.intersection_flag[i, k]: #penetration
                            t = self.intersection_infos[k + i * self.max_control_length][0]
                            l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                            l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                            cur_x = self.get_position_with_index(kp_id[l1])
                            next_x = self.get_position_with_index(kp_id[l2])
                            intersect = cur_x + (next_x - cur_x) * t
                            # intersect = self.intersection_points[k + i * self.max_control_length]
                            length = (intersect - end_point).norm()
                            self.current_length_per_string[i, k] += length
                            self.constraint_length[i] += length

                    penetration2 = 0.
                    before_kp_id = ti.Vector.zero(int, self.unit_edge_max)
                    before_unit_kp_num = self.unit_edge_max
                    if additional_id >= 0:
                        before_kp_id = self.unit_indices[additional_id]
                        before_unit_kp_num = self.unit_kp_num_list[additional_id]
                        hole_direction = -self.hole_dir[i, 0]
                        nm = self.calculateNormalVectorWithUnitId(before_kp_id)
                            
                        penetration2 = force_dir_2.dot(nm) * hole_direction / force_dir_2.norm(1e-6)
                        length = 0.0
                        # if 0: #penetration
                        if force_dir_2.norm() < self.origami_thickness:
                            if self.intersection_flag2_initial[i] and penetration2 <= 0:
                                penetration2 = 0.01
                            elif not self.intersection_flag2_initial[i] and penetration2 > 0:
                                penetration2 = -0.01
                            
                        if not use_history_info:
                            if penetration2 > 0: #penetration
                                find = 0
                                for l in range(before_unit_kp_num):
                                    cur_x = self.get_position_with_index(before_kp_id[l])
                                    next_x = self.get_position_with_index(before_kp_id[(l + 1) % before_unit_kp_num])
                                    if (not self.intersection_flag2_initial[i]) or \
                                        (self.intersection_flag2_initial[i] and self.intersection_infos2_initial[i][0] >= 1e-2 and \
                                        self.intersection_infos2_initial[i][0] <= 0.99 and \
                                        l == self.intersection_infos2_initial[i][1]):
                                            t = self.intersect4p(start_point, cur_x, next_x, force_dir_2, 1.)
                                            if t < -1e-2 and self.intersection_flag2_initial[i]:
                                                t = self.intersect4p(start_point, cur_x, next_x, force_dir_2, -1.)
                                            if t > -1e-2 and t < 1.01:
                                                intersect2 = cur_x + (next_x - cur_x) * t
                                                # print(f"{intersect2}, {before_kp_id[l]}, {before_kp_id[(l + 1) % before_unit_kp_num]}, {t}")
                                                
                                                self.intersection_points2_initial[i] = intersect2
                                                self.intersection_infos2_initial[i] = ti.Vector([t, l, (l + 1) % before_unit_kp_num])
                                                if not line_search:
                                                    self.intersection_flag2_initial[i] = 1
                                                length = (intersect2 - start_point).norm()
                                                self.constraint_length[i] += length
                                                self.current_length_per_string[i, k] += length
                                                # print("end intersection ok")
                                                find = 1
                                                break
                                    
                                if not find:
                                    if self.intersection_flag2_initial[i]:
                                        t = self.intersection_infos2_initial[i][0]
                                        l1 = int(self.intersection_infos2_initial[i][1])
                                        l2 = int(self.intersection_infos2_initial[i][2])
                                        cur_x = self.get_position_with_index(before_kp_id[l1])
                                        next_x = self.get_position_with_index(before_kp_id[l2])
                                        intersect2 = cur_x + (next_x - cur_x) * t
                                        self.intersection_points2_initial[i] = intersect2

                                        length = (intersect2 - start_point).norm()
                                        self.constraint_length[i] += length
                                        self.current_length_per_string[i, k] += length
                                    else:
                                        penetration2 = 0
                        else:
                            if self.intersection_flag2_initial[i]: #penetration
                                t = self.intersection_infos2_initial[i][0]
                                l1 = int(self.intersection_infos2_initial[i][1])
                                l2 = int(self.intersection_infos2_initial[i][2])
                                cur_x = self.get_position_with_index(before_kp_id[l1])
                                next_x = self.get_position_with_index(before_kp_id[l2])
                                intersect2 = cur_x + (next_x - cur_x) * t
                                # intersect2 = self.intersection_points2_initial[i]
                                length = (intersect2 - start_point).norm()
                                self.constraint_length[i] += length
                                self.current_length_per_string[i, k] += length

                    if additional_id >= 0:
                        if use_history_info:
                            if self.intersection_flag[i, k]:
                                penetration = 0.01
                            else:
                                penetration = -0.01
                            if self.intersection_flag2_initial[i]:
                                penetration2 = 0.01
                            else:
                                penetration2 = -0.01
                        if penetration <= 0 and penetration2 <= 0:  
                            # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, end_point, before_kp_id, before_unit_kp_num):
                            #     self.error_status_buffer[i] = True
                            # else:
                            if not line_search and not use_history_info:
                                if self.calculateOtherPanelIntersection(start_point, end_point, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                    self.error_status_buffer[i] = True
                            if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                                self.error_status[0] = True
                            if not line_search and not use_history_info:
                                self.intersection_flag[i, k] = 0
                                self.intersection_flag2_initial[i] = 0    
                            length = (end_point - start_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length
                            # print("end ok")
                        elif penetration > 0 and penetration2 <= 0:  
                            # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, intersect, before_kp_id, before_unit_kp_num):
                            #     self.error_status_buffer[i] = True
                            # else:
                            if not line_search and not use_history_info:
                                if self.calculateOtherPanelIntersection(start_point, intersect, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                    self.error_status_buffer[i] = True
                            if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                                self.error_status[0] = True
                            if not line_search and not use_history_info:
                                self.intersection_flag2_initial[i] = 0    
                            length = (intersect - start_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length
                            # print("end ok")
                        elif penetration <= 0 and penetration2 > 0:  
                            # if self.calculateOtherIntersection(before_kp_id, before_unit_kp_num, intersect2, end_point, kp_id, unit_kp_num):
                            #     self.error_status_buffer[i] = True
                            # else:
                            if not line_search and not use_history_info:
                                if self.calculateOtherPanelIntersection(intersect2, end_point, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                    self.error_status_buffer[i] = True
                            if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                                self.error_status[0] = True
                            if not line_search and not use_history_info:
                                self.intersection_flag[i, k] = 0 
                            length = (end_point - intersect2).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length
                            # print("end ok")
                        else:
                            length = (intersect - intersect2).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length

                    else:
                        if use_history_info:
                            if self.intersection_flag[i, k]:
                                penetration = 0.01
                            else:
                                penetration = -0.01
                        if penetration <= 0:
                            # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, intersect, kp_id, 0):
                            #     self.error_status_buffer[i] = True
                            # else:
                            if not line_search and not use_history_info:
                                if self.calculateOtherPanelIntersection(start_point, end_point, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                    self.error_status_buffer[i] = True
                            if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                                self.error_status[0] = True
                            if not line_search and not use_history_info:
                                self.intersection_flag[i, k] = 0
                            length = (end_point - start_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length
                        else:
                            # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, intersect, kp_id, 0):
                            #     self.error_status_buffer[i] = True
                            # else:
                            if not line_search and not use_history_info:
                                if self.calculateOtherPanelIntersection(start_point, intersect, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                    self.error_status_buffer[i] = True
                            if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                                self.error_status[0] = True
                            length = (intersect - start_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length

                    # print(f"{i}, {k}, {self.constraint_length[i]}")
                else:
                    # self.intersection_points[k + i * self.max_control_length] = [0., 0., 1000.]
                    kp_id = self.unit_indices[self.unit_control[i, k]]
                    unit_kp_num = self.unit_kp_num_list[self.unit_control[i, k]]
                    end_point = self.calculateCenterPoint3DWithUnitId(kp_id, self.unit_control[i, k])
                    before_kp_id = self.unit_indices[self.unit_control[i, k - 1]]
                    before_unit_kp_num = self.unit_kp_num_list[self.unit_control[i, k - 1]]

                    force_dir_1 = end_point - start_point
                    
                    if self.intersection_flag2[i, k - 1] and (not self.intersection_flag[i, k] or \
                                                              (self.intersection_flag[i, k] and \
                                                               (self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][1])] != \
                                                                self.unit_indices[self.unit_control[i, k - 1]][int(self.intersection_infos2[k - 1 + i * self.max_control_length][2])] or \
                                                                self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][2])] != \
                                                                self.unit_indices[self.unit_control[i, k - 1]][int(self.intersection_infos2[k - 1 + i * self.max_control_length][1])]))):
                        t = self.intersection_infos2[k - 1 + i * self.max_control_length][0]
                        l1 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][1])
                        l2 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][2])
                        cur_x = self.get_position_with_index(before_kp_id[l1])
                        next_x = self.get_position_with_index(before_kp_id[l2])
                        intersect2 = cur_x + (next_x - cur_x) * t
                        force_dir_1 = end_point - intersect2

                    force_dir_2 = end_point - start_point
                    if self.intersection_flag[i, k] and (not self.intersection_flag2[i, k - 1] or \
                                                         (self.intersection_flag2[i, k - 1] and \
                                                           (self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][1])] != \
                                                            self.unit_indices[self.unit_control[i, k - 1]][int(self.intersection_infos2[k - 1 + i * self.max_control_length][2])] or \
                                                            self.unit_indices[self.unit_control[i, k]][int(self.intersection_infos[k + i * self.max_control_length][2])] != \
                                                            self.unit_indices[self.unit_control[i, k - 1]][int(self.intersection_infos2[k - 1 + i * self.max_control_length][1])]))):
                        t = self.intersection_infos[k + i * self.max_control_length][0]
                        l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                        l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                        cur_x = self.get_position_with_index(kp_id[l1])
                        next_x = self.get_position_with_index(kp_id[l2])
                        intersect = cur_x + (next_x - cur_x) * t
                        force_dir_2 = intersect - start_point
                    
                    hole_direction = self.hole_dir[i, k]
                    nm = self.calculateNormalVectorWithUnitId(kp_id)
                    penetration = force_dir_1.dot(nm) * hole_direction / force_dir_1.norm(1e-6)
                    length = 0.0
                    self.current_length_per_string[i, k] = 0.0
                    # if 0: #penetration
                    if force_dir_1.norm() < self.origami_thickness:
                        if self.intersection_flag[i, k] and penetration <= 0:
                            penetration = 0.01
                        elif not self.intersection_flag[i, k] and penetration > 0:
                            penetration = -0.01
                    # print(f"{i}, {k}, {penetration}, {force_dir.norm(1e-6)}, {self.intersection_flag[i, k]}")
                    intersect = end_point
                    intersect2 = start_point
                    if not use_history_info:
                        if penetration > 0: #penetration
                            find = 0
                            for l in ti.ndrange(unit_kp_num):
                                cur_x = self.get_position_with_index(kp_id[l])
                                next_x = self.get_position_with_index(kp_id[(l + 1) % unit_kp_num])
                                if (not self.intersection_flag[i, k]) or \
                                    (self.intersection_flag[i, k] and self.intersection_infos[k + i * self.max_control_length][0] >= 1e-2 and \
                                    self.intersection_infos[k + i * self.max_control_length][0] <= 0.99 and \
                                    l == self.intersection_infos[k + i * self.max_control_length][1]):
                                        t = self.intersect4p(end_point, cur_x, next_x, force_dir_1, -1.)
                                        if t < -1e-2 and self.intersection_flag[i, k]:
                                            t = self.intersect4p(end_point, cur_x, next_x, force_dir_1, 1.)
                                        # if not use_history_info and k == 1 and i == 0:
                                        #     print(t)
                                        if t > -1e-2 and t < 1. + 1e-2:
                                            intersect = cur_x + (next_x - cur_x) * t
                                            # if not use_history_info and k == 1 and i == 0:
                                            #     print(f"find: {intersect}, {kp_id[l]}, {kp_id[(l + 1) % unit_kp_num]}, {t}")
                                            
                                            self.intersection_points[k + i * self.max_control_length] = intersect
                                            self.intersection_infos[k + i * self.max_control_length] = ti.Vector([t, l, (l + 1) % unit_kp_num])
                                            if not line_search:
                                                self.intersection_flag[i, k] = 1
                                            length = (intersect - end_point).norm()
                                            self.constraint_length[i] += length
                                            self.current_length_per_string[i, k] += length
                                            find = 1
                                            break
                            if not find:
                                if self.intersection_flag[i, k]:
                                    t = self.intersection_infos[k + i * self.max_control_length][0]
                                    l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                                    l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                                    cur_x = self.get_position_with_index(kp_id[l1])
                                    next_x = self.get_position_with_index(kp_id[l2])
                                    intersect = cur_x + (next_x - cur_x) * t
                                    self.intersection_points[k + i * self.max_control_length] = intersect
                                    length = (intersect - end_point).norm()
                                    self.constraint_length[i] += length
                                    self.current_length_per_string[i, k] += length
                                else:
                                    penetration = 0
                    else:
                        if self.intersection_flag[i, k]: #penetration
                            t = self.intersection_infos[k + i * self.max_control_length][0]
                            l1 = int(self.intersection_infos[k + i * self.max_control_length][1])
                            l2 = int(self.intersection_infos[k + i * self.max_control_length][2])
                            cur_x = self.get_position_with_index(kp_id[l1])
                            next_x = self.get_position_with_index(kp_id[l2])
                            intersect = cur_x + (next_x - cur_x) * t
                            # intersect = self.intersection_points[k + i * self.max_control_length]
                            length = (intersect - end_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length

                    # self.intersection_points2[k - 1 + i * self.max_control_length] = [0., 0., 1000.]
                    hole_direction = self.hole_dir[i, k - 1]

                    #---

                    nm = self.calculateNormalVectorWithUnitId(before_kp_id)
                    penetration2 = force_dir_2.dot(nm) * hole_direction / force_dir_2.norm(1e-6)
                    length = 0.0
                    # if 0: #penetration
                    if force_dir_2.norm() < self.origami_thickness:
                        if self.intersection_flag2[i, k - 1] and penetration2 <= 0:
                            penetration2 = 0.01
                        elif not self.intersection_flag2[i, k - 1] and penetration2 > 0:
                            penetration2 = -0.01
                    
                    if not use_history_info:
                        if penetration2 > 0: #penetration
                            find = 0
                            for l in range(before_unit_kp_num):
                                cur_x = self.get_position_with_index(before_kp_id[l])
                                next_x = self.get_position_with_index(before_kp_id[(l + 1) % before_unit_kp_num])
                                if (not self.intersection_flag2[i, k - 1]) or \
                                    (self.intersection_flag2[i, k - 1] and self.intersection_infos2[k - 1 + i * self.max_control_length][0] >= 1e-2 and \
                                    self.intersection_infos2[k - 1 + i * self.max_control_length][0] <= 0.99 and \
                                    l == self.intersection_infos2[k - 1 + i * self.max_control_length][1]):
                                        t = self.intersect4p(start_point, cur_x, next_x, force_dir_2, 1.)
                                        if t < -1e-2 and self.intersection_flag2[i, k - 1]:
                                            t = self.intersect4p(start_point, cur_x, next_x, force_dir_2, -1.)
                                        if t > -1e-2 and t < 1.01:
                                            intersect2 = cur_x + (next_x - cur_x) * t
                                            # print(f"{intersect2}, {before_kp_id[l]}, {before_kp_id[(l + 1) % before_unit_kp_num]}, {t}")
                                            
                                            self.intersection_points2[k - 1 + i * self.max_control_length] = intersect2
                                            self.intersection_infos2[k - 1 + i * self.max_control_length] = ti.Vector([t, l, (l + 1) % before_unit_kp_num])
                                            if not line_search:
                                                self.intersection_flag2[i, k - 1] = 1
                                            length = (intersect2 - start_point).norm()
                                            self.constraint_length[i] += length
                                            self.current_length_per_string[i, k] += length
                                            # print("end intersection ok")
                                            find = 1
                                            break
                                
                            if not find:
                                if self.intersection_flag2[i, k - 1]:
                                    t = self.intersection_infos2[k - 1 + i * self.max_control_length][0]
                                    l1 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][1])
                                    l2 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][2])
                                    cur_x = self.get_position_with_index(before_kp_id[l1])
                                    next_x = self.get_position_with_index(before_kp_id[l2])
                                    intersect2 = cur_x + (next_x - cur_x) * t
                                    self.intersection_points2[k - 1 + i * self.max_control_length] = intersect2
                                    length = (intersect2 - start_point).norm()
                                    self.constraint_length[i] += length
                                    self.current_length_per_string[i, k] += length
                                else:
                                    penetration2 = 0
                    else:
                        if self.intersection_flag2[i, k - 1]: #penetration
                            t = self.intersection_infos2[k - 1 + i * self.max_control_length][0]
                            l1 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][1])
                            l2 = int(self.intersection_infos2[k - 1 + i * self.max_control_length][2])
                            cur_x = self.get_position_with_index(before_kp_id[l1])
                            next_x = self.get_position_with_index(before_kp_id[l2])
                            intersect2 = cur_x + (next_x - cur_x) * t
                            # intersect2 = self.intersection_points2[k - 1 + i * self.max_control_length]
                            length = (intersect2 - start_point).norm()
                            self.constraint_length[i] += length
                            self.current_length_per_string[i, k] += length
                    
                    if use_history_info:
                        if self.intersection_flag[i, k]:
                            penetration = 0.01
                        else:
                            penetration = -0.01
                        if self.intersection_flag2[i, k - 1]:
                            penetration2 = 0.01
                        else:
                            penetration2 = -0.01
                    
                    if penetration <= 0 and penetration2 <= 0:  
                        # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, end_point, before_kp_id, before_unit_kp_num):
                        #     self.error_status_buffer[i] = True
                        # else:
                        if not line_search and not use_history_info:
                            if self.calculateOtherPanelIntersection(start_point, end_point, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                # print(f"1, start: {start_point}, end: {end_point}, {penetration}, {penetration2}, history: {use_history_info}, {self.intersection_flag[i, k]}, {self.intersection_flag2[i, k - 1]}")
                                self.error_status_buffer[i] = True
                        if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                            self.error_status[0] = True
                        if not line_search and not use_history_info:
                            self.intersection_flag[i, k] = 0
                            self.intersection_flag2[i, k - 1] = 0    
                        length = (end_point - start_point).norm()
                        self.constraint_length[i] += length
                        self.current_length_per_string[i, k] += length
                        # print("end ok")
                    elif penetration > 0 and penetration2 <= 0:  
                        # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, intersect, before_kp_id, before_unit_kp_num):
                        #     self.error_status_buffer[i] = True
                        # else:
                        if not line_search and not use_history_info:
                            if self.calculateOtherPanelIntersection(start_point, intersect, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                # print(f"2, start: {start_point}, intersect: {intersect}, end: {end_point}")
                                self.error_status_buffer[i] = True
                        if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                            self.error_status[0] = True
                        if not line_search and not use_history_info:
                            self.intersection_flag2[i, k - 1] = 0    
                        length = (intersect - start_point).norm()
                        self.constraint_length[i] += length
                        self.current_length_per_string[i, k] += length
                        # print("end ok")
                    elif penetration <= 0 and penetration2 > 0:  
                        # if self.calculateOtherIntersection(before_kp_id, before_unit_kp_num, intersect2, end_point, kp_id, unit_kp_num):
                        #     self.error_status_buffer[i] = True
                        # else:
                        if not line_search and not use_history_info:
                            if self.calculateOtherPanelIntersection(intersect2, end_point, kp_id, before_kp_id, unit_kp_num, before_unit_kp_num):
                                # print(f"3, start: {start_point}, intersect2: {intersect2}, end: {end_point}")
                                self.error_status_buffer[i] = True
                        if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                            self.error_status[0] = True
                        if not line_search and not use_history_info:
                            self.intersection_flag[i, k] = 0 
                        length = (end_point - intersect2).norm()
                        self.constraint_length[i] += length
                        self.current_length_per_string[i, k] += length
                        # print("end ok")
                    else:
                        length = (intersect - intersect2).norm()
                        self.constraint_length[i] += length
                        self.current_length_per_string[i, k] += length
                # print(f"{i}, {k}, {self.constraint_length[i]}")

            else:
                # abondon this
                if self.constraint_end_point_existence[i]:
                    start_point = self.calculateCenterPoint3DWithUnitId(self.unit_indices[self.unit_control[i, current_id]], self.unit_control[i, current_id])
                    # print("start ok")
                    index = current_id
                    # self.intersection_points2[index + i * self.max_control_length] = [0., 0., 1000.]
                    # self.intersection_flag[i, index] = 0
                    kp_id = self.unit_indices[self.unit_control[i, current_id]]
                    end_point = self.constraint_end_point[i]
                    force_dir = end_point - start_point
                    unit_kp_num = self.unit_kp_num_list[self.unit_control[i, current_id]]
                    hole_direction = self.hole_dir[i, current_id]
                    nm = self.calculateNormalVectorWithUnitId(kp_id)
                    penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                    length = 0.0
                    # print(penetration)
                    # if 0: #penetration
                    if force_dir.norm() < self.origami_thickness:
                        if self.intersection_flag2[i, index] and penetration <= 0:
                            penetration = 0.01
                        elif not self.intersection_flag2[i, index] and penetration > 0:
                            penetration = -0.01
                    # print(f"{i}, {index}, {penetration}, {force_dir.norm(1e-6)}, {self.intersection_flag[i, index]}")
                    intersect = start_point
                    if penetration > 0: #penetration
                        find = 0
                        for l in range(unit_kp_num):
                            cur_x = self.get_position_with_index(kp_id[l])
                            next_x = self.get_position_with_index(kp_id[(l + 1) % unit_kp_num])
                            t = self.intersect4p(start_point, cur_x, next_x, force_dir, 1.)
                            if t > -1e-2 and t < 1.01:
                                intersect = cur_x + (next_x - cur_x) * t
                                
                                self.intersection_points2[index + i * self.max_control_length] = intersect
                                self.intersection_infos2[index + i * self.max_control_length] = ti.Vector([t, l, (l + 1) % unit_kp_num])
                                if not line_search:
                                    self.intersection_flag2[i, index] = 1
                                length = (intersect - start_point).norm() + (intersect - end_point).norm()
                                self.constraint_length[i] += length
                                
                                # print("end intersection ok")
                                find = 1
                                break
                        
                        if not find:
                            if self.intersection_flag2[i, index]:
                                intersect = self.intersection_points2[index + i * self.max_control_length]
                                length = (intersect - start_point).norm() + (intersect - end_point).norm()
                                self.constraint_length[i] += length
                            else:
                                length = (end_point - start_point).norm()
                                self.constraint_length[i] += length

                    else:   
                        # if self.calculateOtherIntersection(kp_id, unit_kp_num, start_point, self.constraint_end_point[i], kp_id, 0):
                        #     self.error_status_buffer[i] = True
                        # else:
                        if not line_search:
                            if self.calculateOtherPanelIntersection(start_point, self.constraint_end_point[i], kp_id, kp_id, unit_kp_num, 0):
                                self.error_status_buffer[i] = True
                        if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                            self.error_status[0] = True
                        if not line_search:
                            self.intersection_flag2[i, index] = 0    
                        length = (self.constraint_end_point[i] - start_point).norm()
                        self.constraint_length[i] += length

                    # if self.calculateOtherIntersection(kp_id, unit_kp_num, intersect, self.constraint_end_point[i], kp_id, 0):
                    #     self.error_status_buffer[i] = True
                    # else:
                    if not line_search:
                        if self.calculateOtherPanelIntersection(intersect, self.constraint_end_point[i], kp_id, kp_id, unit_kp_num, 0):
                            self.error_status_buffer[i] = True
                    if self.error_status_buffer[i] and self.history_maximum_force[i] > self.intersection_threshold:
                        self.error_status[0] = True
                            
                    self.current_length_per_string[i, index] = length

                break

    @ti.kernel
    def backupStringLength(self):
        """
        备份绳索长度信息，计算长度变化量。
        Backup string length information and calculate length change.
        """
        for i, j in ti.ndrange(self.constraint_number, self.max_control_length):
            self.delta_length_per_string[i, j] = self.current_length_per_string[i, j] - self.initial_length_per_string[i, j]
            self.initial_length_per_string[i, j] = self.current_length_per_string[i, j]
    
    @ti.func
    def calculateConstraintPoint(self):
        """
        计算约束点的位置（起点和终点）。
        Calculate constraint point positions (start and end points).
        """
        for i in ti.ndrange(self.constraint_number):
            candidate_id = self.constraint_start_point_candidate_id[i]
            unit_id = self.constraint_start_point_candidate_connection[candidate_id]
            # print(unit_id)
            if unit_id >= 0:
                index = self.loc_of_unit[candidate_id]
                unit_indice = self.unit_indices[index]
                center = self.calculateCenterPoint3DWithUnitId(unit_indice, index)
                self.constraint_start_point[i] = center
            if self.constraint_end_point_existence[i]:
                candidate_id = self.constraint_end_point_candidate_id[i]
                unit_id = self.constraint_start_point_candidate_connection[candidate_id]
                # if unit_id >= 0:
                #     unit_indice = self.unit_indices[unit_id]
                #     height = self.constraint_height[candidate_id]
                #     center = self.calculateCenterPoint3DWithUnitId(unit_indice, unit_id)
                #     n = self.calculateNormalVectorWithUnitId(unit_indice)
                #     self.constraint_end_point[i] = center + n * height
                    # print(self.constraint_end_point[i])
        
    @ti.func
    def getEnergy(self, sim_mode, theta, gravitational_acc, facet_k, enable_ground, mode, step):
        """
        计算系统总能量（包括STVK能量、弯曲能量、绳索能量等）。
        Calculate total system energy (including STVK energy, bending energy, string energy, etc.).
        
        :param sim_mode: 仿真模式 / Simulation mode
        :param theta: 目标折叠角度 / Target folding angle
        :param gravitational_acc: 重力加速度 / Gravitational acceleration
        :param facet_k: 面片弯曲刚度 / Facet bending stiffness
        :param enable_ground: 是否启用地面 / Whether to enable ground
        :param mode: 能量计算模式 / Energy calculation mode
        :param step: 当前步数 / Current step
        :return: 总能量 / Total energy
        """
        total_energy = 0.0

        backup_energy = total_energy
        for i in ti.ndrange(self.div_indices_num):
            x0 = self.get_position_with_index(self.indices[3 * i])
            x1 = self.get_position_with_index(self.indices[3 * i + 1])
            x2 = self.get_position_with_index(self.indices[3 * i + 2])
            n = (x1 - x0).cross(x2 - x0)
            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n / (n.norm())])
            f = ds @ self.dm[i]
            green_tensor = 0.5 * (f.transpose() @ f - self.I3[None])
            energy_tensor = green_tensor.norm_sqr()
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] * 0.5 * ti.Matrix.trace(green_tensor) ** 2
            total_energy += psi * self.A[i]

        if self.print:
            print(f"LINE SEARCH :: STVK energy: {total_energy - backup_energy}")
        
        backup_energy = total_energy
        for j in ti.ndrange(self.MAXIMUM_FIX_PANEL):
            if self.fix_id_list[j] >= 0:
                fix_unit_indices = self.unit_indices[self.fix_id_list[j]]
                for i in ti.ndrange(self.unit_edge_max):
                    kp_id = fix_unit_indices[i]
                    if kp_id != -1:
                        total_energy += 0.5 * self.controller_k * (self.x[kp_id] - self.original_vertices[kp_id]).norm_sqr()
        if self.print:
            print(f"LINE SEARCH :: FIXED energy: {total_energy - backup_energy}")
                    
        backup_energy = total_energy
        
        for i in ti.ndrange(self.facet_crease_pairs_num):
            if self.facet_bending_pairs_num > 0:
                new_facet_k = facet_k
                if not self.facet_mode_flag[0]:
                    # avg_folding_angle = 0.0
                    # for j in ti.ndrange(self.crease_pairs_num):
                    #     avg_folding_angle += abs(self.crease_angle[j])
                    # avg_folding_angle /= self.crease_pairs_num
                    # new_facet_k /= (1. + (self.facet_bending_pairs_distance[i] - 1.) * avg_folding_angle)
                    new_facet_k /= self.facet_bending_pairs_distance[i]
                energy = self.getBendingEnergy(
                    self.get_position_with_index(self.facet_crease_pairs[i, 0]), self.get_position_with_index(self.facet_crease_pairs[i, 1]), 
                    self.get_position_with_index(self.facet_bending_pairs[i, 0]), self.get_position_with_index(self.facet_bending_pairs[i, 1]), 
                    new_facet_k, 0.0, 0, False, True, 
                    self.facet_bending_pairs_area[i, 0], self.facet_bending_pairs_area[i, 1], self.facet_crease_initial_length[i], -1,
                    self.facet_bending_pairs_distance[i]
                )
                total_energy += energy
        if self.print:
            print(f"LINE SEARCH :: FACET energy: {total_energy - backup_energy}")

        backup_energy = total_energy
        for k, i, j in ti.ndrange(self.P_number, 4, self.unit_edge_max):
            if self.connection_number[0] > 0:
                if self.constraint_start_point_candidate_connection[k] >= 0:
                    source_indices = self.unit_indices[self.loc_of_unit[k]]
                    connected_indices = self.unit_indices[self.constraint_start_point_candidate_connection[k]]
                    # for i, j in ti.ndrange(4, self.unit_edge_max):
                    if connected_indices[j] != -1:
                        current_distance = self.x[connected_indices[j]] - self.x[source_indices[i]]
                        initial_distance = self.original_vertices[connected_indices[j]] - self.original_vertices[source_indices[i]]
                        delta_l = current_distance.norm() - initial_distance.norm()
                        total_energy += 0.5 * self.controller_k * delta_l ** 2

        for k, i, j in ti.ndrange(self.unit_indices_num * self.unit_edge_max, self.unit_edge_max, self.unit_edge_max):
            if self.thick_panel_additional_connection_id[k][0] != -1:
                source_indices = self.unit_indices[self.thick_panel_additional_connection_id[k][0]]
                connected_indices = self.unit_indices[self.thick_panel_additional_connection_id[k][1]]
                if source_indices[i] != -1 and connected_indices[j] != -1:
                    current_distance = self.x[connected_indices[j]] - self.x[source_indices[i]]
                    initial_distance = self.original_vertices[connected_indices[j]] - self.original_vertices[source_indices[i]]
                    delta_l = current_distance.norm() - initial_distance.norm()
                    total_energy += 0.5 * self.thick_panel_k * delta_l ** 2
        if self.print:
            print(f"LINE SEARCH :: DFDX energy: {total_energy - backup_energy}")
            
        backup_energy = total_energy
        ground_energy = 0.0
        friction_energy = 0.0

        # if sim_mode:
        #     if enable_ground:
        c = self.epsilon_v[0]
        for i in ti.ndrange(self.kp_num):
            if sim_mode and enable_ground:
                ground_barrier = self.ground_barrier
                if self.kp_add_height[i]:
                    ground_barrier += self.additional_height[0]
                if self.x[i][Z] < ground_barrier:
                    barrier_exceed = self.x[i][Z] - ground_barrier
                    barrier_left = (self.x[i][Z] - (ground_barrier - 1.)) + self.collision_d
                    if self.x[i][Z] > ground_barrier - 1.:
                        total_energy += -self.ground_collision_indice * barrier_exceed ** 2 * tm.log(barrier_left / ground_barrier)
                        ground_energy += -self.ground_collision_indice * barrier_exceed ** 2 * tm.log(barrier_left / ground_barrier)
                        # self.ground_force[i] = self.ground_collision_indice * (2 * barrier_exceed * tm.log(barrier_left / ground_barrier) + barrier_exceed ** 2 / barrier_left) * tm.vec3([0., 0., 1.])
                    else:
                        total_energy += self.ground_barrier_energy_maximum - (2. * self.ground_force_maximum + (self.x[i][Z] - (ground_barrier - 1.)) * self.df_ground_force_maximum) * (self.x[i][Z] - (ground_barrier - 1.)) * 0.5
                        ground_energy += self.ground_barrier_energy_maximum - (2. * self.ground_force_maximum + (self.x[i][Z] - (ground_barrier - 1.)) * self.df_ground_force_maximum) * (self.x[i][Z] - (ground_barrier - 1.)) * 0.5
                        # self.ground_force[i] = (self.ground_force_maximum + (self.x[i][Z] - (ground_barrier - 1.)) * self.df_ground_force_maximum) * tm.vec3([0., 0., 1.])
                # if self.back_up_x[i][Z] < ground_barrier:
                # if step:
                self.u[i] = [self.x[i][X] - self.back_up_x[i][X], self.x[i][Y] - self.back_up_x[i][Y]]
                ui_norm = self.u[i].norm()
                term = ui_norm
                if ui_norm < c:
                    term = (-ui_norm ** 3 / (3 * c ** 2) + ui_norm ** 2 / c + c / 3.)
                total_energy += self.ground_miu_param[0] * self.backup_ground_force[i][Z] * term
                friction_energy += self.ground_miu_param[0] * self.backup_ground_force[i][Z] * term

        if self.print:
            print(f"LINE SEARCH :: GROUND energy: {ground_energy}")
            print(f"LINE SEARCH :: GROUND FRICTION energy: {friction_energy}")

        backup_energy = total_energy
        if sim_mode and self.P_number > 0:
            self.calculateUnitCenter()
            self.calculateConstraintPoint()

        for i in ti.ndrange(self.constraint_number):
            if sim_mode:
                self.calculateStringLength(i, True, use_history_info=True)
                delta_length = self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]
                # print(i, delta_length)   
                if self.friction_mode_enable[0] == 1:
                    length = 0.0
                    for j in ti.ndrange(self.max_control_length):
                        if self.string_force_current_discount[i, j] > 0.0:
                            length += (self.current_length_per_string[i, j] * self.string_force_current_discount[i, j])
                    delta_length = length - self.target_string_length_decrease[i, 0]
                elif self.friction_mode_enable[0] == 2:
                    if self.backup_delta_length[i] > 0:
                        additional_force_value = self.getStringForce(self.backup_delta_length[i], self.epsilon_string[i], i)
                        length = 0.0
                        for j in ti.ndrange(self.max_control_length):
                            if self.string_force_current_discount[i, j] > 0.0:
                                length += (self.current_length_per_string[i, j] * (1.0 - self.string_force_current_discount[i, j]))
                        total_energy -= additional_force_value * length
                        # print(self.backup_delta_length[i], additional_force_value * length)  
                if delta_length > 0:
                    total_energy += self.getStringEnergy(delta_length, self.epsilon_string[i], i)
                
        if self.print:
            print(f"LINE SEARCH :: STRING energy: {total_energy - backup_energy}")
                
        backup_energy = total_energy
        
        for i in ti.ndrange(self.crease_pairs_num):
            if sim_mode == self.FOLD_SIM:
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]
                target_folding_angle = self.calculateTargetAngle(i, theta)
            
                energy = self.getBendingEnergy(
                    self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                    self.bending_k_list[i], target_folding_angle, self.crease_type[i], False, False, 
                    self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
                )
                # if i == 0:
                #     print(f"E, id:{i}, energy:{energy}, previous_dir:{self.previous_dir[i]}, angle:{self.crease_angle[i]}, target:{target_folding_angle}")
                total_energy += energy

            else:
                equal_torque_angle = 0.0
                # for j, k in ti.ndrange(self.constraint_number, self.max_control_length):
                #     if self.equivalent_torque_index[j, i, k] > 0:
                #         equivalent_arm_dis = self.h_hole + self.string_thickness
                #         angle = abs(self.backup_crease_angle[i]) * tm.pi
                #         if self.backup_crease_angle[i] >= 0.0:
                #             if angle < 2. * self.beta:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - (self.d_hole - self.string_thickness) * tm.sin(angle * 0.5)
                #             else:
                #                 equivalent_arm_dis = 0.0
                #         else:
                #             if angle < 2. * self.kappa:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - 2. * self.panel_bias * tm.sin(angle * 0.5)
                #             else:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(self.kappa) - 2. * self.panel_bias * tm.sin(self.kappa)
                #         equal_torque_angle += self.equivalent_torque_index[j, i, k] * self.backup_string_force_each[j] * equivalent_arm_dis / (self.bending_k_list[i] * self.crease_initial_length[i])
                #     elif self.equivalent_torque_index[j, i, k] < 0:
                #         equivalent_arm_dis = self.h_hole + self.string_thickness
                #         angle = abs(self.backup_crease_angle[i]) * tm.pi
                #         if self.backup_crease_angle[i] <= 0.0:
                #             if angle < 2. * self.beta:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - (self.d_hole - self.string_thickness) * tm.sin(angle * 0.5)
                #             else:
                #                 equivalent_arm_dis = 0.0
                #         else:
                #             if angle < 2. * self.kappa:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - 2. * self.panel_bias * tm.sin(angle * 0.5)
                #             else:
                #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(self.kappa) - 2. * self.panel_bias * tm.sin(self.kappa)
                #         equal_torque_angle += self.equivalent_torque_index[j, i, k] * self.backup_string_force_each[j] * equivalent_arm_dis / (self.bending_k_list[i] * self.crease_initial_length[i])
                        
                # if self.crease_type[i]:
                #     equal_torque_angle = -equal_torque_angle
                
                energy = self.getBendingEnergy(
                    self.get_position_with_index(self.crease_pairs[i, 0]), self.get_position_with_index(self.crease_pairs[i, 1]), 
                    self.get_position_with_index(self.bending_pairs[i, 0]), self.get_position_with_index(self.bending_pairs[i, 1]),
                    self.bending_k_list[i], self.random_folding_target_angle[i] * tm.pi + equal_torque_angle, self.crease_type[i], False, True, 
                    self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i, 1, True
                )
                total_energy += energy
        if self.print:
            print(f"LINE SEARCH :: BENDING energy: {total_energy - backup_energy}")
          
        backup_energy = total_energy

        # if step == 0:
        #     total_energy += 0.5 * self.calculate_M_norm_velocity_plus_ext()
        # else:
        total_energy += 0.5 * self.calculate_M_norm_velocity()   

        for i in ti.ndrange(self.kp_num):
            total_energy += -self.masses[i] * gravitational_acc.dot(self.get_position_with_index(i))
        
        if self.print:
            print(f"LINE SEARCH :: KINEMATIC & POTENTIAL energy: {total_energy - backup_energy}")
            print(f"LINE SEARCH :: TOTAL energy: {total_energy}\n")

        return total_energy

    @ti.kernel
    def stvkForce(self):
        """
        计算STVK（Saint Venant-Kirchhoff）弹性力，包括应力和刚度矩阵。
        Calculate STVK (Saint Venant-Kirchhoff) elastic force, including stress and stiffness matrix.
        """
        for i in ti.ndrange(self.div_indices_num):
            i0 = self.indices[3 * i]
            i1 = self.indices[3 * i + 1]
            i2 = self.indices[3 * i + 2]

            x0 = self.get_position_with_index(i0)
            x1 = self.get_position_with_index(i1)
            x2 = self.get_position_with_index(i2)

            n = (x1 - x0).cross(x2 - x0)
            dndx1, dndx2 = self.calculatedDs(x0, x1, x2, n, n.norm())
            self.dndx1[i] = dndx1
            self.dndx2[i] = dndx2

            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n / (n.norm())])
            f = ds @ self.dm[i]
            self.f[i] = f

            #stvk model
            green_tensor = 0.5 * (f.transpose() @ f - self.I3[None])
            energy_tensor = green_tensor.norm_sqr()
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] * 0.5 * ti.Matrix.trace(green_tensor) ** 2
            
            #force
            piola_temp = 2.0 * self.lames_bonus[0] * green_tensor + self.lames_bonus[1] * ti.Matrix.trace(green_tensor) * self.I3[None]
            self.piola_temp[i] = piola_temp

            piola = f @ piola_temp 
            self.piola[i] = piola

            dm = ti.Vector([self.dm[i][2, 0], self.dm[i][2, 1], self.dm[i][2, 2]])
            self.dm2[i] = dm

            H = -self.A[i] * (piola @ self.dm[i].transpose() + ti.Matrix.cols([dndx1.transpose() @ piola @ dm, dndx2.transpose() @ piola @ dm, [0., 0., 0.]]))

            #Cauchy stress tensor
            avg_A = self.A[i] / 3
            cauchy = piola @ f.transpose() / f.determinant()
            self.cauchy_stress_tensor[i0] += cauchy * avg_A
            self.cauchy_stress_tensor[i1] += cauchy * avg_A
            self.cauchy_stress_tensor[i2] += cauchy * avg_A
            self.cauchy_area[i0] += avg_A
            self.cauchy_area[i1] += avg_A
            self.cauchy_area[i2] += avg_A
            # if i == 0:
            #     print(H)

            f1 = tm.vec3([H[0, 0], H[1, 0], H[2, 0]])
            f2 = tm.vec3([H[0, 1], H[1, 1], H[2, 1]])
            f0 = -f1 - f2

            self.stvk_force[i0] += f0
            self.stvk_force[i1] += f1
            self.stvk_force[i2] += f2

            # self.stress[i0] += f0 / self.A[i]
            # self.stress[i1] += f1 / self.A[i]
            # self.stress[i2] += f2 / self.A[i]

            self.triplets[i] = [3 * i0, 3 * i1, 3 * i2]
            self.total_energy[0] += psi * self.A[i]
            
            # derivative
            self.fill_ddn_ddx(x0, x1, x2, n, n.norm())

            self.fill_K_element(i)

    @ti.func
    def fill_K_element(self, i):
        for j, k, l in ti.ndrange(3, 3, 3):
            dF = self.dDs[j, k] @ self.dm[i]
            dE = 0.5 * (dF.transpose() @ self.f[i] + self.f[i].transpose() @ dF)
            dP = dF @ self.piola_temp[i] + self.f[i] @ (2.0 * self.lames_bonus[0] * dE + self.lames_bonus[1] * ti.Matrix.trace(dE) * self.I3[None])
            ddn1 = self.ddn1_ddx[j, k]
            ddn2 = self.ddn2_ddx[j, k]

            dH = -self.A[i] * (dP @ self.dm[i].transpose() + \
                            ti.Matrix.cols([self.dndx1[i].transpose() @ dP @ self.dm2[i], self.dndx2[i].transpose() @ dP @ self.dm2[i], [0., 0., 0.]]) + \
                                ti.Matrix.cols([ddn1.transpose() @ self.piola[i] @ self.dm2[i], ddn2.transpose() @ self.piola[i] @ self.dm2[i], [0., 0., 0.]]))
            df1 = tm.vec3([dH[0, 0], dH[1, 0], dH[2, 0]])
            df2 = tm.vec3([dH[0, 1], dH[1, 1], dH[2, 1]])
            df0 = -df1 - df2
            self.K_element[i][0 + l, 3 * j + k] = df0[l]
            self.K_element[i][3 + l, 3 * j + k] = df1[l]
            self.K_element[i][6 + l, 3 * j + k] = df2[l] 

    @ti.func
    def calculateTargetAngle(self, i, theta):
        """
        计算目标折叠角度，考虑序列折叠的进度。
        Calculate target folding angle, considering the progress of sequential folding.
        
        :param i: 折痕索引 / Crease index
        :param theta: 当前角度 / Current angle
        :return: 目标折叠角度 / Target folding angle
        """
        target_folding_angle = 0.0
        percent_low = (self.sequence_level[0] - self.crease_level[i]) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_high = (self.sequence_level[0] - self.crease_level[i] + 1.) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_theta = abs(theta) / tm.pi

        if percent_theta < percent_low:
            target_folding_angle = 0.0
        elif percent_theta > percent_high:
            target_folding_angle = tm.pi
        else:
            coeff = self.crease_coeff[i]
            target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
            target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)

        if self.crease_type[i]:
            target_folding_angle = -target_folding_angle
        
        true_level = self.sequence_level[0]
        current_level_need_to_be_fold = self.sequence_level[0] - percent_theta * (self.sequence_level[0] - self.sequence_level[1] + 1.)
        for level in ti.ndrange((self.sequence_level[1], self.sequence_level[0])):
            if level - current_level_need_to_be_fold <= 1 and level - current_level_need_to_be_fold > 0:
                true_level = level
        
        previous_angle = 0.0 if self.crease_level[i] < true_level + 1 else tm.pi
        if self.crease_type[i]:
            previous_angle = -previous_angle
        recover_angle = previous_angle
        find_recover_level = False
        for j in ti.ndrange(self.maximum_level_number):
            if self.recover_level_need[i, j]:
                if self.recover_level[i, j] == true_level + 1:
                    previous_angle = self.recover_angle[i, j]
                elif self.recover_level[i, j] == true_level:
                    recover_angle = self.recover_angle[i, j]
                    find_recover_level = True

        if find_recover_level:
            coeff = self.crease_coeff[i]
            target_folding_angle = previous_angle + (recover_angle - previous_angle) * (true_level - current_level_need_to_be_fold)
            target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)

        # print(i, self.crease_level[i], target_folding_angle)
            
        return target_folding_angle
    
    @ti.func
    def fillBendingMatrix(self, i, total_k, dqdx0, dqdx1, dqdx2, dqdx3, need_damping=True):
        block_00 = (dqdx0.outer_product(dqdx0))
        block_01 = (dqdx0.outer_product(dqdx1))
        block_02 = (dqdx0.outer_product(dqdx2))
        block_03 = (dqdx0.outer_product(dqdx3))

        block_10 = (dqdx1.outer_product(dqdx0))
        block_11 = (dqdx1.outer_product(dqdx1))
        block_12 = (dqdx1.outer_product(dqdx2))
        block_13 = (dqdx1.outer_product(dqdx3))

        block_20 = (dqdx2.outer_product(dqdx0))
        block_21 = (dqdx2.outer_product(dqdx1))
        block_22 = (dqdx2.outer_product(dqdx2))
        block_23 = (dqdx2.outer_product(dqdx3))

        block_30 = (dqdx3.outer_product(dqdx0))
        block_31 = (dqdx3.outer_product(dqdx1))
        block_32 = (dqdx3.outer_product(dqdx2))
        block_33 = (dqdx3.outer_product(dqdx3))

        for j, k in ti.ndrange(3, 3):
            self.K_element_bending[i][j, k + 0] = total_k * block_00[j, k]
            self.K_element_bending[i][j, k + 3] = total_k * block_01[j, k]
            self.K_element_bending[i][j, k + 6] = total_k * block_02[j, k]
            self.K_element_bending[i][j, k + 9] = total_k * block_03[j, k]

            self.K_element_bending[i][j + 3, k + 0] = total_k * block_10[j, k]
            self.K_element_bending[i][j + 3, k + 3] = total_k * block_11[j, k]
            self.K_element_bending[i][j + 3, k + 6] = total_k * block_12[j, k]
            self.K_element_bending[i][j + 3, k + 9] = total_k * block_13[j, k]

            self.K_element_bending[i][j + 6, k + 0] = total_k * block_20[j, k]
            self.K_element_bending[i][j + 6, k + 3] = total_k * block_21[j, k]
            self.K_element_bending[i][j + 6, k + 6] = total_k * block_22[j, k]
            self.K_element_bending[i][j + 6, k + 9] = total_k * block_23[j, k]

            self.K_element_bending[i][j + 9, k + 0] = total_k * block_30[j, k]
            self.K_element_bending[i][j + 9, k + 3] = total_k * block_31[j, k]
            self.K_element_bending[i][j + 9, k + 6] = total_k * block_32[j, k]
            self.K_element_bending[i][j + 9, k + 9] = total_k * block_33[j, k]
            
            if need_damping:
                self.K_element_bending_damping[i][j, k + 0] = -self.bending_k_damping_list[i] * block_00[j, k]
                self.K_element_bending_damping[i][j, k + 3] = -self.bending_k_damping_list[i] * block_01[j, k]
                self.K_element_bending_damping[i][j, k + 6] = -self.bending_k_damping_list[i] * block_02[j, k]
                self.K_element_bending_damping[i][j, k + 9] = -self.bending_k_damping_list[i] * block_03[j, k]

                self.K_element_bending_damping[i][j + 3, k + 0] = -self.bending_k_damping_list[i] * block_10[j, k]
                self.K_element_bending_damping[i][j + 3, k + 3] = -self.bending_k_damping_list[i] * block_11[j, k]
                self.K_element_bending_damping[i][j + 3, k + 6] = -self.bending_k_damping_list[i] * block_12[j, k]
                self.K_element_bending_damping[i][j + 3, k + 9] = -self.bending_k_damping_list[i] * block_13[j, k]

                self.K_element_bending_damping[i][j + 6, k + 0] = -self.bending_k_damping_list[i] * block_20[j, k]
                self.K_element_bending_damping[i][j + 6, k + 3] = -self.bending_k_damping_list[i] * block_21[j, k]
                self.K_element_bending_damping[i][j + 6, k + 6] = -self.bending_k_damping_list[i] * block_22[j, k]
                self.K_element_bending_damping[i][j + 6, k + 9] = -self.bending_k_damping_list[i] * block_23[j, k]

                self.K_element_bending_damping[i][j + 9, k + 0] = -self.bending_k_damping_list[i] * block_30[j, k]
                self.K_element_bending_damping[i][j + 9, k + 3] = -self.bending_k_damping_list[i] * block_31[j, k]
                self.K_element_bending_damping[i][j + 9, k + 6] = -self.bending_k_damping_list[i] * block_32[j, k]
                self.K_element_bending_damping[i][j + 9, k + 9] = -self.bending_k_damping_list[i] * block_33[j, k]

    @ti.kernel
    def bendingForceFoldSim(self, theta: data_type, dt: data_type):
        """
        计算折叠仿真模式下的弯曲力和阻尼力。
        Calculate bending force and damping force in folding simulation mode.
        
        :param theta: 目标折叠角度 / Target folding angle
        :param dt: 时间步长 / Time step
        """
        for i in ti.ndrange(self.crease_pairs_num):
            crease_start_index = self.crease_pairs[i, 0]
            crease_end_index = self.crease_pairs[i, 1]
            related_p1 = self.bending_pairs[i, 0]
            related_p2 = self.bending_pairs[i, 1]
            target_folding_angle = self.calculateTargetAngle(i, theta) * self.percent[0]

            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                self.bending_k_list[i], target_folding_angle, self.crease_type[i], False, False, 
                self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
            )
            # 增加至force
            self.bending_force[crease_start_index] += csf
            self.bending_force[crease_end_index] += cef
            self.bending_force[related_p1] += rpf1
            self.bending_force[related_p2] += rpf2

            # self.stress[crease_start_index] += 2. * csf / (self.bending_pairs_area[i, 0] + self.bending_pairs_area[i, 1])
            # self.stress[crease_end_index] += 2. * cef / (self.bending_pairs_area[i, 0] + self.bending_pairs_area[i, 1])
            # self.stress[related_p1] += rpf1 / self.bending_pairs_area[i, 0]
            # self.stress[related_p2] += rpf2 / self.bending_pairs_area[i, 1]

            theta_dot = (dqdx0.dot((self.x[crease_start_index] - self.back_up_x[crease_start_index])) + dqdx1.dot((self.x[related_p2] - self.back_up_x[related_p2])) \
                      + dqdx2.dot((self.x[crease_end_index] - self.back_up_x[crease_end_index])) + dqdx3.dot((self.x[related_p1] - self.back_up_x[related_p1]))) / dt
            
            self.crease_velocity[i] = theta_dot
            self.bending_damping_force[crease_start_index] += -self.bending_k_damping_list[i] * dqdx0 * theta_dot
            self.bending_damping_force[crease_end_index] += -self.bending_k_damping_list[i] * dqdx2 * theta_dot
            self.bending_damping_force[related_p1] += -self.bending_k_damping_list[i] * dqdx3 * theta_dot
            self.bending_damping_force[related_p2] += -self.bending_k_damping_list[i] * dqdx1 * theta_dot

            self.crease_angle[i] *= n_value
            self.total_energy[0] += energy

            # print(f"F, id:{i}, energy:{energy}, previous_dir:{self.previous_dir[i]}, angle:{self.crease_angle[i]}, target:{target_folding_angle}")

            self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]
            
            bending_k = -self.bending_k_list[i] * self.crease_initial_length[i]
            barrier_k = -self.barrier_k[i]
            total_k = bending_k + barrier_k

            self.fillBendingMatrix(i, total_k, dqdx0, dqdx1, dqdx2, dqdx3, need_damping=True)

    @ti.kernel
    def bendingForceTSASim(self, dt: data_type):
        """
        计算TSA（Twisted String Actuator）仿真模式下的弯曲力。
        Calculate bending force in TSA (Twisted String Actuator) simulation mode.
        
        :param dt: 时间步长 / Time step
        """
        for i in ti.ndrange(self.crease_pairs_num):
            crease_start_index = self.crease_pairs[i, 0]
            crease_end_index = self.crease_pairs[i, 1]
            related_p1 = self.bending_pairs[i, 0]
            related_p2 = self.bending_pairs[i, 1]

            equal_torque_angle = 0.0
            # for j, k in ti.ndrange(self.constraint_number, self.max_control_length):
            #     if self.equivalent_torque_index[j, i, k] > 0:
            #         equivalent_arm_dis = self.h_hole + self.string_thickness
            #         angle = abs(self.backup_crease_angle[i]) * tm.pi
            #         if self.backup_crease_angle[i] >= 0.0:
            #             if angle < 2. * self.beta:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - (self.d_hole - self.string_thickness) * tm.sin(angle * 0.5)
            #             else:
            #                 equivalent_arm_dis = 0.0
            #         else:
            #             if angle < 2. * self.kappa:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - 2. * self.panel_bias * tm.sin(angle * 0.5)
            #             else:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(self.kappa) - 2. * self.panel_bias * tm.sin(self.kappa)
            #         equal_torque_angle += self.equivalent_torque_index[j, i, k] * self.backup_string_force_each[j] * equivalent_arm_dis / (self.bending_k_list[i] * self.crease_initial_length[i])
            #     elif self.equivalent_torque_index[j, i, k] < 0:
            #         equivalent_arm_dis = self.h_hole + self.string_thickness
            #         angle = abs(self.backup_crease_angle[i]) * tm.pi
            #         if self.backup_crease_angle[i] <= 0.0:
            #             if angle < 2. * self.beta:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - (self.d_hole - self.string_thickness) * tm.sin(angle * 0.5)
            #             else:
            #                 equivalent_arm_dis = 0.0
            #         else:
            #             if angle < 2. * self.kappa:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(angle * 0.5) - 2. * self.panel_bias * tm.sin(angle * 0.5)
            #             else:
            #                 equivalent_arm_dis = (self.h_hole + self.string_thickness) * tm.cos(self.kappa) - 2. * self.panel_bias * tm.sin(self.kappa)
            #         equal_torque_angle += self.equivalent_torque_index[j, i, k] * self.backup_string_force_each[j] * equivalent_arm_dis / (self.bending_k_list[i] * self.crease_initial_length[i])
                        
            # if self.crease_type[i]:
            #     equal_torque_angle = -equal_torque_angle
            # print(i, self.crease_type[i], self.crease_angle[i], self.backup_crease_angle[i], equal_torque_angle * 180.0 / tm.pi)
     
            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                self.bending_k_list[i], self.random_folding_target_angle[i] * tm.pi + equal_torque_angle, self.crease_type[i], False, True, 
                self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i, 1, True
            )
            
            # 增加至force
            self.bending_force[crease_start_index] += csf
            self.bending_force[crease_end_index] += cef
            self.bending_force[related_p1] += rpf1
            self.bending_force[related_p2] += rpf2

            # v1 = (self.x[related_p1] - self.x[crease_start_index])
            # v2 = (self.x[related_p2] - self.x[crease_start_index])
            # vc = (self.x[crease_end_index] - self.x[crease_start_index])

            # n1 = tm.normalize(vc.cross(v1))
            # n2 = tm.normalize(v2.cross(vc))

            # csf_n1 = csf.dot(n1) * n1
            # csf_n2 = csf.dot(n2) * n2
            # cef_n1 = cef.dot(n1) * n1
            # cef_n2 = cef.dot(n2) * n2

            # self.stress[crease_start_index] += csf_n1 / self.bending_pairs_area[i, 0] + csf_n2 / self.bending_pairs_area[i, 1]
            # self.stress[crease_end_index] += cef_n1 / self.bending_pairs_area[i, 0] + cef_n2 / self.bending_pairs_area[i, 1]
            # self.stress[related_p1] += rpf1 / self.bending_pairs_area[i, 0]
            # self.stress[related_p2] += rpf2 / self.bending_pairs_area[i, 1]

            theta_dot = (dqdx0.dot((self.x[crease_start_index] - self.back_up_x[crease_start_index])) + dqdx1.dot((self.x[related_p2] - self.back_up_x[related_p2])) \
                      + dqdx2.dot((self.x[crease_end_index] - self.back_up_x[crease_end_index])) + dqdx3.dot((self.x[related_p1] - self.back_up_x[related_p1]))) / dt
            
            self.crease_velocity[i] = theta_dot
            self.bending_damping_force[crease_start_index] += -self.bending_k_damping_list[i] * dqdx0 * theta_dot
            self.bending_damping_force[crease_end_index] += -self.bending_k_damping_list[i] * dqdx2 * theta_dot
            self.bending_damping_force[related_p1] += -self.bending_k_damping_list[i] * dqdx3 * theta_dot
            self.bending_damping_force[related_p2] += -self.bending_k_damping_list[i] * dqdx1 * theta_dot
        
            self.crease_angle[i] *= n_value
            self.total_energy[0] += energy

            self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]

            bending_k = -self.bending_k_list[i] * self.crease_initial_length[i]

            barrier_k = -self.barrier_k[i]

            total_k = bending_k + barrier_k

            self.fillBendingMatrix(i, total_k, dqdx0, dqdx1, dqdx2, dqdx3, need_damping=True)
    
    @ti.kernel
    def facetBendingForce(self, facet_k: data_type, dt: data_type):
        """
        计算面片的弯曲力（厚板模式）。
        Calculate facet bending force (thick panel mode).
        
        :param facet_k: 面片弯曲刚度 / Facet bending stiffness
        :param dt: 时间步长 / Time step
        """
        for i in ti.ndrange(self.facet_crease_pairs_num):
            new_facet_k = facet_k
            if not self.facet_mode_flag[0]:
                # avg_folding_angle = 0.0
                # for j in ti.ndrange(self.crease_pairs_num):
                #     avg_folding_angle += abs(self.crease_angle[j])
                # avg_folding_angle /= self.crease_pairs_num
                # new_facet_k /= (1. + (self.facet_bending_pairs_distance[i] - 1.) * avg_folding_angle)
                new_facet_k /= self.facet_bending_pairs_distance[i]
            crease_start_index = self.facet_crease_pairs[i, 0]
            crease_end_index = self.facet_crease_pairs[i, 1]
            related_p1 = self.facet_bending_pairs[i, 0]
            related_p2 = self.facet_bending_pairs[i, 1]

            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index),
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                new_facet_k, 0.0, 0, False, True, 
                self.facet_bending_pairs_area[i, 0], self.facet_bending_pairs_area[i, 1], self.facet_crease_initial_length[i], -1,
                self.facet_bending_pairs_distance[i]
            )
            
            # 增加至force
            self.facet_bending_force[crease_start_index] += csf
            self.facet_bending_force[crease_end_index] += cef
            self.facet_bending_force[related_p1] += rpf1
            self.facet_bending_force[related_p2] += rpf2

            # v1 = (self.x[related_p1] - self.x[crease_start_index])
            # v2 = (self.x[related_p2] - self.x[crease_start_index])
            # vc = (self.x[crease_end_index] - self.x[crease_start_index])

            # n1 = tm.normalize(vc.cross(v1))
            # n2 = tm.normalize(v2.cross(vc))

            # csf_n1 = csf.dot(n1) * n1
            # csf_n2 = csf.dot(n2) * n2
            # cef_n1 = cef.dot(n1) * n1
            # cef_n2 = cef.dot(n2) * n2

            # self.stress[crease_start_index] += csf_n1 / self.facet_bending_pairs_area[i, 0] + csf_n2 / self.facet_bending_pairs_area[i, 1]
            # self.stress[crease_end_index] += cef_n1 / self.facet_bending_pairs_area[i, 0] + cef_n2 / self.facet_bending_pairs_area[i, 1]
            # self.stress[related_p1] += rpf1 / self.facet_bending_pairs_area[i, 0]
            # self.stress[related_p2] += rpf2 / self.facet_bending_pairs_area[i, 1]

            # theta_dot = (dqdx0.dot((self.x[crease_start_index] - self.back_up_x[crease_start_index])) + dqdx1.dot((self.x[related_p2] - self.back_up_x[related_p2])) \
            #           + dqdx2.dot((self.x[crease_end_index] - self.back_up_x[crease_end_index])) + dqdx3.dot((self.x[related_p1] - self.back_up_x[related_p1]))) / dt
            
            # self.facet_bending_damping_force[crease_start_index] += -self.bending_k_damping_list[i] * dqdx0 * theta_dot
            # self.facet_bending_damping_force[crease_end_index] += -self.bending_k_damping_list[i] * dqdx2 * theta_dot
            # self.facet_bending_damping_force[related_p1] += -self.bending_k_damping_list[i] * dqdx3 * theta_dot
            # self.facet_bending_damping_force[related_p2] += -self.bending_k_damping_list[i] * dqdx1 * theta_dot

            self.total_energy[0] += energy
            
            index = i + self.bending_pairs_num
            self.triplets_bending[index] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]

            bending_k = -new_facet_k * self.facet_crease_initial_length[i]

            total_k = bending_k

            self.fillBendingMatrix(index, total_k, dqdx0, dqdx1, dqdx2, dqdx3, need_damping=False)

    @ti.kernel
    def updateStringParameters(self):
        """
        更新绳索参数，包括刚度系数和奖励系数。
        Update string parameters, including stiffness coefficient and bonus coefficient.
        """
        for i in ti.ndrange(self.constraint_number):
            self.string_params_bonus[i] = 1.0
            self.string_params[i] = self.ks((self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]) / (self.constraint_initial_length[i] + self.additional_length_of_string)) / (self.constraint_initial_length[i] + self.additional_length_of_string)
            # if self.system_type[0] == 3:
            #     base_clip = self.history_maximum_force[i] / self.string_params[i]
            #     if base_clip > 0:
            #         self.string_params_clip[i] = 0.03502 * self.history_maximum_force[i] ** 2 * 0.00333333 * self.constraint_initial_length[i]
            #         if self.string_params_clip[i] > base_clip * 0.6666666:
            #             self.string_params_clip[i] = base_clip * 0.6666666
            #         self.string_params_bonus[i] = (self.history_maximum_force[i] / (base_clip - self.string_params_clip[i])) / self.string_params[i]
            #     else:
            #         self.string_params_clip[i] = 0.0
            #         self.string_params_bonus[i] = 1.0
            #     if self.max_force[i] >= self.history_maximum_force[i]:
            #         self.enable_plasticity[i] = False
            #     else:
            #         self.enable_plasticity[i] = True
                
            # print(f"{i}, {self.max_force[i]}, {self.string_params[i]}, {self.string_params_clip[i]}, {self.string_params_bonus[i]}")
            
    @ti.func
    def getStringEnergy(self, delta_length, epsilon, id):
        """
        计算绳索的弹性势能。
        Calculate the elastic potential energy of the string.
        
        :param delta_length: 绳索伸长量 / String elongation
        :param epsilon: 绳索松弛量 / String slack
        :param id: 绳索ID / String ID
        :return: 弹性势能 / Elastic potential energy
        """
        value = 0.5 * (delta_length - self.string_params_clip[id]) ** 2 - epsilon * (delta_length - self.string_params_clip[id]) / 3. + 1. / 12. * epsilon ** 2
            
        # print(delta_length, id)
        if not self.enable_plasticity[id]:
            value = 0.5 * (delta_length) ** 2 - epsilon * (delta_length) / 3. + 1. / 12. * epsilon ** 2
            if delta_length < epsilon:
                value = -delta_length ** 4 / (12. * epsilon ** 2) + delta_length ** 3 / (3. * epsilon)
        else:
            if delta_length < (epsilon + self.string_params_clip[id]) and delta_length > self.string_params_clip[id]:
                value = -(delta_length - self.string_params_clip[id]) ** 4 / (12. * epsilon ** 2) + (delta_length - self.string_params_clip[id]) ** 3 / (3. * epsilon)
            elif delta_length < self.string_params_clip[id]:
                value = 0.0
        value *= self.string_params_bonus[id]

        return self.string_params[id] * value

    @ti.func
    def getStringForce(self, delta_length, epsilon, id):
        """
        计算绳索的弹性力。
        Calculate the elastic force of the string.
        
        :param delta_length: 绳索伸长量 / String elongation
        :param epsilon: 绳索松弛量 / String slack
        :param id: 绳索ID / String ID
        :return: 弹性力 / Elastic force
        """
        value = delta_length - self.string_params_clip[id] - epsilon / 3.

        if not self.enable_plasticity[id]:
            value = delta_length - epsilon / 3.
            if delta_length < epsilon:
                value = -delta_length ** 3 / (3. * epsilon ** 2) + delta_length ** 2 / epsilon
        else:
            if delta_length < (epsilon + self.string_params_clip[id]) and delta_length > self.string_params_clip[id]:
                value = -(delta_length - self.string_params_clip[id]) ** 3 / (3. * epsilon ** 2) + (delta_length - self.string_params_clip[id]) ** 2 / epsilon
            elif delta_length < self.string_params_clip[id]:
                value = 0.0
        value *= self.string_params_bonus[id]

        # print(f"force: {value}, {delta_length}, {epsilon}")
        return self.string_params[id] * value
    
    @ti.func
    def getStringDF(self, delta_length, epsilon, id):
        """
        计算绳索力的导数（刚度）。
        Calculate the derivative of string force (stiffness).
        
        :param delta_length: 绳索伸长量 / String elongation
        :param epsilon: 绳索松弛量 / String slack
        :param id: 绳索ID / String ID
        :return: 力的导数 / Force derivative
        """
        value = 1.
        if delta_length < epsilon:
            value = -delta_length ** 2 / (epsilon ** 2) + 2 * delta_length / epsilon
        if self.enable_plasticity[id]:
            if delta_length < (epsilon + self.string_params_clip[id]) and delta_length > self.string_params_clip[id]:
                value = -(delta_length - self.string_params_clip[id]) ** 2 / (epsilon ** 2) + 2 * (delta_length - self.string_params_clip[id]) / epsilon
            elif delta_length < self.string_params_clip[id]:
                value = 0.0
        value *= self.string_params_bonus[id]
        
        # print(f"df: {value}, {delta_length}, {epsilon}")
        return tm.sqrt(self.string_params[id] * value)
    
    @ti.func
    def getDn(self, p, c): # derivative of (p-c)/|p-c|
        """
        计算归一化向量 (p-c)/|p-c| 的导数。
        Calculate the derivative of normalized vector (p-c)/|p-c|.
        
        :param p: 点p坐标 / Point p coordinates
        :param c: 点c坐标 / Point c coordinates
        :return: 3x3导数矩阵 / 3x3 derivative matrix
        """
        ret = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        x = p - c
        # if x.norm() > 1e-6:
        #     x_norm = x.norm()
        #     ret = (x.outer_product(x) / x_norm ** 2 - self.I3[None]) / x_norm
        x_norm = x.norm(1e-6)
        ret = (x.outer_product(x) / x_norm ** 2 - self.I3[None]) / x_norm
        # if x.norm() < 1e-6:
        #     print(ret)
        return ret

    @ti.kernel
    def stringForce(self):
        """
        计算绳索力并施加到相关单元上。
        Calculate string force and apply to related units.
        """
        self.exist_grad_number[0] = 0
        self.ddl_ddx_num[0] = 0

        self.calculateUnitCenter()
        self.calculateConstraintPoint()
        for i in ti.ndrange(self.constraint_number):
            self.calculateStringLength(i)
            self.string_force_each[i] = 0.
            delta_length = self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]
     
            if delta_length > 0: # add force
                # print(i, delta_length)   
                # self.delta_length_of_string[0] = delta_length
                self.total_energy[0] += self.getStringEnergy(delta_length, self.epsilon_string[i], i)
                ref_force = self.getStringForce(delta_length, self.epsilon_string[i], i)
                self.string_force_each[i] = ref_force
                
                # print(i, ref_force)
                
                if ref_force > self.max_force[i]:
                    self.max_force[i] = ref_force
                if ref_force > self.overall_string_maximum_force[0]:
                    self.overall_string_maximum_force[0] = ref_force
                ref_bonus = self.getStringDF(delta_length, self.epsilon_string[i], i)

                before_force_dir = tm.vec3([0.0, 0.0, 0.0])

                # if self.print:
                #     print(force, bonus)

                # current_delta_length = self.current_length_per_string[i][0] - self.initial_length_per_string[i][0]
                friction_force = 0.0
                
                start_point = self.constraint_start_point[i]
                candidate_id = self.constraint_start_point_candidate_id[i]
                unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                if unit_connection_id >= 0:
                    index = self.loc_of_unit[candidate_id]
                    kp_id = self.unit_indices[index]
                    unit_kp_num = self.unit_kp_num_list[index]
                    next_kp_id = self.unit_indices[self.unit_control[i, 0]]
                    center = self.calculateCenterPoint3DWithUnitId(next_kp_id, self.unit_control[i, 0])
                    if self.intersection_flag[i, 0]:
                        center = self.intersection_points[i * self.max_control_length]
                    f_basic = (center - start_point) / (center - start_point).norm(1e-6)
                    for k in ti.ndrange(unit_kp_num):
                        self.string_force[kp_id[k]] += f_basic * (ref_force + friction_force) * self.unit_contributions[index][k]
                        self.dldx_force[i, kp_id[k]] += f_basic * ref_bonus * self.unit_contributions[index][k]
                    dndc = self.getDn(center, start_point)

                    self.dndc[self.ddl_ddx_num[0]] = ref_force * dndc
                    self.unit_id_list[self.ddl_ddx_num[0]] = index
                    self.need_ta[self.ddl_ddx_num[0]] = False
                    self.need_tb[self.ddl_ddx_num[0]] = False
                    self.ddl_ddx_num[0] += 1
                    
                    # self.dldx_friction_force[i, 0, kp_id[k]] += f_basic * bonus
                    # if self.print:
                    #     print(f"{i}, start_unit: {unit_connection_id}, center: {center}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, kp_id: {kp_id}")

                # print(f"{i}, force: {force}")

                for j in ti.ndrange(self.max_control_length):
                    if self.unit_control[i, j] != -1:
                        # current_delta_length = self.current_length_per_string[i][j] - self.initial_length_per_string[i][j]
                        # friction_force = self.miu * current_delta_length
                        # self.total_energy[0] += 0.5 * self.miu * (current_delta_length) ** 2

                        kp_id = self.unit_indices[self.unit_control[i, j]]
                        unit_kp_num = self.unit_kp_num_list[self.unit_control[i, j]]
                        hole_direction = self.hole_dir[i, j]
                        nm = self.calculateNormalVectorWithUnitId(kp_id)
                        end_point = self.calculateCenterPoint3DWithUnitId(kp_id, self.unit_control[i, j])
                        force_dir = end_point - start_point
                        # axis_force = tm.vec3([0., 0., 0.])
                            
                        if j == 0: # 1 point
                            penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point

                            dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)
                            self.unit_id_list[self.ddl_ddx_num[0]] = self.unit_control[i, j]
                            self.need_ta[self.ddl_ddx_num[0]] = False
                            # print(f"{i}, {j}, penetration: {penetration}")
                            if self.intersection_flag[i, j]: #penetration
                                if penetration > 0:
                                    # if self.print:
                                    #     print(f"{i}, {j}, penetration")
                                    t = self.intersection_infos[j + i * self.max_control_length][0]
                                    intersect = self.intersection_points[j + i * self.max_control_length]
                                    self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                    
                                    index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                    index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                    id1 = kp_id[index1]
                                    id2 = kp_id[index2]
                                        
                                    f_basic = (start_point - self.outer_PI[self.exist_grad_number[0]]) / (start_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6) + \
                                                (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                    # print(f"{i}, {j}, {f_basic}, 1")
                                                
                                    f1 = (1 - t) * f_basic
                                    f2 = t * f_basic

                                    self.string_force[id1] += f1 * (ref_force + friction_force)
                                    self.string_force[id2] += f2 * (ref_force + friction_force)
                                    self.dldx_force[i, id1] += f1 * ref_bonus
                                    self.dldx_force[i, id2] += f2 * ref_bonus

                                    dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)
                                    dnda = self.getDn(start_point, self.outer_PI[self.exist_grad_number[0]])

                                    self.need_ta[self.ddl_ddx_num[0]] = True
                                    self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                    self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                    self.dnda[self.ddl_ddx_num[0]] = ref_force * dnda

                                    # self.dldx_friction_force[i, j, id1] += f1 * bonus
                                    # self.dldx_friction_force[i, j, id2] += f2 * bonus
                            self.dndc[self.ddl_ddx_num[0]] = ref_force * dndc
                            self.ddl_ddx_num[0] += 1
                            f_basic = (self.outer_PI[self.exist_grad_number[0]] - end_point) / (self.outer_PI[self.exist_grad_number[0]] - end_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 2")
                            for k in ti.ndrange(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (ref_force + friction_force) * self.unit_contributions[self.unit_control[i, j]][k]
                                self.dldx_force[i, kp_id[k]] += f_basic * ref_bonus * self.unit_contributions[self.unit_control[i, j]][k]
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic

                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                            # if self.print:
                            #     print(f"{i}, PI: {self.outer_PI[self.exist_grad_number[0]]}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, kp_id: {kp_id}")
                            self.exist_grad_number[0] += 1

                        else:
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point
                            self.outer_Q[self.exist_grad_number[0] - 1] = end_point
                            self.outer_QI[self.exist_grad_number[0] - 1] = end_point

                            before_unit_id = self.unit_control[i, j - 1]
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.unit_kp_num_list[before_unit_id]
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)
                            dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)

                            dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)
                            self.unit_id_list[self.ddl_ddx_num[0]] = self.unit_control[i, j]
                            self.need_ta[self.ddl_ddx_num[0]] = False
                            self.need_tb[self.ddl_ddx_num[0] - 1] = False
                            
                            force = ref_force
                            bonus = ref_bonus
                            
                            # print(i, j, force)

                            if self.intersection_flag[i, j] and self.intersection_flag2[i, j - 1]: #与1/1板有穿透
                                t = self.intersection_infos[j + i * self.max_control_length][0]
                                t2 = self.intersection_infos2[j - 1 + i * self.max_control_length][0]

                                intersect = self.intersection_points[j + i * self.max_control_length]
                                intersect2 = self.intersection_points2[j - 1 + i * self.max_control_length]

                                self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect2 + before_n * self.string_width * hole_direction
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                if (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]).norm() > 1e-3:
                                    f_basic += (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]) / (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 1-1, {self.outer_QI[self.exist_grad_number[0] - 1]}, {self.outer_PI[self.exist_grad_number[0]]}, {(self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]).norm()}")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                self.need_ta[self.ddl_ddx_num[0]] = True
                                self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                index1 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                if (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]).norm() > 1e-3:
                                    f_basic += (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]) / (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 1-2")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t2) * f_basic
                                f2 = t2 * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                                self.need_tb[self.ddl_ddx_num[0] - 1] = True

                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t2
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t2

                                self.dnda[self.ddl_ddx_num[0]] = force * self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], self.outer_PI[self.exist_grad_number[0]])
                                self.dndb[self.ddl_ddx_num[0] - 1] = force * self.getDn(self.outer_PI[self.exist_grad_number[0]], self.outer_QI[self.exist_grad_number[0] - 1])


                            elif self.intersection_flag[i, j]: #与0/1板有穿透
                                t = self.intersection_infos[j + i * self.max_control_length][0]

                                intersect = self.intersection_points[j + i * self.max_control_length]

                                self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (start_point - self.outer_PI[self.exist_grad_number[0]]) / (start_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6) + \
                                           (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 2, 1")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_PI[self.exist_grad_number[0]], start_point)
                                dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                                self.need_ta[self.ddl_ddx_num[0]] = True
                                self.need_tb[self.ddl_ddx_num[0] - 1] = False

                                self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                self.dnda[self.ddl_ddx_num[0]] = force * self.getDn(start_point, self.outer_PI[self.exist_grad_number[0]])
                                
                            elif self.intersection_flag2[i, j - 1]: #与1/0板有穿透
                                t2 = self.intersection_infos2[j - 1 + i * self.max_control_length][0]
                                # print(t2)
                                intersect2 = self.intersection_points2[j - 1 + i * self.max_control_length]

                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect2 + before_n * self.string_width * hole_direction

                                index1 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6) + \
                                           (end_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (end_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 2, 2")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t2) * f_basic
                                f2 = t2 * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], end_point)

                                self.need_ta[self.ddl_ddx_num[0]] = False
                                self.need_tb[self.ddl_ddx_num[0] - 1] = True

                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t2
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t2

                                self.dndb[self.ddl_ddx_num[0] - 1] = force * self.getDn(end_point, self.outer_QI[self.exist_grad_number[0] - 1])

                            self.dndc2[self.ddl_ddx_num[0] - 1] = force * dndc_before
                            self.dndc[self.ddl_ddx_num[0]] = force * dndc
                            self.ddl_ddx_num[0] += 1

                            f_basic = (self.outer_QI[self.exist_grad_number[0] - 1] - start_point) / (self.outer_QI[self.exist_grad_number[0] - 1] - start_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 3, 1")
                            for k in ti.ndrange(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[before_unit_id][k]
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus * self.unit_contributions[before_unit_id][k]
                                # self.dldx_friction_force[i, j, before_kp_id[k]] += f_basic

                            f_basic = (self.outer_PI[self.exist_grad_number[0]] - end_point) / (self.outer_PI[self.exist_grad_number[0]] - end_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 3, 2")
                            # print(f"fbasic: {f_basic}")
                            for k in ti.ndrange(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[self.unit_control[i, j]][k]
                                self.dldx_force[i, kp_id[k]] += f_basic * bonus * self.unit_contributions[self.unit_control[i, j]][k]
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic

                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                        
                            # if self.print:
                            #     print(f"{i}, start: {start_point}, BEFORE QI: {self.outer_QI[self.exist_grad_number[0] - 1]}, PI: {self.outer_PI[self.exist_grad_number[0]]}, endpoint: {end_point}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, before_kp_id: {before_kp_id}, kp_id: {kp_id}")
                            self.exist_grad_number[0] += 1
                            
                            # print(f"{i}, {j}, {self.ddl_ddx_num[0]}, {penetration}, {penetration2}, {self.intersection_flag[i, j]}, {self.intersection_flag2[i, j - 1]}, {self.need_ta[self.ddl_ddx_num[0] - 1]}, {self.need_tb[self.ddl_ddx_num[0] - 2]}")
                            
                                
                        start_point = end_point
                        before_force_dir = force_dir
                        
                    else: #end
                        if self.constraint_end_point_existence[i]:
                            index = j - 1
                            self.outer_Q[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]
                            self.outer_QI[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]

                            before_unit_id = self.unit_control[i, j - 1]
                            force_dir = self.constraint_end_point[i] - start_point
                            
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.unit_kp_num_list[before_unit_id]
                            before_n = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                            dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                            self.unit_id_list[self.ddl_ddx_num[0]] = before_unit_id
                            self.need_tb[self.ddl_ddx_num[0] - 1] = False

                            #引导力
                            hole_direction = self.hole_dir[i, j - 1]
                            
                            force = ref_force
                            bonus = ref_bonus

                            if self.intersection_flag2[i, index]: #penetration
                                intersect = self.intersection_points2[index + i * self.max_control_length]
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect + before_n * self.string_width * hole_direction  
                                # print(intersect)

                                t = self.intersection_infos2[index + i * self.max_control_length][0]
                                # print(t)
                                
                                index1 = int(self.intersection_infos2[index + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[index + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6) + \
                                        (self.constraint_end_point[i] - self.outer_QI[self.exist_grad_number[0] - 1]) / (self.constraint_end_point[i] - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic
                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)

                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus
                                
                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], self.constraint_end_point[i])

                                self.need_tb[self.ddl_ddx_num[0] - 1] = True
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t

                                self.dndb[self.ddl_ddx_num[0] - 1] = force * dndc

                            self.dndc2[self.ddl_ddx_num[0] - 1] = force * dndc_before
                            self.dndc[self.ddl_ddx_num[0]] = ti.Matrix.zero(data_type, 3, 3)
                            # self.dndc[self.ddl_ddx_num[0]] = force * dndc
                            self.ddl_ddx_num[0] += 1
                            self.end_force[0] = force

                            # avg_vel = tm.vec3([.0, .0, .0])
                            # for k in range(before_kp_num):
                            #     avg_vel += self.get_velocity_with_index(before_kp_id[k]) / before_kp_num
                            # avg_vel /= avg_vel.norm(1e-6)

                            # if penetration <= 0:
                            f_basic = (self.outer_QI[self.exist_grad_number[0] - 1] - start_point) / (self.outer_QI[self.exist_grad_number[0] - 1] - start_point).norm(1e-6)
                            # print(f_basic)

                            for k in ti.ndrange(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[before_unit_id][k]
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus * self.unit_contributions[before_unit_id][k]
                                # self.dldx_friction_force[i, index, before_kp_id[k]] += f_basic

                            # candidate_id = self.constraint_end_point_candidate_id[i]
                            # unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                            # if unit_connection_id >= 0:
                            #     kp_id = self.unit_indices[unit_connection_id]
                            #     unit_kp_num = self.unit_kp_num_list[unit_connection_id]
                            #     center = self.calculateCenterPoint3DWithUnitId(kp_id, unit_connection_id)
                            #     if self.intersection_flag[i, index]:
                            #         center = self.intersection_points[index + i * self.max_control_length]
                            #     f_basic = tm.normalize(center - self.constraint_end_point[i]) 
                            #     for k in ti.ndrange(unit_kp_num):
                            #         self.string_force[kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[unit_connection_id][k]
                            #         self.dldx_force[i, kp_id[k]] += f_basic * bonus * self.unit_contributions[unit_connection_id][k]
                            #         # self.dldx_friction_force[i, index, kp_id[k]] += f_basic
                                
                            #     dndc = self.getDn(center, self.constraint_end_point[i])

                            #     self.dndc[self.ddl_ddx_num[0]] = force * dndc
                            #     self.unit_id_list[self.ddl_ddx_num[0]] = unit_connection_id
                            #     self.need_ta[self.ddl_ddx_num[0]] = False
                            #     self.ddl_ddx_num[0] += 1
    
                        break
    
    @ti.kernel
    def stringForce3(self, step: int):
        self.exist_grad_number[0] = 0
        self.ddl_ddx_num[0] = 1

        self.calculateUnitCenter()
        self.calculateConstraintPoint()
        for i in ti.ndrange(self.constraint_number):
            self.calculateStringLength(i, use_history_info=step)
            self.string_force_each[i] = 0.
            delta_length = self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]

            additional_force_value = 0.0
            length = 0.0
            if self.backup_delta_length[i] > 0:
                additional_force_value = self.getStringForce(self.backup_delta_length[i], self.epsilon_string[i], i)
                for j in ti.ndrange(self.max_control_length):
                    if self.string_force_current_discount[i, j] > 0.0:
                        length += (self.current_length_per_string[i, j] * (1.0 - self.string_force_current_discount[i, j]))
                self.total_energy[0] -= additional_force_value * length
                        
            # print(i, delta_length, self.backup_delta_length[i], additional_force_value * length)  
            if delta_length > 0: # add force
                # self.delta_length_of_string[0] = delta_length
                self.total_energy[0] += (self.getStringEnergy(delta_length, self.epsilon_string[i], i))
                ref_force = self.getStringForce(delta_length, self.epsilon_string[i], i)
                self.string_force_each[i] = ref_force
                
                # print(i, delta_length, ref_force)
                
                if ref_force > self.max_force[i]:
                    self.max_force[i] = ref_force
                if ref_force > self.overall_string_maximum_force[0]:
                    self.overall_string_maximum_force[0] = ref_force
                ref_bonus = self.getStringDF(delta_length, self.epsilon_string[i], i)

                before_force_dir = tm.vec3([0.0, 0.0, 0.0])

                # if self.print:
                #     print(force, bonus)

                # current_delta_length = self.current_length_per_string[i][0] - self.initial_length_per_string[i][0]
                friction_force = -additional_force_value * (1.0 - self.string_force_current_discount[i, 0])
                start_point = self.constraint_start_point[i]

                additional_id = -1
                if self.equivalent_internal_point_id[i] >= 0:
                    additional_id = self.equivalent_internal_point_id[i]
                else:
                    candidate_id = self.constraint_start_point_candidate_id[i]
                    unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                    if unit_connection_id >= 0:
                        additional_id = self.loc_of_unit[candidate_id]

                if additional_id >= 0:
                    kp_id = self.unit_indices[additional_id]
                    start_point = self.calculateCenterPoint3DWithUnitId(kp_id, additional_id)
                    unit_kp_num = self.unit_kp_num_list[additional_id]
                    next_kp_id = self.unit_indices[self.unit_control[i, 0]]
                    center = self.calculateCenterPoint3DWithUnitId(next_kp_id, self.unit_control[i, 0])

                    if self.intersection_flag2_initial[i]:
                        center = self.intersection_points2_initial[i]
                    elif self.intersection_flag[i, 0]:
                        center = self.intersection_points[i * self.max_control_length]

                    f_basic = (center - start_point) / (center - start_point).norm(1e-6)
                    for k in ti.ndrange(unit_kp_num):
                        self.string_force[kp_id[k]] += f_basic * (ref_force + friction_force) * self.unit_contributions[additional_id][k]
                        self.dldx_force[i, kp_id[k]] += f_basic * ref_bonus * self.unit_contributions[additional_id][k]
                    dndc = self.getDn(center, start_point)

                    self.dndc[self.ddl_ddx_num[0]] = (ref_force + friction_force) * dndc
                    self.unit_id_list[self.ddl_ddx_num[0]] = additional_id
                    self.need_ta[self.ddl_ddx_num[0]] = False
                    self.need_tb[self.ddl_ddx_num[0]] = False

                    if self.intersection_flag2_initial[i]:
                        t2 = self.intersection_infos2_initial[i][0]
                        # print(t2)
                        intersect2 = self.intersection_points2_initial[i]

                        index1 = int(self.intersection_infos2_initial[i][1])
                        index2 = int(self.intersection_infos2_initial[i][2])

                        id1 = kp_id[index1]
                        id2 = kp_id[index2]

                        end_point = self.calculateCenterPoint3DWithUnitId(next_kp_id, self.unit_control[i, 0])
                        if self.intersection_flag[i, 0]:
                            end_point = self.intersection_points[i * self.max_control_length]

                        f_basic = (start_point - intersect2) / (start_point - intersect2).norm(1e-6)
                        if (end_point - intersect2).norm() > 1e-6:
                            f_basic += (end_point - intersect2) / (end_point - intersect2).norm()
                            self.dndb[self.ddl_ddx_num[0] - 1] = (ref_force + friction_force) * self.getDn(end_point, intersect2)
                        else:
                            self.dndb[self.ddl_ddx_num[0] - 1] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                        
                        # print(f"fbasic: {f_basic}")
                        f1 = (1 - t2) * f_basic
                        f2 = t2 * f_basic

                        self.string_force[id1] += f1 * (ref_force + friction_force)
                        self.string_force[id2] += f2 * (ref_force + friction_force)
                        # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, j])
                        # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, j])
                        self.dldx_force[i, id1] += f1 * ref_bonus
                        self.dldx_force[i, id2] += f2 * ref_bonus

                        dndc_before = self.getDn(intersect2, start_point)

                        self.need_ta[self.ddl_ddx_num[0]] = False
                        self.need_tb[self.ddl_ddx_num[0] - 1] = True

                        self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t2
                        self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t2

                        # self.dndb[self.ddl_ddx_num[0] - 1] = (ref_force + friction_force) * (-dndc)
  
                        # print(f"{i}, {f_basic}, {dndc_before}, {dndc}, 0")
                else:
                    self.unit_id_list[self.ddl_ddx_num[0]] = -1
                    self.need_ta[self.ddl_ddx_num[0]] = False
                    self.need_tb[self.ddl_ddx_num[0] - 1] = False

                self.ddl_ddx_num[0] += 1
                # self.exist_grad_number[0] += 1
                    
                    # self.dldx_friction_force[i, 0, kp_id[k]] += f_basic * bonus
                    # if self.print:
                    #     print(f"{i}, start_unit: {unit_connection_id}, center: {center}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, kp_id: {kp_id}")

                # print(f"{i}, force: {force}")

                for j in ti.ndrange(self.max_control_length):
                    if self.unit_control[i, j] != -1:
                        # current_delta_length = self.current_length_per_string[i][j] - self.initial_length_per_string[i][j]
                        # friction_force = self.miu * current_delta_length
                        # self.total_energy[0] += 0.5 * self.miu * (current_delta_length) ** 2

                        kp_id = self.unit_indices[self.unit_control[i, j]]
                        unit_kp_num = self.unit_kp_num_list[self.unit_control[i, j]]
                        hole_direction = self.hole_dir[i, j]
                        nm = self.calculateNormalVectorWithUnitId(kp_id)
                        end_point = self.calculateCenterPoint3DWithUnitId(kp_id, self.unit_control[i, j])
                        force_dir = end_point - start_point
                        # axis_force = tm.vec3([0., 0., 0.])
                        friction_force = -additional_force_value * (1.0 - self.string_force_current_discount[i, j])    
                        if j == 0: # 1 point                               
                            if self.intersection_flag2_initial[i]:
                                start_point = self.intersection_points2_initial[i]
                            # penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point

                            dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                            self.need_ta[self.ddl_ddx_num[0]] = False
                            self.dnda[self.ddl_ddx_num[0]] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                            
                            self.need_tb[self.ddl_ddx_num[0] - 1] = False
                            self.dndb[self.ddl_ddx_num[0] - 1] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

                            self.unit_id_list[self.ddl_ddx_num[0]] = self.unit_control[i, j]
                            # print(f"{i}, {j}")
                            if self.intersection_flag[i, j]: #penetration
                                # if penetration > 0:
                                    # if self.print:
                                    #     print(f"{i}, {j}, penetration")
                                t = self.intersection_infos[j + i * self.max_control_length][0]
                                intersect = self.intersection_points[j + i * self.max_control_length]
                                self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                if (start_point - self.outer_PI[self.exist_grad_number[0]]).norm() > 1e-6:
                                    f_basic += (start_point - self.outer_PI[self.exist_grad_number[0]]) / (start_point - self.outer_PI[self.exist_grad_number[0]]).norm()
                                    self.dnda[self.ddl_ddx_num[0]] = (ref_force + friction_force) * self.getDn(start_point, self.outer_PI[self.exist_grad_number[0]])
                                else:
                                    self.dnda[self.ddl_ddx_num[0]] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                                # print(f"{i}, {j}, {f_basic}, 1")
                                            
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (ref_force + friction_force)
                                self.string_force[id2] += f2 * (ref_force + friction_force)
                                self.dldx_force[i, id1] += f1 * ref_bonus
                                self.dldx_force[i, id2] += f2 * ref_bonus

                                dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                                self.need_ta[self.ddl_ddx_num[0]] = True
                                self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                # self.dnda[self.ddl_ddx_num[0]] = (ref_force + friction_force) * dnda
                                # print(f"{i}, {j}, {f_basic}, {dnda}, {dndc}, 0")              
                                    # self.dldx_friction_force[i, j, id1] += f1 * bonus
                                    # self.dldx_friction_force[i, j, id2] += f2 * bonus

                            self.dndc[self.ddl_ddx_num[0]] = (ref_force + friction_force) * dndc
                            self.ddl_ddx_num[0] += 1
                            f_basic = (self.outer_PI[self.exist_grad_number[0]] - end_point) / (self.outer_PI[self.exist_grad_number[0]] - end_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 2")
                            for k in ti.ndrange(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (ref_force + friction_force) * self.unit_contributions[self.unit_control[i, j]][k]
                                self.dldx_force[i, kp_id[k]] += f_basic * ref_bonus * self.unit_contributions[self.unit_control[i, j]][k]
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic

                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                            # if self.print:
                            #     print(f"{i}, PI: {self.outer_PI[self.exist_grad_number[0]]}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, kp_id: {kp_id}")
                            self.exist_grad_number[0] += 1

                        else:
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point
                            self.outer_Q[self.exist_grad_number[0] - 1] = end_point
                            self.outer_QI[self.exist_grad_number[0] - 1] = end_point

                            before_unit_id = self.unit_control[i, j - 1]
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.unit_kp_num_list[before_unit_id]
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)
                            dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)

                            dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)
                            self.unit_id_list[self.ddl_ddx_num[0]] = self.unit_control[i, j]
                            self.need_ta[self.ddl_ddx_num[0]] = False
                            self.dnda[self.ddl_ddx_num[0]] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                            
                            self.need_tb[self.ddl_ddx_num[0] - 1] = False
                            self.dndb[self.ddl_ddx_num[0] - 1] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                            
                            force = ref_force
                            bonus = ref_bonus
                            
                            # print(i, j, self.intersection_flag[i, j], self.intersection_flag2[i, j - 1])

                            if self.intersection_flag[i, j] and self.intersection_flag2[i, j - 1]: #与1/1板有穿透
                                t = self.intersection_infos[j + i * self.max_control_length][0]
                                t2 = self.intersection_infos2[j - 1 + i * self.max_control_length][0]

                                intersect = self.intersection_points[j + i * self.max_control_length]
                                intersect2 = self.intersection_points2[j - 1 + i * self.max_control_length]

                                self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect2 + before_n * self.string_width * hole_direction
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                if (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]).norm() > 1e-6:
                                    f_basic += (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]) / (self.outer_QI[self.exist_grad_number[0] - 1] - self.outer_PI[self.exist_grad_number[0]]).norm()
                                    self.dnda[self.ddl_ddx_num[0]] = (force + friction_force) * self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], self.outer_PI[self.exist_grad_number[0]])
                                else:
                                    self.dnda[self.ddl_ddx_num[0]] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, j])
                                # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, j])
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                self.need_ta[self.ddl_ddx_num[0]] = True
                                self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                index1 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                if (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]).norm() > 1e-6:
                                    f_basic += (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]) / (self.outer_PI[self.exist_grad_number[0]] - self.outer_QI[self.exist_grad_number[0] - 1]).norm()
                                    self.dndb[self.ddl_ddx_num[0] - 1] = (force + friction_force) * self.getDn(self.outer_PI[self.exist_grad_number[0]], self.outer_QI[self.exist_grad_number[0] - 1])
                                else:
                                    self.dndb[self.ddl_ddx_num[0] - 1] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                                # print(f"{i}, {j}, {f_basic}, 1-2")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t2) * f_basic
                                f2 = t2 * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, j])
                                # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, j])
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                                self.need_tb[self.ddl_ddx_num[0] - 1] = True
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t2
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t2

                            elif self.intersection_flag[i, j]: #与0/1板有穿透
                                t = self.intersection_infos[j + i * self.max_control_length][0]

                                intersect = self.intersection_points[j + i * self.max_control_length]

                                self.outer_PI[self.exist_grad_number[0]] = intersect + nm * self.string_width * hole_direction
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect + nm * self.string_width * hole_direction
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][2])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (start_point - self.outer_PI[self.exist_grad_number[0]]) / (start_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6) + \
                                           (end_point - self.outer_PI[self.exist_grad_number[0]]) / (end_point - self.outer_PI[self.exist_grad_number[0]]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 2, 1")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, j])
                                # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, j])
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_PI[self.exist_grad_number[0]], start_point)
                                dndc = self.getDn(self.outer_PI[self.exist_grad_number[0]], end_point)

                                self.need_ta[self.ddl_ddx_num[0]] = True
                                self.need_tb[self.ddl_ddx_num[0] - 1] = False

                                self.ta_vector[self.ddl_ddx_num[0]][index1] = 1 - t
                                self.ta_vector[self.ddl_ddx_num[0]][index2] = t

                                self.dnda[self.ddl_ddx_num[0]] = (force + friction_force) * self.getDn(start_point, self.outer_PI[self.exist_grad_number[0]])
                                
                            elif self.intersection_flag2[i, j - 1]: #与1/0板有穿透
                                t2 = self.intersection_infos2[j - 1 + i * self.max_control_length][0]
                                # print(t2)
                                intersect2 = self.intersection_points2[j - 1 + i * self.max_control_length]

                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect2 + before_n * self.string_width * hole_direction
                                self.outer_PI[self.exist_grad_number[0]] = intersect2 + before_n * self.string_width * hole_direction

                                index1 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[j - 1 + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6) + \
                                           (end_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (end_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                # print(f"{i}, {j}, {f_basic}, 2, 2")
                                
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t2) * f_basic
                                f2 = t2 * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, j])
                                # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, j])
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus

                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], end_point)

                                self.need_ta[self.ddl_ddx_num[0]] = False
                                self.need_tb[self.ddl_ddx_num[0] - 1] = True

                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t2
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t2

                                self.dndb[self.ddl_ddx_num[0] - 1] = (force + friction_force) * self.getDn(end_point, self.outer_QI[self.exist_grad_number[0] - 1])

                            self.dndc2[self.ddl_ddx_num[0] - 1] = (force + friction_force) * dndc_before
                            self.dndc[self.ddl_ddx_num[0]] = (force + friction_force) * dndc
                            self.ddl_ddx_num[0] += 1

                            f_basic = (self.outer_QI[self.exist_grad_number[0] - 1] - start_point) / (self.outer_QI[self.exist_grad_number[0] - 1] - start_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 3, 1")
                            for k in ti.ndrange(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[before_unit_id][k]
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus * self.unit_contributions[before_unit_id][k]
                                # self.hole_friction_force[before_kp_id[k]] -= f_basic * force * (1 - self.string_force_current_discount[i, j])
                                # self.dldx_friction_force[i, j, before_kp_id[k]] += f_basic

                            f_basic = (self.outer_PI[self.exist_grad_number[0]] - end_point) / (self.outer_PI[self.exist_grad_number[0]] - end_point).norm(1e-6)
                            # print(f"{i}, {j}, {f_basic}, 3, 2")
                            # print(f"fbasic: {f_basic}")
                            for k in ti.ndrange(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[self.unit_control[i, j]][k]
                                self.dldx_force[i, kp_id[k]] += f_basic * bonus * self.unit_contributions[self.unit_control[i, j]][k]
                                # self.hole_friction_force[kp_id[k]] -= f_basic * force * (1 - self.string_force_current_discount[i, j])
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic

                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                        
                            # if self.print:
                            #     print(f"{i}, start: {start_point}, BEFORE QI: {self.outer_QI[self.exist_grad_number[0] - 1]}, PI: {self.outer_PI[self.exist_grad_number[0]]}, endpoint: {end_point}, intersection: {self.intersection_flag[i, 0]}, f_basic: {f_basic}, before_kp_id: {before_kp_id}, kp_id: {kp_id}")
                            self.exist_grad_number[0] += 1
                            
                            # print(f"{i}, {j}, {self.ddl_ddx_num[0]}, {penetration}, {penetration2}, {self.intersection_flag[i, j]}, {self.intersection_flag2[i, j - 1]}, {self.need_ta[self.ddl_ddx_num[0] - 1]}, {self.need_tb[self.ddl_ddx_num[0] - 2]}")
                            
                        start_point = end_point
                        before_force_dir = force_dir
                        
                    else: #end
                        if self.constraint_end_point_existence[i]:
                            index = j - 1
                            self.outer_Q[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]
                            self.outer_QI[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]

                            before_unit_id = self.unit_control[i, j - 1]
                            force_dir = self.constraint_end_point[i] - start_point
                            
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.unit_kp_num_list[before_unit_id]
                            before_n = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                            dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                            self.unit_id_list[self.ddl_ddx_num[0]] = before_unit_id
                            self.need_tb[self.ddl_ddx_num[0] - 1] = False

                            #引导力
                            hole_direction = self.hole_dir[i, j - 1]
                            
                            force = ref_force
                            bonus = ref_bonus


                            if self.intersection_flag2[i, index]: #penetration
                                intersect = self.intersection_points2[index + i * self.max_control_length]
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect + before_n * self.string_width * hole_direction  
                                # print(intersect)

                                t = self.intersection_infos2[index + i * self.max_control_length][0]
                                # print(t)
                                
                                index1 = int(self.intersection_infos2[index + i * self.max_control_length][1])
                                index2 = int(self.intersection_infos2[index + i * self.max_control_length][2])

                                id1 = before_kp_id[index1]
                                id2 = before_kp_id[index2]

                                f_basic = (start_point - self.outer_QI[self.exist_grad_number[0] - 1]) / (start_point - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6) + \
                                        (self.constraint_end_point[i] - self.outer_QI[self.exist_grad_number[0] - 1]) / (self.constraint_end_point[i] - self.outer_QI[self.exist_grad_number[0] - 1]).norm(1e-6)
                                
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic
                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                # self.hole_friction_force[id1] -= f1 * force * (1 - self.string_force_current_discount[i, index])
                                # self.hole_friction_force[id2] -= f2 * force * (1 - self.string_force_current_discount[i, index])
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus
                                
                                dndc_before = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], start_point)
                                dndc = self.getDn(self.outer_QI[self.exist_grad_number[0] - 1], self.constraint_end_point[i])

                                self.need_tb[self.ddl_ddx_num[0] - 1] = True
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index1] = 1 - t
                                self.tb_vector[self.ddl_ddx_num[0] - 1][index2] = t

                                self.dndb[self.ddl_ddx_num[0] - 1] = (force + friction_force) * dndc

                            self.dndc2[self.ddl_ddx_num[0] - 1] = (force + friction_force) * dndc_before
                            self.dndc[self.ddl_ddx_num[0]] = ti.Matrix.zero(data_type, 3, 3)
                            # self.dndc[self.ddl_ddx_num[0]] = force * dndc
                            self.ddl_ddx_num[0] += 1
                            self.end_force[0] = (force + friction_force)

                            # avg_vel = tm.vec3([.0, .0, .0])
                            # for k in range(before_kp_num):
                            #     avg_vel += self.get_velocity_with_index(before_kp_id[k]) / before_kp_num
                            # avg_vel /= avg_vel.norm(1e-6)

                            # if penetration <= 0:
                            f_basic = (self.outer_QI[self.exist_grad_number[0] - 1] - start_point) / (self.outer_QI[self.exist_grad_number[0] - 1] - start_point).norm(1e-6)
                            # print(f_basic)

                            for k in ti.ndrange(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[before_unit_id][k]
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus * self.unit_contributions[before_unit_id][k]
                                # self.hole_friction_force[before_kp_id[k]] -= f_basic * force * (1 - self.string_force_current_discount[i, index])
                                # self.dldx_friction_force[i, index, before_kp_id[k]] += f_basic

                            # candidate_id = self.constraint_end_point_candidate_id[i]
                            # unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                            # if unit_connection_id >= 0:
                            #     kp_id = self.unit_indices[unit_connection_id]
                            #     unit_kp_num = self.unit_kp_num_list[unit_connection_id]
                            #     center = self.calculateCenterPoint3DWithUnitId(kp_id, unit_connection_id)
                            #     if self.intersection_flag[i, index]:
                            #         center = self.intersection_points[index + i * self.max_control_length]
                            #     f_basic = tm.normalize(center - self.constraint_end_point[i]) 
                            #     for k in ti.ndrange(unit_kp_num):
                            #         self.string_force[kp_id[k]] += f_basic * (force + friction_force) * self.unit_contributions[unit_connection_id][k]
                            #         self.dldx_force[i, kp_id[k]] += f_basic * bonus * self.unit_contributions[unit_connection_id][k]
                            #         # self.dldx_friction_force[i, index, kp_id[k]] += f_basic
                                
                            #     dndc = self.getDn(center, self.constraint_end_point[i])

                            #     self.dndc[self.ddl_ddx_num[0]] = force * dndc
                            #     self.unit_id_list[self.ddl_ddx_num[0]] = unit_connection_id
                            #     self.need_ta[self.ddl_ddx_num[0]] = False
                            #     self.ddl_ddx_num[0] += 1
    
                        break
                    
    @ti.kernel
    def groundForce(self, mode: int, step: int):
        # kp_id = self.unit_indices[0]
        self.ground_attachment[0] = False
        for i in ti.ndrange(self.kp_num):
            # if i == 28 or i == 29 or i == 30 or i == 33:
            #     self.kp_add_height[i] = True
            # else:
                self.kp_add_height[i] = False
        # if mode == 1:
        #     if 1:
        #         kp_id = self.unit_indices[self.constraint_start_point_candidate_connection[0]]
        #         for i in ti.ndrange(self.unit_edge_max):
        #             if kp_id[i] != -1:
        #                 self.kp_add_height[kp_id[i]] = True
                
        for i in ti.ndrange(self.kp_num):
            ground_barrier = self.ground_barrier
            if self.kp_add_height[i]:
                ground_barrier += self.additional_height[0]
            if self.x[i][Z] < ground_barrier:
                self.ground_attachment[0] = True
                barrier_exceed = self.x[i][Z] - ground_barrier
                barrier_left = self.x[i][Z] + self.collision_d - (ground_barrier - 1.)
                if self.x[i][Z] > ground_barrier - 1.:
                    self.total_energy[0] += -self.ground_collision_indice * barrier_exceed ** 2 * tm.log(barrier_left / ground_barrier)
                    self.ground_force[i] = self.ground_collision_indice * (2 * barrier_exceed * tm.log(barrier_left / ground_barrier) + barrier_exceed ** 2 / barrier_left) * tm.vec3([0., 0., 1.])
                    self.df_ground_force[i] = self.ground_collision_indice * (4. * barrier_exceed / barrier_left + 2 * tm.log(barrier_left / ground_barrier) - barrier_exceed ** 2 / barrier_left ** 2)
                else:
                    self.total_energy[0] += self.ground_barrier_energy_maximum - (2. * self.ground_force_maximum + (self.x[i][Z] - (ground_barrier - 1.)) * self.df_ground_force_maximum) * (self.x[i][Z] - (ground_barrier - 1.)) * 0.5
                    self.ground_force[i] = (self.ground_force_maximum + (self.x[i][Z] - (ground_barrier - 1.)) * self.df_ground_force_maximum) * tm.vec3([0., 0., 1.])
                    self.df_ground_force[i] = self.df_ground_force_maximum        

    # @ti.func
    # def friction_f0(self, x, c):
    #     return x if x >= c else -x ** 3 / (3. * c ** 2) + x ** 2 / c + c / 3.

    # @ti.func
    # def friction_f1(self, x, c):
    #     return 1.0 if x >= c else -x ** 2 / (c ** 2) + 2. * x / c
    
    # @ti.func
    # def friction_f2(self, x, c):
    #     return 0.0 if x >= c else -2 * x / (c ** 2) + 2. / c

    @ti.kernel
    def frictionForce(self):
        """
        计算地面摩擦力。
        Calculate ground friction force.
        """
        c = self.epsilon_v[0]
        for i in ti.ndrange(self.kp_num):
            # ground_barrier = self.ground_barrier
            # if self.kp_add_height[i]:
            #     ground_barrier += self.additional_height[0]
            # if self.back_up_x[i][Z] < ground_barrier:
            self.u[i] = [self.x[i][X] - self.back_up_x[i][X], self.x[i][Y] - self.back_up_x[i][Y]]
            ui_norm = self.u[i].norm()
            term1 = ui_norm
            term2 = self.u[i]
            if ui_norm < c:
                term1 = (-ui_norm ** 3 / (3. * c ** 2) + ui_norm ** 2 / c + c / 3.)
                term2 = (-ui_norm ** 2 / c ** 2 + 2. * ui_norm / c) * self.u[i]
            self.total_energy[0] += self.ground_miu_param[0] * self.backup_ground_force[i][Z] * term1
            friction_force = 0.0
            if ui_norm > 0:
                friction_force = -self.ground_miu_param[0] * self.backup_ground_force[i][Z]
                self.ground_friction[i][X] = friction_force * (term2[X] / ui_norm)
                self.ground_friction[i][Y] = friction_force * (term2[Y] / ui_norm)
            # print(f"{i}, {c}, {self.u[i]}, {ui_norm}, {term1}, {term2}, {friction_force}, {self.ground_friction}")

    @ti.kernel
    def dfdxConnectionForce(self): 
        """
        计算单元间连接力及其导数。
        Calculate connection force between units and its derivative.
        """
        for k, i, j in ti.ndrange(self.P_number, 4, self.unit_edge_max):
            if self.connection_number[0] > 0:
                if self.constraint_start_point_candidate_connection[k] >= 0:
                    source_indices = self.unit_indices[self.loc_of_unit[k]]
                    connected_indices = self.unit_indices[self.constraint_start_point_candidate_connection[k]]
                    if connected_indices[j] != -1:
                        current_distance = self.x[connected_indices[j]] - self.x[source_indices[i]]
                        initial_distance = self.original_vertices[connected_indices[j]] - self.original_vertices[source_indices[i]]
                        delta_l = current_distance.norm() - initial_distance.norm()
                        self.total_energy[0] += 0.5 * self.controller_k * delta_l ** 2
                        force = self.controller_k * delta_l * current_distance / current_distance.norm(1e-6)
                        self.connection_force[source_indices[i]] += force
                        self.connection_force[connected_indices[j]] -= force
                        self.dfdx_connection[k, i, j] = -self.controller_k / current_distance.norm(1e-6) * (initial_distance.norm() * current_distance.outer_product(current_distance) / current_distance.norm(1e-6) ** 2 + delta_l * self.I3[None])
        
        for k, i, j in ti.ndrange(self.unit_indices_num * self.unit_edge_max, self.unit_edge_max, self.unit_edge_max):
            if self.thick_panel_additional_connection_id[k][0] != -1:
                source_indices = self.unit_indices[self.thick_panel_additional_connection_id[k][0]]
                connected_indices = self.unit_indices[self.thick_panel_additional_connection_id[k][1]]
                if source_indices[i] != -1 and connected_indices[j] != -1:
                    current_distance = self.x[connected_indices[j]] - self.x[source_indices[i]]
                    initial_distance = self.original_vertices[connected_indices[j]] - self.original_vertices[source_indices[i]]
                    delta_l = current_distance.norm() - initial_distance.norm()
                    self.total_energy[0] += 0.5 * self.thick_panel_k * delta_l ** 2
                    force = self.thick_panel_k * delta_l * current_distance / current_distance.norm(1e-6)
                    self.connection_force[source_indices[i]] += force
                    self.connection_force[connected_indices[j]] -= force
                    self.dfdx_thick_panel_connection[k, i, j] = -self.thick_panel_k / current_distance.norm(1e-6) * (initial_distance.norm() * current_distance.outer_product(current_distance) / current_distance.norm(1e-6) ** 2 + delta_l * self.I3[None])
        
    @ti.kernel
    def fixForce(self, mode: int, step: int):
        """
        计算固定约束力，用于固定特定单元。
        Calculate fixed constraint force to fix specific units.
        
        :param mode: 固定模式 / Fix mode
        :param step: 当前步数 / Current step
        """
        if mode == 1:
            if step == 0:
                self.fix_id_list[0] = self.constraint_start_point_candidate_connection[0]
            else:
                self.fix_id_list[0] = -1
        for j, i in ti.ndrange(self.MAXIMUM_FIX_PANEL, self.unit_edge_max):
            if self.fix_id_list[j] >= 0:
                fix_unit_indices = self.unit_indices[self.fix_id_list[j]]
                kp_id = fix_unit_indices[i]
                # area = 0.
                # for k in ti.ndrange(self.unit_kp_num_list[self.fix_id_list[j]] - 2):
                #     x1 = self.x[0]
                #     x2 = self.x[k + 1]
                #     x3 = self.x[k + 2]
                #     area += ((x2 - x1).cross(x3 - x1)).norm() * 0.5
                if kp_id != -1:
                    self.total_energy[0] += 0.5 * self.controller_k * (self.x[kp_id] - self.original_vertices[kp_id]).norm_sqr()
                    self.fix_force[kp_id] += -self.controller_k * (self.x[kp_id] - self.original_vertices[kp_id])
                        # print((self.x[kp_id] - self.original_vertices[kp_id]).norm())
    
    @ti.kernel
    def clearNodalForce(self):
        for i in ti.ndrange(self.kp_num):
            self.stvk_force[i] = [0., 0., 0.]
            self.bending_force[i] = [0., 0., 0.]
            self.bending_damping_force[i] = [0., 0., 0.]
            self.facet_bending_force[i] = [0., 0., 0.]
            self.facet_bending_damping_force[i] = [0., 0., 0.]
            # self.viscosity_force.fill(0.)
            self.string_force[i] = [0., 0., 0.]
            self.ground_force[i] = [0., 0., 0.]
            self.ground_friction[i] = [0., 0., 0.]
            self.u[i] = [0., 0.]
            self.df_ground_force[i] = 0.
            # self.dldx_force_single.fill(0.)
            self.connection_force[i] = [0., 0., 0.]
            self.fix_force[i] = [0., 0., 0.]
            # self.record_force.fill(0.)
            # self.stress.fill(0.)
            self.cauchy_stress_tensor[i] = tm.mat3([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            self.cauchy_area[i] = 0.

    @ti.kernel
    def clearForce(self):
        """
        清除所有力场和能量，为新一轮计算做准备。
        Clear all force fields and energy in preparation for a new round of calculation.
        """
        self.total_energy[0] = 0.0
        self.overall_string_maximum_force[0] = 0.0
        self.max_force.fill(0.)
        self.dfdx_connection.fill(0.)
        self.dfdx_thick_panel_connection.fill(0.)
        self.dldx_force.fill(0.)
        # self.ta_vector.fill(0.)
        # self.tb_vector.fill(0.)
        # self.dndc.fill(0.)
        # self.dndc2.fill(0.)
        # self.dnda.fill(0.)
        # self.dndb.fill(0.)
        self.unit_id_list.fill(-1)
        # self.need_ta.fill(False)
        # self.need_tb.fill(False)

    @ti.kernel
    def mergeForce(self, gravitational_acc: tm.vec3, mode: int, step: int, t: data_type):
        if step == 0:
            self.calculate_M_inv_f_ext()
            self.total_energy[0] += 0.5 * self.calculate_M_norm_velocity_plus_ext(t)
        else:
            self.total_energy[0] += 0.5 * self.calculate_M_norm_velocity()

        for i in ti.ndrange(self.kp_num): 
            self.force[i] = self.stvk_force[i] + self.bending_force[i] + self.facet_bending_force[i] + self.string_force[i] + self.ground_force[i] + self.ground_friction[i] + \
                self.bending_damping_force[i] + self.facet_bending_damping_force[i] + self.connection_force[i] + self.fix_force[i]# + self.backup_hole_friction_force[i]
            for j in ti.ndrange(self.constraint_number):
                if self.max_force[j] > self.history_maximum_force[j]:
                    self.history_maximum_force[j] = self.max_force[j]
            # print(i, self.string_force[i])
            self.record_force[i] = self.bending_force[i] + self.stvk_force[i] + self.facet_bending_force[i] + self.fix_force[i]
            self.force[i] += gravitational_acc * self.masses[i]

            cauchy_real = ti.Matrix.zero(data_type, 3, 3)
            mises = 0.
            if self.cauchy_area[i] > 0.:
                cauchy_real = self.cauchy_stress_tensor[i] / self.cauchy_area[i]
                mises = tm.sqrt(((cauchy_real[X, X] - cauchy_real[Y, Y]) ** 2 + \
                                (cauchy_real[Y, Y] - cauchy_real[Z, Z]) ** 2 + \
                                (cauchy_real[Z, Z] - cauchy_real[X, X]) ** 2 + \
                                    6 * (cauchy_real[X, Y] ** 2 + cauchy_real[X, Z] ** 2 + cauchy_real[Y, Z] ** 2)) * 0.5) 
            self.mises[i] = mises

            # ext_f = tm.vec3([0., 0., 0.])
            # if step == 0:
            #     ext_f = tm.vec3([self.M_inv_f_ext[3 * i], self.M_inv_f_ext[3 * i + 1], self.M_inv_f_ext[3 * i + 2]]) * t
            self.total_energy[0] += -self.masses[i] * gravitational_acc.dot(self.get_position_with_index(i))
        
        # total_force = tm.vec3([0., 0., 0.])
        # for i in ti.ndrange(self.kp_num):
        #     total_force += self.field_force
        # print(f"Total force: {total_force}")
    
    @ti.func
    def backup_v(self):
        for i in ti.ndrange(self.kp_num):
            self.back_up_v[i] = self.v[i]

    @ti.func
    def backup_startpoint(self):
        for i in ti.ndrange(self.constraint_number):
            self.back_up_start_point[i] = self.constraint_start_point[i]

    @ti.func
    def backup_angle(self):
        for i in ti.ndrange(self.crease_pairs_num):
            self.backup_crease_angle[i] = self.crease_angle[i]
            self.backup_crease_velocity[i] = self.crease_velocity[i]

    @ti.func
    def backup_x(self):
        for i in ti.ndrange(self.kp_num):
            self.back_up_x[i] = self.x[i]
    
    @ti.func
    def backup_startx(self):
        for i in ti.ndrange(self.kp_num):
            self.start_x[i] = self.x[i]
        for i in ti.ndrange(self.constraint_number):
            self.start_start_point[i] = self.constraint_start_point[i]

    @ti.kernel
    def backup_xv(self):
        self.backup_v()
        self.backup_x()
        
    @ti.kernel
    def backup_string_force(self):
        for i in ti.ndrange(self.constraint_number):
            self.backup_string_force_each[i] = self.string_force_each[i]

    @ti.kernel
    def backupGroundForce(self):
        for i in ti.ndrange(self.kp_num):
            self.backup_ground_force[i] = self.ground_force[i]

    @ti.kernel
    def backup_equivalent(self):
        self.backup_angle()
        for i in ti.ndrange(self.bending_pairs_num):
            self.backup_enable_equivalent_torque[i] = self.enable_equivalent_torque[i]
            angle = abs(self.crease_angle[i]) * 180.0
            self.bending_k_list[i] = self.kb(angle)

            if (self.crease_angle[i] >= 0. and self.crease_type[i] == VALLEY) or (self.crease_angle[i] < 0. and self.crease_type[i] == MOUNTAIN):
                self.barrier_coeff_list[i] = self.getBarrierBonus(abs(self.crease_angle[i]) * tm.pi, abs(self.folding_angle_upper_bound[i]))
            else:
                self.barrier_coeff_list[i] = self.getBarrierBonus(abs(self.crease_angle[i]) * tm.pi, abs(self.folding_angle_lower_bound[i]))

            self.bending_k_damping_list[i] = self.kb_damping(angle, abs(self.backup_crease_velocity[i])) * self.crease_initial_length[i]
            if self.const_stiff_of_crease_param[0]:
                self.bending_k_list[i] = self.bending_param[0]
                self.bending_k_damping_list[i] = self.k_bending_damping_param[0]

    @ti.kernel
    def update_vertices(self):
        amp = 1.
        for i in ti.ndrange(self.kp_num):
            self.vertices[i] = self.get_position_with_index(i)
            force = self.mises[i]
            # print(i, force)
            if force > amp + 1.:
                self.vertices_color[i] = [1., 0.5, 0.5]
            elif force > amp * 0.5 + 0.5:
                middle_value = 0.5 / (amp * 0.5 + 0.5) * (force - (amp * 0.5 + 0.5))
                self.vertices_color[i] = [0.5 + middle_value, 1. - middle_value, 0.5]
            else:
                middle_value = 0.25 * force / (amp * 0.5 + 0.5)
                self.vertices_color[i] = [0.75 - middle_value, 1., 1. - 2. * middle_value]
    
    @ti.kernel
    def update_string_vertices(self):
        # i = 0
        # k = 0
        if self.constraint_number:
            current_string_index = 0
            endpoint_index = 0
            for i in ti.ndrange(self.constraint_number):
                self.string_vertex[current_string_index] = self.constraint_start_point[i]
                current_string_index += 1
                current_string_id = 0
                for k in ti.ndrange(self.max_control_length):
                    if self.unit_control[i, k] != -1:
                        current_string_id = k
                        kp_id = self.unit_indices[self.unit_control[i, k]]
                        hole_direction = self.hole_dir[i, k]
                        nm = self.calculateNormalVectorWithUnitId(kp_id)
                        before_kp_id = kp_id
                        before_n = tm.vec3([0., 0., 0.])
                        if k >= 1:
                            before_kp_id = self.unit_indices[self.unit_control[i, k - 1]]
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)

                        if k == 0:
                            if self.intersection_flag2_initial[i]:
                                self.string_vertex[current_string_index] = self.intersection_points2_initial[i] + self.BIAS * hole_direction * nm
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_points2_initial[i] + self.BIAS * hole_direction * nm
                                current_string_index += 1

                        if self.intersection_flag[i, k]:
                            if before_n.norm() > 0:
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * (nm + before_n) * 2. / ((nm + before_n).norm()) ** 2
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * (nm + before_n) * 2. / ((nm + before_n).norm()) ** 2
                                current_string_index += 1
                            else:
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * nm
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * nm
                                current_string_index += 1
                    
                        self.string_vertex[current_string_index] = self.unit_center[self.unit_control[i, k]] + self.BIAS * hole_direction * nm
                        current_string_index += 1
                        
                        if self.unit_control[i, k + 1] != -1 or self.constraint_end_point_existence[i]:
                            next_hole_direction = self.hole_dir[i, k + 1]
                            self.string_vertex[current_string_index] = self.unit_center[self.unit_control[i, k]] + self.BIAS * next_hole_direction * nm
                            current_string_index += 1
                            
                            next_kp_id = self.unit_indices[self.unit_control[i, k + 1]]
                            next_n = self.calculateNormalVectorWithUnitId(next_kp_id)                  
                            if self.intersection_flag2[i, k]:
                                if next_n.norm() > 0:
                                    self.string_vertex[current_string_index] = self.intersection_points2[k + i * self.max_control_length] + self.BIAS * next_hole_direction * (nm + next_n) * 2. / ((nm + next_n).norm()) ** 2
                                    current_string_index += 1
                                    self.string_vertex[current_string_index] = self.intersection_points2[k + i * self.max_control_length] + self.BIAS * next_hole_direction * (nm + next_n) * 2. / ((nm + next_n).norm()) ** 2
                                    current_string_index += 1
                                else:
                                    self.string_vertex[current_string_index] = self.intersection_points2[k + i * self.max_control_length] + self.BIAS * next_hole_direction * nm
                                    current_string_index += 1
                                    self.string_vertex[current_string_index] = self.intersection_points2[k + i * self.max_control_length] + self.BIAS * next_hole_direction * nm
                                    current_string_index += 1
                        
                        # current_string_index += 1
                        self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[i, k]]
                    else:
                        if self.constraint_end_point_existence[i]:
                            # print(current_string_index)
                            # print("-0-")
                            hole_direction = self.hole_dir[i, current_string_id]
                            before_kp_id = self.unit_indices[self.unit_control[i, current_string_id]]
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)
                            # # print("1")
                            # if self.intersection_flag2[i, k - 1]:
                            #     # print("2")
                            #     self.string_vertex[current_string_index] = self.intersection_points2[k - 1 + i * self.max_control_length] - self.BIAS * hole_direction * before_n
                            #     current_string_index += 1
                            #     self.string_vertex[current_string_index] = self.intersection_points2[k - 1 + i * self.max_control_length] - self.BIAS * hole_direction * before_n
                            #     current_string_index += 1
                            self.string_vertex[current_string_index] = self.constraint_end_point[i]
                            current_string_index += 1
                            # print("3")
                            self.endpoint_vertex[endpoint_index] = self.constraint_end_point[i]
                            endpoint_index += 1
                        else:
                            # print(k)
                            # self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[i, k - 1]]
                            endpoint_index += 1
                        break

            for i in ti.ndrange((current_string_index, 6 * self.tsa_string_number)):
                self.string_vertex[i] = self.string_vertex[current_string_index - 1]

    @ti.func
    def calculate_M_inv_f_ext(self):
        for i in ti.ndrange(3 * self.kp_num):
            self.M_inv_f_ext[i] = 0.
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            self.M_inv_f_ext[i] += self.mass_matrix_inv[i, j] * self.field_force[j // 3][j % 3]

    @ti.func
    def calculate_M_norm_velocity(self):
        ret = 0.
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            ret += self.v[i // 3][i % 3] * self.v[j // 3][j % 3] * self.mass_matrix[i, j]
        return ret

    @ti.func
    def calculate_M_norm_velocity_plus_ext(self, t):
        ret = 0.
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            ret += (self.v[i // 3][i % 3] + t * self.M_inv_f_ext[i]) * (self.v[j // 3][j % 3] + t * self.M_inv_f_ext[j]) * self.mass_matrix[i, j]
        return ret
    
    # @ti.kernel
    # def fill_K(self, h: data_type, builder: ti.types.sparse_matrix_builder(), sim_mode: bool):
    #     # lumped mass
    #     for index in ti.ndrange(3 * self.kp_num):
    #         builder[index, index] += self.masses[index // 3] / (h ** 2)

    #     for i, j, k, jj, kk in ti.ndrange(self.div_indices_num, 3, 3, 3, 3):
    #         val = self.K_element[i][3 * j + jj, 3 * k + kk]
    #         if val != 0:
    #             builder[self.triplets[i][j] + jj, self.triplets[i][k] + kk] -= val
        
    #     for ii, ij, ik, ijj, ikk in ti.ndrange(self.bending_pairs_num + self.facet_bending_pairs_num, 4, 4, 3, 3):
    #         val1 = self.K_element_bending[ii][3 * ij + ijj, 3 * ik + ikk]
    #         if val1 != 0:
    #             builder[self.triplets_bending[ii][ij] + ijj, self.triplets_bending[ii][ik] + ikk] -= val1
    #         val2 = self.K_element_bending_damping[ii][3 * ij + ijj, 3 * ik + ikk] / h
    #         if val2 != 0:
    #             builder[self.triplets_bending[ii][ij] + ijj, self.triplets_bending[ii][ik] + ikk] -= val2

    #     for j, i in ti.ndrange(self.MAXIMUM_FIX_PANEL, self.unit_edge_max):
    #         if self.fix_id_list[j] >= 0:
    #             fix_unit_indices = self.unit_indices[self.fix_id_list[j]]
    #             kp_id = fix_unit_indices[i]
    #             if kp_id != -1:
    #                 builder[kp_id * 3 + 0, kp_id * 3 + 0] += self.controller_k
    #                 builder[kp_id * 3 + 1, kp_id * 3 + 1] += self.controller_k
    #                 builder[kp_id * 3 + 2, kp_id * 3 + 2] += self.controller_k
        
    #     for t, i, j, ii, jj in ti.ndrange(self.P_number, 4, self.unit_edge_max, 3, 3):
    #         if self.P_number:
    #             if self.constraint_start_point_candidate_connection[t] >= 0:
    #                 source_indices = self.unit_indices[self.loc_of_unit[t]]
    #                 connected_indices = self.unit_indices[self.constraint_start_point_candidate_connection[t]]
    #                 if connected_indices[j] != -1:
    #                     builder[3 * source_indices[i] + ii, 3 * source_indices[i] + jj] -= self.dfdx_connection[t, i, j][ii, jj]
    #                     builder[3 * connected_indices[j] + ii, 3 * connected_indices[j] + jj] -= self.dfdx_connection[t, i, j][ii, jj]
    #                     builder[3 * source_indices[i] + ii, 3 * connected_indices[j] + jj] += self.dfdx_connection[t, i, j][ii, jj]
    #                     builder[3 * connected_indices[j] + ii, 3 * source_indices[i] + jj] += self.dfdx_connection[t, i, j][ii, jj]

        
    #     for t, i, j, ii, jj in ti.ndrange(self.unit_indices_num * self.unit_edge_max, self.unit_edge_max, self.unit_edge_max, 3, 3):
    #         if self.thick_panel_additional_connection_id[t][0] != -1:
    #             source_indices = self.unit_indices[self.thick_panel_additional_connection_id[t][0]]
    #             connected_indices = self.unit_indices[self.thick_panel_additional_connection_id[t][1]]
    #             if source_indices[i] != -1 and connected_indices[j] != -1:
    #                 builder[3 * source_indices[i] + ii, 3 * source_indices[i] + jj] -= self.dfdx_thick_panel_connection[t, i, j][ii, jj]
    #                 builder[3 * connected_indices[j] + ii, 3 * connected_indices[j] + jj] -= self.dfdx_thick_panel_connection[t, i, j][ii, jj]
    #                 builder[3 * source_indices[i] + ii, 3 * connected_indices[j] + jj] += self.dfdx_thick_panel_connection[t, i, j][ii, jj]
    #                 builder[3 * connected_indices[j] + ii, 3 * source_indices[i] + jj] += self.dfdx_thick_panel_connection[t, i, j][ii, jj]

    #     # if sim_mode:
    #     #     if self.constraint_number:
    #     for ii, i, j, jj, kk in ti.ndrange(self.constraint_number, self.kp_num, self.kp_num, 3, 3):
    #         val = self.dldx_force[ii, i].outer_product(self.dldx_force[ii, j])[jj, kk]
    #         if val != 0:
    #             builder[i * 3 + jj, j * 3 + kk] += val
        

    #     for i, j, k, jj, kk in ti.ndrange(self.ddl_ddx_num[0], self.unit_edge_max, self.unit_edge_max, 3, 3):
    #         unit_id = self.unit_id_list[i]
    #         if unit_id != -1:
    #             unit_kp_id = self.unit_indices[unit_id]
    #             unit_kp_num = self.unit_kp_num_list[unit_id]
    #             if j < unit_kp_num and k < unit_kp_num:
    #             # for j, k, jj, kk in ti.ndrange(unit_kp_num, unit_kp_num, 3, 3):
    #                 builder[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= self.unit_contributions[unit_id][j] * self.unit_contributions[unit_id][k] * (self.dndc[i][jj, kk] + self.dndc2[i][jj, kk])
    #                 if self.need_ta[i]:
    #                     coeff_dndc = self.ta_vector[i][j] * self.ta_vector[i][k] - self.ta_vector[i][j] * self.unit_contributions[unit_id][k] - self.ta_vector[i][k] * self.unit_contributions[unit_id][j]
    #                     coeff_dnda = self.ta_vector[i][j] * self.ta_vector[i][k]
    #                     val = coeff_dndc * self.dndc[i][jj, kk] + coeff_dnda * self.dnda[i][jj, kk]
    #                     if val != 0:
    #                         builder[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= val
    #                 if self.need_tb[i]:
    #                     coeff_dndc2 = self.tb_vector[i][j] * self.tb_vector[i][k] - self.tb_vector[i][j] * self.unit_contributions[unit_id][k] - self.tb_vector[i][k] * self.unit_contributions[unit_id][j]
    #                     coeff_dndb = self.tb_vector[i][j] * self.tb_vector[i][k]
    #                     val = coeff_dndc2 * self.dndc2[i][jj, kk] + coeff_dndb * self.dndb[i][jj, kk]
    #                     if val != 0:
    #                         builder[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= val

    #     c = self.epsilon_v[0]
    #     for index in ti.ndrange(self.kp_num):
    #         if self.x[index][Z] < self.ground_barrier:
    #             builder[index * 3 + 2, index * 3 + 2] -= self.df_ground_force[index]
    #         # if self.back_up_x[index][Z] < self.ground_barrier: 
    #         ui_norm = self.u[index].norm()
    #         term1 = ti.Matrix.zero(data_type, 2, 2)
    #         term2 = (-ui_norm / c ** 2 + 2. / c)
    #         if ui_norm > 0:
    #             term1 = -self.u[index].outer_product(self.u[index]) / (ui_norm * c ** 2)
    #         if ui_norm >= c:
    #             term1 = -self.u[index].outer_product(self.u[index]) / (ui_norm ** 3)
    #             term2 = 1. / ui_norm
    #         friction_matrix = self.ground_miu_param[0] * self.backup_ground_force[index][Z] * (term1 + term2 * self.I2[None])
    #         for jj, kk in ti.ndrange(2, 2):
    #             val = friction_matrix[jj, kk]
    #             if val != 0:
    #                 builder[index * 3 + jj, index * 3 + kk] += val

    @ti.kernel
    def clear_AK_field(self):
        self.AK_field.fill(0.)
    
    @ti.kernel
    def common_build(self, h: data_type):
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            self.AK_field[i, j] += self.mass_matrix[i, j] / (h ** 2)

        for i, j, k, jj, kk in ti.ndrange(self.div_indices_num, 3, 3, 3, 3):
            self.AK_field[self.triplets[i][j] + jj, self.triplets[i][k] + kk] -= self.K_element[i][3 * j + jj, 3 * k + kk]
        
        for ii, ij, ik, ijj, ikk in ti.ndrange(self.bending_pairs_num + self.facet_bending_pairs_num, 4, 4, 3, 3):
            self.AK_field[self.triplets_bending[ii][ij] + ijj, self.triplets_bending[ii][ik] + ikk] -= self.K_element_bending[ii][3 * ij + ijj, 3 * ik + ikk]
            self.AK_field[self.triplets_bending[ii][ij] + ijj, self.triplets_bending[ii][ik] + ikk] -= self.K_element_bending_damping[ii][3 * ij + ijj, 3 * ik + ikk] / h

    @ti.kernel
    def fixforce_build(self):
        for j, i in ti.ndrange(self.MAXIMUM_FIX_PANEL, self.unit_edge_max):
            if self.fix_id_list[j] >= 0:
                fix_unit_indices = self.unit_indices[self.fix_id_list[j]]
                kp_id = fix_unit_indices[i]
                if kp_id != -1:
                    self.AK_field[kp_id * 3 + 0, kp_id * 3 + 0] += self.controller_k
                    self.AK_field[kp_id * 3 + 1, kp_id * 3 + 1] += self.controller_k
                    self.AK_field[kp_id * 3 + 2, kp_id * 3 + 2] += self.controller_k

    @ti.kernel
    def dfdx_build(self):
        for t, i, j, ii, jj in ti.ndrange(self.P_number, 4, self.unit_edge_max, 3, 3):
            source_indices = self.unit_indices[self.loc_of_unit[t]]
            connected_indices = self.unit_indices[self.constraint_start_point_candidate_connection[t]]
            if connected_indices[j] != -1:
                self.AK_field[3 * source_indices[i] + ii, 3 * source_indices[i] + jj] -= self.dfdx_connection[t, i, j][ii, jj]
                self.AK_field[3 * connected_indices[j] + ii, 3 * connected_indices[j] + jj] -= self.dfdx_connection[t, i, j][ii, jj]
                self.AK_field[3 * source_indices[i] + ii, 3 * connected_indices[j] + jj] += self.dfdx_connection[t, i, j][ii, jj]
                self.AK_field[3 * connected_indices[j] + ii, 3 * source_indices[i] + jj] += self.dfdx_connection[t, i, j][ii, jj]

    @ti.kernel
    def thickpanel_build(self):
        for t, i, j, ii, jj in ti.ndrange(self.unit_indices_num * self.unit_edge_max, self.unit_edge_max, self.unit_edge_max, 3, 3):
            if self.thick_panel_additional_connection_id[t][0] != -1:
                source_indices = self.unit_indices[self.thick_panel_additional_connection_id[t][0]]
                connected_indices = self.unit_indices[self.thick_panel_additional_connection_id[t][1]]
                if source_indices[i] != -1 and connected_indices[j] != -1:
                    self.AK_field[3 * source_indices[i] + ii, 3 * source_indices[i] + jj] -= self.dfdx_thick_panel_connection[t, i, j][ii, jj]
                    self.AK_field[3 * connected_indices[j] + ii, 3 * connected_indices[j] + jj] -= self.dfdx_thick_panel_connection[t, i, j][ii, jj]
                    self.AK_field[3 * source_indices[i] + ii, 3 * connected_indices[j] + jj] += self.dfdx_thick_panel_connection[t, i, j][ii, jj]
                    self.AK_field[3 * connected_indices[j] + ii, 3 * source_indices[i] + jj] += self.dfdx_thick_panel_connection[t, i, j][ii, jj]

    @ti.kernel
    def stringforce_build(self):
        for ii, i, j, jj, kk in ti.ndrange(self.constraint_number, self.kp_num, self.kp_num, 3, 3):
            self.AK_field[i * 3 + jj, j * 3 + kk] += self.dldx_force[ii, i].outer_product(self.dldx_force[ii, j])[jj, kk]

        for i, j, k, jj, kk in ti.ndrange(self.ddl_ddx_num[0], self.unit_edge_max, self.unit_edge_max, 3, 3):
            unit_id = self.unit_id_list[i]
            if unit_id != -1:
                unit_kp_id = self.unit_indices[unit_id]
                unit_kp_num = self.unit_kp_num_list[unit_id]
                if j < unit_kp_num and k < unit_kp_num:
                # for j, k, jj, kk in ti.ndrange(unit_kp_num, unit_kp_num, 3, 3):
                    self.AK_field[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= self.unit_contributions[unit_id][j] * self.unit_contributions[unit_id][k] * (self.dndc[i][jj, kk] + self.dndc2[i][jj, kk])
                    if self.need_ta[i]:
                        coeff_dndc = self.ta_vector[i][j] * self.ta_vector[i][k] - self.ta_vector[i][j] * self.unit_contributions[unit_id][k] - self.ta_vector[i][k] * self.unit_contributions[unit_id][j]
                        coeff_dnda = self.ta_vector[i][j] * self.ta_vector[i][k]
                        self.AK_field[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= coeff_dndc * self.dndc[i][jj, kk] + coeff_dnda * self.dnda[i][jj, kk]
                    if self.need_tb[i]:
                        coeff_dndc2 = self.tb_vector[i][j] * self.tb_vector[i][k] - self.tb_vector[i][j] * self.unit_contributions[unit_id][k] - self.tb_vector[i][k] * self.unit_contributions[unit_id][j]
                        coeff_dndb = self.tb_vector[i][j] * self.tb_vector[i][k]
                        self.AK_field[3 * unit_kp_id[j] + jj, 3 * unit_kp_id[k] + kk] -= coeff_dndc2 * self.dndc2[i][jj, kk] + coeff_dndb * self.dndb[i][jj, kk]
    
    @ti.kernel
    def ground_build(self):
        c = self.epsilon_v[0]
        for index in ti.ndrange(self.kp_num):
            if self.x[index][Z] < self.ground_barrier:
                self.AK_field[index * 3 + 2, index * 3 + 2] -= self.df_ground_force[index]
            # if self.back_up_x[index][Z] < self.ground_barrier: 
            # if iter:
            ui_norm = self.u[index].norm()
            # if ui_norm > 0:
            term1 = ti.Matrix.zero(data_type, 2, 2)
            term2 = (-ui_norm / c ** 2 + 2. / c)
            if ui_norm > 0:
                term1 = -self.u[index].outer_product(self.u[index]) / (ui_norm * c ** 2)
            if ui_norm >= c:
                term1 = -self.u[index].outer_product(self.u[index]) / (ui_norm ** 3)
                term2 = 1. / ui_norm
            friction_matrix = self.ground_miu_param[0] * self.backup_ground_force[index][Z] * (term1 + term2 * self.I2[None])
            # if ui_norm == 0.:
            #     print(index, friction_matrix)
            for jj, kk in ti.ndrange(2, 2):
                self.AK_field[index * 3 + jj, index * 3 + kk] += friction_matrix[jj, kk]

    def fill_AK_field(self, h: data_type, iter: int):
        self.clear_AK_field()
        self.common_build(h)
        if self.fix_id_list[0] >= 0:
            self.fixforce_build()
        if self.P_number and self.constraint_start_point_candidate_connection[0] >= 0:
            self.dfdx_build()
        if self.thick_panel_additional_connection_id[0][0] != -1:
            self.thickpanel_build()
        if self.constraint_number > 0:
            self.stringforce_build()
        if self.enable_ground:
            self.ground_build()

    @ti.kernel
    def fill_K_from_field(self, builder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            if self.AK_field[i, j] != 0:
                builder[i, j] += self.AK_field[i, j]
   
    @ti.kernel
    def fill_b(self, h: data_type):
        for i in ti.ndrange(3 * self.kp_num):
            self.b[i] = self.force[i // 3][i % 3]
        for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
            self.b[i] += -self.mass_matrix[i, j] * (self.x[j // 3][j % 3] - self.back_up_x[j // 3][j % 3] - h * self.back_up_v[j // 3][j % 3] - h ** 2 * self.M_inv_f_ext[j]) / (h ** 2)

    @ti.kernel
    def step_x(self, time_step: data_type):
        for i in ti.ndrange(3 * self.kp_num):
            self.x[i // 3][i % 3] = self.x[i // 3][i % 3] + self.u0[i] * time_step
    
    @ti.func
    def updateAllStringLength(self, line_search, history):
        for i in ti.ndrange(self.constraint_number):
            self.calculateStringLength(i, line_search=line_search, use_history_info=history)

    @ti.kernel
    def step_xv(self, time_step: data_type, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3, facet_k: data_type, basic_energy: data_type) -> data_type:
        bonus = 0.9 if sim_mode == self.FOLD_SIM else 1.0

        for i in ti.ndrange(self.kp_num):
            self.v[i] = (self.x[i] - self.back_up_x[i]) / time_step * bonus

        # energy1 = self.getEnergy(sim_mode, theta, gravitational_acc, facets_k)
        self.total_energy[0] = basic_energy
        
        self.calculateUnitCenter()
        
        if sim_mode:  
            self.updateAllStringLength(False, False)
        
        return 1.
        # return time_step

    @ti.func
    def step_xv_using_start_info(self, t):
        for i in ti.ndrange(3 * self.kp_num):
            self.x[i // 3][i % 3] = self.start_x[i // 3][i % 3] + self.u0[i] * self.dt_bonus[0]
        for i in ti.ndrange(self.kp_num):
            self.v[i] = (self.x[i] - (self.back_up_x[i] + t * self.back_up_v[i] + t ** 2 * tm.vec3([self.M_inv_f_ext[3 * i], self.M_inv_f_ext[3 * i + 1], self.M_inv_f_ext[3 * i + 2]]))) / t

    @ti.kernel
    def line_search(self, t: data_type, base_energy: data_type, dx_norm: data_type, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3, facet_k: data_type, enable_ground: bool, mode: int, step: int):
        # self.dt_bonus[0] = 2. * self.dt_bonus[0] if self.dt_bonus[0] <= 0.5 else 1.
        self.backup_startx()
        new_energy = base_energy
        previous_false = False

        while 1:
            if self.print:
                print(self.dt_bonus[0])
            self.step_xv_using_start_info(t)
            new_energy = self.getEnergy(sim_mode, theta, gravitational_acc, facet_k, enable_ground, mode, step)

            if new_energy > base_energy + 1e-6:
                previous_false = True
                self.dt_bonus[0] *= 0.5
                if self.dt_bonus[0] < 1e-6:
                    self.calculateUnitCenter()
                    if sim_mode:
                        self.updateAllStringLength(True, True)
                    break
            else:
                self.dt_bonus[0] = 2. * self.dt_bonus[0] if self.dt_bonus[0] <= 0.5 else 1.
                if self.dt_bonus[0] > 0.999999 or previous_false:
                    self.calculateUnitCenter()
                    if sim_mode:
                        self.updateAllStringLength(True, True)
                    break
    @ti.kernel
    def recordCenterTrajectory(self, step: ti.int32, x: data_type, y: data_type, z: data_type):
        self.center_trajectory[step] = [x, y, z]
    
    @ti.kernel
    def updateGroundLines(self, x: data_type, y: data_type, q: data_type, d: data_type):
        # ===================== 终极核心：旋转坐标系网格（彻底消灭小于d的跳变）=====================
        half_len = 64.0 * d
        line_cnt = 65
        half_line_cnt = int(line_cnt / 2)
        z = 0.25 * self.ground_barrier
        c, s = tm.cos(q), tm.sin(q)

        # ============= 第一步：将世界坐标 逆旋转 到局部坐标系（水平垂直，无旋转）=============
        # 逆旋转矩阵：消除角度q的影响，网格变回轴对齐，跳变严格等于d
        x_local =  x * c + y * s
        y_local = -x * s + y * c

        # ============= 第二步：局部坐标系严格d阶梯取整（永久固定，无微小跳变）=============
        # 只有移动完整d，才会跳跃！局部坐标系步长=d，世界坐标系旋转后步长也=d
        grid_i = tm.floor(x_local / d)
        grid_j = tm.floor(y_local / d)
        center_local_x = grid_i * d
        center_local_y = grid_j * d

        # ============= 第三步：局部网格点 旋转回 世界坐标（最终固定基准点）=============
        cx = center_local_x * c - center_local_y * s
        cy = center_local_x * s + center_local_y * c

        # ===================== 第一组：与X轴成q角（朝向正确，保留）=====================
        for i in ti.ndrange(line_cnt):
            offset = (i - half_line_cnt) * d
            x0_l, x1_l = -half_len, half_len
            y0_l, y1_l = offset, offset
            x0 = cx + x0_l * c - y0_l * s
            y0 = cy + x0_l * s + y0_l * c
            x1 = cx + x1_l * c - y1_l * s
            y1 = cy + x1_l * s + y1_l * c
            self.ground_line_vertex[2*i]   = [x0, y0, z]
            self.ground_line_vertex[2*i+1] = [x1, y1, z]

        # ===================== 第二组：与Y轴成q角（朝向正确，保留）=====================
        for i in ti.ndrange(line_cnt):
            offset = (i - half_line_cnt) * d
            y0_l, y1_l = -half_len, half_len
            x0_l, x1_l = offset, offset
            x0 = cx + x0_l * c - y0_l * s
            y0 = cy + x0_l * s + y0_l * c
            x1 = cx + x1_l * c - y1_l * s
            y1 = cy + x1_l * s + y1_l * c
            self.ground_line_vertex[2 * line_cnt + 2*i]   = [x0, y0, z]
            self.ground_line_vertex[2 * line_cnt + 2*i+1] = [x1, y1, z]

    @ti.kernel
    def checkErrorBuffer(self) -> bool:
        error_flag = False
        for i in ti.ndrange(self.constraint_number):
            if self.error_status_buffer[i]:
                error_flag = True
        return error_flag
    
    @ti.kernel
    def clearErrorBuffer(self):
        for i in ti.ndrange(self.constraint_number):
            self.error_status_buffer[i] = False

    @ti.kernel
    def initialize(     self, 
                        indices:                    ti.types.ndarray(), 
                        kps:                        ti.types.ndarray(), 
                        mass_list:                  ti.types.ndarray(),
                        tri_indices:                ti.types.ndarray(),
                        connection_matrix:          ti.types.ndarray(),
                        bending_pairs:              ti.types.ndarray(),
                        crease_pairs:               ti.types.ndarray(),
                        line_indices:               ti.types.ndarray(),
                        facet_bending_pairs:        ti.types.ndarray(),
                        facet_crease_pairs:         ti.types.ndarray(),
                        sim_mode:                   int,
                        string_number_each:         ti.types.ndarray(),
                        string_total_information:   ti.types.ndarray(),
                        tsa_end:                    ti.types.ndarray(),
                        original_kps:               ti.types.ndarray(), 
                        tb_lines:                   ti.types.ndarray(),
                        lame_k:                     data_type,
                        p_candidate:                ti.types.ndarray(),
                        p_candidate_connections:    ti.types.ndarray(),
                        epsilon_v:                  data_type,
                        epsilon_string:             data_type,
                        contributions:              ti.types.ndarray(),
                        recover_level_need:         ti.types.ndarray(),
                        recover_level:              ti.types.ndarray(),
                        recover_angle:              ti.types.ndarray(),
                        noise:                      ti.types.ndarray(),
                        equivalent_torque_id:       ti.types.ndarray(),
                        equivalent_torque_index:    ti.types.ndarray(),
                        z_bias:                     data_type,
                        fix_id:                     ti.types.ndarray(),
                        dt:                         data_type,
                        crease_noise:               ti.types.ndarray(),
                        string_force_discount:      ti.types.ndarray(),
                        equivalent_internal_id:     ti.types.ndarray(),
                        connected_unit_id:          ti.types.ndarray(),
                        facet_mode:                 bool,
                        rot_z:                      data_type,
                        numpy_target_angles:         ti.types.ndarray(),
        ):
        self.facet_mode_flag[0] = facet_mode
        self.epsilon_v[0] = epsilon_v
        self.unit_control.fill(-1)
        self.intersection_flag.fill(0)
        self.intersection_flag2.fill(0)
        self.intersection_flag2_initial.fill(0)
        self.hole_dir.fill(0.)
        self.ta_vector.fill(0.)
        self.tb_vector.fill(0.)
        self.dndc.fill(0.)
        self.dndc2.fill(0.)
        self.dnda.fill(0.)
        self.dndb.fill(0.)
        self.unit_id_list.fill(-1)
        self.need_ta.fill(False)
        self.need_tb.fill(False)
        self.fix_id_list.fill(-1)
        self.constraint_start_point_candidate_connection.fill(-1)
        self.loc_of_unit.fill(-1)
        self.connection_number[0] = 0
        self.enable_equivalent_torque.fill(False)
        self.backup_enable_equivalent_torque.fill(False)
        self.string_force_each.fill(0.)
        self.backup_string_force_each.fill(0.)
        self.initial_length_per_string.fill(0.)
        self.current_length_per_string.fill(0.)
        self.backup_delta_length.fill(0.)
        self.K_element_bending.fill(0.)
        self.K_element_bending_damping.fill(0.)
        # self.backup_hole_friction_force.fill(0.)
        # self.hole_friction_force.fill(0.)
        # 定义初始点候选集及连接关系
        self.bending_param[0] = 0.5
        self.facet_bending_param[0] = 1.0
        self.k_bending_damping_param[0] = 0.005
        self.controller_param[0] = 1e3
        self.intersection_penalty[0] = 0
        self.neglect_initial_penetration[0] = True
        self.neglect_count[0] = 0
        self.backup_ground_force.fill(0.)
        self.ground_force.fill(0.)

        self.I2[None] = ti.Matrix.identity(data_type, 2)
        self.I3[None] = ti.Matrix.identity(data_type, 3)

        for i in ti.ndrange(self.constraint_number):
            self.error_status_buffer[i] = False
        
        for i in ti.ndrange(min(self.MAXIMUM_FIX_PANEL, fix_id.shape[0])):
            self.fix_id_list[i] = fix_id[i]
        
        for i, j in ti.ndrange(connected_unit_id.shape[0], 2):
            self.thick_panel_additional_connection_id[i][j] = connected_unit_id[i, j]
            
        for i in ti.ndrange(self.maximum_recorded_trajectory_point):
            self.center_trajectory[i] = [0., 0., 0.]

        for i in ti.ndrange(self.P_number):
            self.constraint_start_point_candidate[i][X] = p_candidate[i, X]
            self.constraint_start_point_candidate[i][Y] = p_candidate[i, Y]
            self.constraint_start_point_candidate[i][Z] = p_candidate[i, Z]
            self.constraint_start_point_candidate_connection[i] = p_candidate_connections[i]
            if p_candidate_connections[i] >= 0:
                self.connection_number[0] += 1
                self.constraint_height[i] = p_candidate[i, Z] - z_bias
                # print(self.constraint_height[i])
            else:
                self.constraint_height[i] = 0.0
                
        count = 0
        for i in ti.ndrange(self.P_number):
            if p_candidate_connections[i] >= 0:
                self.loc_of_unit[i] = self.unit_indices_num - self.connection_number[0] + count
                count += 1
                
        for i in ti.ndrange(self.constraint_number):
            self.equivalent_internal_point_id[i] = equivalent_internal_id[i]
            
        if self.constraint_number == 0:
            self.system_type[0] = 0
            print("Unknown system type, no strings are inputted.")
        else:
            if self.fix_id_list[0] >= 0:
                self.system_type[0] = 2
                print("System type: IA-Static")
            else:
                if self.connection_number[0] > 0:
                    self.system_type[0] = 3
                    print("System type: IA")
                else:
                    self.system_type[0] = 1
                    print("System type: EA")
                    
        for i, j in ti.ndrange(4, 4):
            self.ground_vertices[4 * i + j] = [-3750. + 2500. * j, -3750. + 2500. * i, 0.]
            if j % 2 or i % 2:
                self.ground_vertices_color[4 * i + j] = [.75, .75, .75]
            else:
                self.ground_vertices_color[4 * i + j] = [.8, .8, .8]

        for i, j in ti.ndrange(3, 3):
            if (i + j) % 2:
                self.ground_indices[18 * i + 6 * j + 0] = 4 * i + j
                self.ground_indices[18 * i + 6 * j + 1] = 4 * i + j + 1
                self.ground_indices[18 * i + 6 * j + 2] = 4 * i + j + 1 + 4
                self.ground_indices[18 * i + 6 * j + 3] = 4 * i + j
                self.ground_indices[18 * i + 6 * j + 4] = 4 * i + j + 1 + 4
                self.ground_indices[18 * i + 6 * j + 5] = 4 * i + j + 4
            else:
                self.ground_indices[18 * i + 6 * j + 0] = 4 * i + j
                self.ground_indices[18 * i + 6 * j + 1] = 4 * i + j + 1
                self.ground_indices[18 * i + 6 * j + 2] = 4 * i + j + 4
                self.ground_indices[18 * i + 6 * j + 3] = 4 * i + j + 1
                self.ground_indices[18 * i + 6 * j + 4] = 4 * i + j + 1 + 4
                self.ground_indices[18 * i + 6 * j + 5] = 4 * i + j + 4

        self.ground_fix_line_vertex[0] = [-5000. * tm.cos(rot_z), -5000. * tm.sin(rot_z), 0.5 * self.ground_barrier]
        self.ground_fix_line_vertex[1] = [5000. * tm.cos(rot_z), 5000. * tm.sin(rot_z), 0.5 * self.ground_barrier]
        self.ground_fix_line_vertex[2] = [5000. * tm.sin(rot_z), -5000. * tm.cos(rot_z), 0.5 * self.ground_barrier]
        self.ground_fix_line_vertex[3] = [-5000. * tm.sin(rot_z), 5000. * tm.cos(rot_z), 0.5 * self.ground_barrier]
        
        # 初始化单元索引
        for i, j in ti.ndrange(self.unit_indices_num, self.unit_edge_max):
            self.unit_indices[i][j] = indices[i, j]
        
        for i in ti.ndrange(self.unit_indices_num):
            self.unit_kp_num_list[i] = self.calculateKpNumWithUnitId(self.unit_indices[i])

        for i, j in ti.ndrange(self.unit_indices_num, self.unit_edge_max):
            self.unit_contributions[i][j] = 1. / self.unit_kp_num_list[i]
        
        if contributions.shape[0] > 0:
            for i, j in ti.ndrange(contributions.shape[0], contributions.shape[1]):   
                self.unit_contributions[i][j] = contributions[i, j]

        # 初始化节点位置与质量
        for i in ti.ndrange(self.kp_num):
            self.original_vertices[i] = [kps[i, X], kps[i, Y], kps[i, Z]]
            self.masses[i] = mass_list[i]

        # 初始化质量矩阵
        if self.mass_mode == 0:
            for i, j in ti.ndrange(3 * self.kp_num, 3 * self.kp_num):
                self.mass_matrix[i, j] = 0.0
                self.mass_matrix_inv[i, j] = 0.0

        if self.mass_mode == 0:
            for i in ti.ndrange(3 * self.kp_num):
                self.mass_matrix[i, i] += self.masses[i // 3]
                self.mass_matrix_inv[i, i] += 1. / self.masses[i // 3]

        # 初始化三角面索引
        for i in ti.ndrange(self.indices_num):
            self.indices[i] = tri_indices[i]
            
        # # 初始化连接矩阵
        # for i, j in ti.ndrange(self.kp_num, self.kp_num):
        #     self.connection_matrix[i, j] = connection_matrix[i, j]
        
        # 初始化弯曲对和折痕对
        for i, j in ti.ndrange(self.bending_pairs_num, 2):
            self.bending_pairs[i, j] = bending_pairs[i, j]
            self.crease_pairs[i, j] = crease_pairs[i, j]

        for i in ti.ndrange(self.bending_pairs_num):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.crease_pairs[i, 0]]
            ce = self.original_vertices[self.crease_pairs[i, 1]]
            p1 = self.original_vertices[self.bending_pairs[i, 0]]
            p2 = self.original_vertices[self.bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            self.bending_pairs_area[i, 0] = a1 * 0.5
            self.bending_pairs_area[i, 1] = a2 * 0.5
            self.crease_initial_length[i] = (ce - cs).norm()
            # 初始化随机预应力
            self.random_folding_target_angle[i] = crease_noise[i]
            for j in ti.ndrange(self.constraint_number):
                self.equivalent_torque_influence_id[j, i] = equivalent_torque_id[j, i]
                for k in ti.ndrange(self.max_control_length):
                    self.equivalent_torque_coeff[j, i, k] = equivalent_torque_index[j, i, k]
            # 初始化目标角度
            self.target_angles[i] = numpy_target_angles[i]

        # 初始化线段对
        for i, j in ti.ndrange(self.line_total_indice_num, 2):
            self.line_pairs[i, j] = int(line_indices[i, j])
        # 初始化面折痕对
        for i, j in ti.ndrange(self.facet_bending_pairs_num, 2):
            self.facet_bending_pairs[i, j] = facet_bending_pairs[i, j]
            self.facet_crease_pairs[i, j] = facet_crease_pairs[i, j]

        for i in ti.ndrange(self.facet_bending_pairs_num):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.facet_crease_pairs[i, 0]]
            ce = self.original_vertices[self.facet_crease_pairs[i, 1]]
            p1 = self.original_vertices[self.facet_bending_pairs[i, 0]]
            p2 = self.original_vertices[self.facet_bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            # assert a1 > 0 and a2 > 0
            self.facet_bending_pairs_area[i, 0] = a1 * 0.5
            self.facet_bending_pairs_area[i, 1] = a2 * 0.5
            self.facet_bending_pairs_distance[i] = ((p1 - p2) - (p1 - p2).dot(ce - cs) / (ce - cs).norm_sqr() * (ce - cs)).norm()
            # print(self.facet_bending_pairs_distance[i])
            self.facet_crease_initial_length[i] = (ce - cs).norm()

        #初始化折痕折角
        for i in ti.ndrange(self.crease_pairs_num):
            self.crease_angle[i] = 0.0
            self.crease_folding_angle[i] = 0.0
            self.crease_folding_accumulate[i] = 0.0
            self.previous_dir[i] = 0.0

        # 初始化折痕类型
        for i in ti.ndrange(self.crease_pairs_num):
            for j in ti.ndrange(self.line_total_indice_num):
                if crease_pairs[i, 0] == int(line_indices[j, 0]) and crease_pairs[i, 1] == int(line_indices[j, 1]):
                    self.crease_type[i] = int(line_indices[j, 2])

                    if line_indices[j, 3] > tm.pi * self.folding_angle_maximum_ratio:
                        self.folding_angle_upper_bound[i] = tm.pi * self.folding_angle_maximum_ratio
                    else:
                        self.folding_angle_upper_bound[i] = line_indices[j, 3]

                    if line_indices[j, 4] < -tm.pi * self.folding_angle_maximum_ratio:
                        self.folding_angle_lower_bound[i] = -tm.pi * self.folding_angle_maximum_ratio
                    else:
                        self.folding_angle_lower_bound[i] = line_indices[j, 4]

                    if not self.double_side_folding_enable:
                        if int(line_indices[j, 2]) == MOUNTAIN:
                            self.folding_angle_upper_bound[i] = 1. / 36 * tm.pi
                        else:
                            self.folding_angle_lower_bound[i] = -1. / 36 * tm.pi
                    # print(f'{i}, {crease_pairs[i, 0]}, {crease_pairs[i, 1]}, {self.crease_type[i]}, {self.folding_angle_upper_bound[i]}, {self.folding_angle_lower_bound[i]}')
                    break
                    
        # ti.loop_config(serialize=True)
        self.sequence_level[0] = 0
        self.sequence_level[1] = 0
        if sim_mode == self.FOLD_SIM:
            # 初始化折叠等级和系数
            for i, j in ti.ndrange(self.crease_pairs_num, tb_lines.shape[0]):
                kp1 = [original_kps[crease_pairs[i, 0], X], original_kps[crease_pairs[i, 0], Y]]
                kp2 = [original_kps[crease_pairs[i, 1], X], original_kps[crease_pairs[i, 1], Y]]
                kp11 = [tb_lines[j, 0], tb_lines[j, 1]]
                kp22 = [tb_lines[j, 2], tb_lines[j, 3]]
                if (((kp1[X] - kp11[X]) ** 2 + (kp1[Y] - kp11[Y]) ** 2) <= 16. and \
                    ((kp2[X] - kp22[X]) ** 2 + (kp2[Y] - kp22[Y]) ** 2) <= 16.) or \
                    (((kp1[X] - kp22[X]) ** 2 + (kp1[Y] - kp22[Y]) ** 2) <= 16. and \
                    ((kp2[X] - kp11[X]) ** 2 + (kp2[Y] - kp11[Y]) ** 2) <= 16.):

                    self.crease_level[i] = int(tb_lines[j, 4])
                    self.crease_coeff[i] = tb_lines[j, 5]
                    
                    for k in ti.static(range(self.maximum_level_number)):
                        self.recover_level_need[i, k] = recover_level_need[j, k]
                        self.recover_level[i, k] = recover_level[j, k]
                        self.recover_angle[i, k] = recover_angle[j, k]
                        if self.recover_level[i, k] > self.sequence_level[0]:
                            self.sequence_level[0] = self.recover_level[i, k]
                        if self.recover_level[i, k] < self.sequence_level[1]:
                            self.sequence_level[1] = self.recover_level[i, k]
                    
                if self.crease_level[i] > self.sequence_level[0]:
                    self.sequence_level[0] = self.crease_level[i]
                if self.crease_level[i] < self.sequence_level[1]:
                    self.sequence_level[1] = self.crease_level[i]
                
            self.folding_micro_step[0] = tm.pi / 900.0 / (1 + self.sequence_level[0] - self.sequence_level[1])
        # ti.loop_config(serialize=False)

        # 初始化渲染的线的颜色信息
        for i in ti.ndrange(self.line_total_indice_num):
            if line_indices[i, 2] == BORDER:
                self.line_color[2 * i] = [0, 0, 0]
                self.line_color[2 * i + 1] = [0, 0, 0]
            elif line_indices[i, 2] == VALLEY:
                self.line_color[2 * i] = [0, 0.17, 0.83]
                self.line_color[2 * i + 1] = [0, 0.17, 0.83]
                if self.control_mode and sim_mode:
                    self.line_color[2 * i] = [0.5, 0.5, 0.5]
                    self.line_color[2 * i + 1] = [0.5, 0.5, 0.5]
            elif line_indices[i, 2] == MOUNTAIN:
                self.line_color[2 * i] = [0.75, 0.2, 0.05]
                self.line_color[2 * i + 1] = [0.75, 0.2, 0.05]
            else:
                self.line_color[2 * i] = [0.5, 0.5, 0.5]
                self.line_color[2 * i + 1] = [0.5, 0.5, 0.5]
                
            pointer = 0
            for j in ti.ndrange(self.unit_indices_num):
                unit_ids = self.unit_indices[j]
                exist1 = False
                exist2 = False
                for k in ti.ndrange(self.unit_edge_max):
                    if unit_ids[k] == line_indices[i, 0]:
                        exist1 = True
                    if unit_ids[k] == line_indices[i, 1]:
                        exist2 = True
                if exist1 and exist2:
                    self.line_connection_unit[i][pointer] = j
                    pointer += 1

        # 暂不考虑ABAB, 初始化控制单元和穿孔方向
        for i in ti.ndrange(self.constraint_number):
            index = 0
            self.string_number_each[i] = string_number_each[i]
            for j in ti.ndrange(string_number_each[i] - 1):
                if string_total_information[i, j + 1, 0] != 0:
                    self.unit_control[i, index] = string_total_information[i, j + 1, 1]
                    self.hole_dir[i, index] = string_total_information[i, j + 1, 2]
                    index += 1
                    # 摩擦力衰减
                self.string_force_initial_discount[i, j] = string_force_discount[i, j]
                self.string_force_current_discount[i, j] = string_force_discount[i, j]

        # calculate initial center point
        for i in ti.ndrange(self.unit_indices_num):
            unit_indice = self.unit_indices[i]
            center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice, i)
            # print(f"{i}, {center_point}")
            self.unit_center_initial_point[i] = center_point
            self.unit_center[i] = center_point

        # 计算初始点
        if self.P_number > 0:
            for i in ti.ndrange(self.constraint_number):
                dir = string_total_information[i, 0, 2]
                start_id = 0
                if self.system_type[0] == 3:
                    if dir == -1:
                        start_id = string_total_information[i, 0, 1]
                        # print(f"Because of IA-Dynamic system, the initial side is {dir}, so the start id is {start_id}")
                    else:
                        start_id = string_total_information[i, 0, 1] + 1
                        # print(f"Because of IA-Dynamic system, the initial side is {dir}, so the start id is {start_id}")
                else:
                    start_id = string_total_information[i, 0, 1]
                self.string_length_decrease[i] = 0.0
                for k in ti.ndrange(self.max_control_length):
                    self.target_string_length_decrease[i, k] = 0.0
                self.constraint_start_point[i] = self.constraint_start_point_candidate[start_id]
                self.constraint_start_point_candidate_id[i] = start_id
                # print(self.constraint_start_point_candidate_id[i])

                if self.constraint_start_point_candidate_connection[start_id] >= 0:
                    index = self.loc_of_unit[start_id]
                    unit_indice = self.unit_indices[index]
                    center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice, index)
                    self.constraint_start_point[i] = center_point

        # 计算末端点信息
        if self.P_number > 0:
            for i in ti.ndrange(self.constraint_number):
                if tsa_end[i] != -1:
                    id = int(tsa_end[i])
                    self.constraint_end_point[i] = self.constraint_start_point_candidate[id]
                    self.constraint_end_point_candidate_id[i] = id
                    self.constraint_end_point_existence[i] = True
                    # if self.constraint_start_point_candidate_connection[id] >= 0:
                    #     unit_indice = self.unit_indices[self.constraint_start_point_candidate_connection[id]]
                    #     center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice, self.constraint_start_point_candidate_connection[id])
                    #     self.constraint_end_point[i] = center_point + tm.vec3([0., 0., self.constraint_height[id]])
                else:
                    self.constraint_end_point[i] = [0.0, 0.0, 0.0]
                    self.constraint_end_point_existence[i] = False

        # 初始化结束条件
        self.folding_angle_reach_pi[0] = False
        
        # self.stable_state = 0
        # self.past_move_indice = 0.0

        # self.can_rotate = False
        # self.tsa_turning_angle = 0.0

        # # 初始化边缘信息
        for i in ti.ndrange(self.P_number):
            self.border_vertex[i] = self.constraint_start_point_candidate[i]
        # for i in ti.ndrange(30):
        #     self.border_vertex[2 * i] = [tm.cos(i / 15. * tm.pi) * self.panel_size, tm.sin(i / 15. * tm.pi) * self.panel_size, self.origami_z_bias]
        #     self.border_vertex[2 * i + 1] = [tm.cos((i + 1) / 15. * tm.pi) * self.panel_size, tm.sin((i + 1) / 15. * tm.pi) * self.panel_size, self.origami_z_bias]

        # 拉梅常数增益
        self.lames_bonus[0] = lame_k * self.mu
        self.lames_bonus[1] = lame_k * self.landa

        # 初始化形状微分矩阵
        self.dDs[0, 0] = ti.Matrix.rows([[-1., -1., 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[0, 1] = ti.Matrix.rows([[0, 0, 0], [-1., -1., 0], [0, 0, 0]])
        self.dDs[0, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [-1., -1., 0]])
        self.dDs[1, 0] = ti.Matrix.rows([[1., 0, 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[1, 1] = ti.Matrix.rows([[0, 0, 0], [1., 0, 0], [0, 0, 0]])
        self.dDs[1, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [1., 0, 0]])
        self.dDs[2, 0] = ti.Matrix.rows([[0, 1., 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[2, 1] = ti.Matrix.rows([[0, 0, 0], [0, 1., 0], [0, 0, 0]])
        self.dDs[2, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [0, 1., 0]])

        self.error_status[0] = False
    
        for i in ti.ndrange(self.kp_num):
            self.x[i] = self.original_vertices[i]
            self.v[i] = [0., 0., 0.]
            self.dv[i] = [0., 0., 0.]

        # precompute dm and A
        for i in ti.ndrange(self.div_indices_num):
            x0 = self.original_vertices[self.indices[3 * i]]
            x1 = self.original_vertices[self.indices[3 * i + 1]]
            x2 = self.original_vertices[self.indices[3 * i + 2]]
            dm = ti.Matrix.cols([x1 - x0, x2 - x0, tm.normalize((x1 - x0).cross(x2 - x0))])
            self.dm[i] = tm.inverse(dm)
            self.A[i] = 0.5 * tm.determinant(dm)
            self.A_total[0] += self.A[i]
        
        # print("Pattern area:", self.A_total[0], "mm^2")

        self.percent[0] = 1.0

        self.updateAllStringLength(False, False)

        # calculate initial length
        self.equal_arm_distance[0] = 0.0
        count = 0
        for i in ti.ndrange(self.constraint_number):
            self.constraint_initial_length[i] = self.constraint_length[i]
            index = 0
            for j in ti.ndrange(self.max_control_length):
                if self.unit_control[i, j] != -1:
                    self.initial_length_per_string[i, j] = self.current_length_per_string[i, j]
                else:
                    index = j
                    break
            if self.constraint_end_point_existence[i]:
                self.initial_length_per_string[i, index] = self.current_length_per_string[i, index]
                # print(f"{i}, end, {length}")
            self.equal_arm_distance[0] += self.constraint_initial_length[i]
            self.backup_constraint_length[i] = self.constraint_initial_length[i]
            self.constraint_initial_length[i] *= noise[i]
        self.equal_arm_distance[0] /= (count * 2)
        self.equal_arm_distance[0] = self.origami_thickness / self.equal_arm_distance[0]

        # # 初始化线的信息
        for i in ti.ndrange(self.constraint_number):
            self.string_params[i] = self.ks(0.) / (self.constraint_initial_length[i] + self.additional_length_of_string)
            self.string_params_bonus[i] = 1.0 
            self.string_params_clip[i] = 0.0
            self.enable_plasticity[i] = False
            self.epsilon_string[i] = self.string_params[i] * dt
            self.tolerance[i] = (self.constraint_initial_length[i] + self.additional_length_of_string) * self.maximum_epsilon_of_string * self.tension_bonus
            self.history_maximum_force[i] = 0.0

    def initializeRunning(self, strict=0, noise=None, show_index=False, noise_type="string", clean_ratio=True):
        # parameters reset
        if clean_ratio:
            self.string_contract_ratio_initial = INITIAL_RATIO
        self.dead_count = 0
        self.positive_count = 0
        self.candidate_method = True #strict=-1
        self.recorded_t = []
        self.recorded_string_decrease_length_control = [[] for _ in range(self.constraint_number)]
        self.recorded_string_decrease_length = [[] for _ in range(self.constraint_number)]
        self.string_length_each = [0 for _ in range(self.constraint_number)]
        self.recorded_max_force = []
        self.recorded_nodal_maximum_force = []
        self.recorded_string_force = [[] for _ in range(self.constraint_number)]
        self.recorded_folding_percent = []
        self.recorded_folding_error = []
        self.recorded_maximum_folding_percent = []
        self.recorded_minimum_folding_percent = []
        self.recorded_maximum_folding_error = []
        self.recorded_minimum_folding_error = []
        self.recorded_movement_x = []
        self.recorded_movement_y = []
        self.recorded_movement_z = []
        self.first_phase_enable = False
        self.recorded_interval_velocity = []
        if self.control_mode and self.sim_mode:
            self.recorded_indices = [self.kp_num - 4, self.kp_num - 3, self.kp_num - 2, self.kp_num - 1]
        else:
            self.recorded_indices = []
        
        self.stable_state = 0
        self.past_move_indice = 0.0
        self.folding_percent = 0.0
        self.abs_folding_percent = 0.0
        self.folding_error = 0.0
        self.offset_x = 0.
        self.offset_y = 0.
        self.allow_initialize = False

        self.current_t = 0

        self.can_rotate = False

        self.dt_bonus[0] = 1.
            
        self.control_signals = [
            [30 for _ in range(self.constraint_number)],
            [3000 for _ in range(self.constraint_number)]
        ]
        
        self.wait_time = 0.0
        self.wait_interval = 0.0
        
        self.recorded_signals = [
            0 for _ in range(self.constraint_number)
        ]

        if self.sim_mode == self.FOLD_SIM:
            self.n = 24 #仿真的时间间隔
            self.dt = 0.1 / self.n #仿真的时间间隔
            self.substeps = 1 #子步长，用于渲染
            self.basic_dt = self.substeps * self.dt
            self.now_t = 0.
            self.lame_k = 1000.

            if self.use_gui:
                if "diamond-II-acc" in self.origami_name:
                    self.camera.position(-52.5, -45, max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, -1, 0)
                    self.camera.lookat(-52.5, -45, self.origami_z_bias)
                elif "diamond-III-acc" in self.origami_name:
                    self.camera.position(30, -155, max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, -1, -0)
                    self.camera.lookat(-52.5, -45, self.origami_z_bias)
                # elif "box" in self.origami_name:
                #     self.camera.position(-50, 75, max(1.2 * self.max_size, 400) + self.origami_z_bias)
                #     self.camera.up(0, 1, 0)
                #     self.camera.lookat(0, -25, self.origami_z_bias)
                elif "mountain-big-fix-new" in self.origami_name:
                    self.camera.position(-75, -150, max(2.8 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-1, 0, 0)
                    self.camera.lookat(-75, -58, self.origami_z_bias)
                elif "resch" in self.origami_name:
                    self.camera.position(70, -120, max(2.8 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, -1, 0)
                    self.camera.lookat(-20, -60, self.origami_z_bias)
                else:
                    self.camera.position(-1.2 * self.max_size, min(-1.2 * self.max_size, -400), max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, 0, 1.0)
                    self.camera.lookat(0, 0, self.origami_z_bias)
                self.camera.z_far(max(10 * self.max_size, 5000.))
            self.viscousity = 0.0
            self.ITER = 10
            self.enable_ground = False
        else:
            if strict <= 2:
                self.n = 24 #仿真的时间间隔
                self.dt = 0.1 / self.n #仿真的时间间隔
                self.substeps = 1 #子步长，用于渲染
                self.basic_dt = self.substeps * self.dt
            else:
                self.n = 60 #仿真的时间间隔
                self.dt = 0.1 / self.n #仿真的时间间隔
                self.substeps = 1
                self.basic_dt = self.substeps * self.dt
            self.now_t = 0.
            self.lame_k = 1000.
            
            if self.control_mode == 0:
                ################################# USER DEFINED #################################
                self.base_decrease_step = self.motor_speed * 24. / self.n * self.speed_bonus
                # self.base_decrease_step = 10.0 / 240.0
                ################################# USER DEFINED #################################
            else:
                self.base_decrease_step = self.motor_speed * 24. / self.n
            
            self.DEAD_MAXIMUM = int(self.n * 10)

            self.string_length_decrease_step = self.base_decrease_step
            if self.strict == 3:
                self.string_length_decrease_step = self.base_decrease_step * self.strict_speed

            if self.use_gui:
                if "f-diamond-II-acc" in self.origami_name:
                    self.camera.position(-52.5, -45, max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, -1, 0)
                    self.camera.lookat(-52.5, -45, self.origami_z_bias)
                elif "f-diamond-III-acc" in self.origami_name:
                    self.camera.position(-52.5, -55, max(1.2 * self.max_size, 450) + self.origami_z_bias)
                    self.camera.up(0, -1, 0)
                    self.camera.lookat(-52.5, -55, self.origami_z_bias)
                # elif "f-box" in self.origami_name:
                #     self.camera.position(-50, 75, max(1.2 * self.max_size, 400) + self.origami_z_bias)
                #     self.camera.up(0, 1, 0)
                #     self.camera.lookat(0, -25, self.origami_z_bias)
                elif "f-mountain-big-fix-new" in self.origami_name:
                    self.camera.position(-75, -58, max(2.8 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-1, 0, 0)
                    self.camera.lookat(-75, -58, self.origami_z_bias)
                elif "f-miura-ori-16" in self.origami_name:
                    self.camera.position(-75, -58, max(2.8 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-1, 0, 0)
                    self.camera.lookat(-75, -58, self.origami_z_bias)
                elif "f-resch" in self.origami_name:
                    self.camera.position(-20, -60, max(2.8 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, -1, 0)
                    self.camera.lookat(-20, -60, self.origami_z_bias)
                elif "f-robot3" in self.origami_name:
                    self.camera.position(100, -120, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, 1, 0)
                    self.camera.lookat(100, -120, self.origami_z_bias)
                elif "f-robot4" in self.origami_name:
                    self.camera.position(300, -360, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, 1, 0)
                    self.camera.lookat(300, -360, self.origami_z_bias)
                elif "f-robot8-big" in self.origami_name:
                    self.camera.position(300, -360, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, 1, 0)
                    self.camera.lookat(300, -360, self.origami_z_bias)
                elif "f-robot5" in self.origami_name:
                    self.camera.position(100, -120, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(0, 1, 0)
                    self.camera.lookat(100, -120, self.origami_z_bias)
                elif "f-robot8" in self.origami_name:
                    self.camera.position(0, 0, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-math.sin(self.rot_z), math.cos(self.rot_z), 0)
                    self.camera.lookat(0, 0, self.origami_z_bias)
                elif "f-robot9" in self.origami_name:
                    self.camera.position(-100, 100, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-math.sin(self.rot_z), math.cos(self.rot_z), 0)
                    self.camera.lookat(-100, 100, self.origami_z_bias)
                elif "f-robot" in self.origami_name:
                    self.camera.position(0, 0, max(5 * self.max_size, 500) + self.origami_z_bias)
                    self.camera.up(-math.sin(self.rot_z), math.cos(self.rot_z), 0)
                    self.camera.lookat(0, 0, self.origami_z_bias)
                elif "f-miura" in self.origami_name:
                    self.camera.position(0, min(-1.2 * self.max_size, -400), max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, 0, 1.0)
                    self.camera.lookat(0, 0, self.origami_z_bias)
                elif "bird" in self.origami_name:
                    self.camera.position(160, -370, 320 + self.origami_z_bias)
                    self.camera.up(-0.05, 0.3, 0.7)
                    self.camera.lookat(20, -10, self.origami_z_bias)
                elif "f-5panel-robot" in self.origami_name:
                    self.camera.position(0.8 * self.max_size, min(-1.2 * self.max_size, -400), max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, 0, 1.0)
                    self.camera.lookat(0.2 * self.max_size, 0, self.origami_z_bias)
                else:
                    self.camera.position(-1.2 * self.max_size, min(-1.2 * self.max_size, -400), max(1.2 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.up(0, 0, 1.0)
                    self.camera.lookat(0, 0, self.origami_z_bias)
                self.camera.z_far(max(10 * self.max_size, 5000.))
            self.viscousity = 0.0
            self.ITER = 10
            self.enable_ground = self.default_ground
        self.current_t = 0.0
        self.actuation_start_t = 0.0
        self.image_id = 0
        self.control_steps = 0
        self.current_center_trajectory_step = 0
        epsilon_v = self.velocity_barrier * self.dt
        epsilon_string = 0.0 #abandon this
        self.minimum_z = self.origami_z_bias
        self.friction_mode_enable[0] = self.friction_mode
        self.const_stiff_of_crease_param[0] = self.const_stiff_of_crease
        self.robot_stand = True

        self.paused = False
        self.step_once = False

        self.image_id = 0
        if self.control_mode == 1:
            self.additional_folder = 'phase1'
            if self.origami_name == 'f-5panel-robot':
                self.additional_folder = 'phase2'
        else:
            self.additional_folder = ''
        self.additional_height[0] = 20
        self.ground_miu_param[0] = self.ground_miu

        self.rotation_list = np.array([1.0 for _ in range(self.constraint_number)])
        self.rotation_list_backup = np.array([1.0 for _ in range(self.constraint_number)])
        self.staggering_flag = np.array([0 for _ in range(self.constraint_number)])

        numpy_noise = np.array([1.0 for _ in range(self.constraint_number)])
        numpy_crease_noise = np.array([0.0 for _ in range(self.bending_pairs_num)])
        equivalent_torque_influence_id = np.array([[0 for _ in range(self.bending_pairs_num)] for _ in range(max(1, self.constraint_number))])
        equivalent_torque_index = np.array([[[0.0 for _ in range(self.max_control_length)] for _ in range(self.bending_pairs_num)] for _ in range(max(1, self.constraint_number))])
        
        if strict == 2:
            if noise_type == "crease":
                if type(noise) == None:
                    for i in range(self.bending_pairs_num):
                        numpy_crease_noise[i] = 0.0
                else:
                    numpy_crease_noise = noise
            elif noise_type == "string":
                if type(noise) == None:
                    for i in range(self.constraint_number):
                        numpy_noise[i] = 1.0
                else:
                    for i in range(self.constraint_number):
                        numpy_noise[i] = (1.0 + noise[i] * 0.6863)
            elif noise_type == "stiffness":
                pass
        
        numpy_indices                       = np.array(self.ori_sim.indices, dtype=np.int32)
        for i in range(len(self.contributions)):
            if self.units[i].repaired:
                self.contributions[i] = self.units[i].getContribution()
            if len(self.contributions[i]):
                for j in range(len(self.contributions[i]), self.unit_edge_max):
                    self.contributions[i].append(0.)
                    
        numpy_fix_id = np.array(self.fix_id, dtype=np.int32)
        numpy_connected_unit_id = np.array([[-1, -1]])
        if len(self.connected_unit_pairs):
            numpy_connected_unit_id = np.array(self.connected_unit_pairs, dtype=np.int32)

        # maximum_length = 0
        # for i in range(len(self.contributions)):
        #     self.contributions[i] = self.units[i].getContribution()
        #     if len(self.contributions[i]) > maximum_length:
        #         maximum_length = len(self.contributions[i])

        # for i in range(len(self.contributions)):
        #     while len(self.contributions[i]) < maximum_length:
        #         self.contributions[i].append(0.)

        numpy_contributions                 = np.array(self.contributions)
        numpy_kps                           = np.array(self.kps) - np.array(self.total_bias + [-self.origami_z_bias])
        numpy_original_kps                  = np.array(self.kps)
        numpy_mass_list                     = np.array(self.mass_list)
        total_mass = sum(numpy_mass_list)
        numpy_tri_indices                   = np.array(self.tri_indices, dtype=np.int32)
        numpy_connection_matrix             = np.array(self.ori_sim.connection_matrix)
        numpy_bending_pairs                 = np.array(self.ori_sim.bending_pairs, dtype=np.int32)
        numpy_crease_pairs                  = np.array(self.ori_sim.crease_pairs, dtype=np.int32)
        numpy_line_indices                  = np.array(self.ori_sim.getNewLineIndices(), dtype=np.float64)

        if self.mass_mode == 1:
            c_mass_matrix = np.zeros((3 * self.kp_num, 3 * self.kp_num))
            for i in range(self.unit_indices_num):
                cur_indice = numpy_indices[i]
                sub_c_mass_matrix = self.consistent_mass_list[i]
                for j in range(self.unit_edge_max):
                    for k in range(self.unit_edge_max):
                        if cur_indice[j] != -1 and cur_indice[k] != -1:
                            c_mass_matrix[3 * cur_indice[j]][3 * cur_indice[k]] += sub_c_mass_matrix[j][k]
                            c_mass_matrix[3 * cur_indice[j] + 1][3 * cur_indice[k] + 1] += sub_c_mass_matrix[j][k]
                            c_mass_matrix[3 * cur_indice[j] + 2][3 * cur_indice[k] + 2] += sub_c_mass_matrix[j][k]
            
            c_mass_inv = np.linalg.inv(c_mass_matrix)
            self.mass_matrix.from_numpy(c_mass_matrix)
            self.mass_matrix_inv.from_numpy(c_mass_inv)


        if len(self.ori_sim.facet_bending_pairs) == 0:
            numpy_facet_bending_pairs           = np.array([[0, 0]], dtype=np.int32)
            numpy_facet_crease_pairs            = np.array([[0, 0]], dtype=np.int32)
        else:
            numpy_facet_bending_pairs           = np.array(self.ori_sim.facet_bending_pairs, dtype=np.int32)
            numpy_facet_crease_pairs            = np.array(self.ori_sim.facet_crease_pairs, dtype=np.int32)
        
        if show_index and self.use_gui:
            fig, ax = plt.subplots(figsize=(4, 4))
            for i in range(len(self.kps)):
                ax.scatter(self.kps[i][X], self.kps[i][Y], marker='o', c='black')
                plt.annotate(str(i), xy = (self.kps[i][X], self.kps[i][Y]), xytext = (self.kps[i][X] + 0.1, self.kps[i][Y] + 0.1))
            for j in range(self.bending_pairs_num):
                xy = (0.5 * (self.kps[self.ori_sim.crease_pairs[j][0]][X] + self.kps[self.ori_sim.crease_pairs[j][1]][X]), 
                                   0.5 * (self.kps[self.ori_sim.crease_pairs[j][0]][Y] + self.kps[self.ori_sim.crease_pairs[j][1]][Y]))
                xytext = (0.5 * (self.kps[self.ori_sim.crease_pairs[j][0]][X] + self.kps[self.ori_sim.crease_pairs[j][1]][X]) + 0.1, 
                                       0.5 * (self.kps[self.ori_sim.crease_pairs[j][0]][Y] + self.kps[self.ori_sim.crease_pairs[j][1]][Y]) + 0.1)
                plt.annotate(str(j), 
                             xy = xy, 
                             xytext = xytext, c='r')
            plt.show()
        
        # construct tb_line information which contains start, end, level and coeff
        if self.sim_mode == self.FOLD_SIM:
            tb_line = []
            for line in self.lines:
                tb_line.append([line[START][X], line[START][Y], line[END][X], line[END][Y], line.level, line.coeff])
        else:
            tb_line = [[]]
        numpy_tb_line                       = np.array(tb_line)

        maximum_recover_level_length = max([len(line.recover_level) for line in self.lines])
        if maximum_recover_level_length > self.maximum_level_number:
            raise NotImplementedError
        numpy_recover_level_need = np.zeros(shape=(len(self.lines), self.maximum_level_number), dtype=bool)
        numpy_recover_level = np.zeros(shape=(len(self.lines), self.maximum_level_number), dtype=int)
        numpy_recover_angle = np.zeros(shape=(len(self.lines), self.maximum_level_number))
        for i in range(len(self.lines)):
            length = len(self.lines[i].recover_level)
            for j in range(maximum_recover_level_length):
                if j < length:
                    numpy_recover_level_need[i][j] = 1
                    numpy_recover_level[i][j] = self.lines[i].recover_level[j]
                    numpy_recover_angle[i][j] = self.lines[i].recover_angle[j]
                else:
                    numpy_recover_level_need[i][j] = 0
        # construct string information including string number in each completed constraint and every id and dir
        numpy_string_number                 = np.array([len(self.method["id"][i]) for i in range(self.constraint_number)], dtype=np.int32)
        self.tsa_string_number = 0
        for i in range(self.constraint_number):
            self.tsa_string_number += len(self.method["id"][i]) - 1 # will multi 2

        if numpy_string_number.size != 0:
            max_string_number = max(numpy_string_number)
        else:
            max_string_number = 0
            numpy_string_number = np.array([1])
        
        string_force_initial_discount = np.array([[0.0 for _ in range(max_string_number - 1)] for _ in range(max(1, self.constraint_number))])
        parsed_string_information = []
        
        #check individual panel
        individual_crease_pair_list = []
        # for unit in self.ori_sim.unit_list:
        #     crease_num = len(unit.crease)
        #     for i in range(crease_num):
        #         previous_id = (i - 1 + crease_num) % crease_num
        #         next_id = (i + 1) % crease_num
        #         if unit.crease[i].getType() != BORDER and unit.crease[previous_id].getType() == BORDER and unit.crease[next_id].getType() == BORDER:
        #             kp1 = unit.crease[i][START]
        #             kp2 = unit.crease[i][END]
        for i in range(self.crease_pairs_num):
            target_kp1 = self.kps[numpy_crease_pairs[i][0]]
            target_kp2 = self.kps[numpy_crease_pairs[i][1]]
            have_border = [False, False]
            
            for unit in self.ori_sim.unit_list:
                crease_num = len(unit.crease)
                for ii in range(crease_num):
                    if unit.crease[ii].getType() == BORDER:
                        kp1 = unit.crease[ii][START]
                        kp2 = unit.crease[ii][END]

                        if (distance3D(kp1, target_kp1) < 1e-3 or distance3D(kp2, target_kp1) < 1e-3):
                            have_border[0] = True

                        if (distance3D(kp1, target_kp2) < 1e-3 or distance3D(kp2, target_kp2) < 1e-3):
                            have_border[1] = True

            if have_border[0] and have_border[1] and i not in individual_crease_pair_list:
                individual_crease_pair_list.append(i)
        
        self.individual_crease_num = len(individual_crease_pair_list)
                
        numpy_equivalent_internal_id = np.array([-1 for _ in range(max(1, self.constraint_number))])
        self.all_individual_crease_flag = [True for _ in range(max(1, self.constraint_number))]
        self.control_crease_ids = [[] for _ in range(max(1, self.constraint_number))]
        self.error_back_up = [2 * math.pi for _ in range(max(1, self.constraint_number))]
        
        for i in range(self.constraint_number):
            parsed_string = []
            previous_vector = np.array([0., 0., 0.])
            current_vector = np.array([0., 0., 0.])
            bonus_left = 1.0
            for j in range(numpy_string_number[i]):
                parsed_string.append([0 if self.method["type"][i][j] == 'A' else 1, self.method["id"][i][j], self.method["reverse"][i][j]])
                if j >= 1:
                    previous_type = parsed_string[j - 1][0]
                    current_type = parsed_string[j][0]
                    find_equal_internal_point = -1
                    if j == 1:
                        external_point = np.array(self.P_candidate[parsed_string[0][1]])
                        for k in range(self.unit_indices_num):
                            internal_indices = self.ori_sim.indices[k]
                            internal_center = sum([numpy_kps[internal_indices[l]] * self.contributions[k][l] for l in range(self.unit_edge_max)]) + np.array(self.total_bias + [-self.origami_z_bias])
                            if distance3D(internal_center, external_point) < 5:
                                find_equal_internal_point = k
                                break
                    if find_equal_internal_point >= 0:
                        previous_id = find_equal_internal_point
                        numpy_equivalent_internal_id[i] = previous_id
                    else:
                        previous_id = parsed_string[j - 1][1]
                    current_id = parsed_string[j][1]
                    if (j > 1) or find_equal_internal_point >= 0:
                        previous_indices = self.ori_sim.indices[previous_id]
                        previous_center = sum([numpy_kps[previous_indices[k]] * self.contributions[previous_id][k] for k in range(self.unit_edge_max)])
                    else:
                        previous_center = self.P_candidate[parsed_string[0][1]] - np.array(self.total_bias + [-self.origami_z_bias])
                    current_indices = self.ori_sim.indices[current_id]
                    current_center = sum([numpy_kps[current_indices[k]] * self.contributions[current_id][k] for k in range(self.unit_edge_max)])
                    current_vector = current_center - previous_center
                    if j == 1:
                        discount_factor = 1
                    else:
                        angle = np.arccos(np.clip(previous_vector.dot(current_vector) / (np.linalg.norm(previous_vector) * np.linalg.norm(current_vector)), -1.0, 1.0))
                        if angle < np.pi * 0.25:
                            discount_factor = self.gamma_bound_0_degree[0] + (angle / (np.pi * 0.25)) * (self.gamma_bound_45_degree[0] - self.gamma_bound_0_degree[0])
                        elif angle < np.pi * 0.5:
                            discount_factor = self.gamma_bound_45_degree[0] + ((angle - np.pi * 0.25) / (np.pi * 0.25)) * (self.gamma_bound_90_degree[0] - self.gamma_bound_45_degree[0])
                        elif angle < np.pi * 0.75:
                            discount_factor = self.gamma_bound_90_degree[0] + ((angle - np.pi * 0.5) / (np.pi * 0.25)) * (self.gamma_bound_135_degree[0] - self.gamma_bound_90_degree[0])
                        else:
                            discount_factor = self.gamma_bound_135_degree[0] + ((angle - np.pi * 0.75) / (np.pi * 0.25)) * (self.gamma_bound_180_degree[0] - self.gamma_bound_135_degree[0])
                    
                    bonus_left = discount_factor * bonus_left - self.static_friction
                    string_force_initial_discount[i][j - 1] = discount_factor
                    if bonus_left < 0:
                        bonus_left = 0
                    previous_vector = -current_vector
                    if (j > 1) or find_equal_internal_point >= 0:    
                        share_indices = [x for x in previous_indices if x != -1 and x in current_indices]
                        if len(share_indices) == 2:
                            for k in range(self.crease_pairs_num):
                                if (self.ori_sim.crease_pairs[k][0] == share_indices[0] and self.ori_sim.crease_pairs[k][1] == share_indices[1]) or \
                                    (self.ori_sim.crease_pairs[k][1] == share_indices[0] and self.ori_sim.crease_pairs[k][0] == share_indices[1]):
                                        x1 = numpy_kps[share_indices[0]]
                                        x2 = numpy_kps[share_indices[1]]
                                        angle = np.arccos(np.clip(current_vector.dot(x2 - x1) / (np.linalg.norm(current_vector) * np.linalg.norm(x2 - x1)), -1.0, 1.0))
                                        equivalent_torque_influence_id[i][k] += 1 << (j - 1)

                                        s = 0
                                        for s in range(numpy_line_indices.shape[0]):
                                            if (numpy_line_indices[s][0] == self.ori_sim.crease_pairs[k][0] and numpy_line_indices[s][1] == self.ori_sim.crease_pairs[k][1]) or \
                                                (numpy_line_indices[s][1] == self.ori_sim.crease_pairs[k][0] and numpy_line_indices[s][0] == self.ori_sim.crease_pairs[k][1]):
                                                break
                                        crease_type = numpy_line_indices[s][2]
                                
                                        if (crease_type == VALLEY and parsed_string[-1][2] == 1) or \
                                            (crease_type == MOUNTAIN and parsed_string[-1][2] == -1):
                                            equivalent_torque_index[i][k][j] = np.sin(angle)
                                        else:
                                            equivalent_torque_index[i][k][j] = -np.sin(angle)

                                        if k not in individual_crease_pair_list:
                                            self.all_individual_crease_flag[i] = False
                                        else:
                                            self.control_crease_ids[i].append(k)
                                        break
                        else:
                            # use geometry info
                            # previous_share_indices = [previous_indices[0], previous_indices[1]]
                            recorded_angles = []
                            recorded_need_torque_ids = []
                            
                            for kk in range(self.crease_pairs_num):
                                kp1 = numpy_kps[self.ori_sim.crease_pairs[kk][0]]
                                kp2 = numpy_kps[self.ori_sim.crease_pairs[kk][1]]
                                v1 = kp2 - previous_center
                                v2 = kp1 - previous_center
                                v3 = kp2 - current_center
                                v4 = kp1 - current_center
                                val = (v1[X] * v2[Y] - v2[X] * v1[Y]) * (v3[X] * v4[Y] - v4[X] * v3[Y])
                                val2 = (v1[X] * v3[Y] - v3[X] * v1[Y]) * (v2[X] * v4[Y] - v4[X] * v2[Y])
                                
                                if val < 1e-3 and val2 < 1e-3:
                                    s = 0
                                    for s in range(numpy_line_indices.shape[0]):
                                        if (numpy_line_indices[s][0] == self.ori_sim.crease_pairs[kk][0] and numpy_line_indices[s][1] == self.ori_sim.crease_pairs[kk][1]) or \
                                            (numpy_line_indices[s][1] == self.ori_sim.crease_pairs[kk][0] and numpy_line_indices[s][0] == self.ori_sim.crease_pairs[kk][1]):
                                            break
                                    crease_type = numpy_line_indices[s][2]

                                    if (crease_type == VALLEY and parsed_string[-1][2] == 1) or \
                                        (crease_type == MOUNTAIN and parsed_string[-1][2] == -1):
                                            angle = np.arccos(np.clip(current_vector.dot(kp2 - kp1) / (np.linalg.norm(current_vector) * np.linalg.norm(kp2 - kp1)), -1.0, 1.0))
                                            recorded_need_torque_ids.append(kk)
                                            recorded_angles.append(np.sin(angle))

                                    elif (crease_type == MOUNTAIN and parsed_string[-1][2] == 1) or \
                                        (crease_type == VALLEY and parsed_string[-1][2] == -1):
                                            angle = np.arccos(np.clip(current_vector.dot(kp2 - kp1) / (np.linalg.norm(current_vector) * np.linalg.norm(kp2 - kp1)), -1.0, 1.0))
                                            recorded_need_torque_ids.append(kk)
                                            recorded_angles.append(-np.sin(angle))

                                    if kk not in individual_crease_pair_list:
                                        self.all_individual_crease_flag[i] = False
                                    else:
                                        self.control_crease_ids[i].append(kk)

                            for kk in range(len(recorded_need_torque_ids)):
                                equivalent_torque_influence_id[i][recorded_need_torque_ids[kk]] += 1 << (j - 1)
                                equivalent_torque_index[i][recorded_need_torque_ids[kk]][j] = recorded_angles[kk] / len(recorded_need_torque_ids)
   
                # parsed_string.append([0 if self.string_total_information[i][j].point_type == 'A' else 1, self.string_total_information[i][j].id, self.string_total_information[i][j].dir])
            for j in range(numpy_string_number[i], max_string_number):
                parsed_string.append([0, -1, 0])
            parsed_string_information.append(parsed_string)

        for i in range(self.constraint_number):
            for j in range(self.crease_pairs_num):
                first_time = True
                sign = 0
                for k in range(self.max_control_length):
                    if first_time and equivalent_torque_index[i][j][k] != 0:
                        sign = 1 if equivalent_torque_index[i][j][k] > 0 else -1
                        first_time = False
                        continue
                    if not first_time and sign * equivalent_torque_index[i][j][k] < 0:
                        print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + 'Warning routing. Strings overpass and underpass the crease at the same time.')
                        self.dead_count = 4800
                        self.candidate_method = False

        numpy_parsed_string_information     = np.array(parsed_string_information, dtype=np.int32) if len(parsed_string_information) != 0 else np.array([[[0, -1, 0]]], dtype=np.int32)
        numpy_string_end                    = np.array([self.method["id"][i][-1] if self.method["type"][i][-1] == 'A' else -1 for i in range(self.constraint_number)])

        if self.P_number > 0:
            numpy_p_candidator = np.array(self.P_candidate) - np.array(self.total_bias + [-self.origami_z_bias])
        else:
            numpy_p_candidator = np.array([[]])
        numpy_p_candidator_connection = np.array(self.P_candidate_connection, dtype=np.int32)

        numpy_target_angle = np.ones(self.crease_pairs_num, dtype=np.float64)
        if len(self.targets):
            numpy_target_angle = np.array(self.targets)
                    
        # initialize!
        self.initialize(
            numpy_indices, numpy_kps, numpy_mass_list, numpy_tri_indices, 
            numpy_connection_matrix, 
            numpy_bending_pairs, numpy_crease_pairs, 
            numpy_line_indices, numpy_facet_bending_pairs, numpy_facet_crease_pairs, self.sim_mode, numpy_string_number, 
            numpy_parsed_string_information, numpy_string_end, numpy_original_kps, numpy_tb_line, self.lame_k,
            numpy_p_candidator, numpy_p_candidator_connection, epsilon_v, epsilon_string, numpy_contributions,
            numpy_recover_level_need, numpy_recover_level, numpy_recover_angle, numpy_noise, equivalent_torque_influence_id, 
            equivalent_torque_index, self.origami_z_bias, numpy_fix_id, self.dt, numpy_crease_noise,
            string_force_initial_discount, numpy_equivalent_internal_id, numpy_connected_unit_id, self.facet_mode, self.rot_z, numpy_target_angle
        )
        
        # 优化：厚板折纸快速模式下，设置更大的folding_micro_step以实现5°步进
        # Optimization: In thick origami fast mode, set larger folding_micro_step for 5° stepping
        if self.sim_mode == self.FOLD_SIM and getattr(self, 'thick_mode_flag', False) and self.check_connection_matrix:
            # 5度对应的弧度值 / 5 degrees in radians
            self.folding_micro_step[0] = 5.0 * np.pi / 180.0  # = 0.0873 rad ≈ 5°
            print(f"[优化] 厚板折纸快速模式：folding_micro_step 设置为 {self.folding_micro_step[0]} rad ({5.0}°)")
            print(f"[Optimization] Thick origami fast mode: folding_micro_step set to {self.folding_micro_step[0]} rad ({5.0}°)")
        
        if self.sim_mode == self.TSA_SIM:
            self.initial_string_length_list = [(self.constraint_initial_length[i] + self.additional_length_of_string) for i in range(self.constraint_number)]
            max_length = max(self.initial_string_length_list)
            self.basic_contraction = max_length * self.string_contract_ratio_initial
        
            for i in range(self.constraint_number):
                self.control_signals[0][i] = self.basic_contraction
                self.control_signals[1][i] = max_length
                # self.control_signals[1][i] = self.basic_contraction + 3.0
            
            if self.origami_name == 'f-5panel-robot':
                self.control_signals = [
                    [4, 4, 4, 4],
                    [24, 2, 2, 4],
                    [4, 10, 10, 4],
                ]
            print(self.control_signals)
                
        # valid actions
        if self.sim_mode == self.FOLD_SIM and self.check_connection_matrix and not self.thick_mode_flag:
            self.valid_matrix_list = [[] for _ in range(self.connection_matrix_interval - 1)]
            self.initial_matrix = np.zeros((self.unit_indices_num - self.connection_number[0] + self.P_number, self.unit_indices_num - self.connection_number[0] + self.P_number), int)
            self.O_points = [
                np.array(self.units[i].getCenterUsingContribution(self.contributions[i])) for i in range(self.unit_indices_num - self.connection_number[0])
            ]
            all_connection_id = [self.constraint_start_point_candidate_connection[i] for i in range(self.P_number)]

            for old_stand_point_id in range(self.unit_indices_num - self.connection_number[0] + self.P_number):
                for current_stand_point_id in range(old_stand_point_id, self.unit_indices_num - self.connection_number[0] + self.P_number):
                    if (old_stand_point_id >= self.P_number or current_stand_point_id >= self.P_number) and \
                        (
                            ((old_stand_point_id - self.P_number) >= 0 and (old_stand_point_id - self.P_number) in all_connection_id or \
                            (current_stand_point_id - self.P_number) >= 0 and (current_stand_point_id - self.P_number) in all_connection_id) or \
                            ((old_stand_point_id - self.P_number) >= 0 and (old_stand_point_id - self.P_number) in self.fix_id) or \
                            ((current_stand_point_id - self.P_number) >= 0 and (current_stand_point_id - self.P_number) in self.fix_id)
                        ):
                        self.initial_matrix[old_stand_point_id][current_stand_point_id] = -10
                        self.initial_matrix[current_stand_point_id][old_stand_point_id] = -10
                        continue
                    if old_stand_point_id < self.P_number and current_stand_point_id < self.P_number:
                        self.initial_matrix[old_stand_point_id][current_stand_point_id] = -10
                        self.initial_matrix[current_stand_point_id][old_stand_point_id] = -10
                    else:
                        old_stand_point = self.P_candidate[old_stand_point_id] if old_stand_point_id < self.P_number else self.O_points[old_stand_point_id - self.P_number]
                        new_stand_point = self.P_candidate[current_stand_point_id] if current_stand_point_id < self.P_number else self.O_points[current_stand_point_id - self.P_number]
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
                            elif intersection_crease_type[BORDER] == 0 and intersection_crease_type[MOUNTAIN] > 0 and intersection_crease_type[MOUNTAIN] <= (max((CREASE_CROSS - 1), 1) if self.control_mode else CREASE_CROSS) and intersection_crease_type[VALLEY] == 0:
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
                            self.initial_matrix[old_stand_point_id][current_stand_point_id] = side
                            self.initial_matrix[current_stand_point_id][old_stand_point_id] = side
                        else:
                            self.initial_matrix[old_stand_point_id][current_stand_point_id] = -10
                            self.initial_matrix[current_stand_point_id][old_stand_point_id] = -10
        self.dx_list = []
    
    def calculateIntersectionWithCreases(self, P_choice, O_choice, creases):
        ids = []
        unsure_ids = []
        for i in range(len(creases)):
            # if creases[i].getLength() < 4:
            #     continue
            k1 = creases[i].k()
            k2 = (P_choice[Y] - O_choice[Y]) / (P_choice[X] - O_choice[X]) if P_choice[X] != O_choice[X] else math.inf
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
                if abs(k1 - k2) < 1e-3:
                    max_x_1 = max(crease_start_point[X], crease_end_point[X])
                    min_x_1 = min(crease_start_point[X], crease_end_point[X])
                    max_x_2 = max(P_choice[X], O_choice[X])
                    min_x_2 = min(P_choice[X], O_choice[X])
                    if not (max_x_1 <= min_x_2 or max_x_2 <= min_x_1):
                        unsure_ids.append(i)
                else:
                    unsure_ids.append(i)
        return ids, unsure_ids

    def deal_with_key(self, key):
        self.dt_bonus[0] = 1
        self.key = ''
        if key == 'r':
            self.origami_z_bias = self.h
            noise = None
            if self.strict == 2:
                noise_dict = self.noise_dict
                noise = np.random.normal(loc=0, scale=0.01, size=ori.bending_pairs_num)
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
            self.initializeRunning(self.strict, noise)
            if self.control_mode == 1:
                self.additional_folder = 'phase1'
            else:
                self.additional_folder = ''

            if self.sim_mode == self.FOLD_SIM:
                self.gravitational_acc = tm.vec3([0., 0., 0.]) 
                self.tsa_turning_angle = 0.0
                self.enable_tsa_rotate = 0.0
                self.enable_add_folding_angle = 0.0
                self.folding_angle = 0.0
            else:
                self.gravitational_acc = tm.vec3(self.standard_g) 
                self.folding_angle = 0.0
                self.enable_tsa_rotate = 0.0
                self.enable_add_folding_angle = 0.0

        elif key == 'c':
            if self.sim_mode == self.FOLD_SIM:
                self.sim_mode = self.TSA_SIM
            else:
                self.sim_mode = self.FOLD_SIM

            noise = None
            if self.strict == 2:
                noise_dict = self.noise_dict
                noise = np.random.normal(loc=0, scale=0.01, size=ori.bending_pairs_num)
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
            self.initializeRunning(self.strict, noise)
            self.current_t = 0

            if self.sim_mode == self.TSA_SIM:
                self.gravitational_acc = tm.vec3(self.standard_g) 
                self.folding_angle = 0.0
                self.enable_tsa_rotate = 0.0
                self.enable_add_folding_angle = 0.0
            else:
                self.gravitational_acc = tm.vec3([0., 0., 0.]) 
                # self.string_length_decrease = 0.0
                self.enable_tsa_rotate = 0.0
                self.enable_add_folding_angle = 0.0
                self.folding_angle = 0.0
        else:
            if self.sim_mode == self.FOLD_SIM:
                if key == 'u': 
                    self.folding_angle += self.folding_step
                    if self.folding_angle >= self.folding_max:
                        self.folding_angle = self.folding_max
                
                elif key == 'j': 
                    self.folding_angle -= self.folding_step
                    if self.folding_angle <= 0:
                        self.folding_angle = 0

                elif key == 'i': 
                    self.enable_add_folding_angle = self.folding_micro_step[0]
                
                elif key == 'k': 
                    self.enable_add_folding_angle = 0.0
                
                elif key == 'm': 
                    self.enable_add_folding_angle = -self.folding_micro_step[0]
                elif key == 'p':
                    temp_matrix, _, _ = self.getValidMatrix()
                    self.valid_matrix_list.append(temp_matrix)
            else:
                if key == 'i': 
                    self.enable_tsa_rotate = self.string_length_decrease_step

                elif key == 'k': 
                    self.enable_tsa_rotate = 0.0
            
                elif key == 'm': 
                    self.enable_tsa_rotate = -self.string_length_decrease_step
                    self.control_steps = 0

                elif key == 'p':
                    self.paused = not self.paused

                elif key == ti.ui.SPACE:
                    self.step_once = True

        self.key = key
    
    def deal_with_motion(self, motion):
        pass
  
    def update_folding_target(self):
        if self.sim_mode == self.FOLD_SIM:
            self.folding_angle += self.enable_add_folding_angle
            if self.folding_angle >= self.folding_max:
                self.folding_angle = self.folding_max
            if self.folding_angle <= 0.0:
                self.folding_angle = 0.0
            # 优化：厚板折纸模式下跳过连接矩阵计算，提高计算效率
            # Optimization: Skip connection matrix calculation in thick origami mode for better efficiency
            if self.check_connection_matrix and not getattr(self, 'thick_mode_flag', False) and \
               (int(self.folding_angle * 180 / np.pi) != 0 and int(self.folding_angle * 180 / np.pi) != 180 and int(self.folding_angle * 180 / np.pi) % int(self.connection_matrix_step) == 0):
                temp_matrix, _, _ = self.getValidMatrix()
                self.valid_matrix_list[int(self.folding_angle * self.connection_matrix_interval / np.pi) - 1] = temp_matrix
                self.folding_angle += self.enable_add_folding_angle * self.connection_matrix_interval

        else:
            if self.constraint_number > 0:
                self.rotation = False
                self.exist_non_rotation = False
                self.is_rotating = 0
                # if self.additional_folder == "phase1" and self.ground_attachment[0] and self.control_steps == 0:
                #     self.control_signals[0] = [self.string_length_decrease[ele] for ele in range(self.constraint_number)]
                self.total_string_force = sum([self.string_force_each[i] for i in range(self.constraint_number)])
                self.contract_speed_discount = 1.0
                
                if self.can_rotate:
                    max_initial_length = max([self.string_number_each[i] for i in range(self.constraint_number)])
                    self.dead_count_adder = 0

                    actuated_individual_crease = []
                    
                    for i in range(self.constraint_number):
                        if self.rotation_list_backup[i] and ((self.constraint_initial_length[i] - self.constraint_length[i] + self.tolerance[i] >= self.string_length_decrease[i]) or (self.enable_tsa_rotate < 0)):
                            if self.all_individual_crease_flag[i]:
                                control_crease_i = self.control_crease_ids[i]
                                for ele in control_crease_i:
                                    if ele not in actuated_individual_crease:
                                        actuated_individual_crease.append(ele)
                                current_error = 0.0
                                for ele in control_crease_i:
                                    # if len(self.targets):
                                    error = (self.crease_angle[ele] - self.target_angles[ele]) * np.pi
                                    if abs(error) > current_error:
                                        current_error = abs(error)
                                    # else:
                                    #     error = (self.crease_angle[ele] - 1.0) * np.pi
                                    #     if abs(error) > current_error:
                                    #         current_error = abs(error)
                                # current_error /= len(control_crease_i)
                                if current_error > self.error_back_up[i]:
                                    if error != 0:
                                        if abs(error) > FOLDING_ERROR_MINIMUM:
                                            self.rotation_list[i] = -np.sign(error) 
                                            self.staggering_flag[i] = 0
                                        else:
                                            self.rotation_list[i] = -(error) * 0.5 / np.pi
                                            self.staggering_flag[i] = 1
                                    else:
                                        self.rotation_list[i] = 0.
                                        self.staggering_flag[i] = 1
                                else:
                                    self.rotation_list[i] = 1.
                                    self.error_back_up[i] = current_error
                                    self.staggering_flag[i] = 0
                                self.rotation = True
                            else:
                                self.rotation_list[i] = 1.
                                self.rotation = True
                        else:
                            # self.rotation = False
                            if not (self.system_type[0] == 3 and not self.first_phase_enable):
                                self.rotation_list[i] = 0.
                                self.rotation_list_backup[i] = 0.
                                self.dead_count_adder += 1
                                self.exist_non_rotation = True
                            # break
                    
                    if self.neglect_initial_penetration[0] and ((not self.exist_non_rotation and self.system_type[0] != 3) or (self.first_phase_enable and self.system_type[0] == 3)):
                        self.neglect_count[0] += 1
                        if self.neglect_count[0] >= self.n:
                            self.neglect_initial_penetration[0] = False

                    # if len(actuated_individual_crease) < self.individual_crease_num:
                    #     self.dead_count = 4800
                    #     self.candidate_method = False
                    
                    if self.control_mode:
                        # if self.total_string_force <= 5:
                        #     self.contract_speed_discount = (-124.689 * self.total_string_force + 2215.72)
                        # else:
                        #     self.contract_speed_discount = (-6.78937 * self.total_string_force ** 2 - 56.7953 * self.total_string_force + 2046.76)
                        self.contract_speed_discount = max(0.01, -120.259 * self.total_string_force + 2556.81)
                        self.contract_speed_discount *= 3.9111e-4

                        if not self.actuation_discount:
                            self.contract_speed_discount = 1.
                        
                        if self.total_string_force > self.control_mode_max_force and self.first_phase_enable:
                            self.rotation = False
                            self.exist_non_rotation = True
                            for i in range(self.constraint_number):
                                self.rotation_list[i] = 0.
                        
                        if not self.first_phase_enable and self.total_string_force > self.control_mode_max_force and self.string_length_decrease[0] < self.basic_contraction - 1e-3:
                            self.rotation = False
                            self.exist_non_rotation = True
                            for i in range(self.constraint_number):
                                self.rotation_list[i] = 0.
                            self.dead_count = 4800
                            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | Warning string tensions!")

                    if self.control_mode == 0:
                        if self.rotation:
                            if not self.error_status[0] or self.enable_tsa_rotate < 0 or (self.error_status[0] and len(self.recorded_maximum_folding_percent) and max([abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])]) < 0.1):
                                contract_length = self.enable_tsa_rotate
                                self.is_rotating = np.sign(self.enable_tsa_rotate)
                                for ele in range(self.constraint_number):
                                    self.string_length_decrease[ele] += contract_length * self.rotation_list[ele] #* self.string_number_each[ele] / max_initial_length 
                            
                    else:
                        if not self.exist_non_rotation or not self.first_phase_enable:
                            if self.control_steps == 0 and self.enable_tsa_rotate < 0 and (not self.error_status[0] and not self.exist_non_rotation):
                                self.enable_tsa_rotate = self.string_length_decrease_step
                                self.allow_initialize = False
                                
                            if self.enable_tsa_rotate != 0.0 and (not self.error_status[0] or self.enable_tsa_rotate < 0):
                                if self.control_steps == 1 and not self.first_phase_enable:
                                    contract_length = 0.0
                                    move_indice = self.compute_move_indice()
                                    if move_indice < 40.0:
                                        self.stable_state += 1
                                    else:
                                        self.stable_state = 0
                                        
                                    if self.stable_state >= self.n:
                                        self.first_phase_enable = True
                                        self.actuation_start_t = self.current_t
                                        self.stable_state = 0
                                        if self.additional_folder != "phase1" and len(self.recorded_movement_x):
                                            self.recorded_interval_velocity.clear()
                                            self.recorded_interval_velocity.append([[self.recorded_movement_x[-1], self.recorded_movement_y[-1]], self.current_t])
                                            # if len(self.recorded_interval_velocity) == 1:
                                            self.recorded_movement_x.clear()
                                            self.recorded_movement_y.clear()
                                            self.recorded_movement_z.clear()
                                            self.offset_x = self.recorded_interval_velocity[0][0][X]
                                            self.offset_y = self.recorded_interval_velocity[0][0][Y]

                                    self.robot_stand = False
                                    max_z = self.ground_barrier - 1.
                                    for i in range(self.kp_num - 4 * self.connection_number[0]):
                                        if self.x[i][Z] > max_z:
                                            max_z = self.x[i][Z]

                                    if max_z >= self.ground_barrier + self.controller_z_warning:
                                        self.robot_stand = True
                                    
                                    if not self.robot_stand:
                                        self.allow_initialize = True

                                else:
                                    actuator_on_ground = 0
                                    for i in range(4):
                                        if self.x[self.unit_indices[self.unit_indices_num - 1][i]][Z] < self.ground_barrier + self.controller_z_warning:
                                            actuator_on_ground += 1
                                    if actuator_on_ground >= 2:
                                        if self.additional_folder == "phase1":
                                            self.recorded_signals = [self.string_length_decrease[i] for i in range(self.constraint_number)]
                                            min_val = min(self.recorded_signals)
                                            min_val = max(min_val, self.basic_contraction)
                                            self.recorded_signals = [((min_val - self.basic_contraction) * self.stroke_percent + self.basic_contraction) for _ in range(self.constraint_number)]
                                            # self.recorded_signals = [(min_val) for _ in range(self.constraint_number)]
                                            self.allow_initialize = True
                                            self.enable_tsa_rotate = -self.string_length_decrease_step
                                            self.is_rotating = np.sign(self.enable_tsa_rotate)
                                            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + "Actuator hit the ground, stop contracting...")
                                        elif self.additional_folder == "phase2":
                                            self.dead_count = 4800
                                            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | Actuation failed in phase 2! (ERROR 02)")
                                    contract_length = self.string_length_decrease_step
                                controller_error = 0.0
                                total_ratio = 1.
                                contract_length *= total_ratio
                                true_control_step = (self.control_steps) % len(self.control_signals)
                                for ele in range(self.constraint_number):
                                    if self.enable_tsa_rotate > 0 and not self.allow_initialize: # execute actions
                                        if ele < len(self.control_signals[true_control_step]):
                                            true_contract_length = contract_length
                                            self.is_rotating = 1
                                            if self.string_length_decrease[ele] == self.control_signals[true_control_step][ele]:
                                                true_contract_length = 0
                                                self.is_rotating = 0
                                            elif self.string_length_decrease[ele] > self.control_signals[true_control_step][ele]:
                                                true_contract_length = -contract_length
                                                self.is_rotating = -1
                                                
                                            if true_contract_length > 0:
                                                true_contract_length *= self.contract_speed_discount
                                            
                                            if (self.current_t - self.wait_time) > self.wait_interval:
                                                self.string_length_decrease[ele] += true_contract_length * self.rotation_list[ele]# * self.string_number_each[ele] / max_initial_length
                                                if true_contract_length > 0 and self.string_length_decrease[ele] > self.control_signals[true_control_step][ele]:
                                                    self.string_length_decrease[ele] = self.control_signals[true_control_step][ele]
                                                elif true_contract_length < 0 and self.string_length_decrease[ele] < self.control_signals[true_control_step][ele]:
                                                    self.string_length_decrease[ele] = self.control_signals[true_control_step][ele]
                                            controller_error += (self.string_length_decrease[ele] - self.control_signals[true_control_step][ele]) ** 2
                                            self.is_rotating = np.sign(true_contract_length)
                                    else:
                                        true_contract_length = -contract_length
                                        self.is_rotating = -1
                                        self.string_length_decrease[ele] += true_contract_length #* self.string_number_each[ele] / max_initial_length
                                        if self.string_length_decrease[ele] < self.control_signals[0][ele]:
                                            if abs(self.string_length_decrease[ele] - self.control_signals[0][ele]) < contract_length:
                                                self.string_length_decrease[ele] = self.control_signals[0][ele]
                                            else:
                                                self.string_length_decrease[ele] += 2. * contract_length
                                                self.is_rotating = 1
                                        controller_error += (self.string_length_decrease[ele] - self.control_signals[0][ele]) ** 2
                                if self.enable_tsa_rotate < 0:
                                    self.error_status[0] = False
                                    if controller_error < 1e-5:
                                        self.enable_tsa_rotate = self.string_length_decrease_step
                                        self.is_rotating = np.sign(self.enable_tsa_rotate)
                                else:
                                    if controller_error < 1e-5:
                                        self.enable_tsa_rotate = self.string_length_decrease_step
                                        self.is_rotating = np.sign(self.enable_tsa_rotate)
                                        self.control_steps = self.control_steps + 1
                                        self.wait_time = self.current_t
                                        if self.additional_folder == "phase1" and self.control_steps == 2 and not self.allow_initialize:
                                            self.minimum_z = self.origami_z_bias
                                            self.recorded_signals = [self.string_length_decrease[i] for i in range(self.constraint_number)]
                                            min_val = min(self.recorded_signals)
                                            min_val = max(min_val, self.basic_contraction)
                                            self.recorded_signals = [((min_val - self.basic_contraction) * self.stroke_percent + self.basic_contraction) for _ in range(self.constraint_number)]
                                            # self.recorded_signals = [(min_val) for _ in range(self.constraint_number)]
                                            self.allow_initialize = True
                                            self.enable_tsa_rotate = -self.string_length_decrease_step
                                            self.is_rotating = np.sign(self.enable_tsa_rotate)
                                            # for i in range(self.kp_num):
                                            #     if self.kps[i][Z] < self.minimum_z:
                                            #         self.minimum_z = self.kps[i][Z]
                                        if (self.control_steps) % len(self.control_signals) == 1 and len(self.recorded_movement_x):
                                            self.recorded_interval_velocity.append([[self.recorded_movement_x[-1], self.recorded_movement_y[-1]], self.current_t])
                                            if len(self.recorded_interval_velocity) > 1 and self.recorded_interval_velocity[-1][1] - self.recorded_interval_velocity[-2][1] < self.recorded_interval_velocity[0][1] * 0.5:
                                                self.dead_count = 4800
                                                print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | Actuation failed in phase 2! (ERROR 01) " + str(self.error_status[0]))
                                            # if len(self.recorded_interval_velocity) == 1:
                                            #     self.recorded_movement_x.clear()
                                            #     self.recorded_movement_y.clear()
                                            #     self.recorded_movement_z.clear()
                                            #     self.offset_x = self.recorded_interval_velocity[0][0][X]
                                            #     self.offset_y = self.recorded_interval_velocity[0][0][Y]
                            else:
                                self.recorded_signals = [self.string_length_decrease[i] for i in range(self.constraint_number)]
                                min_val = min(self.recorded_signals)
                                min_val = max(min_val, self.basic_contraction)
                                self.recorded_signals = [((min_val - self.basic_contraction) * self.stroke_percent + self.basic_contraction) for _ in range(self.constraint_number)]
                                # self.recorded_signals = [(min_val) for _ in range(self.constraint_number)]
                                self.allow_initialize = True
                                self.enable_tsa_rotate = -self.string_length_decrease_step
                                self.is_rotating = np.sign(self.enable_tsa_rotate)
                        else:
                            self.recorded_signals = [self.string_length_decrease[i] for i in range(self.constraint_number)]
                            min_val = min(self.recorded_signals)
                            min_val = max(min_val, self.basic_contraction)
                            self.recorded_signals = [((min_val - self.basic_contraction) * self.stroke_percent + self.basic_contraction) for _ in range(self.constraint_number)]
                            # self.recorded_signals = [(min_val) for _ in range(self.constraint_number)]
                            self.allow_initialize = True
                            self.enable_tsa_rotate = -self.string_length_decrease_step
                            self.is_rotating = np.sign(self.enable_tsa_rotate)

    # @ti.kernel
    def Fc(self, facet_k: data_type, dt: data_type, mode: int, step: int):
        self.clearForce()
        self.clearNodalForce()
        #--------#
        self.stvkForce()
        if self.print:
            print(f"FORCE CALCULATE :: STVK energy: {self.total_energy[0]}")
        backup_energy = self.total_energy[0]
        
        self.fixForce(mode, step)
        if self.print:
            print(f"FORCE CALCULATE :: FIXED energy: {self.total_energy[0] - backup_energy}")
            
        if facet_k > 0:
            backup_energy = self.total_energy[0]
            self.facetBendingForce(facet_k, dt)
            if self.print:
                print(f"FORCE CALCULATE :: FACET energy: {self.total_energy[0] - backup_energy}")
        backup_energy = self.total_energy[0]
        
        self.dfdxConnectionForce()
        if self.print:
            print(f"FORCE CALCULATE :: DFDX energy: {self.total_energy[0] - backup_energy}")

    # @ti.kernel  
    def F1(self, folding_angle: data_type, dt: data_type):
        backup_energy = self.total_energy[0]
        self.bendingForceFoldSim(folding_angle, dt)
        if self.print:
            print(f"FORCE CALCULATE :: BENDING energy: {self.total_energy[0] - backup_energy}")

    # @ti.kernel  
    def F2(self, enable_ground: bool, dt: data_type, mode: int, step: int, iter: int):
        backup_energy = self.total_energy[0]
        if self.constraint_number > 0:
            # if self.friction_mode == 1:
            #     pass
            # elif self.friction_mode == 2:
            self.stringForce3(iter)
            # else:
            #     self.stringForce()
            if self.print:
                print(f"FORCE CALCULATE :: STRING energy: {self.total_energy[0] - backup_energy}")
        backup_energy = self.total_energy[0]
        if not iter:
            self.backup_string_force()
        self.bendingForceTSASim(dt)
        if self.print:
            print(f"FORCE CALCULATE :: BENDING energy: {self.total_energy[0] - backup_energy}")
        if enable_ground:
            backup_energy = self.total_energy[0]
            self.groundForce(mode, step)
            if self.print:
                print(f"FORCE CALCULATE :: GROUND energy: {self.total_energy[0] - backup_energy}")
            backup_energy = self.total_energy[0]
            # if iter:
            self.frictionForce()
            if self.print:
                print(f"FORCE CALCULATE :: GROUND FRICTION energy: {self.total_energy[0] - backup_energy}")

    @ti.kernel
    def getdf(self) -> data_type:
        df_norm = 0.0
        for i in ti.ndrange(3 * self.kp_num):
            df_norm += self.b[i] ** 2
        return tm.sqrt(df_norm)
    
    @ti.kernel
    def compute_move_indice(self) -> data_type:
        total = 0.0
        for i in ti.ndrange(self.kp_num):
            total += self.v[i].norm()
        return total / self.kp_num
    
    @ti.kernel
    def calculateFoldingAngle(self):
        folding_percent = 0.0
        abs_folding_percent = 1.0
        min_folding_percent = 1.0
        max_folding_percent = -1.0

        folding_error = 0.0
        min_folding_error = 2. * tm.pi
        max_folding_error = 0.0

        for i in ti.ndrange(self.bending_pairs_num):
            folding_percent += self.crease_angle[i]
            if abs(self.crease_angle[i]) < abs_folding_percent:
                abs_folding_percent = abs(self.crease_angle[i])
            if self.crease_angle[i] < min_folding_percent:
                min_folding_percent = self.crease_angle[i]
            if self.crease_angle[i] > max_folding_percent:
                max_folding_percent = self.crease_angle[i]
            current_error = abs(self.crease_angle[i] - self.target_angles[i]) * tm.pi
            folding_error += current_error
            if current_error > max_folding_error:
                max_folding_error = current_error
            if current_error < min_folding_error:
                min_folding_error = current_error

        self.fae_information[0] = folding_percent
        self.fae_information[1] = abs_folding_percent
        self.fae_information[2] = min_folding_percent
        self.fae_information[3] = max_folding_percent
        self.fae_information[4] = folding_error
        self.fae_information[5] = min_folding_error
        self.fae_information[6] = max_folding_error
    
    @ti.kernel
    def getMaxMises(self) -> data_type:
        max_val = self.mises[0]
        for i in ti.ndrange(self.kp_num):
            if self.mises[i] > max_val:
                max_val = self.mises[i]
        return max_val

    # @ti.kernel
    def Fm(self, gravitational_acc: tm.vec3, mode: int, step: int, t: data_type):
        backup_energy = self.total_energy[0]
        self.mergeForce(gravitational_acc, mode, step, t)
        if self.print:
            print(f"FORCE CALCULATE :: KINEMATIC & POTENTIAL energy: {self.total_energy[0] - backup_energy}")
            print(f"FORCE CALCULATE :: TOTAL energy: {self.total_energy[0]}")
            
    def stop(self):
        stop = False
        if self.control_mode == 0:
            if self.sim_mode:
                if (self.can_rotate and ((len(self.recorded_folding_error) >= 2 and self.recorded_folding_error[-1] >= self.recorded_folding_error[-2] - 1e-6) or \
                                        (len(self.recorded_maximum_folding_error) >= 2 and self.recorded_maximum_folding_error[-1] >= self.recorded_maximum_folding_error[-2] - 1e-6))) or \
                    (self.dead_count >= 2. * self.DEAD_MAXIMUM and not self.can_rotate) or (self.dead_count >= self.DEAD_MAXIMUM and self.can_rotate):
                    if self.can_rotate:
                        if (self.recorded_folding_error[-1] < FOLDING_ERROR_MINIMUM and ((len(self.recorded_folding_error) >= int(self.n * 2.5) and self.recorded_folding_error[-1] >= np.mean(np.array(self.recorded_folding_error[-min(int(self.n * 2.5), len(self.recorded_folding_error)): -1])) - 1e-6) or \
                                                                                        (len(self.recorded_maximum_folding_error) >= int(self.n * 2.5) and self.recorded_maximum_folding_error[-1] >= np.mean(np.array(self.recorded_maximum_folding_error[-min(int(self.n * 2.5), len(self.recorded_maximum_folding_error)): -1])) - 1e-6))) or \
                            self.overall_string_maximum_force[0] > self.maximum_tension or \
                            (self.staggering_flag == True).all() or \
                            np.linalg.norm(self.rotation_list) < 1e-2:
                            if not (self.system_type[0] == 1 and tm.pi * self.abs_folding_percent < self.equal_arm_distance[0] and self.dead_count <= self.DEAD_MAXIMUM):
                                stop = True
                                if self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[0]:
                                    self.candidate_method = False
                                # elif self.recorded_folding_error[-1] > self.recorded_folding_error[0] * FOLDING_ERROR_CANDIDATE_RATIO:
                                #     self.candidate_method = False
                    else:
                        stop = True
                        if self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[0]:
                            self.candidate_method = False

                if self.current_t >= self.simulation_upper_time and (len(self.recorded_maximum_folding_percent) and max([abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])]) < 0.1):
                    stop = True
                    if self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[0]:
                        self.candidate_method = False
                
                if (self.rotation_list_backup == False).all():
                    stop = True
                    if self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[0]:
                        self.candidate_method = False
                
                if self.error_status[0] or self.dead_count >= 4800:
                    if not (self.system_type[0] == 1 and tm.pi * self.abs_folding_percent < self.equal_arm_distance[0] and self.dead_count <= self.DEAD_MAXIMUM):
                        stop = True
                        if self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[0]:
                            self.candidate_method = False
                        # elif self.recorded_folding_error[-1] > self.recorded_folding_error[0] * FOLDING_ERROR_CANDIDATE_RATIO:
                        #     self.candidate_method = False
            else:
                if self.check_connection_matrix and self.folding_angle >= self.folding_angle_maximum_ratio * math.pi:
                    move_indice = self.compute_move_indice()
                    if move_indice < 40.0:
                        self.stable_state += 1
                    else:
                        self.stable_state = 0
                    if self.stable_state >= self.n:
                        stop = True
        else:
            if self.sim_mode:
                if (self.additional_folder == "phase1" and self.control_steps == 0 and (self.error_status[0] or self.exist_non_rotation)):
                    self.enable_tsa_rotate = self.string_length_decrease_step
                    self.is_rotating = np.sign(self.enable_tsa_rotate)
                    self.control_steps = self.control_steps + 1
                    self.wait_time = self.current_t
                    # stop = True
                if (self.dead_count >= 4800):
                    self.candidate_method = False
                    stop = True
                if (self.dead_count >= self.DEAD_MAXIMUM and self.can_rotate and self.additional_folder == 'phase2'):
                    self.candidate_method = False
                    stop = True
                if self.current_t - self.actuation_start_t >= self.control_mode_simulation_time:
                    if self.additional_folder == 'phase2' and self.control_steps >= 2:
                    #     value, value_f, rf, rm, rs, actuator_bonus = ori.reward()
                    #     self.phase_2_value = value
                    #     current_signal = deepcopy(self.control_signals)
                    #     for i in range(self.constraint_number):
                    #         current_signal[1][i] = (current_signal[1][i] + self.basic_contraction) * 0.5
                    #     self.origami_z_bias = self.minimum_z
                    #     self.initializeRunning(strict)
                    #     self.control_signals = current_signal
                    #     self.gravitational_acc = tm.vec3(self.standard_g) 
                    #     self.folding_angle = 0.0
                    #     self.enable_tsa_rotate = 0.0
                    #     self.enable_add_folding_angle = 0.0
                    #     self.additional_folder = 'phase3'
                    # elif self.additional_folder == 'phase3':
                        if self.strict == -1:
                            self.candidate_method = False
                        stop = True
                if self.current_t >= self.control_mode_simulation_time * 1.5:
                    stop = True
                if self.strict == -1:
                    if self.control_steps >= 5 and (self.control_steps % 2):
                        displacement = [
                            self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][X] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][X],
                            self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][Y] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][Y]
                        ]
                        time_interval = self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][1] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][1]
                        displacement_baseline = [
                            self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][X] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 3][0][X],
                            self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][Y] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 3][0][Y]
                        ]
                        time_interval_baseline = self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][1] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 3][1]
                        distance = math.sqrt(displacement[X]**2 + displacement[Y]**2)
                        distance_baseline = math.sqrt(displacement_baseline[X]**2 + displacement_baseline[Y]**2)
                        angle = np.array(displacement) / np.linalg.norm(np.array(displacement))
                        angle_baseline = np.array(displacement_baseline) / np.linalg.norm(np.array(displacement_baseline))
                        if (abs(distance - distance_baseline) / distance_baseline > 0.1) or \
                            (abs(time_interval - time_interval_baseline) / time_interval_baseline > 0.1) or \
                            angle.dot(angle_baseline) < 0.5:
                            if self.control_steps == 7:
                                self.candidate_method = False
                                stop = True
                        else:
                            stop = True
                    
        # if stop:
        #     if self.use_gui:
        #         self.camera.track_user_inputs(self.window, movement_speed=0.23, hold_key=ti.ui.RMB)
        #         if self.sim_mode == self.TSA_SIM and len(self.recorded_movement_x):
        #             self.camera.position(self.recorded_movement_x[-1], self.recorded_movement_y[-1] + min(-1.5 * self.max_size, -400), max(1.5 * self.max_size, 400) + self.origami_z_bias)
        #             self.camera.lookat(self.recorded_movement_x[-1], self.recorded_movement_y[-1], self.recorded_movement_z[-1])
        #         self.scene.set_camera(self.camera)

        #         self.scene.point_light(pos=(0., 0., self.max_size + 3 * self.origami_z_bias), color=(0.8, 0.8, 0.8))
        #         self.scene.ambient_light((0.5, 0.5, 0.5))

        #         if self.enable_ground:
        #             self.scene.mesh(vertices=self.ground_vertices, indices=self.ground_indices, per_vertex_color=self.ground_vertices_color, two_sided=True)
        #             if len(self.recorded_indices):
        #                 # self.updateGroundLines(x, y, self.rot_z, self.ground_d)
        #                 self.scene.lines(vertices=self.ground_line_vertex,
        #                         width=1,
        #                         color=(0.6, 0.6, 0.6))
        #                 self.scene.lines(vertices=self.ground_fix_line_vertex,
        #                         width=4,
        #                         color=(0.25, 0.25, 0.25))
                    
        #         self.scene.mesh(self.vertices,
        #                 indices=self.indices,
        #                 per_vertex_color=self.vertices_color,
        #                 two_sided=True)
                
        #         self.fill_line_vertex()
        #         self.scene.lines(vertices=self.line_vertex,
        #                     width=2,
        #                     per_vertex_color=self.line_color)
                
        #         if len(self.recorded_indices):
        #             self.scene.particles(centers=self.center_trajectory, radius=1, color=(0.5625, 0.94, 0.5625))
        #         try:
        #             self.canvas.scene(self.scene)
        #             self.window.save_image(f'./physResult/TR-' + str(self.ID).zfill(8) + '.png')
        #         except:
        #             pass
        return stop

    def step(self, force_dir=False, check_force_dir=True):
        # self.can_rotate = True
        if self.use_gui:
            if self.window.get_event(ti.ui.PRESS):
                self.deal_with_key(self.window.event.key)
        
        if self.allow_initialize:
            if self.additional_folder == 'phase1':
                move_indice = self.compute_move_indice()
                if move_indice < 40.0:
                    self.stable_state += 1
                else:
                    self.stable_state = 0

                if self.robot_stand:
                    current_signal = [
                        (self.constraint_initial_length[i] - self.constraint_length[i]) for i in range(self.constraint_number)
                    ]
                    error = [
                        current_signal[i] - self.control_signals[0][i] for i in range(self.constraint_number)
                    ]
                    self.origami_z_bias = self.minimum_z
                    recorded_signal = deepcopy(self.recorded_signals)

                    for i in range(len(recorded_signal)):
                        recorded_signal[i] = round(recorded_signal[i], 2)

                    delta_contraction = recorded_signal[0] - self.control_signals[0][0]
                    if delta_contraction > 0.25 * self.basic_contraction:
                        self.initializeRunning(self.strict, clean_ratio=False)
                        self.gravitational_acc = tm.vec3(self.standard_g) 
                        self.folding_angle = 0.0
                        self.enable_tsa_rotate = 0.0
                        self.enable_add_folding_angle = 0.0
                        if abs(max(error)) < self.basic_contraction:
                            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + f"Unfolding successful, new signals: {recorded_signal}, to phase 2")
                            self.control_signals[1] = recorded_signal
                            self.allow_initialize = False
                            self.additional_folder = 'phase2'
                        else:
                            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + f"Can't be unfolded, new signals: {recorded_signal}, to phase 2")
                            self.control_signals[1] = recorded_signal
                            self.allow_initialize = False
                            self.additional_folder = 'phase2'
                    else:
                        self.dead_count = 4800
                        print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | Not enough stroke!")
                else:
                    self.string_contract_ratio_initial += 0.01

                    self.initializeRunning(self.strict, clean_ratio=False)
                    self.gravitational_acc = tm.vec3(self.standard_g) 
                    self.folding_angle = 0.0
                    self.enable_tsa_rotate = 0.0
                    self.enable_add_folding_angle = 0.0

                    print(f"Can't stand, new signals: {self.control_signals}, stay in phase 1")
                    self.allow_initialize = False
                    if self.string_contract_ratio_initial > 0.2:
                        self.dead_count = 4800
                        print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | Robot can't stand!")
            else:
                self.dead_count += 1

        if not self.paused or self.step_once:
            if self.sim_mode == self.FOLD_SIM:
                facet_k = self.facet_bending_bonus * self.facet_bending_param[0]
                if self.facet_bending_pairs_num > 0:
                    i = 0
                    while i < self.substeps:
                        i += 1
                        self.update_folding_target()
                        self.backup_xv()
                        self.backup_equivalent()
                        
                        self.dt_bonus[0] = 1.
                        # backup_energy = 0.0
                        for k in range(self.ITER):
                            self.Fc(facet_k, self.dt, 0, self.control_steps)
                            self.F1(self.folding_angle, self.dt)
                            self.Fm(self.gravitational_acc, 0, k, self.dt)

                            # Solve Equations
                            # self.fill_K(self.dt, self.AK, self.sim_mode)
                            self.fill_AK_field(self.dt, k)
                            self.fill_K_from_field(self.AK)
                            self.AM = self.AK.build()
                            self.sparse_solver.compute(self.AM)
                
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground, 0, k)

                            if dx_norm < self.dx_tolerance:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.current_t += self.dt
                else:
                    i = 0
                    while i < self.substeps:
                        i += 1
                        self.update_folding_target()
                        self.backup_xv()
                        self.backup_equivalent()
                        # backup_energy = 0.0
                        self.dt_bonus[0] = 1.
                        for k in range(self.ITER):
                            self.Fc(-1., self.dt, 0, self.control_steps)
                            self.F1(self.folding_angle, self.dt)
                            self.Fm(self.gravitational_acc, 0, k, self.dt)

                            # Solve Equations
                            # self.fill_K(self.dt, self.AK, self.sim_mode)
                            self.fill_AK_field(self.dt, k)
                            self.fill_K_from_field(self.AK)
                            self.AM = self.AK.build()
                            self.sparse_solver.compute(self.AM)
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)

                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground, 0, k)

                            if dx_norm < self.dx_tolerance:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.current_t += self.dt
            else:
                facet_k = self.facet_bending_bonus * self.facet_bending_param[0]
                if self.facet_bending_pairs_num > 0:
                    i = 0
                    # dx_list = []
                    while i < self.substeps:
                        i += 1
                        self.update_folding_target()   
                        self.backup_xv()
                        self.backupStringLength()
                        self.backup_equivalent()
                        self.updateStringParameters()
                        self.calculateDiscount(self.control_steps)
                        self.backupGroundForce()
                        self.dt_bonus[0] = 1.
                        for k in range(self.ITER):
                            self.Fc(facet_k, self.dt, self.control_mode, self.control_steps)
                            self.F2(self.enable_ground, self.dt, self.control_mode, self.control_steps, k)
                            self.Fm(self.gravitational_acc, self.control_mode, k, self.dt)

                            # Solve Equations
                            
                            # self.fill_K(self.dt, self.AK, self.sim_mode)
                            self.fill_AK_field(self.dt, k)
                            self.fill_K_from_field(self.AK)
                            # self.AK.print_triplets()
                            self.AM = self.AK.build(dtype=data_type)

                            self.sparse_solver.compute(self.AM)
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)
                            df_norm = self.getdf()

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground, self.control_mode, k)
                            if dx_norm < self.dx_tolerance:
                                break
                        
                        # if dx_norm > 0.4:
                        #     print(dx_norm)

                        # if dx_norm > 1:
                        #     a = 1

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        # for i in range(self.kp_num):
                        #     print(f"{i}, {self.v[i]}")
                        # self.backupStringLength()
                        self.current_t += self.dt

                else:
                    i = 0
                    while i < self.substeps:
                        i += 1
                        self.update_folding_target()
                        self.backup_xv()
                        self.backupStringLength()
                        self.backup_equivalent()
                        self.updateStringParameters()
                        self.calculateDiscount(self.control_steps)
                        self.backupGroundForce()
                        self.dt_bonus[0] = 1.
                        for k in range(self.ITER):
                            self.Fc(-1., self.dt, self.control_mode, self.control_steps)
                            self.F2(self.enable_ground, self.dt, self.control_mode, self.control_steps, k)
                            self.Fm(self.gravitational_acc, self.control_mode, k, self.dt)
                            
                            # Solve Equations
                            # self.fill_K(self.dt, self.AK, self.sim_mode)
                            self.fill_AK_field(self.dt, k)
                            self.fill_K_from_field(self.AK)
                            self.AM = self.AK.build()
                            self.sparse_solver.compute(self.AM)
                            if not k:
                                backup_energy = self.total_energy[0]
                            # print(self.total_energy[0])

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)
                            # print(dx_norm)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground, self.control_mode, k)

                            if dx_norm < self.dx_tolerance:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        # for i in range(self.kp_num):
                        #     print(f"{i}, {self.v[i]}")
                        # self.backupStringLength()
                        self.current_t += self.dt

            self.dx_list.append(dx_norm)
            if self.use_gui:             
                self.update_vertices() 

            if self.sim_mode == self.FOLD_SIM and self.enable_add_folding_angle == 0. and self.key == '' and self.check_connection_matrix and not self.control_mode:
                move_indice = self.compute_move_indice()
                if move_indice < 40.0:
                    self.stable_state += 1
                else:
                    self.stable_state = 0

                if (self.stable_state >= self.n):
                    self.enable_add_folding_angle = self.folding_micro_step[0]

                self.past_move_indice = move_indice  

            if self.sim_mode == self.FOLD_SIM:    
                if self.key == '' and self.folding_angle >= self.folding_angle_maximum_ratio * math.pi and self.check_connection_matrix:
                    self.enable_add_folding_angle = 0.0
                    self.stable_state = 0 
                    self.key = 'k'

            self.total_tension = round(sum([self.getActuationForce(self.string_force_each[i]) for i in range(self.constraint_number)]), 3)
            
            # if self.total_tension >= 50.0:
            #     self.dead_count = 4800

            if self.sim_mode == self.TSA_SIM:
                if self.constraint_number:   
                    if not self.can_rotate:
                        self.dead_count += 1
                        # self.folding_percent = 0.0
                        # self.abs_folding_percent = 0.0
                        # self.folding_error = 0.0
                        # for i in range(self.bending_pairs_num):
                        #     self.folding_percent += self.crease_angle[i]
                        #     if len(self.targets):
                        #         current_error = abs(self.crease_angle[i] - self.targets[i]) * np.pi
                        #     else:
                        #         current_error = abs(self.crease_angle[i] - 1.0) * np.pi
                        #     self.folding_error += current_error
                        # self.folding_percent /= self.bending_pairs_num
                        # self.folding_error /= self.bending_pairs_num
                        move_indice = self.compute_move_indice()
                        # if self.print:
                        #     print("Move indice: " + str(round(move_indice, 1)) + ", Current time: " + str(round(self.current_t, 3)) + ", Stable state: " + str(self.stable_state))
                        if move_indice < 40.0:
                            self.stable_state += 1
                        else:
                            self.stable_state = 0

                        if (self.stable_state >= self.n and self.control_mode == 0) or (self.control_mode and self.stable_state >= self.n):# or self.control_mode:
                            self.can_rotate = True
                            self.stable_state = 0
                            self.enable_tsa_rotate = self.string_length_decrease_step
                            self.dead_count = 0
                            for i in range(self.constraint_number):
                                self.string_length_each[i] = self.constraint_initial_length[i]
                            # if self.print:
                            #     print("Actuation ok")

                        self.past_move_indice = move_indice

                    else:
                        exist_error_candidate = self.checkErrorBuffer()
                        if not self.rotation or (exist_error_candidate or self.error_status[0]):
                            self.dead_count += 1
                            if ((self.error_status[0] or exist_error_candidate) and self.rotation) and (not len(self.recorded_maximum_folding_percent) or (len(self.recorded_maximum_folding_percent) and tm.pi * (max([abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])]) < self.equal_arm_distance[0])) and self.neglect_initial_penetration[0]):
                                self.dead_count -= 1
                                self.error_status[0] = False
                                self.clearErrorBuffer()
                            if self.system_type[0] == 3 and not self.first_phase_enable:
                                self.dead_count -= 1
                                self.error_status[0] = False
                                self.clearErrorBuffer()
                        else:
                            if (self.strict == -1 and self.control_mode == 0):
                                if len(self.recorded_maximum_folding_error) >= 2 and self.recorded_maximum_folding_error[-1] > self.recorded_maximum_folding_error[-2]:
                                    self.positive_count = 0
                                    if tm.pi * max(abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])) > self.equal_arm_distance[0] and self.recorded_maximum_folding_error[-1] > tm.pi:
                                        self.dead_count += 1
                                        if self.dead_count >= self.DEAD_MAXIMUM:
                                            self.dead_count = 4800
                                            self.can_rotate = False
                                            self.candidate_method = False
                                else:
                                    if not self.dead_count_adder:
                                        self.dead_count = 0
                                        if tm.pi * max(abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])) > self.equal_arm_distance[0]:
                                            self.positive_count += 1
                                            # if self.positive_count >= 480 and self.strict == -1:
                                            #     self.dead_count = 480
                                            #     self.can_rotate = False
                                            #     self.candidate_method = True
                                    else:
                                        self.positive_count = 0
                                        if tm.pi * max(abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])) > self.equal_arm_distance[0] and self.recorded_maximum_folding_error[-1] > tm.pi:
                                            self.dead_count += self.dead_count_adder
                                            if self.dead_count >= self.DEAD_MAXIMUM:
                                                self.dead_count = 4800
                                                self.can_rotate = False
                                                # self.candidate_method = False
                            else:
                                if not self.dead_count_adder:
                                    if self.dead_count < 4800:
                                        self.dead_count = 0
                                else:
                                    if tm.pi * max(abs(self.recorded_maximum_folding_percent[-1]), abs(self.recorded_minimum_folding_percent[-1])) > self.equal_arm_distance[0] and self.recorded_maximum_folding_error[-1] > tm.pi:
                                        self.dead_count += self.dead_count_adder
                        
                    for i in range(self.constraint_number):
                        self.recorded_string_decrease_length_control[i].append(self.string_length_decrease[i])
                        self.recorded_string_decrease_length[i].append(self.constraint_initial_length[i] - self.constraint_length[i])
                        self.recorded_string_force[i].append(self.getActuationForce(self.string_force_each[i]))

                    self.calculateFoldingAngle()
                    self.folding_percent = self.fae_information[0]
                    self.abs_folding_percent = self.fae_information[1]
                    min_folding_percent = self.fae_information[2]
                    max_folding_percent = self.fae_information[3]
                    
                    self.folding_error = self.fae_information[4]
                    min_folding_error = self.fae_information[5]
                    max_folding_error = self.fae_information[6]
                    
                    # print(self.crease_type)
                    # max_angle = 0.0
                    # col = 0.0
                    # for i in range(self.bending_pairs_num):
                    #     self.folding_percent += self.crease_angle[i]
                    #     # if abs(self.crease_angle[i] > max_angle):
                    #     #     max_angle = self.crease_angle[i]
                    #     #     col = self.barrier_coeff_list[i]
                    #     # if len(self.targets):
                    #     current_error = abs(self.crease_angle[i] - self.target_angles[i]) * np.pi
                    #     # else:
                    #     #     current_error = abs(self.crease_angle[i] - 1.0) * np.pi
                    #     self.folding_error += current_error
                    #     if self.crease_angle[i] > max_folding_percent:
                    #         max_folding_percent = self.crease_angle[i]
                    #     if self.crease_angle[i] < min_folding_percent:
                    #         min_folding_percent = self.crease_angle[i]
                    #     if abs(self.crease_angle[i]) < self.abs_folding_percent:
                    #         self.abs_folding_percent = abs(self.crease_angle[i])
                    #     if current_error > max_folding_error:
                    #         max_folding_error = current_error
                    #     if current_error < min_folding_error:
                    #         min_folding_error = current_error
                        
                    self.folding_percent /= self.bending_pairs_num
                    self.folding_error /= self.bending_pairs_num
                    self.recorded_folding_percent.append(self.folding_percent)
                    self.recorded_folding_error.append(self.folding_error)
                    self.recorded_maximum_folding_percent.append(max_folding_percent)
                    self.recorded_minimum_folding_percent.append(min_folding_percent)
                    
                    self.recorded_maximum_folding_error.append(max_folding_error)
                    self.recorded_minimum_folding_error.append(min_folding_error)
                        
                    self.recorded_max_force.append(self.overall_string_maximum_force[0])
                    self.recorded_nodal_maximum_force.append(self.getMaxMises())

                    if len(self.recorded_indices):
                        self.center_x = sum([self.x[self.recorded_indices[jj]][X] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices) - self.offset_x
                        self.center_y = sum([self.x[self.recorded_indices[jj]][Y] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices) - self.offset_y
                        z = sum([self.x[self.recorded_indices[jj]][Z] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices)
                        nm = self.calculateExtendedVector(self.unit_indices_num - 1, (1. - 1. / math.sqrt(3)) * (self.controller_centroid_bias[Z] - 10.))
                        self.center_x += nm[X]
                        self.center_y += nm[Y]
                        z += nm[Z]
                        self.recorded_movement_x.append(self.center_x)
                        self.recorded_movement_y.append(self.center_y)
                        self.recorded_movement_z.append(z)
                        if self.first_phase_enable:
                            self.recordCenterTrajectory(self.current_center_trajectory_step, self.center_x + self.offset_x, self.center_y + self.offset_y, z)
                        self.current_center_trajectory_step = (self.current_center_trajectory_step + 1) % self.maximum_recorded_trajectory_point
                    self.recorded_t.append(self.current_t)
                    # print(max_angle, col)
                    # print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | H" + str(self.origami_z_bias) + \
                    #       " | Error: " + str(self.error_status[0]) + " | Time: " + "{:.3f}".format(self.current_t) + \
                    #         " | Percent / Error: " + "{:.3f}".format(self.folding_percent) + " / {:.3f}".format(self.folding_error) + " / {:.3f}".format(max_folding_error) + \
                    #             " | DC: " + str(self.dead_count).zfill(3) + " | R: " + str(self.is_rotating) + " | ST: " + str(self.stable_state) + \
                    #                 " | Ten: " + str(self.total_tension))
            else:
                if len(self.recorded_indices):
                    self.center_x = sum([self.x[self.recorded_indices[jj]][X] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices) - self.offset_x
                    self.center_y = sum([self.x[self.recorded_indices[jj]][Y] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices) - self.offset_y
                    z = sum([self.x[self.recorded_indices[jj]][Z] for jj in range(len(self.recorded_indices))]) / len(self.recorded_indices)
                    self.recorded_movement_x.append(self.center_x)
                    self.recorded_movement_y.append(self.center_y)
                    self.recorded_movement_z.append(z)
                    if self.first_phase_enable:
                        self.recordCenterTrajectory(self.current_center_trajectory_step, self.center_x + self.offset_x, self.center_y + self.offset_y, z)
                    self.current_center_trajectory_step = (self.current_center_trajectory_step + 1) % self.maximum_recorded_trajectory_point
                    self.recorded_t.append(self.current_t)

        # for i in range(self.bending_pairs_num):
        #     print(self.crease_angle[i])

        if self.use_gui:
            self.camera.track_user_inputs(self.window, movement_speed=0.23, hold_key=ti.ui.RMB)
            if self.sim_mode == self.TSA_SIM and len(self.recorded_movement_x) and self.tracking_camera and (not self.paused or self.step_once):
                if self.strict != -1:
                    self.camera.position(self.recorded_movement_x[-1], self.recorded_movement_y[-1] + min(-1.5 * self.max_size, -400), max(1.5 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.lookat(self.recorded_movement_x[-1], self.recorded_movement_y[-1], self.recorded_movement_z[-1])
                else:
                    self.camera.position(0, 0 + min(-1.5 * self.max_size, -400), max(1.5 * self.max_size, 400) + self.origami_z_bias)
                    self.camera.lookat(0, 0, 0)
            self.scene.set_camera(self.camera)

            self.scene.point_light(pos=(0., 0., self.max_size + 3 * self.origami_z_bias), color=(0.8, 0.8, 0.8))
            self.scene.ambient_light((0.5, 0.5, 0.5))

            if self.enable_ground:
                self.scene.mesh(vertices=self.ground_vertices, indices=self.ground_indices, per_vertex_color=self.ground_vertices_color, two_sided=True)
                if len(self.recorded_indices):
                    if self.strict != -1:
                        self.updateGroundLines(self.center_x, self.center_y, self.rot_z, self.ground_d)
                        self.scene.lines(vertices=self.ground_line_vertex,
                                width=1,
                                color=(0.6, 0.6, 0.6))
                    self.scene.lines(vertices=self.ground_fix_line_vertex,
                            width=1,
                            color=(0.25, 0.25, 0.25))
                
            self.scene.mesh(self.vertices,
                    indices=self.indices,
                    per_vertex_color=self.vertices_color,
                    two_sided=True)
            
            self.fill_line_vertex()
            self.scene.lines(vertices=self.line_vertex,
                        width=2,
                        per_vertex_color=self.line_color)
            
            if len(self.recorded_indices):
                self.scene.particles(centers=self.center_trajectory, radius=1, color=(0.5625, 0.94, 0.5625))
            
            # for i in range(self.kp_num):
            #     self.force_vertex[2 * i] = self.x[i]
            #     self.force_vertex[2 * i + 1] = self.x[i] + 10 * self.record_force[i]

            # self.scene.lines(vertices=self.force_vertex,
            #                 width=2,
            #                 color=(.9, .03, .9))
            # for i in range(self.bending_pairs_num):
            #     print(self.crease_angle[i])

            self.gui.text(f"System time: {round(self.current_t, 3)}s")
            self.gui.text(f"Error status: {self.error_status[0]}")
            if self.sim_mode == self.TSA_SIM:
                self.gui.text(f"Tension: {self.total_tension} N")
                for i in range(self.constraint_number):
                    self.gui.text(text=f"Delta_length[{i}]: " + str(round(self.constraint_initial_length[i] - self.constraint_length[i], 2)))
                    self.gui.text(text=f"String length decrease[{i}]: " + str(round(self.string_length_decrease[i], 2)))
                self.gui.slider_float('Total folding percent', round(self.folding_percent, 4), -1., 1.)
                if self.control_mode:
                    self.gui.slider_float('Total load', round(self.total_string_force, 4), 0., self.maximum_tension)
                    self.gui.slider_float('Contract speed discount', round(self.contract_speed_discount, 4), 0., 1.)
            #     self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0., 1000000)
            #     self.gui.slider_int('Dead count', self.dead_count, 0, 500)
            #     self.enable_ground = self.gui.slider_int('Enable ground', self.enable_ground, 0, 1)
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 2), 0.0, 1000000.)
                self.update_string_vertices()

                if self.constraint_number and self.P_number:
                    self.scene.lines(vertices=self.string_vertex, width=1.5, color=(.6, .03, .8))
                    self.scene.particles(centers=self.constraint_start_point, radius=1, color=(.6, .03, .8))
                    self.scene.particles(centers=self.endpoint_vertex, radius=1., color=(.6, .03, .8))
                    # self.scene.particles(centers=self.intersection_points, radius=0.4, color=(.6, .03, .8))
                # for i in range(self.constraint_number):
                #     self.string_params[i] = self.gui.slider_float('String_k', self.string_params[i], 0.0, 100.0)
                #     self.tolerance[i] = self.gui.slider_float('Tolerance', self.tolerance[i], 0.0, 100.)
                for i in range(self.constraint_number):
                    self.gui.slider_float('String_k', self.string_params[i] * self.string_params_bonus[i], 0.0, 100.0)
                    self.gui.slider_float('Tolerance', self.tolerance[i], 0.0, 100.)
                    
                # self.lame_k = self.gui.slider_float('Lame_k', self.lame_k, 1., 3000.)

            else:
                self.folding_angle = self.gui.slider_float('Folding angle', self.folding_angle, 0.0, self.folding_max)
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0., 1000000.)
                self.percent[0] = self.gui.slider_float('Percent', self.percent[0], 0.0, 1.0)
                self.lame_k = self.gui.slider_float('Lame_k', self.lame_k, 1., 10000.)
                self.bending_param[0] = self.gui.slider_float('Bending_k', self.bending_param[0], 0.01, 100.00)

                self.lames_bonus[0] = self.lame_k * self.mu
                self.lames_bonus[1] = self.lame_k * self.landa
                self.facet_bending_param[0] = self.lame_k / 1000.0
            
            self.canvas.scene(self.scene)
            if not self.fast_simulation_mode:
                if (self.current_t * 60.0 - float(self.image_id) + 1.) >= 1.:
                    try:
                        self.window.save_image(f'./physResult/{self.origami_name}-{self.time}-{self.mode}/' + self.additional_folder + ('/' if self.control_mode else '') + str(self.image_id).zfill(8) + '.png')
                        self.image_id += 1
                    except:
                        pass
            self.window.show()
        
        self.step_once = False

    def run(self, force_dir=False, check_force_dir=True):
        noise = None
        if self.strict == 2:
            noise_dict = self.noise_dict
            noise = np.random.normal(loc=0, scale=0.01, size=ori.bending_pairs_num)
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
            
        self.initializeRunning(self.strict, noise, show_index=0)
        
        if self.use_gui:
            while self.window.running:
                self.step(force_dir, check_force_dir)

                # ti.profiler.print_scoped_profiler_info()  
                # ti.profiler.print_kernel_profiler_info()  # 看每个kernel的执行时间、线程数
                # ti.profiler.clear_kernel_profiler_info()
                
                if self.stop():
                    break
        else:
            # Robot actuation (control_mode=1) in FOLD_SIM: exit after one step
            if self.control_mode == 1 and self.sim_mode == self.FOLD_SIM:
                self.step(force_dir, check_force_dir)
            else:
                while 1:
                    self.step(force_dir, check_force_dir)
                    
                    if self.stop():
                        break
        
        import pandas as pd
        test = pd.DataFrame(columns=["dx"], data=self.dx_list)
        test.to_csv('./physResult/dx.csv')
        
    
    def reward(self):
        """
        计算奖励值，支持TSA_SIM和FOLD_SIM两种模式。
        Calculate reward value, supporting both TSA_SIM and FOLD_SIM modes.
        
        :return: 
            - TSA_SIM模式: (value, value_f, rf, rm, rs, ra, rh, rforce, rforce2)
            - FOLD_SIM模式（厚板折纸）: (avg_folding_percentage, max_folding_percentage, min_folding_percentage, 
                                      std_folding_percentage, crease_number, 0, 0, 0, 0)
        """
        value = -1.
        value_f = -1.
        rf = 0.
        rm = 0.
        rs = 0.
        ra = 0.
        rh = 0.
        rforce = 0.
        rforce2 = 0.
        a_rs = self.a_s
        a_ra = self.a_a
        a_rforce = self.a_f
        
        # FOLD_SIM模式奖励计算（厚板折纸折叠程度评估）
        # FOLD_SIM mode reward calculation (thick origami folding degree evaluation)
        if self.sim_mode == self.FOLD_SIM and getattr(self, 'thick_mode_flag', False):
            # 计算所有折痕的折叠百分比
            # Calculate folding percentage for all creases
            if self.crease_pairs_num > 0:
                crease_angles = np.array([self.crease_angle[i] for i in range(self.crease_pairs_num)])
                actual_angles = np.abs(crease_angles) * np.pi
                target_angle = np.pi
                folding_percentages = (actual_angles / target_angle) * 100.0
                
                avg_folding_percentage = np.mean(folding_percentages)
                min_folding_percentage = np.min(folding_percentages)
                max_folding_percentage = np.max(folding_percentages)
                std_folding_percentage = np.std(folding_percentages)
                
                # 返回格式：(平均百分比, 最大百分比, 最小百分比, 标准差, 折痕数, 0, 0, 0, 0)
                # Return format: (avg%, max%, min%, std%, crease_num, 0, 0, 0, 0)
                return (avg_folding_percentage, max_folding_percentage, min_folding_percentage,
                        std_folding_percentage, float(self.crease_pairs_num), 0., 0., 0., 0.)
            else:
                return (0., 0., 0., 0., 0., 0., 0., 0., 0.)
        
        if self.control_mode == 0:
            if self.strict == -1 and not self.candidate_method:
                # if self.candidate_method:
                #     value = 1.0
                #     value_f = 1.0
                #     rf = 1.0
                #     rm = 1.0
                #     rs = 0.0
                #     ra = 1.0
                if len(self.recorded_folding_error) == 0:
                    value = -1.
                    value_f = -1.
                else:
                    folding_error = self.recorded_folding_error[-1]
                    maximum_folding_error = self.recorded_maximum_folding_error[-1]
                    string_length_decrease_each_final = sorted([self.recorded_string_decrease_length_control[i][-1] for i in range(self.constraint_number)])
                    actuator_number = 0
                    base_contraction = -10
                    for ele in string_length_decrease_each_final:
                        if abs(ele - base_contraction) > 2.0:
                            actuator_number += 1
                        base_contraction = ele
                    
                    rf = 1 - folding_error / np.pi
                    rm = 1 - maximum_folding_error / np.pi

                    speed_baseline = (self.simulation_upper_time * self.string_length_decrease_step / self.dt)
                    equivalent_length = ((self.recorded_t[-1] - self.recorded_t[0]) * self.string_length_decrease_step / self.dt)

                    rs = max(rf * (1. - equivalent_length / speed_baseline), 0.0)
                    
                    ra = 1. + a_ra * (self.constraint_number - actuator_number)

                    rh = 0.0
                    for i in range(self.kp_num):
                        rh += self.x[i][Z]
                    rh /= self.kp_num

                    rforce = max([sum([self.recorded_string_force[i][j] for i in range(self.constraint_number)]) for j in range(len(self.recorded_max_force))])
                    
                    maximum_stress = max([self.recorded_nodal_maximum_force[i] for i in range(len(self.recorded_nodal_maximum_force))])
                    rforce2 = max((self.mean_stress_baseline - maximum_stress) / self.mean_stress_baseline, 0.)
                    
                    value = min((rf + rm) / 2., VALID_BOUND)
                    value_f = value

            else:
                if len(self.recorded_folding_error) == 0:
                    # print("Batch: " + str(ori_sim.ID) + ", Value: Error Actuation")
                    value = -1.
                    value_f = -1.
                else:
                    folding_error = self.recorded_folding_error[-1]
                    maximum_folding_error = self.recorded_maximum_folding_error[-1]
                    string_length_decrease_each_final = sorted([self.recorded_string_decrease_length_control[i][-1] for i in range(self.constraint_number)])
                    actuator_number = 0
                    base_contraction = -10
                    for ele in string_length_decrease_each_final:
                        if abs(ele - base_contraction) > 2.0:
                            actuator_number += 1
                        base_contraction = ele
                    
                    rf = 1 - folding_error / np.pi
                    rm = 1 - maximum_folding_error / np.pi
                    
                    speed_baseline = (self.simulation_upper_time * self.string_length_decrease_step / self.dt)
                    equivalent_length = ((self.recorded_t[-1] - self.recorded_t[0]) * self.string_length_decrease_step / self.dt)

                    rs = max(rf * (1. - equivalent_length / speed_baseline), 0.0)
                    
                    ra = 1. + a_ra * (self.constraint_number - actuator_number)

                    rh = 0.0
                    for i in range(self.kp_num):
                        rh += self.x[i][Z]
                    rh /= self.kp_num

                    rforce = (self.maximum_tension * self.constraint_number - \
                              max([sum([self.recorded_string_force[i][j] \
                                        for i in range(self.constraint_number)]) \
                                            for j in range(len(self.recorded_max_force))])) \
                                                / (self.maximum_tension * self.constraint_number)
                    
                    maximum_stress = max([self.recorded_nodal_maximum_force[i] for i in range(len(self.recorded_nodal_maximum_force))])
                    rforce2 = max((self.mean_stress_baseline - maximum_stress) / self.mean_stress_baseline, 0.)

                    if folding_error >= FOLDING_ERROR_MINIMUM:
                        value = (rf + rm) / 2
                        value_f = value
                    else:
                        # if maximum_folding_error >= FOLDING_ERROR_MINIMUM:
                        value = (rf + rm) / 2 + a_rs * rs
                        value_f = (rf + rm) / 2 + a_rforce * rforce
                        # else:
                        #     value = ((rf + rm) / 2 + a_rs * rs + a_rforce * rforce2) * ra
                        #     value_f = ((rf + rm) / 2 + a_rs * rs + a_rforce * rforce2)
        else:
            try:
                if 1:
                    # displacement
                    displacement = [
                        self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][X] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][X],
                        self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][Y] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][Y]
                    ]
                    time_interval = self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][1] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][1]

                else:
                    # turn
                    all_displacement = []
                    all_time_interval = []
                    for i in range(1, len(self.recorded_interval_velocity) - 1):
                        displacement = [
                            self.recorded_interval_velocity[i + 1][0][X] - self.recorded_interval_velocity[i][0][X],
                            self.recorded_interval_velocity[i + 1][0][Y] - self.recorded_interval_velocity[i][0][Y]
                        ]
                        time_interval = self.recorded_interval_velocity[i + 1][1] - self.recorded_interval_velocity[i][1]
                        all_displacement.append(displacement)
                        all_time_interval.append(time_interval)
                    # calculate the turning angle between each two displacements
                    turning_angles = []
                    for i in range(len(all_displacement) - 1):
                        v1 = np.array(all_displacement[i])
                        v2 = np.array(all_displacement[i + 1])
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = min(1.0, max(-1.0, cos_angle))
                        angle = np.sign(np.cross(v1, v2)) * math.acos(cos_angle)
                        turning_angles.append(angle)
                    # get the average turning angle
                    average_turning_angle_vel = (sum(turning_angles) / len(turning_angles)) / (sum(all_time_interval) / len(all_time_interval))
                fail = 0
            except:
                displacement = [0, 0]
                time_interval = 1
                fail = 1

            try:
                displacement = [
                    self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][X] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][X],
                    self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][0][Y] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][0][Y]
                ]
                time_interval = self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 1][1] - self.recorded_interval_velocity[len(self.recorded_interval_velocity) - 2][1]
            except:
                displacement = [0, 0]
                time_interval = 1
                fail = 1

            try:
                overall_moving_speed = math.sqrt(displacement[X] ** 2 + displacement[Y] ** 2) / time_interval
            except:
                overall_moving_speed = 0.

            value = 0.0
            value_f = 0.0
            pass_num = 0.0
            if not fail and self.candidate_method:
                pass_num = sum([len(self.method["id"][i]) for i in range(self.constraint_number)])
                try:
                    angular_velocity_reward = (0.1 + 1. / (50.0 * abs(average_turning_angle_vel) + 1.))
                    cycle_time = (sum(all_time_interval) / len(all_time_interval))
                except:
                    angular_velocity_reward = 1.0
                    cycle_time = 0.0
                
                value = (overall_moving_speed / self.max_size) * angular_velocity_reward * 10.0 * (1. - self.intersection_penalty[0] / (self.current_t / self.dt * (pass_num - self.constraint_number)))
            else:
                angular_velocity_reward = 0.0
                cycle_time = 0.0

            actuator_number = 1
            rf = overall_moving_speed / self.max_size
            rm = angular_velocity_reward
            rs = cycle_time
            ra = (1. - self.intersection_penalty[0] / (self.current_t / self.dt * (pass_num - self.constraint_number)))
            rh = 0.0
            for i in range(self.kp_num):
                rh += self.x[i][Z]
            rh /= self.kp_num
            rforce = max([sum([self.recorded_string_force[i][j] for i in range(self.constraint_number)]) for j in range(len(self.recorded_max_force))])
                    
            maximum_stress = max([self.recorded_nodal_maximum_force[i] for i in range(len(self.recorded_nodal_maximum_force))])
            rforce2 = max((self.mean_stress_baseline - maximum_stress) / self.mean_stress_baseline, 0.)
            # only speed
            value_f = overall_moving_speed / self.max_size * angular_velocity_reward * 10.0

        return value, value_f, rf, rm, rs, ra, rh, rforce, rforce2

    def getCenter(self):
        center_initial = []
        center_current = []
            
        for i in range(self.unit_indices_num):
            center_initial.append(np.array(self.unit_center_initial_point[i]))
            center_current.append(np.array(self.unit_center[i]))
        
        for i in range(self.unit_indices_num):
            center_initial[i][X] += self.total_bias[X]
            center_initial[i][Y] += self.total_bias[Y]
            center_initial[i][Z] -= self.origami_z_bias
            center_current[i][X] += self.total_bias[X]
            center_current[i][Y] += self.total_bias[Y]
            center_current[i][Z] -= self.origami_z_bias

        return [ele.tolist() for ele in center_initial], [ele.tolist() for ele in center_current]

    # ================================================================== #
    #  Taichi-accelerated helpers for getValidMatrix
    # ================================================================== #

    @ti.func
    def _vm_normal_fan(self, unit_kps):
        """Fan-triangulation normal (matches original getValidMatrix logic)."""
        n = tm.vec3([0.0, 0.0, 0.0])
        for l in range(self.unit_edge_max):
            if unit_kps[l] == -1:
                break
            p0 = self.x[unit_kps[l]]
            s1 = (l + 1) % self.unit_edge_max
            if unit_kps[s1] == -1:
                s1 = 0
            p1 = self.x[unit_kps[s1]]
            s2 = (l + 2) % self.unit_edge_max
            if unit_kps[s2] == -1:
                s2 = 0
            p2 = self.x[unit_kps[s2]]
            n += (p1 - p0).cross(p2 - p0)
        return tm.normalize(n)

    @ti.func
    def _vm_kp_in_unit(self, unit_kps, val):
        """Check if val appears in unit_kps."""
        found = False
        for i in ti.ndrange(self.unit_edge_max):
            if unit_kps[i] == val:
                found = True
        return found

    @ti.func
    def _vm_is_connection_unit(self, unit_idx):
        """Check if unit_idx is a constraint connection unit."""
        result = False
        for p in ti.ndrange(self.P_number):
            if self.constraint_start_point_candidate_connection[p] == unit_idx:
                result = True
        return result

    @ti.func
    def _vm_find_crease(self, unit_kps_a, unit_kps_b):
        """Find crease between two units. Returns (found: int, angle: float)."""
        found = 0
        angle = 0.0
        for crease_id in range(self.crease_pairs_num):
            cs = self.crease_pairs[crease_id, 0]
            ce = self.crease_pairs[crease_id, 1]
            if self._vm_kp_in_unit(unit_kps_a, cs):
                if self._vm_kp_in_unit(unit_kps_b, cs):
                    if self._vm_kp_in_unit(unit_kps_a, ce):
                        if self._vm_kp_in_unit(unit_kps_b, ce):
                            found = 1
                            angle = self.crease_angle[crease_id]
                            break
        return found, angle

    @ti.func
    def _vm_point_in_unit(self, ip, unit_kps):
        """Check if point ip is inside unit polygon using angle sum = 2*pi."""
        angle = 0.0
        for jj in ti.ndrange(self.unit_edge_max):
            if unit_kps[jj] != -1:
                current = self.x[unit_kps[jj]]
                next_idx = (jj + 1) % self.unit_edge_max
                next_pt = tm.vec3([0.0, 0.0, 0.0])
                if unit_kps[next_idx] != -1:
                    next_pt = self.x[unit_kps[next_idx]]
                else:
                    next_pt = self.x[unit_kps[0]]
                ip_c = ti.sqrt((current - ip).dot(current - ip))
                ip_n = ti.sqrt((next_pt - ip).dot(next_pt - ip))
                nc = ti.sqrt((next_pt - current).dot(next_pt - current))
                val = (ip_c * ip_c + ip_n * ip_n - nc * nc) / (2.0 * ip_c * ip_n)
                if val > 1.0:
                    val = 1.0
                elif val < -1.0:
                    val = -1.0
                angle += ti.acos(val)
        return ti.abs(angle - 2.0 * tm.pi) < 1e-3

    @ti.kernel
    def _compute_valid_matrix_kernel(self, matrix: ti.types.ndarray(), row: int, P_num: int):
        """Taichi kernel for computing the valid action matrix.

        Uses ti.ndrange to flatten nested loops into a single parallel loop.
        Part 1: P constraints (rows 0..P-1) vs units — P_num*row parallel threads
        Part 2: units vs units (upper triangle) — row*row parallel threads
        Inner intersection-detection loop stays serial (has break for early exit).
        """
        # === Part 1: P constraints vs units (flattened 2D parallel) ===
        for i, j in ti.ndrange(P_num, row):
            if j == self.fix_id_list[0]:
                continue
            if self._vm_is_connection_unit(j):
                continue

            unit_kps_j = self.unit_indices[j]
            center_j = self.unit_center[j]

            # Compute center of P constraint i
            unit_id = self.constraint_start_point_candidate_connection[i]
            center_i = tm.vec3([0.0, 0.0, 0.0])
            if unit_id >= 0:
                idx = self.loc_of_unit[i]
                unit_kps_i = self.unit_indices[idx]
                for k in range(self.unit_edge_max):
                    if unit_kps_i[k] == -1:
                        break
                    center_i += self.x[unit_kps_i[k]] * self.unit_contributions[idx][k]
            else:
                center_i = self.constraint_start_point_candidate[i]

            normal_j = self._vm_normal_fan(unit_kps_j)
            direction = (center_j - center_i).dot(normal_j)

            intersection = False
            for ii in range(row):
                if ii != j and ii != self.fix_id_list[0]:
                    if not self._vm_is_connection_unit(ii):
                        unit_kps_col = self.unit_indices[ii]
                        center_c = self.unit_center[ii]

                        exist_j, folding_angle_j = self._vm_find_crease(unit_kps_j, unit_kps_col)

                        normal_col = self._vm_normal_fan(unit_kps_col)
                        h1 = (center_i - center_c).dot(normal_col)
                        h2 = (center_j - center_c).dot(normal_col)

                        if exist_j > 0:
                            if folding_angle_j > 1.0:
                                h2 = -h2

                        if h1 * h2 < 0.0:
                            ratio = ti.abs(h1) / (ti.abs(h1) + ti.abs(h2))
                            ip = center_i + ratio * (center_j - center_i)
                            if self._vm_point_in_unit(ip, unit_kps_col):
                                intersection = True
                                break
                if intersection:
                    break

            if not intersection:
                if direction < 0.0:
                    matrix[i, j + P_num] = 1
                    matrix[j + P_num, i] = 1
                else:
                    matrix[i, j + P_num] = -1
                    matrix[j + P_num, i] = -1

        # === Part 2: units vs units (flattened 2D parallel, upper triangle) ===
        for i, j in ti.ndrange(row, row):
            if j <= i:
                continue
            if i == self.fix_id_list[0] or self._vm_is_connection_unit(i):
                continue
            if j == self.fix_id_list[0] or self._vm_is_connection_unit(j):
                continue

            unit_kps_i = self.unit_indices[i]
            unit_kps_j = self.unit_indices[j]
            center_i = self.unit_center[i]
            center_j = self.unit_center[j]

            normal_j = self._vm_normal_fan(unit_kps_j)
            direction = (center_j - center_i).dot(normal_j)

            intersection = False
            for ii in range(row):
                if ii != i and ii != j:
                    unit_kps_col = self.unit_indices[ii]
                    center_c = self.unit_center[ii]

                    exist_i, folding_angle_i = self._vm_find_crease(unit_kps_i, unit_kps_col)
                    exist_j, folding_angle_j = self._vm_find_crease(unit_kps_j, unit_kps_col)

                    normal_col = self._vm_normal_fan(unit_kps_col)
                    h1 = (center_i - center_c).dot(normal_col)
                    h2 = (center_j - center_c).dot(normal_col)

                    if exist_i > 0:
                        if folding_angle_i > 1.0:
                            h1 = -h1
                    if exist_j > 0:
                        if folding_angle_j > 1.0:
                            h2 = -h2

                    if h1 * h2 < 0.0:
                        ratio = ti.abs(h1) / (ti.abs(h1) + ti.abs(h2))
                        ip = center_i + ratio * (center_j - center_i)
                        if self._vm_point_in_unit(ip, unit_kps_col):
                            intersection = True
                            break
                if intersection:
                    break

            if not intersection:
                if direction < 0.0:
                    matrix[i + P_num, j + P_num] = 1
                    matrix[j + P_num, i + P_num] = 1
                else:
                    matrix[i + P_num, j + P_num] = -1
                    matrix[j + P_num, i + P_num] = -1

    # ================================================================== #
    #  getValidMatrix (optimised: matrix computation via Taichi kernel)
    # ================================================================== #

    def getValidMatrix(self):
        center_initial = []
        center_current = []
            
        for i in range(self.unit_indices_num):
            center_initial.append(np.array(self.unit_center_initial_point[i]))
            center_current.append(np.array(self.unit_center[i]))
        
        if self.fix_id_list[0] >= 0:
            print("IA-Static system detected! Simulator will judge the feasibility of the system...")
            fix_indices = self.unit_indices[self.fix_id_list[0]]
            normal_fix = np.array([0., 0., 0.], dtype=np.float64)
            for l in range(self.unit_edge_max):
                start = l
                if fix_indices[start] == -1:
                    break
                p0 = np.array(self.x[fix_indices[start]])
                start = (start + 1) % self.unit_edge_max
                if fix_indices[start] == -1:
                    start = 0
                p1 = np.array(self.x[fix_indices[start]])
                start = (start + 1) % self.unit_edge_max
                if fix_indices[start] == -1:
                    start = 0
                p2 = np.array(self.x[fix_indices[start]])
                normal_fix += np.cross(p1 - p0, p2 - p0) 

            normal_fix /= np.linalg.norm(normal_fix)
            temp_val = 0.0
            correctness = True
            
            for i in range(self.kp_num):
                v = np.array(self.x[i]) - center_current[self.fix_id_list[0]]
                val = np.dot(v, normal_fix)
                if abs(val) > 1:
                    temp_val = val
                    break
                
            for i in range(self.kp_num):
                v = np.array(self.x[i]) - center_current[self.fix_id_list[0]]
                val = np.dot(v, normal_fix)
                if abs(val) > 1 and val * temp_val < -1e-5:
                    print("Result: IA-Static system can't be defined!")
                    correctness = False
                    break
            
            if correctness:
                print("Result: This IA-Static system is feasible!")
            else:
                print("Calculating proper units for being fixed...")
                proper_ids = []
                for i in range(self.unit_indices_num):
                    unit_indice_i = self.unit_indices[i]
                    normal_i = np.array([0., 0., 0.], dtype=np.float64)
                    for l in range(self.unit_edge_max):
                        start = l
                        if unit_indice_i[start] == -1:
                            break
                        p0 = np.array(self.x[unit_indice_i[start]])
                        start = (start + 1) % self.unit_edge_max
                        if unit_indice_i[start] == -1:
                            start = 0
                        p1 = np.array(self.x[unit_indice_i[start]])
                        start = (start + 1) % self.unit_edge_max
                        if unit_indice_i[start] == -1:
                            start = 0
                        p2 = np.array(self.x[unit_indice_i[start]])
                        normal_i += np.cross(p1 - p0, p2 - p0) 

                    normal_i /= np.linalg.norm(normal_i)
                    temp_val = 0.0
                    correctness = True
                    for j in range(self.kp_num):
                        v = np.array(self.x[j]) - center_current[i]
                        val = np.dot(v, normal_i)
                        if j == 0:
                            temp_val = val
                        else:
                            if abs(val) > 1 and val * temp_val < 0:
                                correctness = False
                                break
                    if correctness:
                        proper_ids.append(i)
                print("Result: Proper units for being fixed: " + str(proper_ids))
                return [], None, None

        elif (self.connection_number[0] > 0):
            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + "IA-Dynamic system detected!")
        else:
            print(str(self.strict) + " | Process: " + "{:05d}".format(self.ID) + " | " + "EA system detected!")

        row = self.unit_indices_num - int(self.connection_number[0])
        
        matrix_size = row + self.P_number
        matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)
        if matrix_size > 0:
            self._compute_valid_matrix_kernel(matrix, int(row), int(self.P_number))

        for i in range(self.unit_indices_num):
            center_initial[i][X] += self.total_bias[X]
            center_initial[i][Y] += self.total_bias[Y]
            center_initial[i][Z] -= self.origami_z_bias
            center_current[i][X] += self.total_bias[X]
            center_current[i][Y] += self.total_bias[Y]
            center_current[i][Z] -= self.origami_z_bias

        return matrix, [ele.tolist() for ele in center_initial], [ele.tolist() for ele in center_current]


if __name__ == "__main__":
    # ---- CLI mode: invoked externally with --origami argument (threading design automation) ----
    if "--origami" in sys.argv:
        parser = argparse.ArgumentParser(description="Origami FOLD_SIM runner for threading design automation")
        parser.add_argument("--origami", type=str, required=True, help="Origami model name")
        parser.add_argument("--height", type=float, default=1.0, help="Structure height (mm)")
        parser.add_argument("--control-mode", type=int, default=0, choices=[0, 1],
                            help="0=structure folding, 1=robot actuation")
        parser.add_argument("--gravity", type=str, default="0,0,0",
                            help="Gravity vector as x,y,z (e.g. 0,0,-9810)")
        parser.add_argument("--material", type=int, default=0, choices=[0, 1],
                            help="0=PLA+TPU, 1=PU foam+TPU")
        parser.add_argument("--stroke-percent", type=float, default=0.75,
                    help="Actuator stroke fraction (0.00~1.00); only used when control-mode=1")
        args = parser.parse_args()

        # Parse gravity vector
        gx, gy, gz = [float(v) for v in args.gravity.split(",")]

        print(f"[FOLD_SIM CLI] origami={args.origami}, h={args.height}, "
              f"control_mode={args.control_mode}, g=[{gx}, {gy}, {gz}], material={args.material}, stroke_percent={args.stroke_percent}")

        ori = OrigamiSimulator(
            origami_name=args.origami,
            use_gui=1,
            fast_simulation=1,
            g=[gx, gy, gz],
            default_ground=0,
            control_mode=args.control_mode,
            ground_miu=0.0,
            h=args.height,
            strict=0,
            friction_mode=2,
            const_stiff_of_crease=True,
            speed_bonus=1.0,
            check_connection_matrix=True,
            robot_type=args.material,
            stroke_percent=args.stroke_percent,
        )
        ori.start(args.origami, 4, ori.FOLD_SIM)
        ori.run(False, False)

        # --- System feature extraction (same as standalone simulation mode) ---
        if ori.sim_mode == ori.FOLD_SIM and ori.check_connection_matrix:
            try:
                if ori.connection_number[0] == 0:
                    final_valid_matrix, initial_center, current_center = ori.getValidMatrix()
                    final_valid_matrix = final_valid_matrix.tolist()
                    for i in range(len(final_valid_matrix)):
                        for j in range(len(final_valid_matrix[i])):
                            for k in range(len(ori.valid_matrix_list)):
                                if len(ori.valid_matrix_list[k]) != 0 and ori.valid_matrix_list[k][i][j] != final_valid_matrix[i][j]:
                                    final_valid_matrix[i][j] = 0
                                    break
                    ori.input_json["valid_actions"] = final_valid_matrix
                    ori.input_json["initial_center"] = initial_center
                    ori.input_json["current_center"] = current_center
                    ori.input_json["crease_angle"] = [
                        ori.crease_angle[i] for i in range(ori.crease_pairs_num)
                    ]
                    ori.input_json["trans_units"] = [
                        [
                            [
                                ori.x[ori.unit_indices[i][(j + ori.unit_kp_num_list[i]) % ori.unit_kp_num_list[i]]][k] for k in range(3)
                            ]   for j in range(0, -ori.unit_kp_num_list[i], -1)
                        ] for i in range(ori.unit_indices_num)
                    ]
                    crease_info = [
                        [ori.kps[ori.crease_pairs[i, 0]], ori.kps[ori.crease_pairs[i, 1]]] for i in range(ori.crease_pairs_num)
                    ]
                    ori.input_json["crease_info"] = crease_info
                    if ori.ref_crease_angle_mode:
                        for i in range(len(ori.input_json["line_features"])):
                            points = ori.lines[i].points
                            for j in range(ori.crease_pairs_num):
                                info = crease_info[j]
                                if (distance3D(points[START], info[START]) < 1e-3 and distance3D(points[END], info[END]) < 1e-3) or \
                                    (distance3D(points[START], info[END]) < 1e-3 and distance3D(points[END], info[START]) < 1e-3):
                                    ori.input_json["line_features"][i]["hard_angle"] = abs(ori.crease_angle[j]) * math.pi
                                    ori.input_json["line_features"][i]["hard_angle_down"] = -abs(ori.crease_angle[j]) * math.pi
                                    break
                    # final connection matrix
                    all_connection_id = [ori.constraint_start_point_candidate_connection[i] for i in range(ori.P_number)]
                    for i in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                        for j in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                            if (i >= ori.P_number or j >= ori.P_number) and \
                                (
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in all_connection_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in all_connection_id) or \
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in ori.fix_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in ori.fix_id)
                                ):
                                ori.initial_matrix[i][j] = -10
                            else:
                                if abs(ori.initial_matrix[i][j]) != 1:
                                    if abs(final_valid_matrix[i][j]) == 1:
                                        if (i < ori.P_number or j < ori.P_number) and ori.system_type[0] != 3:
                                            if final_valid_matrix[i][j] == 1:
                                                ori.initial_matrix[i][j] = 1
                                            elif final_valid_matrix[i][j] == -1:
                                                ori.initial_matrix[i][j] = -1
                                            else:
                                                ori.initial_matrix[i][j] = 0
                                        else: # internal
                                            if ori.initial_matrix[i][j] == 0:
                                                if final_valid_matrix[i][j] == 1:
                                                    ori.initial_matrix[i][j] = 1
                                                elif final_valid_matrix[i][j] == -1:
                                                    ori.initial_matrix[i][j] = -1
                                            else:
                                                ori.initial_matrix[i][j] = -10
                                    else:
                                        ori.initial_matrix[i][j] = -10
                                else:
                                    if abs(final_valid_matrix[i][j]) == 1:
                                        if (i < ori.P_number and ori.P_candidate[i][Z] > 1e-5) or (j < ori.P_number and ori.P_candidate[j][Z] > 1e-5):
                                            ori.initial_matrix[i][j] = 1
                                        elif (i < ori.P_number and ori.P_candidate[i][Z] < -1e-5) or (j < ori.P_number and ori.P_candidate[j][Z] < -1e-5):
                                            ori.initial_matrix[i][j] = -1
                                        elif ori.initial_matrix[i][j] == 0:
                                            if final_valid_matrix[i][j] == 1:
                                                ori.initial_matrix[i][j] = 1
                                            elif final_valid_matrix[i][j] == -1:
                                                ori.initial_matrix[i][j] = -1
                                    else:
                                        ori.initial_matrix[i][j] = -10
                    all_connection_id = [ori.constraint_start_point_candidate_connection[i] for i in range(ori.P_number)]
                    for i in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                        for j in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                            if (i >= ori.P_number or j >= ori.P_number) and \
                                (
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in all_connection_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in all_connection_id) or \
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in ori.fix_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in ori.fix_id)
                                ):
                                ori.initial_matrix[i][j] = -10
                                try:
                                    p_index = all_connection_id.index((i - ori.P_number) if (i - ori.P_number) in all_connection_id else (j - ori.P_number))
                                    ori.initial_matrix[p_index][all_connection_id[p_index] + ori.P_number] = -10
                                    ori.initial_matrix[all_connection_id[p_index] + ori.P_number][p_index] = -10
                                except:
                                    pass
                                try:
                                    p_index = ori.fix_id.index((i - ori.P_number) if (i - ori.P_number) in all_connection_id else (j - ori.P_number))
                                    ori.initial_matrix[p_index][all_connection_id[p_index] + ori.P_number] = -10
                                    ori.initial_matrix[all_connection_id[p_index] + ori.P_number][p_index] = -10
                                except:
                                    pass
                    else:
                        initial_center, current_center = ori.getCenter()
                    ori.input_json["initial_center"] = initial_center
                    ori.input_json["current_center"] = current_center
                    ori.input_json["true_valid_actions"] = ori.initial_matrix.tolist()

                    with open("./descriptionData/" + args.origami + ".json", 'w', encoding='utf-8') as fw:
                        json.dump(ori.input_json, fw, indent=4)

            except Exception as e:
                print(f"[FOLD_SIM CLI] Feature extraction failed: {e}")
                import traceback
                traceback.print_exc()

        if ori.use_gui:
            ori.window.destroy()

        sys.exit(0)

    # ---- Standalone simulation mode (original behaviour) ----
    # ori_name_list = ["mountain-big-fix-new"]#, "diamond-III-acc", "resch-50", "tri-resch-50", "tri-resch-50-wub", "bird4"]
    # ori_name_list = ["f-mountain-big-fix-new", "f-diamond-III-acc", "f-resch-50", "tri-f-resch-50", "tri-f-resch-50-wub", \
    #                  "f-box2", "f-bird4", "f-robot3"]
    # ori_name_list = ["f-robot8-90"]
    
    ori_name_list = ["f-robot14-4"]
    output_fig = 0
    fast_mode = not output_fig

    for idx, ori_name in enumerate(ori_name_list):
        # IA-1 SYSTEM WITH +X GRAVITY
        if "f-miura-ori-16" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[9810., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=100., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH +X GRAVITY
        elif "f-mountain-big-fix-new" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[9810., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=100., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH +Y GRAVITY
        elif "f-diamond-III-acc" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 9810., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=100., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH +Y GRAVITY
        elif "f-resch-50" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 9810., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=100., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH -Z GRAVITY
        elif "f-hexa" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.0, h=1., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH -Z GRAVITY
        elif "f-box" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.0, h=1., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-1 SYSTEM WITH -Z GRAVITY
        elif "f-chair7" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.0, h=11., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=0.0, robot_type=LARGE_THICK)
            ori.start(ori_name, 4, ori.TSA_SIM, thick_mode=1)
        
        # IA-1 SYSTEM WITH -Z GRAVITY
        elif "f-chair8" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.0, h=11., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=0.0, robot_type=LARGE_THICK)
            ori.start(ori_name, 4, ori.TSA_SIM, thick_mode=1)

        # IA-1 SYSTEM WITH -Z GRAVITY
        elif "f-chair" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.0, h=1., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # EA SYSTEM WITH -Z GRAVITY
        elif "f-bird" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.3, h=2., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=400.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # EA SYSTEM WITH -Z GRAVITY
        elif "f-mountain-big" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.3, h=10., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=400.0)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # EA SYSTEM WITH -Z GRAVITY
        elif "f-miura" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.3, h=30., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=400.0)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # EA SYSTEM WITH -Z GRAVITY
        elif "f-8seg-miura" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, use_gui=1,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.3, h=2., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=400.0)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # EA SYSTEM WITH -Z GRAVITY, THICK-PANEL
        elif "f-miura-EA" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=0, ground_miu=0.3, h=30., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=400.0)
            ori.start(ori_name, 4, ori.TSA_SIM, thick_mode=0)
        
        # IA-1 SYSTEM WITH +X GRAVITY, THICK-PANEL
        elif "mountain-EA" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                fast_simulation=fast_mode, g=[9810., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.3, h=100., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1., additional_length=100.0)
            ori.start(ori_name, 4, ori.FOLD_SIM, thick_mode=1)
        
        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "f-robot4" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.3, h=220., 
                                strict=-1, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, simulation_upper_time=65, control_mode_training_time=60)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "f-robot8-big" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.4, h=220., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=(idx)%2, simulation_upper_time=65, control_mode_training_time=60, robot_type=LARGE, type_of_controller=2)
            ori.start(ori_name, 4, ori.TSA_SIM)

        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "f-robot" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.3, h=90., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=0)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "f1" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.3, h=60., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=(idx+1)%2)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "opt_design" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.3, h=60., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=(idx+1)%2, robot_type=3)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # IA-2 SYSTEM WITH -Z GRAVITY
        elif "f-cube-robot" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.3, h=1., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=idx%2, type_of_controller=1)
            ori.start(ori_name, 4, ori.TSA_SIM)

        elif "f-5panel-robot" in ori_name:
            ori = OrigamiSimulator(origami_name=ori_name,
                                fast_simulation=fast_mode, g=[0., 0., -9810.], default_ground=1, control_mode=1, ground_miu=0.6, h=60., 
                                strict=0, friction_mode=2, const_stiff_of_crease=False, speed_bonus=1.0, additional_length=0.0, tracking_camera=0)
            ori.start(ori_name, 4, ori.TSA_SIM)
        
        # TARGET DEFINITION INTERFACE
        elif ori_name in ["mountain-big-new", "mountain-big-fix-new", "diamond-II-acc", "diamond-III-acc", "resch-50", "tri-resch-50", "tri-resch-60-wub", "tri-resch-50-wub", "bird3", "bird4", \
                          "box", "box2", "miura", "8-seg-miura", "chair6", "miura-ori-16", "miura-EA"]:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                    fast_simulation=fast_mode, g=[0., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=1., 
                                    strict=0, friction_mode=2, const_stiff_of_crease=True, speed_bonus=1.0, check_connection_matrix=True)
            ori.start(ori_name, 4, ori.FOLD_SIM)
        
        elif ori_name in ["robot3", "robot4", "robot5", "robot6", "robot7-90", "robot8-90", "robot9-90", "robot8-big", "robot10", "robot11", "robot10-2", "robot11-2", "robot12", "robot13", "robot10-3", "robot10-4", "robot14"]:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                    fast_simulation=fast_mode, g=[0., 0., 0.], default_ground=0, control_mode=1, ground_miu=0.0, h=1., 
                                    strict=0, friction_mode=2, const_stiff_of_crease=True, speed_bonus=1.0, check_connection_matrix=True)
            ori.start(ori_name, 4, ori.FOLD_SIM)
        
        # TARGET DEFINITION INTERFACE
        elif ori_name in ["auxetic", "double_curve_3", "chair_test1", "chair_test2", "chair3", "huffman-box", "phys_sim", "chair6-2", "chair6-5-2times-big", "apple", "mustard", "demo", "demo-2D", \
                          "chair6-0thick", "chair6-thick", "phys_sim-cw", "miura-thick_batch_64", "miyamotoTower", "1", "11"]:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                    fast_simulation=fast_mode, g=[0., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=1., 
                                    strict=0, friction_mode=2, const_stiff_of_crease=True, speed_bonus=1.0, check_connection_matrix=False)
            ori.start(ori_name, 4, ori.FOLD_SIM, thick_mode=0)
        
        # TARGET DEFINITION INTERFACE
        elif ori_name in ["chair7", "chair8"]:
            ori = OrigamiSimulator(origami_name=ori_name, 
                                    fast_simulation=fast_mode, g=[0., 0., 0.], default_ground=0, control_mode=0, ground_miu=0.0, h=1., 
                                    strict=0, friction_mode=2, const_stiff_of_crease=True, speed_bonus=1.0, check_connection_matrix=False, robot_type=LARGE)
            ori.start(ori_name, 4, ori.FOLD_SIM, thick_mode=1)
        
        # TARGET DEFINITION INTERFACE - 厚板折纸折叠程度评估 (Thick Origami Folding Degree Evaluation)
        # 用于评估厚板折纸的折叠程度，计算所有折痕的平均折叠百分比
        # Used to evaluate the folding degree of thick origami, calculating the average folding percentage of all creases
        elif ori_name in ["miura-thick", "thick-eval"]:
            """
            厚板折纸折叠程度评估流程 / Thick Origami Folding Degree Evaluation Process:
            1. 读取JSON文件，获取折纸定义信息 / Read JSON file to get origami definition
            2. 初始化OrigamiSimulator，设置check_connection_matrix=True, FOLD_SIM模式, thick_mode=1
               Initialize OrigamiSimulator with check_connection_matrix=True, FOLD_SIM mode, thick_mode=1
            3. 设置所有折痕目标角度为3.14（完全折叠）/ Set all crease target angles to 3.14 (fully folded)
            4. 运行仿真直到稳定 / Run simulation until stable
            5. 使用reward()函数获取折叠百分比数据 / Get folding percentage data using reward() function
            
            优化说明 / Optimization notes:
            - 厚板模式下角度步进更快（5°为一步进）/ Faster angle step in thick mode (5° per step)
            - 跳过连接矩阵计算以提高效率 / Skip connection matrix calculation for efficiency
            - 奖励值通过reward()函数返回 / Reward values returned via reward() function
            """
            # print(f"\n{'='*60}")
            # print(f"厚板折纸折叠程度评估 / Thick Origami Folding Degree Evaluation")
            # print(f"{'='*60}\n")
            
            # 1. 初始化仿真器 / Initialize simulator
            ori = OrigamiSimulator(
                origami_name=ori_name,
                fast_simulation=fast_mode,           # 使用快速仿真模式 / Use fast simulation mode
                mode='fast',
                g=[0., 0., 0.],                 # 无重力 / No gravity
                default_ground=0,
                control_mode=0,                 # 非控制模式 / Non-control mode
                ground_miu=0.0,
                h=1.,
                strict=0,
                friction_mode=2,
                const_stiff_of_crease=True,     # 恒定折痕刚度 / Constant crease stiffness
                speed_bonus=1.0,
                check_connection_matrix=True    # 启用连接矩阵检查 / Enable connection matrix checking
            )
            
            # 2. 启动仿真，使用FOLD_SIM模式和thick_mode=1（厚板模式）
            # Start simulation with FOLD_SIM mode and thick_mode=1 (thick panel mode)
            ori.start(ori_name, 4, ori.FOLD_SIM, thick_mode=1)
            
            # print(f"折纸模型: {ori_name}")
            # print(f"折痕数量: {ori.crease_pairs_num}")
            # print(f"厚板模式: 已启用 (thick_mode=1)")
            # print(f"仿真模式: FOLD_SIM")
            # print(f"目标角度: 3.14 (完全折叠)")
            # print(f"\n开始仿真...")
            
            # 3. 设置所有折痕的目标角度为3.14（完全折叠）
            ori.folding_angle_maximum_ratio = 1.0  # 设置最大折叠比例 / Set maximum folding ratio
            
            # 4. 运行仿真 / Run simulation
            ori.run(False, False)
            
            # 5. 仿真结束后，使用reward()函数获取折叠百分比数据
            # After simulation, use reward() function to get folding percentage data
            # print(f"\n{'='*60}")
            # print(f"仿真完成，计算折叠程度...")
            # print(f"{'='*60}\n")
            
            # 调用reward()函数获取奖励值（FOLD_SIM厚板模式）
            # Call reward() function to get reward values (FOLD_SIM thick mode)
            # 返回格式: (avg%, max%, min%, std%, crease_num, 0, 0, 0, 0)
            # Return format: (avg%, max%, min%, std%, crease_num, 0, 0, 0, 0)
            avg_pct, max_pct, min_pct, std_pct, crease_num, _, _, _, _ = ori.reward()
            
            # 输出结果 / Output results
            print(f"折叠程度评估结果 / Folding Degree Evaluation Results:")
            print(f"{'-'*60}")
            print(f"折痕总数 / Total creases: {int(crease_num)}")
            print(f"平均折叠百分比 / Average folding percentage: {avg_pct:.2f}%")
            print(f"最大折叠百分比 / Maximum folding percentage: {max_pct:.2f}%")
            print(f"最小折叠百分比 / Minimum folding percentage: {min_pct:.2f}%")
            print(f"折叠百分比标准差 / Standard deviation: {std_pct:.2f}%")
            print(f"{'-'*60}")
            
            print(f"\n{'='*60}")
            print(f"评估结论 / Evaluation Conclusion:")
            print(f"{'='*60}")
            
            # 根据平均折叠百分比给出评估结论
            if avg_pct >= 95:
                conclusion = "完全折叠 / Fully Folded"
            elif avg_pct >= 80:
                conclusion = "大部分折叠 / Mostly Folded"
            elif avg_pct >= 50:
                conclusion = "部分折叠 / Partially Folded"
            else:
                conclusion = "折叠程度较低 / Low Folding Degree"
            
            print(f"折叠程度: {conclusion}")
            print(f"平均折叠百分比: {avg_pct:.2f}%")
            print(f"{'='*60}\n")
            
            print(f"程序正常结束 / Program completed successfully.\n")
            
            # 跳过后续处理 / Skip subsequent processing
            continue
        
        else:
            print(f"No simulation configuration about the origami: {ori_name}")
            continue
        # ori.initializeRunning()
        ori.run(False, False)

        if ori.sim_mode == ori.FOLD_SIM and ori.check_connection_matrix:
            try:
                if ori.connection_number[0] == 0:
                    final_valid_matrix, initial_center, current_center = ori.getValidMatrix()
                    final_valid_matrix = final_valid_matrix.tolist()
                    for i in range(len(final_valid_matrix)):
                        for j in range(len(final_valid_matrix[i])):
                            for k in range(len(ori.valid_matrix_list)):
                                if len(ori.valid_matrix_list[k]) != 0 and ori.valid_matrix_list[k][i][j] != final_valid_matrix[i][j]:
                                    final_valid_matrix[i][j] = 0
                                    break
                    ori.input_json["valid_actions"] = final_valid_matrix
                    ori.input_json["initial_center"] = initial_center
                    ori.input_json["current_center"] = current_center
                    ori.input_json["crease_angle"] = [
                        ori.crease_angle[i] for i in range(ori.crease_pairs_num)
                    ]
                    ori.input_json["trans_units"] = [
                        [
                            [
                                ori.x[ori.unit_indices[i][(j + ori.unit_kp_num_list[i]) % ori.unit_kp_num_list[i]]][k] for k in range(3)
                            ]   for j in range(0, -ori.unit_kp_num_list[i], -1)
                        ] for i in range(ori.unit_indices_num)
                    ]
                    crease_info = [
                        [ori.kps[ori.crease_pairs[i, 0]], ori.kps[ori.crease_pairs[i, 1]]] for i in range(ori.crease_pairs_num)
                    ]
                    ori.input_json["crease_info"] = crease_info
                    if ori.ref_crease_angle_mode:
                        # modify the barrier
                        for i in range(len(ori.input_json["line_features"])):
                            points = ori.lines[i].points
                            for j in range(ori.crease_pairs_num):
                                info = crease_info[j]
                                if (distance3D(points[START], info[START]) < 1e-3 and distance3D(points[END], info[END]) < 1e-3) or \
                                    (distance3D(points[START], info[END]) < 1e-3 and distance3D(points[END], info[START]) < 1e-3):
                                    ori.input_json["line_features"][i]["hard_angle"] = abs(ori.crease_angle[j]) * math.pi
                                    ori.input_json["line_features"][i]["hard_angle_down"] = -abs(ori.crease_angle[j]) * math.pi
                                    break
                    # final connection matrix
                    all_connection_id = [ori.constraint_start_point_candidate_connection[i] for i in range(ori.P_number)]
                    for i in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                        for j in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                            if (i >= ori.P_number or j >= ori.P_number) and \
                                (
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in all_connection_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in all_connection_id) or \
                                    ((i - ori.P_number) >= 0 and (i - ori.P_number) in ori.fix_id) or \
                                    ((j - ori.P_number) >= 0 and (j - ori.P_number) in ori.fix_id)
                                ):
                                ori.initial_matrix[i][j] = -10
                            else:
                                if abs(ori.initial_matrix[i][j]) != 1:
                                    if abs(final_valid_matrix[i][j]) == 1:
                                        if (i < ori.P_number or j < ori.P_number) and ori.system_type[0] != 3:
                                            if final_valid_matrix[i][j] == 1:
                                                ori.initial_matrix[i][j] = 1
                                            elif final_valid_matrix[i][j] == -1:
                                                ori.initial_matrix[i][j] = -1
                                            else:
                                                ori.initial_matrix[i][j] = 0
                                        else: # internal
                                            if ori.initial_matrix[i][j] == 0:
                                                if final_valid_matrix[i][j] == 1:
                                                    ori.initial_matrix[i][j] = 1
                                                elif final_valid_matrix[i][j] == -1:
                                                    ori.initial_matrix[i][j] = -1
                                            else:
                                                ori.initial_matrix[i][j] = -10
                                    else:
                                        ori.initial_matrix[i][j] = -10
                                else:
                                    if abs(final_valid_matrix[i][j]) == 1:
                                        if (i < ori.P_number and ori.P_candidate[i][Z] > 1e-5) or (j < ori.P_number and ori.P_candidate[j][Z] > 1e-5):
                                            ori.initial_matrix[i][j] = 1
                                        elif (i < ori.P_number and ori.P_candidate[i][Z] < -1e-5) or (j < ori.P_number and ori.P_candidate[j][Z] < -1e-5):
                                            ori.initial_matrix[i][j] = -1
                                        elif ori.initial_matrix[i][j] == 0:
                                            if final_valid_matrix[i][j] == 1:
                                                ori.initial_matrix[i][j] = 1
                                            elif final_valid_matrix[i][j] == -1:
                                                ori.initial_matrix[i][j] = -1
                                    else:
                                        ori.initial_matrix[i][j] = -10
                            # if i == j and i >= ori.P_number and (i - ori.P_number) not in all_connection_id:
                            #     ori.initial_matrix[i][j] = -2
                all_connection_id = [ori.constraint_start_point_candidate_connection[i] for i in range(ori.P_number)]
                for i in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                    for j in range(ori.unit_indices_num - ori.connection_number[0] + ori.P_number):
                        if (i >= ori.P_number or j >= ori.P_number) and \
                            (
                                ((i - ori.P_number) >= 0 and (i - ori.P_number) in all_connection_id) or \
                                ((j - ori.P_number) >= 0 and (j - ori.P_number) in all_connection_id) or \
                                ((i - ori.P_number) >= 0 and (i - ori.P_number) in ori.fix_id) or \
                                ((j - ori.P_number) >= 0 and (j - ori.P_number) in ori.fix_id)
                            ):
                            ori.initial_matrix[i][j] = -10
                            try:
                                p_index = all_connection_id.index((i - ori.P_number) if (i - ori.P_number) in all_connection_id else (j - ori.P_number))
                                ori.initial_matrix[p_index][all_connection_id[p_index] + ori.P_number] = -10
                                ori.initial_matrix[all_connection_id[p_index] + ori.P_number][p_index] = -10
                            except:
                                pass
                            try:
                                p_index = ori.fix_id.index((i - ori.P_number) if (i - ori.P_number) in all_connection_id else (j - ori.P_number))
                                ori.initial_matrix[p_index][all_connection_id[p_index] + ori.P_number] = -10
                                ori.initial_matrix[all_connection_id[p_index] + ori.P_number][p_index] = -10
                            except:
                                pass
                else:
                    initial_center, current_center = ori.getCenter()
                ori.input_json["initial_center"] = initial_center
                ori.input_json["current_center"] = current_center
                ori.input_json["true_valid_actions"] = ori.initial_matrix.tolist()
                
                with open("./descriptionData/" + ori_name + ".json", 'w', encoding='utf-8') as fw:
                    json.dump(ori.input_json, fw, indent=4)
            except:
                pass
        else:
            if len(ori.recorded_t):
                if ori.control_mode == 0:
                    # folding_percent = ori.recorded_folding_percent[-1]
                    # folding_error = ori.recorded_folding_error[-1]
                    # maximum_folding_error = ori.recorded_maximum_folding_error[-1]
                    # minimum_folding_percent = ori.recorded_minimum_folding_percent[-1]
                    # folding_speed = (ori.recorded_folding_percent[-1] - ori.recorded_folding_percent[0]) / (ori.recorded_t[-1] - ori.recorded_t[0])
                    # # value = folding_speed * folding_percent
                    # string_length_decrease_each_final = sorted([ori.recorded_string_decrease_length_control[i][-1] for i in range(ori.constraint_number)])
                    # actuator_number = 0
                    # base_contraction = 0
                    # for ele in string_length_decrease_each_final:
                    #     if abs(ele - base_contraction) > 2.0:
                    #         actuator_number += 1
                    #     base_contraction = ele
                    
                    # actuator_bonus = 1. + 0.1 * (ori.constraint_number - actuator_number)
                    # if folding_percent < FOLDING_MAXIMUM:
                    #     value = folding_percent * minimum_folding_percent
                    # else:
                    #     value = (folding_speed + folding_percent) * minimum_folding_percent

                    # value *= actuator_bonus
                    # rf = 1 - folding_error / np.pi
                    # rm = 1 - maximum_folding_error / np.pi
                    # rs = rf / (ori.recorded_t[-1] - ori.recorded_t[0])
                    # new_value = (rf + rs + rm) / 2 * actuator_bonus
                    # print(f"评分: \n平均折叠: {folding_percent}, \n最小折叠: {minimum_folding_percent}, \n折叠速度: {folding_speed}, \n线收缩: {string_length_decrease_each_final}, \n驱动器分: {actuator_bonus}, \n总分: {value}")
                    value, value_f, rf, rm, rs, actuator_bonus, rh, rforce, rforce2 = ori.reward()
                    print(f"\nAvg. FAE. : {ori.recorded_folding_error[-1] * 180 / np.pi} deg, \nMax. FAE. : {ori.recorded_maximum_folding_error[-1] * 180 / np.pi} deg, \nTime: {ori.recorded_t[-1] - ori.recorded_t[0]}, \nSub-reward: rf = {rf}, rm = {rm}, rs = {rs},\nH: {rh}, Tension: {rforce}, Max. Mises: {rforce2}\nReward: speed: {value} / stress: {value_f}")
                else:
                    try:
                        move_x = ori.recorded_movement_x[-1]
                        move_y = ori.recorded_movement_y[-1]
                    except:
                        move_x = 0
                        move_y = 0
                    checkpoints = [[round(ori.recorded_interval_velocity[i][0][X], 3), round(ori.recorded_interval_velocity[i][0][Y], 3)] for i in range(len(ori.recorded_interval_velocity))]
                    time_checkpoints = [round(ori.recorded_interval_velocity[i][1], 3) for i in range(len(ori.recorded_interval_velocity))]
                    
                    print(f"间隔记录点: {checkpoints}, 对应时间点: {time_checkpoints}")
                    print(f"步态: {ori.control_signals}")
                    for i in range(len(ori.recorded_interval_velocity) - 1):
                        displacement = [
                            round(ori.recorded_interval_velocity[i + 1][0][X] - ori.recorded_interval_velocity[i][0][X], 3),
                            round(ori.recorded_interval_velocity[i + 1][0][Y] - ori.recorded_interval_velocity[i][0][Y], 3)
                        ]
                        time_interval = round(ori.recorded_interval_velocity[i + 1][1] - ori.recorded_interval_velocity[i][1], 3)
                        print(f"间隔{i}: 位移: {displacement}, 时间: {time_interval}, 位移量: {np.sqrt(displacement[0] ** 2 + displacement[1] ** 2)}, 速度: {np.sqrt(displacement[0] ** 2 + displacement[1] ** 2) / time_interval} BL/s")
                    print(f"最终: X移动距离: {move_x}, Y移动距离: {move_y}, 总距离: {np.sqrt(move_x ** 2 + move_y ** 2)}")
                    pass_num = sum([len(ori.method["id"][i]) for i in range(ori.constraint_number)])
                    print(f"绳索干涉惩罚: {ori.intersection_penalty[0]}, 基准上限: {int(ori.current_t / ori.dt * (pass_num - ori.constraint_number))}")
                    value, value_f, rf, rm, rs, actuator_bonus, rh, rforce, rforce2 = ori.reward()
                    try:
                        phase_2_value = ori.phase_2_value
                    except:
                        phase_2_value = 0
                    new_value = max(phase_2_value, value)
                    print(f"\n新评分: 移动能力: {rf} BL/s, 转动角速度: {rm} rad/s, reward: {value}")
        
        if len(ori.recorded_t) and ori.sim_mode == ori.TSA_SIM:
            all_dis = {
                "value": value,
                "sub_val": [
                    rf, rm, rs
                ],
                "Avg. FAE": ori.recorded_folding_error[-1] * 180 / np.pi,
                "Max. FAE": ori.recorded_maximum_folding_error[-1] * 180 / np.pi,
                "Total Time": ori.recorded_t[-1] - ori.recorded_t[0],
                "gait": ori.control_signals,
                "final_control_signal_each": [],
                "final_decrease_each": [],
                "string_length_each": [],
                "string_total_length": 0,
                "control_string_decrease": [],
                "string_decrease_each": [],
                "max_force": [],
                "max_nodal_force": [],
                "folding_percent": [],
                "folding_error": [],
                "max_folding_error": [],
                "min_folding_error": [],
                "max_folding_percent": [],
                "min_folding_percent": [],
                "time": [],
                "string_force_each": [],
                "recorded_movement_x": [],
                "recorded_movement_y": [],
                "recorded_movement_z": [],
                "recorded_indices": [],
            }
            all_dis["control_string_decrease"] = ori.recorded_string_decrease_length_control
            all_dis["string_decrease_each"] = ori.recorded_string_decrease_length
            all_dis["final_control_signal_each"] = [ori.recorded_string_decrease_length_control[i][-1] for i in range(ori.constraint_number)]
            all_dis["final_decrease_each"] = [ori.recorded_string_decrease_length[i][-1] for i in range(ori.constraint_number)]
            all_dis["string_length_each"] = ori.string_length_each
            all_dis["string_total_length"] = sum(ori.string_length_each)
            all_dis["folding_percent"] = ori.recorded_folding_percent
            all_dis["folding_error"] = ori.recorded_folding_error
            all_dis["max_folding_error"] = ori.recorded_maximum_folding_error
            all_dis["min_folding_error"] = ori.recorded_minimum_folding_error
            all_dis["max_folding_percent"] = ori.recorded_maximum_folding_percent
            all_dis["min_folding_percent"] = ori.recorded_minimum_folding_percent
            all_dis["max_force"] = ori.recorded_max_force
            all_dis["nodal_force"] = ori.recorded_nodal_maximum_force
            all_dis["time"] = ori.recorded_t
            all_dis["deal"] = 1
            all_dis["string_force_each"] = ori.recorded_string_force
            all_dis["recorded_movement_x"] = ori.recorded_movement_x
            all_dis["recorded_movement_y"] = ori.recorded_movement_y
            all_dis["recorded_movement_z"] = ori.recorded_movement_z
            all_dis["recorded_indices"] = ori.recorded_indices


            with open(os.path.join("./physResult", ori.origami_name + "_" +time.strftime('%Y%m%d-%H%M%S', time.localtime()) + ".json"), 'w', encoding="utf-8") as f:
                json.dump(all_dis, f, indent=4)

        try:
            video_maker = ti.tools.VideoManager(output_dir=f"./physResult/{ori.origami_name}-{ori.time}-fast", framerate=60, automatic_build=False)
            video_maker.make_video(mp4=True)
        except:
            pass

        if ori.use_gui:
            ori.window.destroy()
    
    # video_maker = ti.tools.VideoManager(output_dir=f"./physResult/best1-20241017-232854", framerate=60, automatic_build=False)
    # video_maker.make_video(mp4=True)
