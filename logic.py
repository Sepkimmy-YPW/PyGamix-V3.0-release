# Std API import
import sys
import os
import math
import subprocess
import numpy as np
import json
import time
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
# import pandas as pd
# import multiprocessing
import matplotlib as mpl
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add GUI python file into our application
sys.path.append('./gui')

# Qt API import
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPixmap, QPen, QCursor, QPolygon, QFont, QIcon
from PyQt5.QtCore import QObject, Qt, QTimer, QPoint, QThread, QProcess, pyqtSignal
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

# Window/Dialog sub-module import
from gui.Ui_window import Ui_MainWindow
from tm_window import TmWindow
# from stl_dialog import StlSettingDialog
from pref_pack import PreferencePackWindow
from threading_design_dialog import ThreadingDesignDialog
FOLDING_MAXIMUM = 163.0 / 180.0 * 0.95
FOLDING_ERROR_MINIMUM = 5.0 / 180.0 * np.pi

# Sub-module import

# --Description module for describing the source of design process-- #
from desc import *
# --Designer module for designing origami from source-- #
from designer import *
# --Units module for representing origami units like Miura and LeanMiura-- #
from units import *
# --Utils module for providing necessary mathematical tools for application-- #
from utils import *
# --Dxftool module for outputing 2D drawing file of the origami-- #
from dxftool import *
# --Cdftool module for doing computational design -- #
from cdftool import *
# --Stltool module for outputing 3D model file of the origami-- #
from stltool2 import StlMaker

# Window for user to design origami
class Mainwindow(Ui_MainWindow, QMainWindow):
    def __init__(self, parent=None) -> None:
        """
        @ function: Program initialization
        @ version: 0.1111
        @ developer: py
        @ progress: on road
        @ date: 20230415
        @ spec: Add notes
        """
        # Set up for parent and set up for UI
        super(Mainwindow, self).__init__(parent)
        self.setupUi(self)
        icon_path = "./setting/icon.png"
        window_icon = QIcon(icon_path)
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)

        # Define action-function conncetion for application
        # 1 Action slot
        self.defineAction()
        # 2 Button slot
        self.defineButton()
        # 3 Spinbox slot
        self.defineSpinbox()
        
        # --Define flags for application-- #
        self.INITIAL_STATE = 0x0001
        self.IMPORT_SUCCESS = 0x0002
        self.DESIGN_FINISH = 0x0004
        self.DESIGN_ERROR = 0x0008
        self.OUTPUT_FINISH = 0x0010

        # --Define const value for application-- #
        self.KL_JSON = 0x0001
        self.PACKED_DATA = 0x0002
        self.THREADING_METHOD = 0x0004

        self.A4_length = 296
        self.A4_width = 210
        self.A4_half_length = 148
        self.A4_half_width = 105

        self.pixmap_length = 4096
        self.pixmap_width = 2048

        self.draw_panel_x_bias = 169
        self.draw_panel_y_bias = 62

        # --Define parameters for application-- #
        # Axis parameters
        self.cursor_x = 0
        self.cursor_y = 0
        self.real_x = 0
        self.real_y = 0
        self.initial_x = 0          # record initial value of cursor[X]
        self.initial_y = 0          # record initial value of cursor[Y]
        self.enable_moving = False  # represent that whether to enable recording the axis of cursor
        
        self.mirror_x = False
        self.mirror_y = False
        self.xy_rotation = self.horizontal_xy_rotation_slider.value()

        # Current file information
        self.file_type = None
        self.file_path = None
        self.string_file_path = None

        # Origami information
        self.origami_length = 0
        self.origami_width = 0
        self.origami_info = []  # Type:[origin_point, length, width, origami_type]
        self.origami_number = 0 # Number of all design results
        self.unit_number = []

        # The amplitude of operation (1-180)
        self.operation_amp = 0.05

        # Inner value got from Ui widgets
        self.unit_width         = self.spinbox_crease_width.value() # mm
        self.bias_max           = self.unit_width * 0.2
        self.copy_time          = self.spinbox_copy_time.value()
        self.entry_flag         = self.slider_flag.value()
        self.add_hole_mode      = self.checkBox_add_hole_mode.isChecked()
        self.add_string_mode    = self.checkBox_add_string_mode.isChecked()
        self.hole_size          = self.spinbox_hole_size.value()
        self.hole_resolution    = self.spinbox_resolution.value()
        self.state              = self.INITIAL_STATE

        # Storage information for designer
        self.storage = []
        self.rotation = []
        self.paper_info = {
            "unit_width": self.unit_width
        }
        self.design_info = {
            "copy_time": 1,
            "folding_percent": 0.0
        }

        # whether to put the target angles to the barrier
        self.ref_crease_angle_mode = REF_TARGET_ANGLE

        # Pixmap information for drawing origami design result
        self.half_pixmap_length     = self.pixmap_length / 2    #375
        self.half_pixmap_width      = self.pixmap_width / 2     #225
        self.pixmap                 = QPixmap(self.pixmap_length, self.pixmap_width)
        self.A4_pixmap              = QPixmap(self.A4_length, self.A4_width)
        self.output_pixmap          = QPixmap(1920, 1920)
        self.pixmap.fill(QColor(255, 255, 255))
        self.A4_pixmap.fill(QColor(255, 255, 255))
        self.output_pixmap.fill(QColor(255, 255, 255))

        self.draw_panel.mouseDoubleClickEvent = self.mouseDoubleClickEvent_panel
        self.draw_panel.mouseMoveEvent = self.mouseMoveEvent_panel
        self.draw_panel.mousePressEvent = self.mousePressEvent_panel
        self.draw_panel.mouseReleaseEvent = self.mouseReleaseEvent_panel

        # Set painter and painting device
        self.painter = QPainter(self.pixmap)
        self.painter.end()

        # Realworld coordinate information
        self.kps = []                           # keypoints
        self.lines = []                         # key lines
        self.units = []                         # origami units, type: [Unit...]
        self.additional_lines = []              # additional line for warning of the add-hole operation
        self.hole_kps = []                      # hole keypoints
        self.connection_hole_kps = []           # connection hole keypoints 
        self.crease_lines = []                  # crease line for output crease dxf file
        self.backup_connection_hole_kps = []    # back up the connection keypoint for exporting stl
        self.strings = []                       # TSA strings
        self.unit_bias_list = []
        self.backup_unit_bias_list = []
        self.backup_crease_type_list = []

        self.transformation_matrix = []
        self.enable_read_list_from_backup = False
        self.enable_read_contribution_from_backup = False
    
        # Pixelworld coordinate information
        self.pixel_kps = []                     # keypoints
        self.pixel_lines = []                   # key lines
        self.pixel_additional_lines = []        # additional line for warning of the add-hole operation
        self.pixel_hole_kps = []                # hole keypoints
        self.pixel_connection_hole_kps = []     # connection hole keypoints 
        self.pixel_string_kps = []

        # Parameters for axisConverter()
        self.pixel_bias = [25, 25]                  # transform 2D: transform[X, Y]
        self.old_pixel_bias_x = self.pixel_bias[0]  # old bias[X]
        self.old_pixel_bias_y = self.pixel_bias[1]  # old bias[Y]
        self.pixel_scale = [
            15, 12.5, 10, 9.0, 7.8, 6.5, 5.5, 5, 3.8, 3.1, 2.5, 2.0, 1.6, 1.3, 1.1, 1.0, 
            0.93, 0.87, 0.81, 0.76, 0.71, 0.66, 0.62, 0.58, 0.54, 0.51, 0.48, 0.45, 0.42, 0.4, 0.2, 0.1, 0.05, 0.03, 0.01
        ]                                           # scale factor list
        self.pixel_scale_ranking = 14               # scale level referencing pixel_scale list
        self.current_pixel_scale = 1.0              # value got from pixel_scale[scale_ranking]
        self.pixel_scale_min_ranking = len(self.pixel_scale)

        # Window update method
        self.show_square = None
        self.show_process = True
        self.enable_design = True

        # Start QT-Thread flag
        self.enable_output_stl = False
        self.enable_cdf_curve_fitting = False
        self.enable_phys_data_collecting = False
        self.enable_mcts = False
        self.enable_threading_design = False

        # Define QT-Thread object which is belong to the application
        self.stl_output_thread = None
        self.cdf_curve_fitting_thread = None
        self.phys_data_collecting_thread = None
        self.mcts_thread = None
        self.threading_design_process = None

        # Dxf output setting
        self.dxf_split_flag = False

        # Stl output information of file name
        self.output_stl_file_path = ""
        self.output_stl_crease_flag = ""
        self.output_stl_board_flag = ""

        # Additional line information
        self.bias_val = 0.1
        self.show_additional_crease = True # True if add-hole mode is enabled

        # Curve information for TM-Window
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.dir_list = []
        self.curve_name = None # we generate corresponding file name of CDF result
        
        # Tools
        self.designer               = OrigamiDesigner(src=None, paper_info=None, design_info=None)
        self.dxf_writer             = OrigamiToDxfConverter()
        self.stl_writer             = StlMaker()
        self.additional_line_maker  = StlMaker() # this maker only generate additional lines, not for 3D model output
        self.pref_pack_window       = PreferencePackWindow(parent=self)
        self.string_generator       = TSAString() 

        # connection information
        self.enable_connection  = self.checkBox_connection.isChecked()
        self.connection_radius  = self.spinbox_connection_radius.value()
        self.con_left_length    = self.spinbox_con_left_length.value()
        self.con_right_length   = self.spinbox_con_right_length.value()
        self.add_bias_flag = [] # list of bool ([bool...]) of whether to add width-bias for every origami design result

        # Parameters of hole information we choose by double-clicking a hole
        self.choose_hole_flag = False
        self.choose_hole_id = 0
        self.choose_hole_index = 0

        # Parameters of origami information we choose by double-clicking a origami
        self.choose_origami_flag = False
        self.choose_origami_index = 0

        self.choose_unit_id = -1
        self.choose_crease_id = -1
        self.choose_kl_id = -1
        self.choose_line_id = -1
        self.choose_crease_sequence_id = -1
        self.choose_string_id = -1
        self.expert_mode = False
        self.edit_kl_mode = False
        self.edit_sequence_mode = False

        # Parameters of hard crease
        self.hard_crease_index = []

        # Parameters of string start&end
        self.exist_string_start = False
        self.string_start_point = []
        self.string_type = BOTTOM

        self.a_string = [] # one string
        self.string_total_information = [] # list of list
        
        # Use unified initialization for application
        self.pref_pack_window.readFile()                        # import setting(or preference)
        self.pref_pack = self.pref_pack_window.getPrefPack()    # get preference pack from pref_window
        self.limitation = self.pref_pack_window.getLimitation() # get CDF-Limitation pack from pref_window


        self.widget.setVisible(False)
        self.widget_edit_kl.setVisible(False)
        self.widget_edit_sequence.setVisible(False)

        self.P_candidate = []
        self.P_candidate_connection_index = []

        self.fixed_panel = -1

        self.show_index = False

        self.full_description_mode = True

        self.unit_center_contribute_coeff = []
        self.backup_unit_center_contribute_coeff = []
        
        self.backup_open_file_path = None
        
        self.crease_angle = [] # crease angle for each crease line
        self.crease_info = []

        self.setupTimer()                                       # enable timer to update information of application
        self.updateState("Ready", self.state)                   # update state bar 1
        self.updateMessage("Ready")                             # update state bar 2(message bar)
        self.setFocusPolicy(Qt.StrongFocus)                     # strong focus 

    def addConnectionHole(self): # real axis
        self.connection_hole_kps.clear()
        col_num = self.copy_time
        for j in range(len(self.origami_info)):
            if type(self.origami_info[j][-1]) == KinematicLine:
                origami = self.origami_info[j]
                all_unit_number_of_origami = self.unit_number[j]
                copy_time = self.copy_time
                row_num = int(all_unit_number_of_origami / copy_time)
                start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)])
                for i in range(col_num):
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + 0.5 * self.con_left_length,
                            origami[0][Y] + self.unit_width / 4 + self.unit_width * i
                        ],
                        start_unit_id + i * row_num, True, LEFT #left
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + 0.5 * self.con_left_length,
                            origami[0][Y] + self.unit_width * 3 / 4 + self.unit_width * i
                        ],
                        start_unit_id + i * row_num + 1, True, LEFT #left
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + self.con_left_length + origami[1] + 1 / 2 * self.con_right_length,
                            origami[0][Y] + self.unit_width / 4 + self.unit_width * i
                        ],
                        start_unit_id + row_num - 2 + i * row_num, True, RIGHT #right
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + self.con_left_length + origami[1] + 1 / 2 * self.con_right_length,
                            origami[0][Y] + self.unit_width * 3 / 4 + self.unit_width * i
                        ],
                        start_unit_id + row_num - 1 + i * row_num, True, RIGHT #right
                    ])
            elif type(self.origami_info[j][-1]) == ModuleLeanMiura:
                origami = self.origami_info[j]
                copy_time = origami[-1].copy_time
                if origami[-1].half_flag != NO_HALF:
                    start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)]) + self.unit_number[j] - 2 * copy_time
                    # Get the global bias
                    if origami[-1].half_flag == LEFT_HALF:
                        initial_x = origami[0][X] + origami[-1].con_left_length / 2
                        initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    elif origami[-1].half_flag == RIGHT_HALF:
                        initial_x = origami[0][X] + origami[-1].unit_width * copy_time / 2 + origami[-1].stretch_length + origami[-1].con_right_length / 2
                        initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    f = LEFT # all at left board
                    # Add connection hole
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1, True, f
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index, True, f
                        ])
                else:
                    start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)]) + self.unit_number[j] - 4 * copy_time
                    initial_x = origami[0][X] + origami[-1].con_left_length / 2
                    initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    # Add connection hole
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1, True, LEFT
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index, True, LEFT
                        ])
                    # Add connection hole
                    initial_x = origami[0][X] + origami[-1].con_left_length + origami[-1].unit_width * copy_time + origami[-1].stretch_length + origami[-1].con_right_length / 2
                    initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1 + 2 * copy_time, True, LEFT
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index + 2 * copy_time, True, LEFT
                        ])
                        
        self.backup_connection_hole_kps = deepcopy(self.connection_hole_kps)

    def addCrease(self, creases):
        for new_crease in creases:
            if new_crease.getLength() < 1e-5:
                continue
            same_crease = False
            for crease in self.lines:
                if sameCrease(new_crease, crease):
                    # same kp
                    same_crease = True
                    break
            if not same_crease:
                self.lines.append(new_crease)

    def addHoleToUnit(self, pixel_x, pixel_y):
        """
        @ function: Convert some axis fro
        m pixel to real, and add hole to unit
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230305
        @ spec: None
        """
        real_x, real_y = self.mapFromPixelToReal(pixel_x, pixel_y)
        unit_id = self.pointInUnit([real_x, real_y])
        if unit_id != None:
            # point, id, valid
            self.hole_kps.append([[real_x, real_y], unit_id, True])
            self.updateMessage("Successfully add hole on unit " + str(unit_id) + ". Total hole number: " + str(len(self.hole_kps)) + " ...")
        else:
            self.updateMessage("Failed to add hole, check if it is inside some unit...", "WARNING")

    def addHoleToUnitUsingRealAxis(self, real_x, real_y):
        """
        @ function: add hole to unit
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230305
        @ spec: None
        """
        unit_id = self.pointInUnit([real_x, real_y])
        if unit_id != None:
            # point, id, valid
            self.hole_kps.append([[real_x, real_y], unit_id, True])
            self.updateMessage("Total: " + str(len(self.hole_kps)) + " Successfully add hole on unit " + str(unit_id) + " ...")
        else:
            self.updateMessage("Failed to add hole, check if it is inside some unit...", "WARNING")    

    def addKp(self, kps):
        for new_kp in kps:
            same_kp = False
            for kp in self.kps:
                if distance(kp, new_kp) < 1e-5:
                    # same kp
                    same_kp = True
                    break
            if not same_kp:
                self.kps.append(new_kp)

    def addLeanMiuraStorage(self):
        lean_miura_storage = ModuleLeanMiura()
        lean_miura_storage.raiseDialog(self)
        if lean_miura_storage.initial_flag:
            # success to add leanmiura
            self.updateMessage("Success to add LeanMiura")
            self.enable_design = True
            tsp = lean_miura_storage.tsp
            new_storage = [
                tsp, 
                lean_miura_storage,
                False
            ]
            self.storage.append(new_storage)
            self.rotation.append(0.0)
            self.add_bias_flag.append(False)

    def addMiuraStorage(self):
        path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a json file with kl specification", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            self.backup_open_file_path = None
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
            origin_list = input_json['origin']
            add_width_flag_list = input_json['add_width']
            origin_list_length = len(origin_list)
            for i in range(origin_list_length):
                origin = origin_list[i]
                test, ok = QInputDialog.getText(self, "Reset transition start point", "enter tsp splited with ','")
                if ok:
                    ans = re.findall(r"\d+\.?\d*", test)
                    if len(ans) >= 2:
                        origin[0] = ans[0]
                        origin[1] = ans[1]
                add_width_flag = add_width_flag_list[i]
                kl = KinematicLine()
                for j in range(len(input_json['kl'][i])):
                    element = input_json['kl'][i][j]
                    kl.append(element)
                self.storage.append([origin, kl, add_width_flag])
                self.rotation.append(0.0)
                self.add_bias_flag.append(False)
            self.updateState("Succeeded to load Miura json file", self.state)
            self.enable_design = True

    def addPassStringUsingRealAxis(self, real_x, real_y):
        pass

    def addRecoverLevel(self):
        index, ok = QInputDialog.getInt(self, "Recover Level Input: ", "Please input the recover level: ", 0, -99, 99, 1)
        if ok:
            angle, ok = QInputDialog.getInt(self, "Recover Angle Input: ", "Please input the recover angle (deg): ", 0, -180, 180, 1)
            angle *= math.pi / 180.0
            if ok:
                if index not in self.lines[self.choose_crease_sequence_id].recover_level:
                    self.lines[self.choose_crease_sequence_id].recover_level.append(index)
                    self.lines[self.choose_crease_sequence_id].recover_angle.append(angle)
                    self.comboBox_recover_level.clear()
                    self.comboBox_recover_level.addItems([str(self.lines[self.choose_crease_sequence_id].recover_level[i]) for i in range(len(self.lines[self.choose_crease_sequence_id].recover_level))])
                    self.chooseCreaseSequence(self.choose_crease_sequence_id, 1)
                    self.updateState(f"Succeed to add recover level {index} and angle {angle} for crease {self.choose_crease_sequence_id}", self.state)
                else:
                    self.updateState(f"Recover level {index} already exists", self.state)
                    

    def addTSACandidators(self):
        content, ok = QInputDialog.getText(self, "TSA Candidators Input", "Please input the axis of TSA candidator: ")
        if ok:
            ans = re.findall(r"-?\d+\.?\d*", content)
            if len(ans) != 3:
                self.updateState("Failed to add candidator, make sure that you input x, y and z of the axis", self.state)
            else:
                connection, ok = QInputDialog.getInt(self, "Candidator Connection: ", "Please select a unit for connection: ", -1, -1, len(self.units) - 1, 1)
                if ok:
                    self.P_candidate.append([float(ans[0]), float(ans[1]), float(ans[2])])
                    self.P_candidate_connection_index.append(connection)
                    self.updateState(f"Succeed to add candidator {[float(ans[0]), float(ans[1]), float(ans[2])]} connected to {connection}", self.state)

    def addTsaAPoint(self):
        if len(self.P_candidate):
            index, ok = QInputDialog.getInt(self, "TSA Input: ", "Please input TSA A point ID: ", 0, 0, len(self.P_candidate) - 1, 1)
            if ok:
                self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
        else:
            self.updateState(f"Please add TSA A candidators first", self.state)
    
    def addTsaAPointWithResolutionValue(self, resolution_value):
        origami_size = [self.origami_length, self.origami_width]
        cal_x = self.pref_pack["tsa_radius"] * math.cos(resolution_value / self.pref_pack["tsa_resolution"] * 2 * math.pi) + origami_size[X] / 2.0
        cal_y = self.pref_pack["tsa_radius"] * math.sin(resolution_value / self.pref_pack["tsa_resolution"] * 2 * math.pi) + origami_size[Y] / 2.0
        self.addStringPoint(cal_x, cal_y, resolution_value)

    def addStringPoint(self, x, y, id=0, id_type='A', reverse=False):
        if self.exist_string_start:
            if reverse:
                end_point = [x, y, 0.0]
                self.strings.append(TSAString())
                self.strings[-1].type = self.string_type
                self.strings[-1].setStringKeyPoint(self.string_start_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                self.strings.append(TSAString())
                self.strings[-1].type = PASS
                self.strings[-1].setStringKeyPoint(end_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                tsa_point = TSAPoint()
                tsa_point.point = np.array(end_point)
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                if self.string_type == BOTTOM:
                    self.string_type = TOP
                    self.updateMessage("Enable passing from bottom to top. Add 2 strings, currently the type is top...")
                else:
                    self.string_type = BOTTOM
                    self.updateMessage("Enable passing from top to bottom. Add 2 strings, currently the type is bottom...")

                self.string_start_point = end_point
            else:
                end_point = [x, y, 0.0]
                self.strings.append(TSAString())
                self.strings[-1].type = self.string_type
                self.strings[-1].setStringKeyPoint(self.string_start_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                tsa_point = TSAPoint()
                tsa_point.point = np.array(end_point)
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                if self.string_type == BOTTOM:
                    self.updateMessage("Add 1 strings, currently the type is still bottom...")
                else:
                    self.updateMessage("Add 1 strings, currently the type is still top...")

                self.string_start_point = end_point
        else:
            if reverse:
                if self.string_type == BOTTOM:
                    self.string_type = TOP
                    self.updateMessage("Change the z-axis of the string, currently the type is top...")
                else:
                    self.string_type = BOTTOM
                    self.updateMessage("Change the z-axis of the string, currently the type is bottom...")
            else:
                self.string_start_point = [x, y, 0.0]
                self.updateMessage("Record the start point of the string...")

                tsa_point = TSAPoint()
                tsa_point.point = np.array([x, y, 0.0])
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                self.exist_string_start = True  

    def addUnitPackStorage(self):
        unit_pack_storage = UnitPack()
        unit_pack_storage.raiseDialog(self)
        if unit_pack_storage.getLength():
            # success to add leanmiura
            self.updateMessage("Success to add UnitPack")
            tsp = unit_pack_storage.tsp
            self.enable_design = True
            new_storage = [
                tsp, 
                unit_pack_storage,
                False
            ]
            self.storage.append(new_storage)
            self.rotation.append(0.0)
            self.add_bias_flag.append(False)

    def appendSimulationAngles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a description pack", 
            ".", 
            "Json files (*.json);;Txt files (*.txt);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel opening file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.crease_angle = input_json['crease_angle']     
                self.crease_info = input_json['crease_info']
                if self.ref_crease_angle_mode:
                    # modify the barrier
                    for line in self.lines:
                        points = line.points
                        for i in range(len(self.crease_angle)):
                            info = self.crease_info[i]
                            if (distance3D(points[START], info[START]) < 1e-3 and distance3D(points[END], info[END]) < 1e-3) or \
                                (distance3D(points[START], info[END]) < 1e-3 and distance3D(points[END], info[START]) < 1e-3):
                                line.folding_angle_upper_bound = abs(self.crease_angle[i]) * math.pi
                                line.folding_angle_lower_bound = -abs(self.crease_angle[i]) * math.pi
                                break
                    # modify the trim
                    for i in range(len(self.units)):
                        creases = self.units[i].getCrease()
                        for j in range(len(creases)):
                            crease = creases[j]
                            points = crease.points
                            for k in range(len(self.crease_angle)):
                                info = self.crease_info[k]
                                if (distance3D(points[START], info[START]) < 1e-3 and distance3D(points[END], info[END]) < 1e-3) or \
                                    (distance3D(points[START], info[END]) < 1e-3 and distance3D(points[END], info[START]) < 1e-3):
                                    angle = abs(self.crease_angle[k]) * math.pi
                                    half_angle = angle * 0.5
                                    new_bias = min(math.tan(half_angle) * self.pref_pack["layer_of_panel"] * self.pref_pack["print_accuracy"], self.bias_val)
                                    self.unit_bias_list[i][j] = new_bias
                                    break
                    self.choose_hole_flag = False
                    self.choose_origami_flag = False
                    self.expert_mode = False
                    self.edit_kl_mode = False
                    self.edit_sequence_mode = False
                    self.choose_unit_id = 0
                    self.choose_crease_id = 0
                    self.choose_kl_id = 0
                    self.choose_line_id = 0
                    self.choose_crease_sequence_id = 0
                    self.choose_string_id = -1
                    self.widget.setVisible(False)
                    self.widget_edit_kl.setVisible(False)
                    self.widget_edit_sequence.setVisible(False)
                    self.updateMessage("Back to normal view...")
                self.updateState(f"Add {len(self.crease_angle)} crease angle information.", self.state)    
            except:
                self.crease_angle = []
                self.crease_info = []
                self.updateState("No crease angle information.", self.state, "ERROR")

    def appendTransformationMatrix(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a description pack", 
            ".", 
            "Json files (*.json);;Txt files (*.txt);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel opening file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)

                # 提取键值，校验键是否存在
                if "units" not in input_json or "trans_units" not in input_json:
                    raise ValueError("Lack 'units' or 'trans_units' key!")
                
                units = input_json["units"]
                trans_units = input_json["trans_units"]

                # 校验两个列表长度一致
                if len(units) != len(trans_units):
                    raise ValueError(f"units长度({len(units)})与trans_units长度({len(trans_units)})不匹配！")
            
                num_units = len(units)
                # print(f"✅ 成功读取数据，共 {num_units} 个单元")

                # 2. 初始化变换矩阵列表
                self.transformation_matrix = []

                # 3. 遍历每个单元，计算对应变换矩阵
                for idx in range(num_units):
                    # print(f"\n┌── 计算第 {idx+1}/{num_units} 个单元的变换矩阵")
                    
                    # 提取当前单元的点集（转为numpy数组）
                    src_points = np.array(units[idx], dtype=np.float64)
                    tgt_points = np.array(trans_units[idx], dtype=np.float64)

                    # 计算变换矩阵
                    matrix = compute_3d_rigid_transform(src_points, tgt_points)
                    self.transformation_matrix.append(matrix)

                    # # 打印当前矩阵（保留4位小数）
                    # print(f"└── 4x4齐次变换矩阵：\n{np.round(matrix, 4)}")
                
                self.updateState(f"Add {num_units} transformation matrix.", self.state)   
            except:
                self.transformation_matrix = []
                self.updateState("No matrix information.", self.state, "ERROR")

    def axisConverter(self):
        """
        @ function: Convert axis from real to pixel
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230107
        @ spec: Add scale standard
        """
        self.pixel_kps.clear()
        self.pixel_lines.clear()
        self.pixel_hole_kps.clear()
        self.pixel_additional_lines.clear()
        self.pixel_connection_hole_kps.clear()
        self.pixel_string_kps.clear()

        # keypoints
        for kp in self.kps:
            self.pixel_kps.append(self.toPixel(kp))
        
        # lines
        for line in self.lines:
            self.pixel_lines.append(Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            ))

        # additional lines
        for line in self.additional_lines:
            self.pixel_additional_lines.append(Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType()
            ))

        # hole keypoints
        for kp in self.hole_kps:
            self.pixel_hole_kps.append([self.toPixel(kp[0]), kp[1], kp[2]])

        # connection hole keypoints
        for kp in self.connection_hole_kps:
            self.pixel_connection_hole_kps.append([self.toPixel(kp[0]), kp[1], kp[2]]
            )

        # string keypoints
        for s in self.strings:
            self.pixel_string_kps.append([self.toPixel(s.start_point), self.toPixel(s.end_point),
                s.type, s.width, s.id
            ])

    def calculateSequence(self):
        try:
            tb = TreeBasedOrigamiGraph(self.kps, self.lines)
            tb.calculateTreeBasedGraph()
            if self.edit_sequence_mode:
                self.chooseCreaseSequence(self.choose_crease_sequence_id, 1)
            self.updateMessage("Succeed to calculate sequence")
        except:
            self.updateMessage("Failed to calculate sequence")

    def calculateUnitCenterUsingContribution(self, unit_id):
        contribution = self.unit_center_contribute_coeff[unit_id]
        seq_point = self.units[unit_id].getSeqPoint()
        center = [0, 0]
        for i in range(len(contribution)):
            point = seq_point[i]
            center[X] += contribution[i] * point[X]
            center[Y] += contribution[i] * point[Y]
        return center
    
    def cdfCurveFitting(self):
        if self.limitation["match_mode"] < 3:
            if len(self.x_list) == 0:
                self.updateMessage("No curve has been imported, please import a curve file first...")
                return
            if self.limitation["direction_enable"] and len(self.dir_list) == 0:
                self.updateMessage("No direction curve has been imported but direction match enabled, please import direction first...")
                return
        if self.enable_output_stl:
            self.updateMessage("A stl file is being outputed, please wait...")
            return
        if self.enable_cdf_curve_fitting:
            self.updateMessage("A cdf process is running, please wait...")
            return
        if self.enable_phys_data_collecting:
            self.updateMessage("A physical simulation is running, please wait...")
            return
        if self.enable_mcts:
            self.updateMessage("A MCTS Searching process is running, please wait...")
            return
        if self.enable_threading_design:
            self.updateMessage("A threading design simulation is running, please wait...")
            return
        
        self.enable_cdf_curve_fitting = True
        if self.limitation["match_mode"] < 3:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  =self.curve_name,
                pref_pack   =self.limitation,
                curve_x     =self.x_list,
                curve_y     =self.y_list,
                curve_z     =self.z_list,
                curve_dir   =self.dir_list if self.limitation["direction_enable"] else None
            )
        elif self.limitation["match_mode"] == 3:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  ="exoskeleton",
                pref_pack   =self.limitation,
                curve_x     =None,
                curve_y     =None,
                curve_z     =None,
                curve_dir   =None
            )
        elif self.limitation["match_mode"] == 4:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  ="zerodistance",
                pref_pack   =self.limitation,
                curve_x     =None,
                curve_y     =None,
                curve_z     =None,
                curve_dir   =None
            )
        self.cdf_curve_fitting_thread._emit.connect(self.drawProcess)
        self.cdf_curve_fitting_thread.start()

    def changeCoeff(self):
        self.lines[self.choose_crease_sequence_id].coeff = self.doubleSpinBox_coeff_value.value()

    def changeCopyTime(self):
        self.copy_time = self.spinbox_copy_time.value()
        self.design_info['copy_time'] = self.copy_time
        self.recalculateContributions()
        self.enable_design = True

    def changeConLeftLength(self):
        """
        @ function: Change stretch length of left side of the origami
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230314
        @ spec: None
        """
        new_length = self.spinbox_con_left_length.value()
        bias_length = len(self.add_bias_flag)
        
        for kp in self.hole_kps:
            kp[0][X] -= self.con_left_length
            kp[0][X] += new_length

        for i in range(bias_length):
            if self.add_bias_flag[i]:
                src = self.origami_info[i][-1]
                if type(src) == ModuleLeanMiura:
                    self.storage[i][1].stretch_length -= self.con_left_length
                    self.storage[i][1].stretch_length += new_length
        self.con_left_length = self.spinbox_con_left_length.value()
        self.spinbox_connection_radius.setMaximum(min(self.con_right_length, self.con_left_length) / 4)
        self.enable_design = True
        
    def changeConnectionRadius(self):
        self.connection_radius = self.spinbox_connection_radius.value()
        self.enable_design = True

    def changeConRightLength(self):
        self.con_right_length = self.spinbox_con_right_length.value()
        self.spinbox_connection_radius.setMaximum(min(self.con_right_length, self.con_left_length) / 4)
        self.enable_design = True

    def changeCreaseWidth(self):
        self.unit_width = self.spinbox_crease_width.value()
        self.doubleSpinBox_bias_val.setMaximum(self.unit_width / 4.)
        self.paper_info['unit_width'] = self.unit_width
        self.enable_design = True

    def changeFlag(self):
        self.entry_flag = self.slider_flag.value()
        self.enable_design = True

    def changeFoldingPercent(self):
        val = self.horizontal_folding_slider.value()
        
        self.design_info["folding_percent"] = val / 1000.0
        self.label_folding_percent.setText(str(round(100.0 - self.design_info["folding_percent"] * 100.0, 1)) + "%")
        self.label_folding_percent.setGeometry(int(180 + 0.74 * val), 494, 51, 16)
        # self.enable_design = True

    def changeHard(self):
        previous = self.lines[self.choose_crease_sequence_id].hard
        if previous:
            self.lines[self.choose_crease_sequence_id].hard = False
            self.radioButton_hard_crease.setChecked(False)
            for i in range(len(self.units)):
                unit = self.units[i]
                lines = unit.getCrease()
                for j in range(len(lines)):
                    crease = lines[j]
                    if (distance(self.lines[self.choose_crease_sequence_id][START], crease[START]) < 1e-5 and distance(self.lines[self.choose_crease_sequence_id][END], crease[END]) < 1e-5) or \
                        (distance(self.lines[self.choose_crease_sequence_id][START], crease[END]) < 1e-5 and distance(self.lines[self.choose_crease_sequence_id][END], crease[START]) < 1e-5):
                            self.unit_bias_list[i][j] = None
        else:
            self.lines[self.choose_crease_sequence_id].hard = True
            self.radioButton_hard_crease.setChecked(True)
            for i in range(len(self.units)):
                unit = self.units[i]
                lines = unit.getCrease()
                for j in range(len(lines)):
                    crease = lines[j]
                    if (distance(self.lines[self.choose_crease_sequence_id][START], crease[START]) < 1e-5 and distance(self.lines[self.choose_crease_sequence_id][END], crease[END]) < 1e-5) or \
                        (distance(self.lines[self.choose_crease_sequence_id][START], crease[END]) < 1e-5 and distance(self.lines[self.choose_crease_sequence_id][END], crease[START]) < 1e-5):
                            self.unit_bias_list[i][j] = 1e-3
        # self.enable_design

    def changeHardAngle(self):
        self.lines[self.choose_crease_sequence_id].folding_angle_upper_bound = self.doubleSpinBox_hard_angle_value.value() * math.pi / 180.0
    
    def changeHardAngleDown(self):
        self.lines[self.choose_crease_sequence_id].folding_angle_lower_bound = self.doubleSpinBox_hard_angle_value_down.value() * math.pi / 180.0

    def changeHoleSize(self):
        self.hole_size = self.spinbox_hole_size.value()

    def changeHoleResolution(self):
        self.hole_resolution = self.spinbox_resolution.value()
    
    def changeKlLength(self):
        self.storage[self.choose_kl_id][DATA][self.choose_line_id][0] = self.doubleSpinBox_length.value()
        self.enable_design = True

    def changeKlSectorAngle(self):
        self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] = self.doubleSpinBox_sector_angle.value() / 180.0 * math.pi
        self.enable_design = True

    def changeLevel(self):
        self.lines[self.choose_crease_sequence_id].level = self.spinBox_level_value.value()

    def changeOperationStep(self):
        self.operation_amp = (self.verticalSlider_operation_step.value() + 1) * 0.05
        self.label_operation_step.setText(str(round(self.operation_amp, 2)) + " mm")

    def changePanelOffset(self):
        self.doubleSpinBox_bias_val.setMaximum(self.unit_width * 0.2)
        self.bias_val = self.doubleSpinBox_bias_val.value()
        self.enable_design = True
        
    def changeRecoverAngle(self):
        text = self.comboBox_recover_level.currentText()
        if text != '':
            ans = int(re.findall(r"-?\d+\.?\d*", text)[0])
            index = self.lines[self.choose_crease_sequence_id].recover_level.index(ans)
            self.lines[self.choose_crease_sequence_id].recover_angle[index] = self.doubleSpinBox_recover_angle.value() * math.pi / 180.0

    def changeThickPanelHeight(self):
        self.lines[self.choose_crease_sequence_id].thick_panel_height = self.doubleSpinBox_crease_height.value()
            
    def changeUnitBias(self):
        self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = self.doubleSpinBox_expert_mode.value()

    def changeUnitCenterContributeCoeff(self, not_dialog=False):
        if not_dialog:
            self.unit_center_contribute_coeff.clear()
            for unit in self.units:
                coeff = unit.getContribution()
                self.unit_center_contribute_coeff.append(coeff)
        else:
            crease_number = len(self.units[self.choose_unit_id].getCrease())
            seq_point = self.units[self.choose_unit_id].getSeqPoint()
            items = []
            coeffs = []
            # List all coeff choose
            coeff_equal = self.unit_center_contribute_coeff[self.choose_unit_id]
            items.append(f"Current | {coeff_equal}")
            coeffs.append(coeff_equal)
            for i in range(crease_number):
                coeff = [coeff_equal[j] - 0.02 * (self.operation_amp * 0.18 + 0.991) for j in range(len(coeff_equal))]
                coeff[i] += 0.02 * crease_number * (self.operation_amp * 0.18 + 0.991)
                coeffs.append(coeff)
                items.append(f"Closer to Point {i} {seq_point[i]} | {coeff}")
            for i in range(crease_number):
                coeff = [coeff_equal[j] - 0.02 * (self.operation_amp * 0.18 + 0.991) for j in range(len(coeff_equal))]
                coeff[i] += 0.01 * crease_number * (self.operation_amp * 0.18 + 0.991)
                coeff[(i + 1) % crease_number] += 0.01 * crease_number * (self.operation_amp * 0.18 + 0.991)
                coeffs.append(coeff)
                items.append(f"Closer to Crease {i} (P{i} ~ P{(i + 1) % crease_number}) | {coeff}")
            selected_item, ok = QInputDialog.getItem(self, "Select Contribution Item", "Select a Contribution combo:", items)
            if ok:
                index = items.index(selected_item)
                self.unit_center_contribute_coeff[self.choose_unit_id] = coeffs[index]
                self.updateState("Succeed to change the mode of keypoint contribution", self.state)
    
    def changeXMirror(self):
        val = self.checkBox_x_mirror.isChecked()
        if val:
            self.mirror_x = True
        else:
            self.mirror_x = False
    
    def changeXYRotation(self):
        val = self.horizontal_xy_rotation_slider.value()
        self.xy_rotation = val
        self.label_xy_rotation.setText(f"XY-rotation: {val - 180} deg")
    
    def changeYMirror(self):
        val = self.checkBox_y_mirror.isChecked()
        if val:
            self.mirror_y = True
        else:
            self.mirror_y = False

    def checkHoleKpIsValid(self):
        for ele in self.hole_kps:
            try:
                in_unit_id = ele[1]
                # find which origami is falls
                origami_id = -1
                all_unit = 0
                while all_unit <= in_unit_id:
                    all_unit += self.unit_number[origami_id + 1]
                    origami_id += 1
                origami = self.origami_info[origami_id]
                lower_x_bound = origami[0][X] + self.con_left_length
                upper_x_bound = origami[0][X] + origami[1] + self.con_left_length
                unit = self.units[in_unit_id].getSeqPoint()
                if self.enable_connection:
                    min_dis = pointInPolygon(ele[0], unit, return_min_distance=True, lower_x_bound=lower_x_bound, upper_x_bound=upper_x_bound)
                else:
                    min_dis = pointInPolygon(ele[0], unit, return_min_distance=True)
                if min_dis and min_dis > self.hole_size:
                    ele[2] = True
                else:
                    ele[2] = False
            except:
                ele[2] = False

    def chooseKl(self, id):
        self.choose_kl_id = id
        self.choose_line_id = 0
        self.label_current_kl_number.setText(str(id))
        self.label_current_line_number.setText("0")
        self.doubleSpinBox_length.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][0])
        self.doubleSpinBox_sector_angle.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] * 180.0 / math.pi)

    def chooseKlLine(self, id):
        self.choose_line_id = id
        self.label_current_line_number.setText(str(id))
        self.doubleSpinBox_length.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][0])
        self.doubleSpinBox_sector_angle.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] * 180.0 / math.pi)

    def chooseNextKl(self):
        for i in range(self.choose_kl_id + 1, len(self.storage)):
            if type(self.storage[i][DATA]) == KinematicLine:
                self.chooseKl(i)
                break

    def chooseNextKlLine(self):
        line_number = len(self.storage[self.choose_kl_id][DATA])
        choose_line_id = (self.choose_line_id + 1) % line_number
        self.chooseKlLine(choose_line_id)

    def choosePreviousKl(self):
        for i in range(self.choose_kl_id - 1, -1, -1):
            if type(self.storage[i][DATA]) == KinematicLine:
                self.chooseKl(i)
                break

    def choosePreviousKlLine(self):
        line_number = len(self.storage[self.choose_kl_id][DATA])
        choose_line_id = (self.choose_line_id - 1 + line_number) % line_number
        self.chooseKlLine(choose_line_id)

    def chooseUnit(self, id):
        self.choose_unit_id = id
        self.choose_crease_id = 0
        self.label_current_unit_number.setText(str(id))
        self.label_current_crease_number.setText("0")
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.radioButton_use_default.setChecked(True)
            self.radioButton_expert_mode.setChecked(False)
            self.doubleSpinBox_expert_mode.setVisible(False)
        else:
            self.radioButton_use_default.setChecked(False)
            self.radioButton_expert_mode.setChecked(True)
            self.doubleSpinBox_expert_mode.setVisible(True)
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])

    def chooseCreaseSequence(self, id, step):
        counter = 0
        while counter < len(self.lines):
            if self.lines[id].getType() == BORDER:
                counter += 1
                id = (id + step) % len(self.lines)
            else:
                break

        self.choose_crease_sequence_id = id
        self.label_crease_sequence_id.setText(str(id))
        self.spinBox_level_value.setValue(self.lines[id].level)
        self.doubleSpinBox_coeff_value.setValue(self.lines[id].coeff)
        # self.spinBox_recover_level_value.setValue(self.lines[id].recover_level)
        self.radioButton_hard_crease.setChecked(self.lines[id].hard)
        self.doubleSpinBox_hard_angle_value.setValue(self.lines[id].folding_angle_upper_bound * 180.0 / math.pi)
        self.doubleSpinBox_hard_angle_value_down.setValue(self.lines[id].folding_angle_lower_bound * 180.0 / math.pi)
        self.doubleSpinBox_crease_height.setValue(self.lines[id].thick_panel_height)
        self.comboBox_recover_level.clear()
        if len(self.lines[id].recover_level):
            self.doubleSpinBox_recover_angle.setEnabled(True)
            self.comboBox_recover_level.addItems([str(self.lines[id].recover_level[i]) for i in range(len(self.lines[id].recover_level))])
            text = self.comboBox_recover_level.currentText()
            ans = int(re.findall(r"-?\d+\.?\d*", text)[0])
            index = self.lines[id].recover_level.index(ans)
            self.doubleSpinBox_recover_angle.setValue(self.lines[id].recover_angle[index] * 180.0 / math.pi)
        else:
            self.doubleSpinBox_recover_angle.setValue(0.0)
            self.doubleSpinBox_recover_angle.setEnabled(False)
            

    def chooseCrease(self, id):
        self.choose_crease_id = id
        self.label_current_crease_number.setText(str(id))
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.radioButton_use_default.setChecked(True)
            self.radioButton_expert_mode.setChecked(False)
            self.doubleSpinBox_expert_mode.setVisible(False)
        else:
            self.radioButton_use_default.setChecked(False)
            self.radioButton_expert_mode.setChecked(True)
            self.doubleSpinBox_expert_mode.setVisible(True)
            self.doubleSpinBox_expert_mode.setMinimum(0)
            self.doubleSpinBox_expert_mode.setMaximum(self.unit_width / 6.0)
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])

    def chooseNextUnit(self):
        unit_number = len(self.units)
        choose_unit_id = (self.choose_unit_id + 1) % unit_number
        self.chooseUnit(choose_unit_id)

    def chooseNextCrease(self):
        crease_number = len(self.units[self.choose_unit_id].getCrease())
        choose_crease_id = (self.choose_crease_id + 1) % crease_number
        self.chooseCrease(choose_crease_id)
    
    def chooseNextCreaseSequence(self):
        id = (self.choose_crease_sequence_id + 1) % len(self.lines)
        self.chooseCreaseSequence(id, 1)

    def choosePreviousUnit(self):
        unit_number = len(self.units)
        choose_unit_id = (self.choose_unit_id - 1 + unit_number) % unit_number
        self.chooseUnit(choose_unit_id)

    def choosePreviousCrease(self):
        crease_number = len(self.units[self.choose_unit_id].getCrease())
        choose_crease_id = (self.choose_crease_id - 1 + crease_number) % crease_number
        self.chooseCrease(choose_crease_id)

    def choosePreviousCreaseSequence(self):
        id = (self.choose_crease_sequence_id - 1 + len(self.lines)) % len(self.lines)
        self.chooseCreaseSequence(id, -1)

    def chooseRecoverLevel(self):
        text = self.comboBox_recover_level.currentText()
        if text != '':
            ans = int(re.findall(r"-?\d+\.?\d*", text)[0])
            index = self.lines[self.choose_crease_sequence_id].recover_level.index(ans)
            self.doubleSpinBox_recover_angle.setValue(self.lines[self.choose_crease_sequence_id].recover_angle[index] * 180.0 / math.pi)

    def closeEvent(self, event):
        self.stopThread()
        self.timer.stop()
        self.deleteLater()

    def compute_mean_and_stats(self, trajectories, smooth_mean_sigma=1):
        """
        Compute smoothed mean trajectory + perpendicular statistics.

        Short trajectories are padded to the maximum length by repeating their
        last point, so the mean and variance are computed over aligned indices.
        """
        trajectories = [np.asarray(t, dtype=float) for t in trajectories]
        max_len = max(len(t) for t in trajectories)
        padded = []
        for t in trajectories:
            if len(t) < max_len:
                last = t[-1:]
                pad = np.tile(last, (max_len - len(t), 1))
                t = np.vstack([t, pad])
            padded.append(t)
        stacked = np.stack(padded, axis=0)
        raw_mean = np.mean(stacked, axis=0)

        if smooth_mean_sigma > 0:
            mean_traj = np.column_stack([
                gaussian_filter1d(raw_mean[:, 0], sigma=smooth_mean_sigma),
                gaussian_filter1d(raw_mean[:, 1], sigma=smooth_mean_sigma),
            ])
        else:
            mean_traj = raw_mean

        tangents = np.gradient(mean_traj, axis=0)
        tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        n_points = mean_traj.shape[0]
        perp_devs = np.zeros((len(trajectories), n_points))
        for i, traj in enumerate(padded):
            perp_devs[i] = np.sum((traj - mean_traj) * normals, axis=1)

        perp_std = np.std(perp_devs, axis=0, ddof=1)
        perp_std_smoothed = gaussian_filter1d(perp_std, sigma=smooth_mean_sigma)
        avg_std = float(np.mean(perp_std))

        return mean_traj, perp_std_smoothed, avg_std, normals

    def compute_uniform_tube(self, mean_traj, avg_std, perp_std, normals):
        """
        Build upper/lower boundaries with constant half-width = avg_std.
        Produces a perfectly smooth tube (no local width variation).
        """
        upper = mean_traj + np.array([perp_std[i] * normals[i] for i in range(len(perp_std))])
        lower = mean_traj - np.array([perp_std[i] * normals[i] for i in range(len(perp_std))])
        return upper, lower
    # def dataConv(self):
    #     pass
    #     path, _ = QFileDialog.getOpenFileName(
    #             self, 
    #             "Choose a json file with kl specification", 
    #             "./", 
    #             "CSV files (*.csv);;All Files (*.*)"
    #         )
    #     data = pd.read_csv(path)
    #     column = data["true_reward"]
    #     size = column.size

    #     # paths, _ = QFileDialog.getOpenFileNames(
    #     #         self, 
    #     #         "Choose a json file with kl specification", 
    #     #         "./", 
    #     #         "Json files (*.json);;All Files (*.*)"
    #     #     )
    #     # for path in paths:
    #     #     with open(path, 'r', encoding="utf-8") as fw:
    #     #         input_json = json.load(fw)
    #     #     deal = 0
    #     #     try:
    #     #         deal = input_json["deal"]
    #     #     except:
    #     #         pass
    #     #     if 1:
    #     #         old_maximum_force = input_json["max_force"]
    #     #         new_maximum_force = [(-(ele / 500.)**3/12.+(ele / 500.)**2/2.)*500. for ele in old_maximum_force]
    #     #         input_json["max_force"] = new_maximum_force
    #     #         input_json["deal"] = 1
    #     #         with open(path, 'w', encoding="utf-8") as f:
    #     #             json.dump(input_json, f, indent=4)

    #     path, _ = QFileDialog.getOpenFileName(
    #             self, 
    #             "Choose a json file with kl specification", 
    #             "./", 
    #             "Json files (*.json);;All Files (*.*)"
    #         )

    #     with open(path, 'r', encoding="utf-8") as fw:
    #         input_json = json.load(fw)

    #     for i in range(0, size):
    #         reward = column[i]
    #         input_json["method"][i]["reward"] = reward
    #         input_json["simulated_number"] = i

    #     with open(path.split(".")[0] + "_true_reward.json", 'w', encoding="utf-8") as f:
    #         json.dump(input_json, f, indent=4)

    def defineAction(self):
        """
        @ function: define actions and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.actionNew_file.triggered.connect(self.newFile)
        self.actionImport.triggered.connect(self.importKL)
        self.actionPrint_P.triggered.connect(self.printOrigami)
        self.actionAs_Dxf.triggered.connect(self.exportAsDxf)
        self.actionAs_Stl.triggered.connect(self.exportAsStl)
        self.actionAll_As_Stl.triggered.connect(self.exportAllAsStl)
        self.actionAs_Split_Dxf.triggered.connect(self.exportAsSplitDxf)
        self.actionSettings.triggered.connect(self.setting)
        self.actionSave_result.triggered.connect(self.saveResult)
        self.actionOpen_file_O.triggered.connect(self.openFile)
        self.actionLeanMiura.triggered.connect(self.addLeanMiuraStorage)
        self.actionMiura.triggered.connect(self.addMiuraStorage)
        self.actionTransition_T.triggered.connect(self.showTG)
        self.actionView_Curve.triggered.connect(self.showCurve)
        self.actionCDF_Curve_Fitting_F.triggered.connect(self.cdfCurveFitting)
        self.actionStop_Thread_S.triggered.connect(self.stopThread)
        self.actionImport_dxf.triggered.connect(self.importDxf)
        self.actionAdd_Holes.triggered.connect(self.oneClickAddHoles)
        self.actionPhysical_Simulation_P.triggered.connect(self.physicalSimulation)
        self.actionImport_Directions_D.triggered.connect(self.showDirection)
        self.actionAdd_TSA_A_point.triggered.connect(self.addTsaAPoint)
        self.actionCollect_Physical_Data_C.triggered.connect(self.physicalDataCollecting)
        self.actionPlot_Physical_Data.triggered.connect(self.plotJson)
        self.actionPlot_Evolution_Data.triggered.connect(self.plotEvolutionJson)
        self.actionExplicit_Simulation_E.triggered.connect(self.physicalSimulationExplicit)
        self.actionExpert_Mode_E.triggered.connect(self.expertModeEnable)
        self.actionEdit_kl_E.triggered.connect(self.editKl)
        self.actionCalculate_Sequence.triggered.connect(self.calculateSequence)
        self.actionAs_Full_description_Data.triggered.connect(self.exportDescriptionData)
        self.actionEdit_Sequence_S.triggered.connect(self.editSequence)
        self.actionImport_string_path.triggered.connect(self.importStringPath)
        self.actionAdd_TSA_A_candidators.triggered.connect(self.addTSACandidators)
        self.actionShow_Index.triggered.connect(self.showIndex)
        self.actionDelete_TSA_A_Candidators.triggered.connect(self.deleteTSACandidators)
        self.actionPlot_Simulation_Data.triggered.connect(self.plotSimulationJson)
        self.actionFix_Panel_F.triggered.connect(self.fixPanel)
        self.actionRecalculate_Contributions.triggered.connect(self.recalculateContributions)
        self.actionImport_Trajectory_T.triggered.connect(self.showTrajectory)
        self.actionAppend_simulation_angles.triggered.connect(self.appendSimulationAngles)
        self.actionAs_Rotational_Origami.triggered.connect(self.exportRotationalDescriptionData)
        self.actionAs_Rotational_Origami_CW.triggered.connect(self.exportRotationalDescriptionData_Reverse)
        self.actionAppend_unit_transformation.triggered.connect(self.appendTransformationMatrix)

    def defineButton(self):
        """
        @ function: define buttons and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.button_design.clicked.connect(self.onDesign)
        self.button_threading_design.clicked.connect(self.onDesignThreadingMethod)
        self.button_reset_view.clicked.connect(self.resetView)
        self.radiobutton_A4.clicked.connect(self.showA4Square)
        self.radiobutton_none.clicked.connect(self.showNone)
        self.checkBox_add_hole_mode.clicked.connect(self.onAddHoleMode)
        self.checkBox_connection.clicked.connect(self.onAddConnection)
        self.checkBox_add_string_mode.clicked.connect(self.onAddStringMode)
        self.pushButton_next_unit.clicked.connect(self.chooseNextUnit)
        self.pushButton_next_crease.clicked.connect(self.chooseNextCrease)
        self.pushButton_previous_unit.clicked.connect(self.choosePreviousUnit)
        self.pushButton_previous_crease.clicked.connect(self.choosePreviousCrease)
        self.radioButton_use_default.clicked.connect(self.setBiasAsDefault)
        self.radioButton_expert_mode.clicked.connect(self.setBiasAsExpertModified)
        self.pushButton_next_line.clicked.connect(self.chooseNextKlLine)
        self.pushButton_previous_line.clicked.connect(self.choosePreviousKlLine)
        self.pushButton_previous_kl.clicked.connect(self.choosePreviousKl)
        self.pushButton_next_kl.clicked.connect(self.chooseNextKl)
        self.pushButton_previous_crease_sequence.clicked.connect(self.choosePreviousCreaseSequence)
        self.pushButton_next_crease_sequence.clicked.connect(self.chooseNextCreaseSequence)
        self.radioButton_hard_crease.clicked.connect(self.changeHard)
        self.pushButton_contribution.clicked.connect(self.changeUnitCenterContributeCoeff)
        self.pushButton_add_level.clicked.connect(self.addRecoverLevel)
        self.pushButton_delete_level.clicked.connect(self.deleteRecoverLevel)
        self.checkBox_x_mirror.clicked.connect(self.changeXMirror)
        self.checkBox_y_mirror.clicked.connect(self.changeYMirror)

    def defineSpinbox(self):
        """
        @ function: define spinboxes and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.spinbox_crease_width.valueChanged.connect(self.changeCreaseWidth)
        self.spinbox_copy_time.valueChanged.connect(self.changeCopyTime)
        self.doubleSpinBox_bias_val.valueChanged.connect(self.changePanelOffset)
        self.spinbox_hole_size.valueChanged.connect(self.changeHoleSize)
        self.spinbox_resolution.valueChanged.connect(self.changeHoleResolution)
        self.spinbox_connection_radius.valueChanged.connect(self.changeConnectionRadius)
        self.spinbox_con_left_length.valueChanged.connect(self.changeConLeftLength)
        self.spinbox_con_right_length.valueChanged.connect(self.changeConRightLength)
        self.horizontal_folding_slider.valueChanged.connect(self.changeFoldingPercent)
        self.verticalSlider_operation_step.valueChanged.connect(self.changeOperationStep)
        self.slider_flag.valueChanged.connect(self.changeFlag)
        self.doubleSpinBox_expert_mode.valueChanged.connect(self.changeUnitBias)
        self.doubleSpinBox_length.valueChanged.connect(self.changeKlLength)
        self.doubleSpinBox_sector_angle.valueChanged.connect(self.changeKlSectorAngle)
        self.spinBox_level_value.valueChanged.connect(self.changeLevel)
        self.doubleSpinBox_coeff_value.valueChanged.connect(self.changeCoeff)
        # self.spinBox_recover_level_value.valueChanged.connect(self.changeRecoverLevel)
        self.doubleSpinBox_hard_angle_value.valueChanged.connect(self.changeHardAngle)
        self.doubleSpinBox_hard_angle_value_down.valueChanged.connect(self.changeHardAngleDown)
        self.doubleSpinBox_recover_angle.valueChanged.connect(self.changeRecoverAngle)
        self.comboBox_recover_level.currentIndexChanged.connect(self.chooseRecoverLevel)
        self.doubleSpinBox_crease_height.valueChanged.connect(self.changeThickPanelHeight)
        self.horizontal_xy_rotation_slider.valueChanged.connect(self.changeXYRotation)

    def deleteRecoverLevel(self):
        pass

    def deleteTSACandidators(self):
        items = []
        # List all P_CANDIDATE
        for i in range(len(self.P_candidate)):
            items.append(f"Candidator {str(i)}, Axis: [{self.P_candidate[i][X]}, {self.P_candidate[i][Y]}, {self.P_candidate[i][Z]}], connecting to {self.P_candidate_connection_index[i]}")
        selected_item, ok = QInputDialog.getItem(self, "Select Miura Item", "Select a Miura combo:", items)
        # If press ok
        if ok:
            index = items.index(selected_item)
            del(self.P_candidate[index])
            del(self.P_candidate_connection_index[index])
            self.updateState(f"Succeed to delete Candidator {index}", self.state)

    def design(self):
        """
        @ function: Design origami crease
        @ version: 0.11
        @ developer: py
        @ progress: on road
        @ spec: Cancel try/except
        """
        self.kps.clear()
        self.lines.clear()
        self.additional_lines.clear()
        self.crease_lines.clear()
        self.connection_hole_kps.clear()
        self.units.clear()
        self.unit_bias_list.clear()

        body_line = []
        # whether to get additional line
        self.additional_line_maker.clearValidCrease()
        if self.add_hole_mode:
            self.additional_line_maker.clear()
            self.additional_line_maker.clearCrease()   
            self.additional_line_maker.setBias(self.bias_val)
            self.additional_line_maker.enable_difference = self.pref_pack["additional_line_option"]
            self.additional_line_maker.border_nobias = not self.pref_pack['stl_asymmetry']
        # Clear origami info 
        self.origami_info.clear()
        self.origami_number = len(self.storage)
        self.unit_number.clear()
        try:
            # for all storage
            for i in range(self.origami_number):
                r = R(self.rotation[i])
                src = self.storage[i][1]
                # ----- Set the transition start point for designer ----- #
                # ----- START ----- #
                if(self.storage[i][2]):
                    self.designer.setTransitionStartPoint([
                        self.storage[i][0][0], 
                        self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                    ])
                else:
                    self.designer.setTransitionStartPoint(self.storage[i][0])
                # ----- END ----- #

                # If LeanMiura is using global data
                # ----- START ----- #
                if type(src) == ModuleLeanMiura:
                    if src.enable_global_modify:
                        src.unit_width = self.unit_width
                        src.copy_time = self.copy_time
                        src.entry_flag = self.entry_flag if self.copy_time % 2 else not self.entry_flag
                        src.connection_flag = self.enable_connection
                        src.con_left_length = self.con_left_length
                        src.con_right_length = self.con_right_length
                        src.connection_hole_size = self.connection_radius
                # ----- END ----- #

                # ----- Set the source for designer ----- #
                # ----- START ----- #
                self.designer.setSource(src)
                # ----- END ----- #
                
                # Clear data of additional_line_maker to generate new additional line
                # ----- START ----- #
                self.additional_line_maker.clear() 
                # ----- END ----- #

                # Set connection at left and right, and change hole position
                self.designer.clearAdditionalLine()
                # ----- START ----- #
                if type(src) == KinematicLine:
                    if self.enable_connection:
                        self.designer.insertLineSource([self.con_left_length, self.designer.src.lines[0][1]], 0)
                        self.designer.insertLineSource([self.con_right_length, self.designer.src.lines[-1][1]], self.designer.getKLNumber())
                        if not self.add_bias_flag[i]:
                            for kp in self.hole_kps:
                                kp[0][X] += self.con_left_length
                            self.add_bias_flag[i] = True
                    else:
                        if self.add_bias_flag[i]:
                            for kp in self.hole_kps:
                                kp[0][X] -= self.con_left_length
                            self.add_bias_flag[i] = False
                elif type(src) == ModuleLeanMiura:
                    if self.enable_connection:
                        if not self.add_bias_flag[i]:
                            if self.storage[i][1].half_flag == RIGHT_HALF:
                                # We have modify the stretch line of LeanMiura
                                self.storage[i][1].stretch_length += src.con_left_length
                                self.storage[i][1].modify_stretch_flag = True
                                self.add_bias_flag[i] = True
                    else:
                        if self.add_bias_flag[i]:
                            if self.storage[i][1].half_flag == RIGHT_HALF:
                                self.storage[i][1].stretch_length -= src.con_left_length
                                self.storage[i][1].modify_stretch_flag = False
                                self.add_bias_flag[i] = False
                # ----- END ----- #

                # Set the data of designer to make new origami
                # ----- START ----- #
                self.designer.setPaperInfo(self.paper_info)
                self.designer.setDesignInfo(self.design_info)
                self.designer.setEntryFlag(self.entry_flag)
                # ----- END ----- #
                
                # <<<<< Before design <<<<< #

                # -------- Design -------- #
                # ----- START ----- #
                self.designer.parseData()
                # ----- END ----- #
                # -------- Design -------- #

                # >>>>> After design >>>>> #

                # Get design data
                # ----- START ----- #
                self.origami_length, self.origami_width = self.designer.getPaperData() # not include connection
                # Set tsp, length, width and type
                self.origami_info.append([self.designer.getTransitionStartPoint(), self.origami_length, self.origami_width, self.storage[i][1]])
                data = self.designer.getDesignData()
                # ----- END ----- #

                # Get kp, line, unit
                # ----- START ----- #
                # LeanMiura Module, only one result at data[0]
                if type(src) == ModuleLeanMiura: 
                    # Pull out the modify stretch length
                    if src.modify_stretch_flag:
                        self.origami_info[i][1] -= src.con_left_length
                    kp = data[0].getKeyPoint()
                    line = data[0].getLine()
                    new_kp_list = []
                    new_line_list = []
                    if abs(self.rotation[i]) > 1e-5:
                        tsp = np.array(self.storage[i])
                        for ele in kp:
                            new_kp = (r @ np.array([kp[X] - self.storage[i][X], kp[Y] - self.storage[i][Y]]) + tsp).tolist()
                            new_kp_list.append(new_kp)
                        self.kps += new_kp_list
                        for ele in line:
                            new_start = (r @ np.array(ele[START]) + tsp).tolist()
                            new_end = (r @ np.array(ele[END]) + tsp).tolist()
                            new_line = Crease(new_start, new_end, ele.getType())
                            new_line_list.append(new_line)
                        self.lines += new_line_list
                    else:
                        self.addKp(kp)
                        self.addCrease(line)
                    u = data[0].getUnits(connection=True)
                    self.unit_number.append(len(u))
                    if self.add_hole_mode:
                        for ele in u:
                            self.additional_line_maker.addPackedOrigamiUnit(ele)
                    else:
                        for ele in u:
                            self.units.append(ele)
                # Miura Module
                elif type(src) == KinematicLine:
                    data_length = len(data)
                    row_data_length = int(data_length / self.copy_time)
                    self.unit_number.append((row_data_length - 1) * self.copy_time * 2)
                    for j in range(self.copy_time):
                        for k in range(0, row_data_length):
                            unit_id = k + j * row_data_length
                            kp = data[unit_id].getKeypoint()
                            # self.addKp(kp)
                            self.addKp(kp)
                            line = data[unit_id].getLine()
                            record_body_line = data[unit_id].getLineConnectToBody()
                            # self.addCrease(line)
                            self.addCrease(line)
                            body_line += record_body_line
                            if self.add_hole_mode:
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
                                    self.additional_line_maker.addPackedOrigamiUnit(u1)
                                    self.additional_line_maker.addPackedOrigamiUnit(u2)
                            else:
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
                                    self.units.append(u1)
                                    self.units.append(u2)

                elif type(src) == DxfDirectGrabber:
                    kp = data[0].getKeyPoint()
                    line = data[0].getLine()
                    for ele in line:
                        if ele.getType() == FACET:
                            ele.crease_type = VALLEY
                            ele.hard = True
  
                    u = data[0].getUnits()

                    self.addKp(kp)
                    self.addCrease(line)
                    self.unit_number.append(len(u))
                    if self.add_hole_mode:
                        for ele in u:
                            self.additional_line_maker.addPackedOrigamiUnit(ele)
                    else:
                        for ele in u:
                            self.units.append(ele)

                self.additional_line_maker.addValidCreases(self.lines)
                if self.add_hole_mode:
                    self.units += self.additional_line_maker.unit_list
                if self.enable_read_list_from_backup:
                    self.unit_bias_list = deepcopy(self.backup_unit_bias_list)
                    for i in range(len(self.lines)):
                        info = self.backup_crease_type_list[i]
                        self.lines[i].hard = info["hard"]
                        self.lines[i].level = info["level"]
                        self.lines[i].coeff = info["coeff"]
                        self.lines[i].folding_angle_upper_bound = info["hard_angle"]
                        self.lines[i].folding_angle_lower_bound = info["hard_angle_down"]
                        self.lines[i].recover_level = info["recover_level"]
                        self.lines[i].recover_angle = info["recover_angle"]
                        try:
                            self.lines[i].thick_panel_height = info["thick_panel_height"]
                        except:
                            pass
                    self.enable_read_list_from_backup = False
                else:
                    self.unit_bias_list = [[None for j in range(len(unit.getCrease()))] for unit in self.units]
                if self.expert_mode:
                    self.chooseUnit(self.choose_unit_id)
                    self.chooseCrease(self.choose_crease_id)
                    
                if self.edit_sequence_mode:
                    self.chooseCreaseSequence(self.choose_crease_sequence_id, 1)
                if self.enable_read_contribution_from_backup:
                    self.unit_center_contribute_coeff = deepcopy(self.backup_unit_center_contribute_coeff)
                    self.enable_read_contribution_from_backup = False
                elif len(self.unit_center_contribute_coeff) != len(self.units):
                    self.changeUnitCenterContributeCoeff(not_dialog=True)
                # UnitPackParser
                
                
                # Collect units done
                # -----END----- #
                  
                # Get additional line based on units
                # -----START----- #           
                if self.add_hole_mode:
                    if self.enable_connection:
                        self.additional_lines += self.additional_line_maker.getAdditionalLineForAllUnit(
                            upper_x_bound=self.origami_info[i][1] + self.con_left_length + self.storage[i][0][0],
                            lower_x_bound=self.con_left_length + self.storage[i][0][0]
                        )
                    else:
                        self.additional_lines += self.additional_line_maker.getAdditionalLineForAllUnit()
                    self.crease_lines += self.additional_line_maker.calculateDrawingForAllCrease()
                # -----END----- #
                
                # Add connection hole
                # -----START----- #
                if self.enable_connection:
                    self.addConnectionHole()
                # -----END----- #

                # Set crease stiffness
                for index in self.hard_crease_index:
                    origin_index = self.additional_line_maker.valid_crease_list[index].origin_index
                    self.lines[origin_index].setHard(True)
                
                for ii in range(len(self.lines)):
                    if self.lines[ii].hard:
                        for i in range(len(self.units)):
                            unit = self.units[i]
                            lines = unit.getCrease()
                            for j in range(len(lines)):
                                crease = lines[j]
                                if (distance(self.lines[ii][START], crease[START]) < 1e-5 and distance(self.lines[ii][END], crease[END]) < 1e-5) or \
                                    (distance(self.lines[ii][START], crease[END]) < 1e-5 and distance(self.lines[ii][END], crease[START]) < 1e-5):
                                        self.unit_bias_list[i][j] = 1e-3
                
            # paint result
            self.drawCreasePattern()
            if self.state == self.DESIGN_FINISH or self.state == self.DESIGN_ERROR:
                self.updateState("Design finished with success", self.DESIGN_FINISH)

        except Exception as e:
            self.updateState("Design failed, Adjusting...", self.DESIGN_ERROR, "ERROR")
            if self.bias_val > 0.1:
                self.bias_val -= 0.05
            else:
                self.spinbox_crease_width.setValue(self.unit_width - 1.)
                self.bias_val = self.unit_width * 0.2
    
    def dictToStorage(self, input_json: json, json_type):
        """
        @ function: Turn input json file to application storage
        @ version: 0.11
        @ developer: py
        @ progress: on road
        @ spec: add packed-data parser
        """
        
        if json_type == self.KL_JSON:
            self.storage.clear()
            self.add_bias_flag.clear()
            self.show_index = False
            self.actionShow_Index.setText("Show Index")
            self.unit_center_contribute_coeff = []

            self.file_type = "KL"
            origin_list = input_json['origin']
            try:
                mode = input_json['mode']
            except:
                mode = 'rad'
            add_width_flag_list = input_json['add_width']
            origin_list_length = len(origin_list)
            for i in range(origin_list_length):
                origin = origin_list[i]
                add_width_flag = add_width_flag_list[i]
                kl = KinematicLine()
                for j in range(len(input_json['kl'][i])):
                    element = input_json['kl'][i][j]
                    if mode == 'deg':
                        element[1] *= math.pi / 180.0
                    kl.append(element)
                self.storage.append([origin, kl, add_width_flag])
                self.rotation.append(0.0)
                self.add_bias_flag.append(False)
        
        elif json_type == self.PACKED_DATA:
            # We clear all storage to generate new one
            self.storage.clear()
            self.add_bias_flag.clear()
            self.show_index = False
            self.actionShow_Index.setText("Show Index")
            self.unit_center_contribute_coeff = []
            # Get main data, hole data and settings
            data = input_json['origami']
            util = input_json['util']
            setting = input_json['setting']
            #1 Setup all the parameters in setting
            self.setGlobalParameter(
                unit_width          =setting["unit_width"],
                copy_time           =setting["copy_time"],
                entry_flag          =setting["entry_flag"],
                add_hole_mode       =setting["add_hole_mode"],
                hole_size           =setting["hole_size"],
                hole_resolution     =setting["hole_resolution"],
                enable_connection   =setting["enable_connection"],
                con_left_length     =setting["con_left_length"],
                con_right_length    =setting["con_right_length"],
                con_radius          =setting["con_radius"],
                bias_val            =setting["bias_val"]
            )
            #2 Setup all storage data in data
            for ele in data:
                self.file_type = ele["type"]
                origin = ele["tsp"]
                add_width_flag = ele["add_width"]
                if self.file_type == "kl":
                    kl = KinematicLine()
                    for j in range(len(ele['data'])):
                        element = ele['data'][j]
                        kl.append(element)
                    self.storage.append([origin, kl, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)
                elif self.file_type == "leanMiura":
                    lean_miura_storage = ModuleLeanMiura()
                    lean_miura_storage.initialize(
                        unit_width          =self.unit_width,
                        copy_time           =self.copy_time,
                        entry_flag          =self.entry_flag,
                        stretch_length      =ele["data"]["stretch_length"],
                        connection_flag     =self.enable_connection,
                        con_left_length     =self.con_left_length,
                        con_right_length    =self.con_right_length,
                        con_radius          =self.connection_radius,
                        half_flag           =ele["data"]["half_type"],
                        tsp                 =origin,
                        enabled             =True
                    )
                    self.storage.append([origin, lean_miura_storage, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)
                elif self.file_type == "dxf":
                    dxf_grabber = DxfDirectGrabber()
                    dxf_grabber.kps = ele["data"]["kps"]
                    dxf_grabber.lines = ele["data"]["lines"]
                    dxf_grabber.lines_type = ele["data"]["lines_type"]
                    try:
                        dxf_grabber.units = ele["data"]["units"]
                    except KeyError:
                        pass
                    self.storage.append([origin, dxf_grabber, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)

            # Start design
            self.design()
            # End design

            #3 Add holes to origami in util
            self.hole_kps.clear()
            for ele in util["hole_axis"]:
                self.addHoleToUnitUsingRealAxis(ele[0][X], ele[0][Y])

            #4 Add strings
            try:
                self.exist_string_start = False
                self.string_start_point = []
                self.string_type = BOTTOM

                self.strings = []  
                self.a_string = [] # one string
                self.string_total_information = [] # list of list

                self.P_candidate = input_json['P_candidators']["points"]
                self.P_candidate_connection_index = input_json['P_candidators']["connections"]

                string_type_list = input_json['strings']['type']
                string_id_list = input_json['strings']['id']
                string_reverse_list = input_json['strings']['reverse']
                for i in range(len(string_type_list)):
                    self.startAddString()
                    if string_reverse_list[i][0] == -1:
                        self.string_type = BOTTOM
                    else:
                        self.string_type = TOP
                    for j in range(len(string_type_list[i])):
                        if string_type_list[i][j] == 'A':
                            index = string_id_list[i][j]
                            if index >= len(self.P_candidate):
                                self.addStringPoint(0, 0, index)
                            else:
                                self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
                        else:
                            # unit_axis = self.units[string_id_list[i][j]].getCenter()
                            unit_axis = self.calculateUnitCenterUsingContribution(string_id_list[i][j])
                            self.addStringPoint(unit_axis[X], unit_axis[Y], string_id_list[i][j], 'B', string_reverse_list[i][j])
                    self.endAddString()
            except:
                pass

            # 5 other information
            try:
                self.backup_unit_bias_list = input_json['unit_bias_list']
                self.backup_crease_type_list = input_json['line_features']
                self.enable_read_list_from_backup = True
            except:
                self.enable_read_list_from_backup = False
            try:
                self.backup_unit_center_contribute_coeff = input_json['contributions']
                self.enable_read_contribution_from_backup = True
            except:
                self.enable_read_contribution_from_backup = False
            try:
                self.fixed_panel = input_json["fix"][0]
            except:
                self.fixed_panel = -1
            try:
                self.crease_angle = input_json["crease_angle"]
            except:
                self.crease_angle = []
            try:
                self.crease_info = input_json["crease_info"]
            except:
                self.crease_info = []
            self.updateState("Succeeded to load json file", self.IMPORT_SUCCESS)
            self.updateMessage("Previewing, click the \"design\" button to continue...")

        elif json_type == self.THREADING_METHOD:
            #4 Add strings
            # Parameters of string start&end
            try:
                methods = input_json["method"]
                items = []
                # try:
                #     control_mode = input_json["control_mode"]
                # except:
                #     control_mode = 0
                
                val_speed = 1.0
                val_force = 0.0
                val, ok = QInputDialog.getDouble(self, "Speed coefficient", "How much you care about the folding speed (default: 1.0):", min=0.0, max=1.0, decimals=2, value=1.0)
                if ok:
                    val_speed = val
                    val_force = 1. - val_speed
                
                try:
                    sorted_methods = sorted(methods, key=(lambda x: val_speed * x['reward_speed'] + val_force * x['reward_force']), reverse=True)
                except:
                    sorted_methods = methods

                for i in range(len(sorted_methods)):
                    # try:
                    #     success = methods[i]["success"]
                    #     fail = methods[i]["fail"]
                    #     soft_success = methods[i]["soft_success"]
                    #     soft_fail = methods[i]["soft_fail"]
                    #     extract = True
                    # except:
                    #     extract = False
                    # try:
                    #     reward = round(methods[i]["reward_without_actuator"], 3)
                    #     score = round(methods[i]["score"], 3)
                    #     if control_mode == 0:
                    #         error = round((1 - methods[i]['rf']) * 180.0, 3)
                    #         maximum_error = round((1 - methods[i]['rm']) * 180.0, 3)
                    #         time = round(methods[i]['rf'] / (methods[i]['rs'] + 1e-6), 3)
                    #         actuator = methods[i]['ra']
                    #     reward_text = True
                    # except:
                    #     reward = round(methods[i]["score"], 3)
                    #     score = reward
                    #     reward_text = False
                    # extract_text = f"Success Rate: Strict {success}/{success + fail}, Soft {soft_success}/{soft_success + soft_fail}" if extract else ""
                    # if reward_text:
                    #     if control_mode == 0:
                    #         string = f"#{i + 1}, F2DC: {score}, {'Reward' if reward_text else ''}: {reward if reward_text else -1.0}, {extract_text}, Error: {error}, Maximum Error: {maximum_error}, Time: {time}, A: {actuator}"
                    #     else:
                    #         string = f"#{i + 1}, F2DC: {score}, {'Reward' if reward_text else ''}: {reward if reward_text else -1.0}, {extract_text}"
                    # else:
                    #     string = f"#{i + 1}, F2DC: {score}, {extract_text}"
                    try:
                        original_reward = round(sorted_methods[i]["score"], 5)
                        reward_speed = round(sorted_methods[i]["reward_speed"], 5)
                        reward_force = round(sorted_methods[i]["reward_force"], 5)
                        recal_reward = val_speed * reward_speed + val_force * reward_force
                        string = f"#{i + 1}, train reward: {original_reward}, re-cal reward: {recal_reward}"
                    except:
                        original_reward = round(sorted_methods[i]["score"], 5)
                        string = f"#{i + 1}, train reward: {original_reward}, re-cal reward: {original_reward}"
                    items.append(string)
                selected_item, ok = QInputDialog.getItem(self, "Select String-routing Paths", "Select a String-routing Path:", items)
                if ok:
                    index = items.index(selected_item)
                    input_json = sorted_methods[index]["method"]
                else:
                    self.updateState("Cancelling string path selection", self.state)
                    return
            except:
                pass

            self.exist_string_start = False
            self.string_start_point = []
            self.string_type = BOTTOM

            self.strings = []  
            self.a_string = [] # one string
            self.string_total_information = [] # list of list

            try:
                self.P_candidate = input_json['P_candidators']["points"]
                self.P_candidate_connection_index = input_json['P_candidators']["connections"]
            except:
                pass

            string_type_list = input_json['type']
            string_id_list = input_json['id']
            string_reverse_list = input_json['reverse']
            for i in range(len(string_type_list)):
                self.startAddString()
                if string_reverse_list[i][0] == -1:
                    self.string_type = BOTTOM
                else:
                    self.string_type = TOP
                for j in range(len(string_type_list[i])):
                    if string_type_list[i][j] == 'A':
                        index = string_id_list[i][j]
                        if index >= len(self.P_candidate):
                            self.addStringPoint(0, 0, index)
                        else:
                            self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
                    else:
                        # unit_axis = self.units[string_id_list[i][j]].getCenter()
                        unit_axis = self.calculateUnitCenterUsingContribution(string_id_list[i][j])
                        self.addStringPoint(unit_axis[X], unit_axis[Y], string_id_list[i][j], 'B', string_reverse_list[i][j])
                self.endAddString()

    def drawA4Pixmap(self):
        """
        @ function: Draw origami crease on A4
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ spec: update color and drawing method
        """
        line_weight = self.pref_pack["line_weight"]
        self.A4_pixmap = QPixmap(round(self.A4_length), round(self.A4_width))
        bias_x = round(self.half_pixmap_length - self.A4_half_length)
        bias_y = round(self.half_pixmap_width - self.A4_half_width)
        A4_pixel_kps = []
        A4_pixel_lines = []
        self.A4_pixmap.fill(QColor(255, 255, 255))
        painter = QPainter(self.A4_pixmap)
        for kp in self.pixel_kps:
            A4_pixel_kps.append([kp[0] - bias_x, kp[1] - bias_y])
        for line in self.pixel_lines:
            A4_pixel_lines.append(Crease(
                [(round(line[0][0] - bias_x)), (round(line[0][1] - bias_y))], 
                [(round(line[1][0] - bias_x)), (round(line[1][1] - bias_y))], 
                line.getType()
            ))
        for line in A4_pixel_lines:
            type_crease = line.getType()
            if type_crease == MOUNTAIN:
                painter.setPen(QPen(QColor(255, 0, 0), line_weight, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            elif type_crease == VALLEY:
                painter.setPen(QPen(QColor(0, 0, 255), line_weight, Qt.DashLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            elif type_crease == CUTTING:
                painter.setPen(QPen(QColor(0, 205, 102), line_weight, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            else:
                painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
        painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
        if self.pref_pack["show_keypoint"]:
            for kp in A4_pixel_kps:
                painter.drawRect(kp[0] - 1, kp[1] - 1, 2, 2)
        painter.end()

    def drawCreasePattern(self, pixmap=None, fontsize=12):
        """
        @ function: Draw origami crease
        @ version: 0.1111
        @ developer: py
        @ progress: waiting
        @ date: 20230228
        @ spec: update color and drawing method
                add hole showing
        """
        if pixmap == None:
            pixmap = self.pixmap
        line_weight = self.pref_pack["line_weight"]
        theme = self.pref_pack["theme"]

        self.painter.begin(pixmap)
        self.axisConverter()

        if self.expert_mode:
            unit = self.units[self.choose_unit_id]
            line = unit.getCrease()[self.choose_crease_id]
            highlight_crease = Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            )
            highlight_unit = [Crease(
                self.toPixel(ele[START]), self.toPixel(ele[END]), 
                ele.getType(), hard=ele.hard
            ) for ele in unit.getCrease()]
        
        if self.edit_sequence_mode:
            line = self.lines[self.choose_crease_sequence_id]
            highlight_crease_sequence = Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            )

        if theme == 0:
            pixmap.fill(QColor(255, 255, 255))
        else:
            pixmap.fill(QColor(0, 0, 0))
        if self.show_square == "A4":
            self.painter.fillRect(
                round(self.half_pixmap_length - self.A4_half_length), 
                round(self.half_pixmap_width - self.A4_half_width), 
                round(self.A4_length), 
                round(self.A4_width),
                QColor(135, 206, 250)
            )
        #draw lines
        for line in self.pixel_lines:
            type_crease = line.getType()
            if line.hard:
                self.painter.setPen(QPen(QColor(128, 128, 128), 1, Qt.SolidLine))
                self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
            else:
                if type_crease == MOUNTAIN:
                    self.painter.setPen(QPen(QColor(192, 0, 0), int(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                elif type_crease == VALLEY:
                    self.painter.setPen(QPen(QColor(0, 0, 192), int(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.DashLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                elif type_crease == CUTTING:
                    self.painter.setPen(QPen(QColor(0, 205, 102), int(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                else:
                    if theme == 0:
                        self.painter.setPen(QPen(QColor(0, 0, 0), int((line_weight + 1) * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                        self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                    else:
                        self.painter.setPen(QPen(QColor(255, 255, 255), int((line_weight + 1) * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                        self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
        # draw kps
        if self.pref_pack["show_keypoint"]:
            width = int(2 * (line_weight + 1) * self.current_pixel_scale) 
            if theme == 0:
                self.painter.setPen(QPen(QColor(0, 0, 0), width, Qt.SolidLine))
                for kp in self.pixel_kps:
                    self.painter.drawPoint(kp[X], kp[Y])
            else:
                self.painter.setPen(QPen(QColor(255, 255, 255), width, Qt.SolidLine))
                for kp in self.pixel_kps:
                    self.painter.drawPoint(kp[X], kp[Y])
        # draw additional lines
        if self.pref_pack["show_additional_lines"]:
            self.painter.setPen(QPen(QColor(255, 228, 181), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.DashDotDotLine))    
            for line in self.pixel_additional_lines:
                self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
        # draw holes
        for kp in self.pixel_hole_kps:
            if kp[2]:
                self.painter.setPen(QPen(QColor(60, 179, 113), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.hole_size))
            else:
                self.painter.setPen(QPen(QColor(192, 192, 192), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.DashLine))
                self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.hole_size))
        # draw connection border
        if self.enable_connection:
            self.painter.setPen(QPen(QColor(155, 35, 155), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.DashLine))
            for ele in self.origami_info:
                pixel_border_left = (self.con_left_length + ele[0][X]) * self.current_pixel_scale + self.pixel_bias[X]
                pixel_border_right = (ele[1] + self.con_left_length + ele[0][X]) * self.current_pixel_scale + self.pixel_bias[X]
                pixel_border_up = (ele[2] + ele[0][Y]) * self.current_pixel_scale + self.pixel_bias[Y]
                pixel_border_down = ele[0][Y] * self.current_pixel_scale + self.pixel_bias[Y]
                self.painter.drawLine(
                    pixel_border_left, 
                    pixel_border_down,
                    pixel_border_left,
                    pixel_border_up
                )
                self.painter.drawLine(
                    pixel_border_right, 
                    pixel_border_down,
                    pixel_border_right,
                    pixel_border_up
                )
            # draw connection hole
            for kp in self.pixel_connection_hole_kps:
                if kp[2]:
                    self.painter.setPen(QPen(QColor(36, 203, 105), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
                    self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.connection_radius))
                else:
                    self.painter.setPen(QPen(QColor(192, 192, 192), round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1), Qt.DashLine))
                    self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.connection_radius))

        # draw strings
        color = QColor(190, 120, 219)
        for ele in self.pixel_string_kps:
            if ele[4] != self.choose_string_id:
                if ele[2] == BOTTOM:
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.DashLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                elif ele[2] == TOP:
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.SolidLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                else:
                    pixel_width = ele[3] * 0.5 * self.current_pixel_scale
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.SolidLine))
                    self.painter.fillRect(int(ele[START][X] - pixel_width), int(ele[START][Y] - pixel_width), int(pixel_width * 2), int(pixel_width * 2), color)
        
        color = QColor(86, 10, 180)
        for ele in self.pixel_string_kps:
            if ele[4] == self.choose_string_id:
                if ele[2] == BOTTOM:
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.DashLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                elif ele[2] == TOP:
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.SolidLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                else:
                    pixel_width = ele[3] * 0.5 * self.current_pixel_scale
                    self.painter.setPen(QPen(color, round((line_weight * 0.5 + 0.5) * self.current_pixel_scale * 0.5 + 1) * 2, Qt.SolidLine))
                    self.painter.fillRect(int(ele[START][X] - pixel_width), int(ele[START][Y] - pixel_width), int(pixel_width * 2), int(pixel_width * 2), color)

        # draw highlight crease and unit
        if self.expert_mode:
            self.painter.setPen(QPen(QColor(36, 203, 105), round(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
            for crease in highlight_unit:
                self.painter.drawLine(crease[START][X], crease[START][Y], crease[END][X], crease[END][Y])
            self.painter.setPen(QPen(QColor(255, 223, 0), round(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
            self.painter.drawLine(highlight_crease[START][X], highlight_crease[START][Y], highlight_crease[END][X], highlight_crease[END][Y])
        
        if self.edit_sequence_mode:
            self.painter.setPen(QPen(QColor(255, 223, 0), round(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))
            self.painter.drawLine(highlight_crease_sequence[START][X], highlight_crease_sequence[START][Y], highlight_crease_sequence[END][X], highlight_crease_sequence[END][Y])
        
        self.painter.setPen(QPen(QColor(0, 0, 0), round(line_weight * self.current_pixel_scale * 0.5 + 1), Qt.SolidLine))

        for i in range(len(self.P_candidate)):
            pixel_center = self.toPixel(self.P_candidate[i])
            center = QPoint(pixel_center[X], pixel_center[Y])
            self.painter.drawPolygon(self.generatePolygonByCenter(pixel_center, self.connection_radius))
            # self.painter.drawRect(pixel_center[X] - fontsize*0.25, pixel_center[Y] - fontsize*0.25, fontsize*0.5, fontsize*0.5)

        if self.show_index:
            font_internal = QFont("Arial", fontsize)
            self.painter.setFont(font_internal)
            for i in range(len(self.units)):
                pixel_center = self.toPixel(self.calculateUnitCenterUsingContribution(i))   
                self.painter.drawText(QPoint(pixel_center[X] + int(fontsize*0.33), pixel_center[Y] - 4), str(i))
            font_internal.setBold(True)
            font_internal.setItalic(True)
            self.painter.setFont(font_internal)
            for i in range(len(self.P_candidate)):
                pixel_center = self.toPixel(self.P_candidate[i])
                center = QPoint(pixel_center[X], pixel_center[Y])
                self.painter.drawText(QPoint(pixel_center[X] + int(fontsize*0.33), pixel_center[Y] + fontsize), 'P' + str(i))
        
        if self.expert_mode:
            self.painter.setPen(QPen(QColor(86, 10, 180), int(2 * line_weight * self.current_pixel_scale), Qt.SolidLine))
            center = self.calculateUnitCenterUsingContribution(self.choose_unit_id)
            pixel_center = self.toPixel(center)
            q_center = QPoint(pixel_center[X], pixel_center[Y])
            self.painter.drawPoint(q_center)
        
        if self.fixed_panel >= 0:
            self.painter.setPen(QPen(QColor(192, 0, 0), int(2 * line_weight * self.current_pixel_scale), Qt.SolidLine))
            seq_point = self.units[self.fixed_panel].getSeqPoint()
            for point in seq_point:
                pixel_point = self.toPixel(point)
                q_point = QPoint(pixel_point[X], pixel_point[Y])
                self.painter.drawPoint(q_point)

        self.draw_panel.setPixmap(self.pixmap)
        self.painter.end()

    def drawProcess(self, process=0.0):
        """
        @ function: draw process bar
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        if self.enable_output_stl:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_output_stl = False
                self.updateMessage("Succeed to export as stl at " + 
                                   self.output_stl_file_path + 
                                   self.output_stl_crease_flag + 
                                   self.output_stl_board_flag)
        elif self.enable_cdf_curve_fitting:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_cdf_curve_fitting = False
                self.updateMessage("Succeed to do cdf process")
        elif self.enable_phys_data_collecting:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_phys_data_collecting = False
                self.updateMessage("Succeed to collect physical simulation data")
        elif self.enable_mcts:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_mcts = False
                self.updateMessage("Succeed to do MCTS")
            else:
                with open(os.path.join(self.string_file_path, "current.json"), 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.THREADING_METHOD)
                self.updateState("Current strings", self.DESIGN_FINISH)

        else:
            self.process_bar.setValue(int(process * 100))

    def editKl(self):
        if self.expert_mode:
            self.updateMessage("Please close the expert mode first...")
            return
        if self.edit_sequence_mode:
            self.updateMessage("Please close the edit sequence mode first...")
            return
        find_kl = False
        for i in range(len(self.storage)):
            if type(self.storage[i][1]) == KinematicLine:
                find_kl = True
                break
        if find_kl:
            self.chooseKl(i)
            self.chooseKlLine(0)
            self.widget_edit_kl.setVisible(True)
            self.edit_kl_mode = True
        else:
            self.edit_kl_mode = False
            self.widget_edit_kl.setVisible(False)
            self.updateMessage("No Kinematic Line are found.", "WARNING")

    def editSequence(self):
        if self.expert_mode:
            self.updateMessage("Please close the expert mode first...")
            return
        if self.edit_kl_mode:
            self.updateMessage("Please close the edit kl mode first...")
            return
        self.chooseCreaseSequence(0, 1)
        self.edit_sequence_mode = True
        self.widget_edit_sequence.setVisible(True)
       
    def endAddString(self):
        self.actionAdd_TSA_A_point.setEnabled(False)
        self.updateMessage("Add-string mode disabled...")
        if len(self.a_string) >= 2:
            self.string_total_information.append(self.a_string)

    def expertModeEnable(self):
        if self.edit_kl_mode:
            self.updateMessage("Please close the edit kl mode first...")
            return
        if self.edit_sequence_mode:
            self.updateMessage("Please close the edit sequence mode first...")
            return
        if self.add_hole_mode:
            self.chooseUnit(0)
            self.chooseCrease(0)
            self.widget.setVisible(True)
            self.expert_mode = True
        else:
            self.expert_mode = False
            self.widget.setVisible(False)
            self.updateMessage("Please enable add-hole mode first.", "WARNING")

    def exportAsDxf(self, file_path):
        """
        @ function: export as dxf
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Name Dxf file and specify a path", 
                ".", 
                "dxf files (*.dxf)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as dxf")
        else:
            try:
                self.dxf_writer.setFileName(file_path)
                
                hole_lines = []
                for ele in self.hole_kps:
                    center = ele[0] + [0.0]
                    self.dxf_writer.addCircle(center, self.hole_size, HOLE)
                    # points = generatePolygonByCenter(center, self.hole_size, self.hole_resolution)
                    # for i in range(-1, len(points) - 1):
                    #     cur = points[i]
                    #     next = points[i + 1]
                    #     hole_lines.append(Crease(
                    #         cur, next, HOLE
                    #     ))         
                        
                self.dxf_writer.ExportAsDxf(self.lines)
                self.updateMessage("Succeed to export as dxf at " + file_path)
            except:
                self.updateMessage("Failed to export dxf file, please check the permission..." + file_path)
    
    def exportAsSplitDxf(self):
        """
        @ function: export as dxf with boards and creases
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230416
        @ spec: None
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Name Dxf file and specify a path", 
            ".", 
            "dxf files (*.dxf)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as dxf")
        else:
            try:
                total_lines = deepcopy(self.additional_lines)
                if self.pref_pack["dxf_hole_enable"]:
                    hole_lines = []
                    for ele in self.hole_kps:
                        center = ele[0] + [0.0]
                        points = generatePolygonByCenter(center, self.hole_size, self.hole_resolution)
                        for i in range(-1, len(points) - 1):
                            cur = points[i]
                            next = points[i + 1]
                            hole_lines.append(Crease(
                                cur, next, HOLE
                            ))
                    total_lines += hole_lines

                    thick_panel_hole_lines = []

                    thick_panel_split_line_maker = deepcopy(self.additional_line_maker)
                    thick_panel_split_line_maker.clearValidCrease()
                    thick_panel_split_line_maker.clear()
                    thick_panel_split_line_maker.clearCrease()   
                    bias = self.pref_pack["normal_hinge_width"] * 0.5
                    thick_panel_split_line_maker.setBias(bias)
                    thick_panel_split_line_maker.border_nobias = False

                    for unit in self.units:
                        thick_panel_split_line_maker.addPackedOrigamiUnit(unit)

                    split_lines = thick_panel_split_line_maker.getAdditionalLineForAllUnit()
                    for line in split_lines:
                        # if line.getType() != BORDER:
                        if 1:
                            thick_panel_hole_lines = []
                            length = line.getLength()
                            k_standard = line.k()
                            mid_point = line.getMidPoint()
                            if k_standard == math.inf:
                                mid_point_bias1 = [
                                    mid_point[X] - bias * 2,
                                    mid_point[Y],
                                ]
                                mid_point_bias2 = [
                                    mid_point[X] + bias * 2,
                                    mid_point[Y],
                                ]
                            else:
                                mid_point_bias1 = [
                                    mid_point[X] - k_standard / math.sqrt(k_standard ** 2 + 1) * bias * 2,
                                    mid_point[Y] + 1.0 / math.sqrt(k_standard ** 2 + 1) * bias * 2,
                                ]
                                mid_point_bias2 = [
                                    mid_point[X] + k_standard / math.sqrt(k_standard ** 2 + 1) * bias * 2,
                                    mid_point[Y] - 1.0 / math.sqrt(k_standard ** 2 + 1) * bias * 2,
                                ]

                            correct_crease = []
                            for additional_crease in split_lines:
                                if (additional_crease.k() == math.inf and k_standard == math.inf) or abs(additional_crease.k() - k_standard) < 1e-5:
                                    if pointOnCrease(mid_point_bias1, additional_crease):
                                        correct_crease.append(additional_crease)
                                        continue
                                    if pointOnCrease(mid_point_bias2, additional_crease):
                                        correct_crease.append(additional_crease)
                                        continue
                            if len(correct_crease) == 1:
                                another_line = correct_crease[0]
                                v1 = np.array([line[END][X] - line[START][X], line[END][Y] - line[START][Y]])
                                v2 = np.array([another_line[END][X] - line[START][X], another_line[END][Y] - line[START][Y]])
                                val = v2.dot(v1) / np.linalg.norm(v1)
                                ratio = val / length
                                if ratio > 0 and ratio < 1:
                                    percent_point = line.getPercentPoint(ratio)
                                    line.points[START] = percent_point
                                elif ratio > 1:
                                    continue
                                v3 = -v1
                                v4 = np.array([another_line[START][X] - line[END][X], another_line[START][Y] - line[END][Y]])
                                val2 = v3.dot(v4) / np.linalg.norm(v3)
                                ratio = val2 / length
                                if ratio > 0 and ratio < 1:
                                    percent_point = line.getPercentPoint(1. - ratio)
                                    line.points[END] = percent_point
                                elif ratio > 1:
                                    continue
                            else:
                                continue
                            length = line.getLength()
                            if 1:
                                if length < 2. * self.pref_pack["normal_hinge_outer_length"]:# and length >= self.pref_pack["normal_hinge_outer_length"]:
                                    # add 1 hinge
                                    ratio = self.pref_pack["normal_hinge_length"] / length * 0.5

                                    center1 = line.getPercentPoint(0.5 - ratio) + [0.0]
                                    center2 = line.getPercentPoint(0.5 + ratio) + [0.0]
                                    center3 = line.getPercentPoint(0.5) + [0.0]
                                    points1 = generatePolygonByCenter(center1, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points2 = generatePolygonByCenter(center2, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points3 = generatePolygonByCenter(center3, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    for i in range(-1, len(points1) - 1):
                                        cur = points1[i]
                                        next = points1[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points2) - 1):
                                        cur = points2[i]
                                        next = points2[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points3) - 1):
                                        cur = points3[i]
                                        next = points3[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))

                                elif length >= 2. * self.pref_pack["normal_hinge_outer_length"]:
                                    # add 2 hinge
                                    ratio = self.pref_pack["normal_hinge_length"] / length * 0.5
                                    center1 = line.getPercentPoint(0.25 - ratio) + [0.0]
                                    center2 = line.getPercentPoint(0.25 + ratio) + [0.0]
                                    center3 = line.getPercentPoint(0.25) + [0.0]
                                    points1 = generatePolygonByCenter(center1, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points2 = generatePolygonByCenter(center2, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points3 = generatePolygonByCenter(center3, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    for i in range(-1, len(points1) - 1):
                                        cur = points1[i]
                                        next = points1[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points2) - 1):
                                        cur = points2[i]
                                        next = points2[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points3) - 1):
                                        cur = points3[i]
                                        next = points3[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    
                                    center1 = line.getPercentPoint(0.75 - ratio) + [0.0]
                                    center2 = line.getPercentPoint(0.75 + ratio) + [0.0]
                                    center3 = line.getPercentPoint(0.75) + [0.0]
                                    points1 = generatePolygonByCenter(center1, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points2 = generatePolygonByCenter(center2, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points3 = generatePolygonByCenter(center3, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    for i in range(-1, len(points1) - 1):
                                        cur = points1[i]
                                        next = points1[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points2) - 1):
                                        cur = points2[i]
                                        next = points2[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points3) - 1):
                                        cur = points3[i]
                                        next = points3[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                total_lines += thick_panel_hole_lines
                            else:
                                ratio = self.pref_pack["normal_hinge_length"] / length * 0.5
                                # center3 = line.getPercentPoint(0.5) + [0.0]
                                # points3 = generatePolygonByCenter(center3, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                # for i in range(-1, len(points3) - 1):
                                #     cur = points3[i]
                                #     next = points3[i + 1]
                                #     thick_panel_hole_lines.append(Crease(
                                #         cur, next, HINGE_HOLE
                                #     ))
                                expand = 1.
                                while expand * ratio < 0.5:
                                    center1 = line.getPercentPoint(0.5 - expand * ratio) + [0.0]
                                    center2 = line.getPercentPoint(0.5 + expand * ratio) + [0.0]
                                    points1 = generatePolygonByCenter(center1, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    points2 = generatePolygonByCenter(center2, self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                                    for i in range(-1, len(points1) - 1):
                                        cur = points1[i]
                                        next = points1[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    for i in range(-1, len(points2) - 1):
                                        cur = points2[i]
                                        next = points2[i + 1]
                                        thick_panel_hole_lines.append(Crease(
                                            cur, next, HINGE_HOLE
                                        ))
                                    expand += 1.
                                total_lines += thick_panel_hole_lines
                            
                    #corner
                    if 0:
                        thick_panel_hole_lines = []
                        thick_panel_split_line_maker.clearValidCrease()
                        thick_panel_split_line_maker.clear()
                        thick_panel_split_line_maker.clearCrease()   
                        bias = self.pref_pack["normal_hinge_hole_radius"] * 2.0
                        thick_panel_split_line_maker.setBias(bias)
                        thick_panel_split_line_maker.border_nobias = False

                        for unit in self.units:
                            thick_panel_split_line_maker.addPackedOrigamiUnit(unit)

                        corner_centers = []
                        split_lines = thick_panel_split_line_maker.getAdditionalLineForAllUnit()
                        for line in split_lines:
                            center1 = line[START]
                            center2 = line[END]
                            find1 = False
                            find2 = False
                            for ele in corner_centers:
                                if distance(ele, center1) < 1e-3:
                                    find1 = True
                                if distance(ele, center2) < 1e-3:
                                    find2 = True
                                if find1 and find2:
                                    break

                            if not find1:
                                corner_centers.append(center1)
                            if not find2:
                                corner_centers.append(center2)
                        
                        for center in corner_centers:
                            points = generatePolygonByCenter(center + [0.0], self.pref_pack["normal_hinge_hole_radius"], self.hole_resolution)
                            for i in range(-1, len(points) - 1):
                                cur = points[i]
                                next = points[i + 1]
                                thick_panel_hole_lines.append(Crease(
                                    cur, next, HINGE_HOLE
                                ))

                        total_lines += thick_panel_hole_lines

                self.dxf_writer.setFileName(file_path.split('.')[0] + '.dxf')
                self.dxf_writer.ExportAsDxf(total_lines)
                self.dxf_writer.setFileName(file_path.split('.')[0] + '_crease.dxf')
                self.dxf_writer.ExportAsDxf(self.crease_lines)
                self.updateMessage("Succeed to export 2 files as dxf at " + file_path)
            except:
                self.updateMessage("Failed to export dxf file, please check the permission..." + file_path)

    def exportAllAsStl(self):
        """
        @ function: export all stl file
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        for ele in self.hole_kps:
            if not ele[2]:
                self.updateMessage("Failed to call stl output, check if any holes are invalid(in gray color)...")
                return
        if self.enable_output_stl:
            self.updateMessage("A stl file is being outputed, please wait...")
            return
        if self.enable_cdf_curve_fitting:
            self.updateMessage("A cdf process is running, please wait...")
            return
        if self.enable_phys_data_collecting:
            self.updateMessage("A physical simulation is running, please wait...")
            return
        if self.enable_mcts:
            self.updateMessage("A MCTS Searching process is running, please wait...")
            return
        if self.enable_threading_design:
            self.updateMessage("A threading design simulation is running, please wait...")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Name Stl file and specify a path", 
            ".", 
            "stl files (*.stl)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as stl")
        else:
            # stl_dialog = StlSettingDialog(parent=self)
            # stl_dialog.spinBox_unit_id.setEnabled(False)
            # stl_dialog.setBiasAndLock(self.bias_val)
            # if stl_dialog.exec_():
            #     pass
            # if not stl_dialog.getOK():
            #     self.updateMessage("Cancel exporting as stl")
            #     return
            # STL settings 
            # -----START----- #
            height = self.pref_pack["layer_of_panel"] * self.pref_pack["print_accuracy"] * 2.
            bias = self.bias_val
            method = "symmetry"
            if method == "symmetry" and self.pref_pack['enable_db_bind']:
                method = 'binding'

            self.enable_output_stl = True
            self.updateMessage("Stl file generating...")
            self.drawProcess(0.0)
            self.repaint()
            
            self.stl_writer.clear()
            self.stl_writer.clearCrease()

            self.stl_writer.setPrintAccuracy(self.pref_pack['print_accuracy'])
            self.stl_writer.setAsym(self.pref_pack["asym"])
            self.stl_writer.setOnlyTwoSides(self.pref_pack["only_two_sides"])

            if len(self.strings):
                self.stl_writer.string_width = self.strings[-1].width
                self.stl_writer.string_list = deepcopy(self.strings)

            self.stl_writer.setHeight(height)
            self.stl_writer.setBias(bias)
            self.stl_writer.min_bias = self.pref_pack['middle_bias']
            self.stl_writer.setMethod(method)
            self.stl_writer.setThinMode(self.pref_pack['thin_mode'])
            self.stl_writer.setDbEnable(self.pref_pack['enable_db'])
            self.stl_writer.setPillarDisable(self.pref_pack['disable_pillars'])
            self.stl_writer.border_nobias = not self.pref_pack['stl_asymmetry']
            connection_enabled = True
            board_enabled = True

            # self.stl_writer.setHoleWidth(stl_dialog.getHoleWidth())
            # self.stl_writer.setHoleLength(stl_dialog.getHoleLength())
            if len(self.transformation_matrix) == len(self.units):
                self.stl_writer.unit_rotation_matrix.clear()
                self.stl_writer.unit_transformation_vector.clear()
                for i in range(len(self.units)):
                    self.stl_writer.unit_rotation_matrix.append(self.transformation_matrix[i][:3, :3])
                    self.stl_writer.unit_transformation_vector.append(self.transformation_matrix[i][:3, 3])

            self.stl_writer.setLayerOfCrease(self.pref_pack['layer'])
            board_height = self.pref_pack["layer"] * self.pref_pack["print_accuracy"]
            self.stl_writer.setBoardHeight(board_height)

            # set hole
            self.stl_writer.setUnitHoles(deepcopy(self.hole_kps))
            self.stl_writer.setUnitHoleSize(self.hole_size)
            self.stl_writer.setUnitHoleResolution(self.hole_resolution)

            self.stl_writer.disableUsingModifiedUnit()

            # set connection hole
            if self.enable_connection:
                self.stl_writer.setConnectionHoleSize(self.connection_radius)
                self.stl_writer.setConnectionHoles(self.backup_connection_hole_kps)
            
            # set crease stiffness
            self.stl_writer.setHardCrease(self.hard_crease_index)

            self.stl_writer.setUnitBias(self.unit_bias_list)

            # -----END----- #

            # Global settings
            # -----START----- #
            if method == "symmetry":
                # Connection of crease
                connection_enabled = True
            # connection_angle = self.pref_pack["connection_angle"] * math.pi / 180.0
            # Copy designer
            designer = deepcopy(self.designer)
            # -----END----- #

            # for all storage
            # try:
            for i in range(self.origami_number):
                src = self.storage[i][1]
                # ----- Set the transition start point for designer ----- #
                # ----- START ----- #
                if(self.storage[i][2]):
                    designer.setTransitionStartPoint([
                        self.storage[i][0][0], 
                        self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                    ])
                else:
                    designer.setTransitionStartPoint(self.storage[i][0])
                # ----- END ----- #

                # ----- Set the source for designer ----- #
                # ----- START ----- #
                designer.setSource(src)
                # ----- END ----- #

                # If LeanMiura is using global data
                # ----- START ----- #
                if type(src) == ModuleLeanMiura:
                    if src.enable_global_modify:
                        src.unit_width = self.unit_width
                        src.copy_time = self.copy_time
                        src.entry_flag = self.entry_flag if self.copy_time % 2 else not self.entry_flag
                        src.connection_flag = self.enable_connection
                        src.con_left_length = self.con_left_length
                        src.con_right_length = self.con_right_length
                        src.connection_hole_size = self.connection_radius
                # ----- END ----- #

                # Clear additional line for designer
                # ----- START ----- #
                designer.clearAdditionalLine()
                # ----- END ----- #

                # Set the data of designer to make new origami
                # ----- START ----- #
                designer.setPaperInfo(self.paper_info)
                designer.setDesignInfo(self.design_info)
                designer.setEntryFlag(self.entry_flag)
                # ----- END ----- #

                # <<<<< Before design <<<<< #

                # -------- Design -------- #
                # ----- START ----- #
                if method != "symmetry": # 1 designer
                    designer.parseData()
                    # ----- END ----- #
                    # -------- Design -------- #

                    # >>>>> After design >>>>> #
                    # get design data
                    data = designer.getDesignData()

                    # Get unit
                    # -----START----- #
                    # LeanMiura Module, only one result at data[0]
                    if type(src) == ModuleLeanMiura: 
                        _ = data[0].getKeyPoint()
                        creases = data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiUnit(ele)

                    # Miura Module
                    elif type(src) == KinematicLine:
                        data_length = len(data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
      
                                    self.stl_writer.addPackedOrigamiUnit(u1)
                                    self.stl_writer.addPackedOrigamiUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = data[0].getLine()
                        u = data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiUnit(ele)

                else: # 2 designer
                    other_designer = deepcopy(designer)
                    # first designer add left
                    if self.enable_connection:
                        if type(src) == KinematicLine:
                            designer.insertLineSource([self.con_left_length, designer.src.lines[0][1]], 0)
                            # second designer add right and set tsp
                            if(self.storage[i][2]):
                                other_designer.setTransitionStartPoint([
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                                ])
                            else:
                                other_designer.setTransitionStartPoint([
                                    self.storage[i][0][0] + self.con_left_length,
                                    self.storage[i][0][1]
                                ])
                            other_designer.insertLineSource([self.con_right_length, designer.src.lines[-1][1]], designer.getKLNumber())
                        elif type(src) == ModuleLeanMiura:
                            if(self.storage[i][2]):
                                transition_initial_point = [
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                                ]
                            else:
                                transition_initial_point = [
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1]
                                ]
                            other_designer.setTransitionStartPoint(transition_initial_point)
                            modified_lean_miura = deepcopy(src)
                            modified_lean_miura.tsp = transition_initial_point
                            modified_lean_miura.connection_flag = False
                            other_designer.setSource(modified_lean_miura)
                    
                    # -------- Design -------- #
                    # ----- START ----- #
                    # parse data at the same time
                    designer.parseData()
                    other_designer.parseData()
                    # ----- END ----- #
                    # -------- Design -------- #

                    # get the first designer data
                    data = designer.getDesignData()

                    # LeanMiura Module
                    if type(src) == ModuleLeanMiura: # 1 data
                        _ = data[0].getKeyPoint()
                        creases = data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiUnit(ele)
                    
                    # Miura module
                    elif type(src) == KinematicLine:    
                        data_length = len(data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    # add to normal unit
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])

                                    self.stl_writer.addPackedOrigamiUnit(u1)
                                    self.stl_writer.addPackedOrigamiUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = data[0].getLine()
                        u = data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiUnit(ele)
                    
                    # get the second designer data
                    second_data = other_designer.getDesignData()

                    # LeanMiura Module
                    if type(src) == ModuleLeanMiura: # 1 data
                        _ = second_data[0].getKeyPoint()
                        creases = second_data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = second_data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiModifiedUnit(ele)

                    # Miura module
                    elif type(src) == KinematicLine: 
                        data_length = len(second_data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = second_data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    # add to normal unit
                                    u1, u2 = getUnitWithinMiura(second_data[unit_id - 1], second_data[unit_id])
   
                                    self.stl_writer.addPackedOrigamiModifiedUnit(u1)
                                    self.stl_writer.addPackedOrigamiModifiedUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = second_data[0].getLine()
                        u = second_data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiModifiedUnit(ele)

                    # can generate different up-down unit when symmetry
                    self.stl_writer.enableUsingModifiedUnit()

            if self.pref_pack['debug_mode']:
                #symmetry method
                if method == "symmetry":
                    if self.stl_writer.db_enable:
                        self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.pref_pack["print_accuracy"])
                        soft_file_path = file_path.split('.')[0] + '_S.stl'

                        self.stl_writer.calculateTriPlaneForAllUnit(inner=True)

                        self.stl_writer.outputAllStl(soft_file_path)
                        hard_file_path = file_path.split('.')[0] + '_H.stl'

                        self.stl_writer.calculateTriPlaneForAllUnit(inner=False)

                        self.stl_writer.outputAllStl(hard_file_path)
                    else:
                        self.stl_writer.setBoardHeight(self.pref_pack["print_accuracy"])

                        self.stl_writer.calculateTriPlaneForAllUnit()

                        self.stl_writer.outputAllStl(file_path)

                    if self.stl_writer.db_enable:
                        self.stl_writer.setHeight(2 * self.pref_pack["print_accuracy"])
                    else:
                        self.stl_writer.setHeight(self.pref_pack["print_accuracy"])
                    self.stl_writer.setBias(self.pref_pack["board_bias"])
                    self.stl_writer.setHoleWidth(0.001)
                    self.stl_writer.setHoleLength(0.001)
                    self.stl_writer.getAdditionalLineForAllUnit()
                    
                    crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'

                    self.stl_writer.calculateTriPlaneForAllCrease()

                    self.stl_writer.outputAllCreaseStl(crease_file_path)

                    board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                    self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

                elif method == "binding":            
                    # set difference
                    self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                    self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)

                    hard_file_path = file_path.split('.')[0] + '.stl'

                    self.stl_writer.calculateTriPlaneForAllUnit(inner=False)

                    self.stl_writer.outputAllStl(hard_file_path)

                    # set difference
                    self.stl_writer.enable_difference = 0
                    self.stl_writer.setBias(self.pref_pack["board_bias"])
                    self.stl_writer.setHoleWidth(1e-5)
                    self.stl_writer.setHoleLength(1e-5)
                    self.stl_writer.getAdditionalLineForAllUnit()
                    
                    crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'
                    crease_file_path_dxf = file_path.split('.')[0] + '_crease.dxf'
                    self.stl_writer.crease_file_path = crease_file_path_dxf

                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)

                    board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                    self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

                    if (len(self.stl_writer.string_list)):
                        self.stl_writer.calculateTriPlaneForString()
                        string_file_path = file_path.split('.')[0] + '_string.stl'
                        self.stl_writer.outputStringStl(string_file_path)
                
                self.drawProcess(1.0)
                self.enable_output_stl = False
                return
            
            else:
                self.stl_output_thread = StlOutputThread(
                    self.stl_writer, 
                    self.show_process,
                    method,
                    connection_enabled,
                    board_enabled,
                    bias,
                    file_path,
                    pref_pack=self.pref_pack
                )
                self.stl_output_thread._emit.connect(self.drawProcess)
                self.stl_output_thread.start()
                self.output_stl_file_path = file_path
                if method == "upper_bias" or method == "both_bias":
                    self.output_stl_crease_flag = ""
                    if connection_enabled:
                        self.output_stl_crease_flag = "(+ *_crease.stl)"
                    self.output_stl_board_flag = ""
                    if board_enabled:
                        self.output_stl_crease_flag = "(+ *_board.stl)"
                elif method == "symmetry":
                    self.output_stl_crease_flag = "(+ *_midlayer_C.stl)"
                    self.output_stl_board_flag = "(+ *_midlayer_B.stl)"
                return

    def exportAsStl(self):
        return

    def exportDescriptionData(self, file_path):
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save the description data", 
                ".", 
                "json files (*.json)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting design result")
        else:
            # the units is compatible with the left-hand coodination
            s = {
                "kps": self.kps,
                "lines": [line.points for line in self.lines],
                "units": [unit.getSeqPoint() for unit in self.units],
                "line_features": [{
                    "type": line.getType(),
                    "level": line.level,
                    "coeff": line.coeff,
                    "recover_level": line.recover_level,
                    "recover_angle": line.recover_angle,
                    "hard": line.hard,
                    "hard_angle": line.folding_angle_upper_bound,
                    "hard_angle_down": line.folding_angle_lower_bound,
                    "thick_panel_height": line.thick_panel_height
                } for line in self.lines],
                "strings": {
                    "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                },
                "P_candidators": {
                    "points": self.P_candidate,
                    "connections": self.P_candidate_connection_index 
                },
                "contributions": self.unit_center_contribute_coeff
            }

            if self.fixed_panel >= 0:
                s["fix"] = [self.fixed_panel]
            
            if len(self.crease_angle):
                s["crease_angle"] = self.crease_angle
            
            if len(self.crease_info):
                s["crease_info"] = self.crease_info

            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(s, f, indent=4)
            self.updateMessage("Succeed to save the description data at " + file_path)
            
    def exportRotationalDescriptionData(self, file_path):
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save the description data", 
                ".", 
                "json files (*.json)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting design result")
        else:
            # the units is compatible with the left-hand coodination
            if type(self.storage[0][1]) == KinematicLine:
                copy_time = self.copy_time
                entry_flag = self.entry_flag
                half_width = self.unit_width * 0.5
                rotational_step = math.pi / copy_time
                
                new_kps = deepcopy(self.kps)
                new_lines = deepcopy(self.lines)
                new_units = deepcopy(self.units)
                
                try:
                    i = 0
                    while i < len(new_kps):
                        if len(new_kps[i]) == 2:
                            new_kps[i] = [new_kps[i][X], new_kps[i][Y], 0.0]
                        x = new_kps[i][X]
                        y = new_kps[i][Y]
                        z = new_kps[i][Z]
                        bonus = round(y / half_width)
                        new_y = 0.
                        new_z = 0.
                        for j in range(bonus):
                            new_y += half_width * math.cos(j * rotational_step)
                            new_z += half_width * math.sin(j * rotational_step)
                        
                        new_kps[i] = [x, new_y, new_z]
                        if bonus == 2 * copy_time:
                            del(new_kps[i])
                        else:
                            i += 1
                    
                    i = 0   
                    
                    no_match_edge = [] #x1, x2, id
                    exceed_mach_edge = [] #x1, x2, crease_type

                    while i < len(new_lines):
                        bonus_list = []
                        current_x = []
                        crease_type = new_lines[i].crease_type
                        for j in range(2):
                            if len(new_lines[i][j]) == 2:
                                new_lines[i].points[j] = [new_lines[i].points[j][X], new_lines[i].points[j][Y], 0.0]
                            x = new_lines[i].points[j][X]
                            y = new_lines[i].points[j][Y]
                            z = new_lines[i].points[j][Z]
                            bonus = round(y / half_width)
                            new_y = 0.
                            new_z = 0.
                            for k in range(bonus):
                                new_y += half_width * math.cos(k * rotational_step)
                                new_z += half_width * math.sin(k * rotational_step)
                            
                            new_lines[i].points[j] = [x, new_y, new_z]

                            current_x.append(x)
                            bonus_list.append(bonus)
                        if bonus_list[0] == 2 * copy_time and bonus_list[1] == 2 * copy_time:
                            del(new_lines[i])
                        elif bonus_list[0] == 0 and bonus_list[1] == 0:
                            no_match_edge.append([current_x[0], current_x[1], i])
                            for other_crease in exceed_mach_edge:
                                if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                    (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                    new_lines[i].crease_type = other_crease[2]
                                    break
                            i += 1
                        elif bonus_list[0] == 2 and bonus_list[1] == 2:
                            exceed_mach_edge.append([current_x[0], current_x[1], crease_type])
                            for other_crease in no_match_edge:
                                if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                    (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                    new_lines[other_crease[2]].crease_type = crease_type
                                    break
                            i += 1
                        else:
                            i += 1
                                
                    for i in range(len(new_units)):
                        creases = new_units[i].crease
                        for j in range(len(creases)):
                            bonus_list = []
                            current_x = []
                            for k in range(2):
                                if len(creases[j][k]) == 2:
                                    creases[j].points[k] = [creases[j].points[k][X], creases[j].points[k][Y], 0.0]
                                x = creases[j].points[k][X]
                                y = creases[j].points[k][Y]
                                z = creases[j].points[k][Z]
                                bonus = round(y / half_width)
                                new_y = 0.
                                new_z = 0.
                                for l in range(bonus):
                                    new_y += half_width * math.cos(l * rotational_step)
                                    new_z += half_width * math.sin(l * rotational_step)
                                
                                creases[j].points[k] = [x, new_y, new_z]
                                current_x.append(x)
                                bonus_list.append(bonus)
                                
                            if bonus_list[0] == 0 and bonus_list[1] == 0:
                                for other_crease in no_match_edge:
                                    if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                        (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                        creases[j].crease_type = other_crease[2]
                            i += 1
                
                except:
                    self.updateMessage("Convertion failed. Please check the input.")
                    return
                    
                s = {
                    "kps": new_kps,
                    "lines": [line.points for line in new_lines],
                    "units": [unit.getSeqPoint() for unit in new_units],
                    "line_features": [{
                        "type": line.getType(),
                        "level": line.level,
                        "coeff": line.coeff,
                        "recover_level": line.recover_level,
                        "recover_angle": line.recover_angle,
                        "hard": line.hard,
                        "hard_angle": line.folding_angle_upper_bound,
                        "hard_angle_down": line.folding_angle_lower_bound,
                        "thick_panel_height": line.thick_panel_height
                    } for line in new_lines],
                    "strings": {
                        "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                        "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                        "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                    },
                    "P_candidators": {
                        "points": self.P_candidate,
                        "connections": self.P_candidate_connection_index 
                    },
                    "contributions": self.unit_center_contribute_coeff
                }

                if self.fixed_panel >= 0:
                    s["fix"] = [self.fixed_panel]
                
                if len(self.crease_angle):
                    s["crease_angle"] = self.crease_angle
                
                if len(self.crease_info):
                    s["crease_info"] = self.crease_info

                with open(file_path, 'w', encoding="utf-8") as f:
                    json.dump(s, f, indent=4)
                self.updateMessage("Succeed to save the description data at " + file_path)
                
            else:
                self.updateMessage("The origami is not in description of Kinematic lines.")
    
    def exportRotationalDescriptionData_Reverse(self, file_path):
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save the description data", 
                ".", 
                "json files (*.json)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting design result")
        else:
            # the units is compatible with the left-hand coodination
            if type(self.storage[0][1]) == KinematicLine:
                copy_time = self.copy_time
                entry_flag = self.entry_flag
                half_width = self.unit_width * 0.5
                rotational_step = -math.pi / copy_time
                
                new_kps = deepcopy(self.kps)
                new_lines = deepcopy(self.lines)
                new_units = deepcopy(self.units)
                
                try:
                    i = 0
                    while i < len(new_kps):
                        if len(new_kps[i]) == 2:
                            new_kps[i] = [new_kps[i][X], new_kps[i][Y], 0.0]
                        x = new_kps[i][X]
                        y = new_kps[i][Y]
                        z = new_kps[i][Z]
                        bonus = round(y / half_width)
                        new_y = 0.
                        new_z = 0.
                        for j in range(bonus):
                            new_y += half_width * math.cos(j * rotational_step)
                            new_z += half_width * math.sin(j * rotational_step)
                        
                        new_kps[i] = [x, new_y, new_z]
                        if bonus == 2 * copy_time:
                            del(new_kps[i])
                        else:
                            i += 1
                    
                    i = 0   
                    
                    no_match_edge = [] #x1, x2, id
                    exceed_mach_edge = [] #x1, x2, crease_type

                    while i < len(new_lines):
                        bonus_list = []
                        current_x = []
                        crease_type = new_lines[i].crease_type
                        for j in range(2):
                            if len(new_lines[i][j]) == 2:
                                new_lines[i].points[j] = [new_lines[i].points[j][X], new_lines[i].points[j][Y], 0.0]
                            x = new_lines[i].points[j][X]
                            y = new_lines[i].points[j][Y]
                            z = new_lines[i].points[j][Z]
                            bonus = round(y / half_width)
                            new_y = 0.
                            new_z = 0.
                            for k in range(bonus):
                                new_y += half_width * math.cos(k * rotational_step)
                                new_z += half_width * math.sin(k * rotational_step)
                            
                            new_lines[i].points[j] = [x, new_y, new_z]

                            current_x.append(x)
                            bonus_list.append(bonus)
                        if bonus_list[0] == 2 * copy_time and bonus_list[1] == 2 * copy_time:
                            del(new_lines[i])
                        elif bonus_list[0] == 0 and bonus_list[1] == 0:
                            no_match_edge.append([current_x[0], current_x[1], i])
                            for other_crease in exceed_mach_edge:
                                if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                    (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                    new_lines[i].crease_type = other_crease[2]
                                    break
                            i += 1
                        elif bonus_list[0] == 2 and bonus_list[1] == 2:
                            exceed_mach_edge.append([current_x[0], current_x[1], crease_type])
                            for other_crease in no_match_edge:
                                if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                    (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                    new_lines[other_crease[2]].crease_type = crease_type
                                    break
                            i += 1
                        else:
                            i += 1
                                
                    for i in range(len(new_units)):
                        creases = new_units[i].crease
                        for j in range(len(creases)):
                            bonus_list = []
                            current_x = []
                            for k in range(2):
                                if len(creases[j][k]) == 2:
                                    creases[j].points[k] = [creases[j].points[k][X], creases[j].points[k][Y], 0.0]
                                x = creases[j].points[k][X]
                                y = creases[j].points[k][Y]
                                z = creases[j].points[k][Z]
                                bonus = round(y / half_width)
                                new_y = 0.
                                new_z = 0.
                                for l in range(bonus):
                                    new_y += half_width * math.cos(l * rotational_step)
                                    new_z += half_width * math.sin(l * rotational_step)
                                
                                creases[j].points[k] = [x, new_y, new_z]
                                current_x.append(x)
                                bonus_list.append(bonus)
                                
                            if bonus_list[0] == 0 and bonus_list[1] == 0:
                                for other_crease in no_match_edge:
                                    if (abs(current_x[0] - other_crease[0]) < 1e-3 and abs(current_x[1] - other_crease[1] < 1e-3)) or \
                                        (abs(current_x[0] - other_crease[1]) < 1e-3 and abs(current_x[1] - other_crease[0]) < 1e-3):
                                        creases[j].crease_type = other_crease[2]
                            i += 1
                
                except:
                    self.updateMessage("Convertion failed. Please check the input.")
                    return
                    
                s = {
                    "kps": new_kps,
                    "lines": [line.points for line in new_lines],
                    "units": [unit.getSeqPoint() for unit in new_units],
                    "line_features": [{
                        "type": line.getType(),
                        "level": line.level,
                        "coeff": line.coeff,
                        "recover_level": line.recover_level,
                        "recover_angle": line.recover_angle,
                        "hard": line.hard,
                        "hard_angle": line.folding_angle_upper_bound,
                        "hard_angle_down": line.folding_angle_lower_bound,
                        "thick_panel_height": line.thick_panel_height
                    } for line in new_lines],
                    "strings": {
                        "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                        "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                        "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                    },
                    "P_candidators": {
                        "points": self.P_candidate,
                        "connections": self.P_candidate_connection_index 
                    },
                    "contributions": self.unit_center_contribute_coeff
                }

                if self.fixed_panel >= 0:
                    s["fix"] = [self.fixed_panel]
                
                if len(self.crease_angle):
                    s["crease_angle"] = self.crease_angle
                
                if len(self.crease_info):
                    s["crease_info"] = self.crease_info

                with open(file_path, 'w', encoding="utf-8") as f:
                    json.dump(s, f, indent=4)
                self.updateMessage("Succeed to save the description data at " + file_path)
                
            else:
                self.updateMessage("The origami is not in description of Kinematic lines.")

    def fixPanel(self):
        fix_id, ok = QInputDialog.getInt(self, "Fix panel: ", "Please select a panel to be fixed: ", -1, -1, len(self.units) - 1, 1)
        if ok:
            self.fixed_panel = fix_id
            if fix_id >= 0:
                self.updateState(f"Succeed to fix panel {fix_id}", self.state)
            else:
                self.updateState(f"Clear fixed panels", self.state)

    def generatePolygonByCenter(self, center, radius):
        """
        @ function: generate polygon for drawing
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230228
        @ spec: None
        """
        points = []
        step = math.pi * 2 / self.hole_resolution
        for i in range(0, self.hole_resolution):
            points.append(
                QPoint(
                    int(center[0] + math.cos(i * step) * radius * self.current_pixel_scale), 
                    int(center[1] + math.sin(i * step) * radius * self.current_pixel_scale)
                )
            )
        polygon = QPolygon(points)
        return polygon
    
    def initialize(self):
        self.kps = []                           # keypoints
        self.lines = []                         # key lines
        self.units = []                         # origami units, type: [Unit...]
        self.additional_lines = []              # additional line for warning of the add-hole operation
        self.hole_kps = []                      # hole keypoints
        self.connection_hole_kps = []           # connection hole keypoints 
        self.crease_lines = []                  # crease line for output crease dxf file
        self.backup_connection_hole_kps = []    # back up the connection keypoint for exporting stl
        self.strings = []                       # TSA strings
        self.add_bias_flag = []
        self.a_string = []
        self.string_total_information = []
        self.P_candidate = []
        self.P_candidate_connection_index = []
        self.unit_center_contribute_coeff = []

        # Pack of add hole mode
        self.checkBox_add_hole_mode.setChecked(False)
        self.checkBox_add_string_mode.setChecked(False)
        self.actionAdd_TSA_A_point.setEnabled(False)

        # Pack of connection
        self.checkBox_connection.setChecked(False)

        self.enable_connection = False
        self.add_hole_mode = False
        self.add_string_mode = False
        self.expert_mode = False

        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.dir_list = []
        self.transformation_matrix = []
        self.curve_name = None # we generate corresponding file name of CDF result

        self.show_index = False
        self.actionShow_Index.setText("Show Index")

        self.choose_hole_flag = False
        self.choose_origami_flag = False
        self.expert_mode = False
        self.edit_kl_mode = False
        self.edit_sequence_mode = False
        self.choose_unit_id = 0
        self.choose_crease_id = 0
        self.choose_kl_id = 0
        self.choose_line_id = 0
        self.choose_crease_sequence_id = 0
        self.choose_string_id = -1
        self.fixed_panel = -1
        
        self.crease_angle = []
        self.crease_info = []
        
        self.widget.setVisible(False)
        self.widget_edit_kl.setVisible(False)
        self.widget_edit_sequence.setVisible(False)
        self.updateMessage("Back to normal view...")

    def importDxf(self, not_call_openfile_dialog=False, file_name=''):
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a dxf file", 
                ".", 
                "Dxf files (*.dxf);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                self.backup_open_file_path = None
                dxf = DxfDirectGrabber()
                dxf.readFile(path)
                if dxf.mode == 'normal_dxf':
                    self.storage.clear()
                    self.add_bias_flag.clear()
                    self.storage.append([
                        [0, 0], dxf, False
                    ])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)
                    self.show_index = False
                    self.actionShow_Index.setText("Show Index")
                    self.unit_center_contribute_coeff = []
                    # self.hole_kps.clear()
                    self.file_path = path
                    self.enable_design = True
                    self.updateState("Succeeded to load dxf file", self.IMPORT_SUCCESS)
                    self.updateMessage("Previewing, click the \"design\" button to continue...")
                else:
                    self.storage.clear()
                    self.add_bias_flag.clear()
                    self.show_index = False
                    self.actionShow_Index.setText("Show Index")
                    self.unit_center_contribute_coeff = []
                    self.file_type = "KL"
                    self.file_path = path
                    kl_lines = dxf.kl
                    origin_list = dxf.origin
                    add_width_flag_list = [False for _ in range(len(origin_list))]
                    origin_list_length = len(origin_list)
                    for i in range(origin_list_length):
                        origin = origin_list[i]
                        add_width_flag = add_width_flag_list[i]
                        kl = KinematicLine()
                        for j in range(len(kl_lines[i])):
                            element = kl_lines[i][j]
                            kl.append(element)
                        self.storage.append([origin, kl, add_width_flag])
                        self.rotation.append(0.0)
                        self.add_bias_flag.append(False)
                        self.enable_design = True
                        self.updateState("Succeeded to load dxf file", self.IMPORT_SUCCESS)
                        self.updateMessage("Previewing, click the \"design\" button to continue...")
            except:
                self.updateState("Failed to load dxf file, check if it is in correct format", self.state, "ERROR")

    def importKL(self, not_call_openfile_dialog=False, file_name=''):
        """
        @ function: Import kinematic lines for design
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230305
        @ spec: add choose dialog
        """
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a json file with kl specification", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.KL_JSON)
                # self.hole_kps.clear()
                self.file_path = path
                self.backup_open_file_path = None
                self.enable_design = True
                self.updateState("Succeeded to load json file", self.IMPORT_SUCCESS)
                self.updateMessage("Previewing, click the \"design\" button to continue...")
            except:
                self.updateState("Failed to load json file, check if it is in correct format", self.state, "ERROR")
        
    def importStringPath(self, not_call_openfile_dialog=False, file_name=''):
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a string path", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.THREADING_METHOD)
                self.updateState("Current strings", self.DESIGN_FINISH)
            except:
                self.updateState("Failed to load json file, check if it is in correct format", self.state, "ERROR")

    def keyPressEvent(self, event):
        """
        @ function: exactly move the pixmap
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230227
        @ spec: None
        """
        self.key = ''
        if self.state == self.DESIGN_FINISH:
            if event.key() == Qt.Key_Up:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Up"
                    self.pixel_bias[Y] -= 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][Y] -= self.operation_amp
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][Y] -= self.operation_amp
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Down:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Down"
                    self.pixel_bias[Y] += 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][Y] += self.operation_amp
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][Y] += self.operation_amp
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Left:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Left"
                    self.pixel_bias[X] -= 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][X] -= self.operation_amp
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][X] -= self.operation_amp
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Right:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Right"
                    self.pixel_bias[X] += 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][X] += self.operation_amp
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][X] += self.operation_amp
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Z:
                if event.modifiers() & Qt.ControlModifier: 
                    if len(self.hole_kps) > 0:
                        self.hole_kps.pop()
                        self.updateMessage("Delete the latest point...")
            elif event.key() == Qt.Key_Q:
                if self.choose_origami_flag: 
                    self.rotation[self.choose_origami_index] -= 1.0 / 180.0 * math.pi
                    self.enable_design = True
            elif event.key() == Qt.Key_E:
                if event.modifiers() & Qt.ControlModifier:
                    self.key = "Control+E"
                    self.exportAllAsStl()
                else:
                    if self.choose_origami_flag: 
                        self.rotation[self.choose_origami_index] += 1.0 / 180.0 * math.pi
                        self.enable_design = True
            elif event.key() == Qt.Key_S:
                if event.modifiers() & Qt.ControlModifier:
                    self.key = "Control+S"
                    self.saveResult()
            elif event.key() == Qt.Key_D:
                # self.enable_design = True
                if len(self.strings):
                    if self.strings[-1].type == PASS:
                        self.strings.pop()
                        self.string_type = self.strings[-1].type
                        self.strings.pop()
                    else:
                        self.string_type = self.strings[-1].type
                        self.strings.pop()
                            
                    if self.add_string_mode:
                        if len(self.a_string) > 2:
                            self.a_string.pop()
                            self.string_start_point = self.a_string[-1].point.tolist()
                        else:
                            self.exist_string_start = False
                            self.string_start_point = []
                            self.a_string = []
                            self.string_type = BOTTOM
                    else:
                        if len(self.string_total_information[-1]) > 2:
                            self.string_total_information[-1].pop()
                        else:
                            self.string_total_information.pop()
                    self.updateMessage("Delete the latest string...")
            elif event.key() == Qt.Key_Escape:
                # self.enable_design = True
                self.choose_hole_flag = False
                self.choose_origami_flag = False
                self.expert_mode = False
                self.edit_kl_mode = False
                self.edit_sequence_mode = False
                self.choose_unit_id = 0
                self.choose_crease_id = 0
                self.choose_kl_id = 0
                self.choose_line_id = 0
                self.choose_crease_sequence_id = 0
                self.choose_string_id = -1
                self.widget.setVisible(False)
                self.widget_edit_kl.setVisible(False)
                self.widget_edit_sequence.setVisible(False)
                self.updateMessage("Back to normal view...")
            elif event.key() == Qt.Key_Delete:
                # self.enable_design = True
                if self.choose_hole_flag:
                    del(self.hole_kps[self.choose_hole_index])
                    self.choose_hole_flag = False
                    self.updateMessage("Delete hole id: " + str(self.choose_hole_id) + ", back to normal view...")

    def mapFromPixelToReal(self, pixel_x, pixel_y):
        real_x = pixel_x
        real_y = pixel_y
        if self.xy_rotation != 180:
            rad_rotation = -(self.xy_rotation - 180.0) / 180.0 * math.pi
            real_x = math.cos(rad_rotation) * (pixel_x - self.pixel_bias[X]) - math.sin(rad_rotation) * (pixel_y - self.pixel_bias[Y]) + self.pixel_bias[X]
            real_y = math.sin(rad_rotation) * (pixel_x - self.pixel_bias[X]) + math.cos(rad_rotation) * (pixel_y - self.pixel_bias[Y]) + self.pixel_bias[Y]
        if self.mirror_y:
            real_y = -real_y + 2 * self.pixel_bias[Y]
        if self.mirror_x:
            real_x = -real_x + 2 * self.pixel_bias[X]
        real_x = (real_x - self.pixel_bias[X]) / self.current_pixel_scale
        real_y = (real_y - self.pixel_bias[Y]) / self.current_pixel_scale
        
        return real_x, real_y

    def mapFromRealToPixel(self, real_x, real_y):
        pixel_x = real_x * self.current_pixel_scale + self.pixel_bias[X]
        pixel_y = real_y * self.current_pixel_scale + self.pixel_bias[Y]
        return pixel_x, pixel_y

    def mouseDoubleClickEvent_panel(self, event):
        x = event.x()
        y = event.y()
        if event.button() == Qt.LeftButton:
            self.cursor_x = x
            self.cursor_y = y
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )
            for i in range(len(self.hole_kps)):
                if distance([self.real_x, self.real_y], self.hole_kps[i][0]) < self.hole_size:
                    self.choose_hole_flag = True
                    self.choose_origami_flag = False
                    self.choose_hole_id = self.hole_kps[i][1]
                    self.choose_hole_index = i
                    self.updateMessage("Choose hole index: " + str(i))
                    return

            for i in range(len(self.storage)):
                if distance([self.real_x, self.real_y], self.storage[i][0]) < self.pref_pack["line_weight"]:
                    if isinstance(self.storage[i][1], ModuleLeanMiura):
                        self.storage[i][1].raiseDialog(self)
                        self.storage[i][0] = self.storage[i][1].tsp
                        self.enable_design = True
                        return
                    if isinstance(self.storage[i][1], KinematicLine):
                        self.choose_origami_flag = True
                        self.choose_hole_flag = False
                        self.choose_origami_index = i
                        self.updateMessage("Choose origami index: " + str(i))
                        self.enable_design = True
                        return
                    if isinstance(self.storage[i][1], DxfDirectGrabber):
                        self.choose_origami_flag = True
                        self.choose_hole_flag = False
                        self.choose_origami_index = i
                        self.updateMessage("Choose origami index: " + str(i))
                        self.enable_design = True
                        return
                    
            valid_crease_list = self.additional_line_maker.valid_crease_list
            for i in range(len(valid_crease_list)):
                if pointOnCrease([self.real_x, self.real_y], valid_crease_list[i], 2):
                    if i not in self.hard_crease_index:
                        self.hard_crease_index.append(i)
                        # self.enable_design = True
                        
                        true_index = -1
                        
                        for ii in range(len(self.lines)):
                            if (distance(self.lines[ii][START], valid_crease_list[i][START]) < 1e-5 and distance(self.lines[ii][END], valid_crease_list[i][END]) < 1e-5) or \
                                (distance(self.lines[ii][START], valid_crease_list[i][END]) < 1e-5 and distance(self.lines[ii][END], valid_crease_list[i][START]) < 1e-5):
                                    true_index = ii
                                    break
                        
                        self.lines[true_index].hard = True
                          
                        for ii in range(len(self.units)):
                            unit = self.units[ii]
                            lines = unit.getCrease()
                            for jj in range(len(lines)):
                                crease = lines[jj]
                                if (distance(self.lines[true_index][START], crease[START]) < 1e-5 and distance(self.lines[true_index][END], crease[END]) < 1e-5) or \
                                    (distance(self.lines[true_index][START], crease[END]) < 1e-5 and distance(self.lines[true_index][END], crease[START]) < 1e-5):
                                        self.unit_bias_list[ii][jj] = 1e-3
                    else:
                        del(self.hard_crease_index[self.hard_crease_index.index(i)])
                        # self.enable_design = True
                        
                        true_index = -1
                        
                        for ii in range(len(self.lines)):
                            if (distance(self.lines[ii][START], valid_crease_list[i][START]) < 1e-5 and distance(self.lines[ii][END], valid_crease_list[i][END]) < 1e-5) or \
                                (distance(self.lines[ii][START], valid_crease_list[i][END]) < 1e-5 and distance(self.lines[ii][END], valid_crease_list[i][START]) < 1e-5):
                                    true_index = ii
                                    break
                        
                        self.lines[true_index].hard = False

                        for ii in range(len(self.units)):
                            unit = self.units[ii]
                            lines = unit.getCrease()
                            for jj in range(len(lines)):
                                crease = lines[jj]
                                if (distance(self.lines[true_index][START], crease[START]) < 1e-5 and distance(self.lines[true_index][END], crease[END]) < 1e-5) or \
                                    (distance(self.lines[true_index][START], crease[END]) < 1e-5 and distance(self.lines[true_index][END], crease[START]) < 1e-5):
                                        self.unit_bias_list[ii][jj] = None
                    break
                
            for i in range(len(self.strings)):
                if self.strings[i].type != PASS:
                    if pointOnCrease([self.real_x, self.real_y], Crease(self.strings[i].start_point, self.strings[i].end_point, BORDER), 2):
                        self.choose_string_id = self.strings[i].id
                        break

    def mouseMoveEvent_panel(self, event) -> None:
        """
        @ function: move the pixmap axis
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230107
        @ spec: None
        """
        if self.enable_moving:
            x = event.x()
            y = event.y()
            self.pixel_bias[0] = self.old_pixel_bias_x + x - self.initial_x
            self.pixel_bias[1] = self.old_pixel_bias_y + y - self.initial_y
            self.cursor_x = event.x()
            self.cursor_y = event.y()
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )

    def mousePressEvent_panel(self, event) -> None:
        """
        @ function: move the pixmap axis
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230107
        @ spec: None
        """
        if self.state == self.DESIGN_FINISH:
            x = event.x()
            y = event.y()
            self.cursor_x = x
            self.cursor_y = y
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )
            if event.button() == Qt.RightButton:
                if self.add_hole_mode and not self.add_string_mode:
                    self.addHoleToUnit(self.cursor_x, self.cursor_y)
                elif self.add_string_mode:
                    unit_id = self.pointInUnit([self.real_x, self.real_y])
                    if unit_id != None:
                        self.addStringPoint(self.real_x, self.real_y, unit_id, 'B', True)
                    else:
                        self.updateMessage("Invalid threading method. The threading point is not in some unit.")
                else:
                    if (self.cursor_x > 0 and self.cursor_x < self.pixmap_length and self.cursor_y > 0 and self.cursor_y < self.pixmap_width):
                        self.initial_x = x
                        self.initial_y = y
                        self.old_pixel_bias_x = self.pixel_bias[0]
                        self.old_pixel_bias_y = self.pixel_bias[1]
                        self.enable_moving = True

            else:
                if self.add_string_mode:
                    unit_id = self.pointInUnit([self.real_x, self.real_y])
                    if unit_id != None:
                        self.addStringPoint(self.real_x, self.real_y, unit_id, 'B', False)
                    else:
                        self.updateMessage("Invalid threading method. The threading point is not in some unit.")
                else:
                    if (self.cursor_x > 0 and self.cursor_x < self.pixmap_length and self.cursor_y > 0 and self.cursor_y < self.pixmap_width):
                        self.initial_x = x
                        self.initial_y = y
                        self.old_pixel_bias_x = self.pixel_bias[0]
                        self.old_pixel_bias_y = self.pixel_bias[1]
                        self.enable_moving = True

    def mouseReleaseEvent_panel(self, event) -> None:
        self.enable_moving = False

    def newFile(self):
        if len(self.storage) == 0:
            self.designer = OrigamiDesigner(src=[], paper_info=self.paper_info, design_info=self.design_info)
            self.updateState("Succeeded to create new file", self.IMPORT_SUCCESS)
            self.updateMessage("Please add some origami...")
            self.enable_design = True
            self.initialize()
        else:
            reply = QMessageBox.question(self, "New file option", "Do you want to replace current design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.designer = OrigamiDesigner(src=[], paper_info=self.paper_info, design_info=self.design_info)
                self.storage.clear()
                self.updateState("Succeeded to create new file", self.IMPORT_SUCCESS)
                self.updateMessage("Please add some origami...")
                self.enable_design = True
                self.initialize()

    def onAddHoleMode(self):
        """
        @ function: add hole to units
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230227
        @ spec: None
        """
        self.add_hole_mode = self.checkBox_add_hole_mode.isChecked()
        if self.add_hole_mode:
            self.add_hole_mode = False
            ok = 1
            self.bias_val = self.doubleSpinBox_bias_val.value()
            if ok:
                self.add_hole_mode = True
                self.show_additional_crease = True
                self.updateMessage("Add-hole mode enabled...")
                # self.doubleSpinBox_bias_val.setEnabled(False)
            else:
                self.add_hole_mode = False
                self.checkBox_add_hole_mode.setChecked(False)
                # self.doubleSpinBox_bias_val.setEnabled(True)
        else:
            if self.expert_mode:
                self.expert_mode = False
                self.choose_unit_id = 0
                self.choose_crease_id = 0
                self.widget.setVisible(False)
            self.updateMessage("Add-hole mode disabled...")
            # self.doubleSpinBox_bias_val.setEnabled(True)
        self.enable_design = True
    
    def onAddConnection(self):
        self.enable_connection = self.checkBox_connection.isChecked()
        self.enable_design = True

    def onAddStringMode(self):
        self.add_string_mode = self.checkBox_add_string_mode.isChecked()
        if self.add_string_mode:
            self.startAddString()
        else:
            self.endAddString()
        # self.enable_design = True

    def onDesign(self):
        self.design()
        self.updateState("Design finished with success", self.DESIGN_FINISH)
        self.updateMessage("Design mode enabled...")

    def onDesignThreadingMethod(self):
        if self.threading_design_process is not None:
            self.updateMessage("A threading design simulation is already running...")
            return

        # --- Threading Design Automation: Settings Dialog ---
        dialog = ThreadingDesignDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.threading_design_settings = dialog.get_settings()
            # Print all settings to console
            print("=" * 60)
            print("[Threading Design Settings]")
            print("=" * 60)
            for key, value in self.threading_design_settings.items():
                print(f"  {key}: {value}")
            print("=" * 60)
        else:
            self.updateMessage("Threading design cancelled.")
            return

        # === Step 1: Convert gravity direction (F) to gravity vector ===
        s = self.threading_design_settings
        gravity_str = s.get("gravity_direction", "0")
        GRAVITY_MAP = {
            "+x": [9810.0, 0.0, 0.0],
            "-x": [-9810.0, 0.0, 0.0],
            "+y": [0.0, 9810.0, 0.0],
            "-y": [0.0, -9810.0, 0.0],
            "+z": [0.0, 0.0, 9810.0],
            "-z": [0.0, 0.0, -9810.0],
            "0":  [0.0, 0.0, 0.0],
        }
        gravity_vec = GRAVITY_MAP.get(gravity_str, [0.0, 0.0, 0.0])
        gravity_str_csv = ",".join(str(v) for v in gravity_vec)
        print(f"[Step 1] Gravity direction '{gravity_str}' -> vector {gravity_vec}")

        origami_name = s["origami_name"]
        height = s["structure_height"]
        control_mode = s["control_mode"]
        material = s["material_type"]

        # === Step 2: Export current design to JSON for the simulator ===
        # The simulator reads ./descriptionData/<origami_name>.json in start().
        json_path = f"./descriptionData/{origami_name}.json"

        # Check if description file already exists — offer to reuse, re-extract, or cancel
        if os.path.exists(json_path):
            reply = QMessageBox.question(
                self,
                "Description File Exists",
                f"System description file already exists:\n  {json_path}\n\n"
                f"How do you want to proceed?\n\n"
                f"  Yes     = Overwrite + re-run FOLD_SIM (uses current design, slower)\n"
                f"  No      = Reuse existing file, skip to threading search (faster)\n"
                f"  Cancel  = Go back to settings dialog",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No,
            )
            if reply == QMessageBox.Cancel:
                print("[Step 2] User cancelled — returning to settings dialog.")
                self.updateMessage("Threading design cancelled.")
                return
            if reply == QMessageBox.No:
                print(f"[Step 2] Reusing existing description file '{json_path}', skipping FOLD_SIM.")
                self.updateMessage(
                    f"Reusing existing system features for '{origami_name}', "
                    f"launching threading search... (Stop: Simulation → Stop Thread)"
                )
                self._start_trainer(origami_name, s)
                return
            print(f"[Step 2] Overwriting existing description file '{json_path}'...")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        self.exportDescriptionData(json_path)
        print(f"[Step 2] Exported design data to '{json_path}'")

        # === Step 3: Invoke FOLD_SIM via phys_sim25.py subprocess (non-blocking) ===
        # FOLD_SIM always uses zero gravity; the user-selected gravity
        # direction (F) is reserved for subsequent steps.
        fold_sim_gravity = "0,0,0"
        cmd_args = [
            "--origami", origami_name,
            "--height", str(height),
            "--control-mode", str(control_mode),
            "--gravity", fold_sim_gravity,
            "--material", str(material),
        ]
        if control_mode == 1:
            cmd_args += ["--stroke-percent", str(s["stroke_percent"])]

        print(f"[Step 3] Extracting system features via FOLD_SIM (gravity={fold_sim_gravity})")

        self.updateMessage(f"Extracting system features for '{origami_name}' via FOLD_SIM ({'structure folding' if control_mode == 0 else 'robot actuation'})..."
                          " (Stop: Simulation → Stop Thread)")

        # Use QProcess for non-blocking execution
        self.threading_design_process = QProcess(self)
        self.threading_design_process.setWorkingDirectory(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.threading_design_process.setProcessChannelMode(QProcess.MergedChannels)
        self.threading_design_process._origami_name = origami_name  # store for callback
        self.enable_threading_design = True

        def on_ready_read():
            proc = self.threading_design_process
            if proc is None:
                return
            while proc.bytesAvailable():
                chunk = bytes(proc.readAllStandardOutput()).decode('utf-8', errors='replace')
                if chunk:
                    print(chunk, end='', flush=True)

        def on_finished(exit_code, exit_status):
            proc = self.threading_design_process
            if proc is None:
                return  # already cleaned up by stopThread
            # flush any remaining output
            while proc.bytesAvailable():
                chunk = bytes(proc.readAllStandardOutput()).decode('utf-8', errors='replace')
                if chunk:
                    print(chunk, end='', flush=True)
            print(f"[FOLD_SIM] Exit code: {exit_code}, status: {exit_status}")
            if exit_status == QProcess.NormalExit and exit_code == 0:
                self.updateMessage(f"System features extracted for '{origami_name}' → ./descriptionData/{origami_name}.json")
                # === Step 4: Launch trainer.py for threading design search ===
                self._start_trainer(origami_name, s)
                return  # trainer handles its own cleanup via on_trainer_finished
            elif exit_status == QProcess.NormalExit:
                self.updateMessage(f"FOLD_SIM for '{origami_name}' finished with errors (code: {exit_code}).")
            else:
                self.updateMessage(f"FOLD_SIM for '{origami_name}' was stopped or crashed.")
            self.threading_design_process = None
            self.enable_threading_design = False

        self.threading_design_process.readyReadStandardOutput.connect(on_ready_read)
        self.threading_design_process.finished.connect(on_finished)
        self.threading_design_process.start(sys.executable, ["-u", "phys_sim25.py"] + cmd_args)
        # if self.enable_output_stl:
        #     self.updateMessage("A stl file is being outputed, please wait...")
        #     return
        # if self.enable_cdf_curve_fitting:
        #     self.updateMessage("A cdf process is running, please wait...")
        #     return
        # if self.enable_phys_data_collecting:
        #     self.updateMessage("A physical simulation is running, please wait...")
        #     return
        # if self.enable_mcts:
        #     self.updateMessage("A MCTS Searching process is running, please wait...")
        #     return
        
        # file_path, _ = QFileDialog.getSaveFileName(
        #     self, 
        #     "Specify the filename of the simulation result"
        # ) 

        # if file_path == '':
        #     self.updateState("Cancel outputing mcts results.", self.state)
        #     return
        # else:
        #     if not os.path.exists(file_path):
        #         os.makedirs(file_path)
        # self.string_file_path = file_path

        # self.enable_mcts = True

        # batch_size = 4
        
        # self.exportAsDxf("./dxfResult/phys_sim.dxf")

        # max_edge = 0
        # for ele in self.units:
        #     edge = len(ele.crease)
        #     if edge > max_edge:
        #         max_edge = edge

        # # 获取点和线段信息
        # dxfg = DxfDirectGrabber()
        # dxfg.readFile("./dxfResult/phys_sim.dxf")

        # # 获取折纸单元信息
        # unit_parser = UnitPackParserReverse(
        #                 tsp=[0.0, 0.0],
        #                 kps=dxfg.kps,
        #                 lines=dxfg.lines,
        #                 lines_type=dxfg.lines_type
        #             )

        # unit_parser.setMaximumNumberOfEdgeInAllUnit(max_edge) #For every units, there exists at most 4 edges
        # input_units = unit_parser.getUnits()

        # # calculate max length of view
        # max_size, max_x, max_y = unit_parser.getMaxDistance()
        # total_bias = unit_parser.getTotalBias(units=input_units)

        # # new_lines = []
        # # for i in range(len(self.lines)):
        # #     line = self.lines[i]
        # #     if distance(line[START], line[END]) < 1e-5:
        # #         continue
        # #     not_duplicate = True
        # #     for j in range(i):
        # #         other_line = self.lines[j]
        # #         if (distance(line[START], other_line[START]) < 1e-5 and distance(line[END], other_line[END]) < 1e-5) or (distance(line[START], other_line[END]) < 1e-5 and distance(line[END], other_line[START]) < 1e-5):
        # #             not_duplicate = False
        # #             break
        # #     if not not_duplicate:
        # #         continue
        # #     new_lines.append(line)

        # # 构建蒙特卡洛搜索树
        # mcts = MCTS_Simplified(self.units, unit_parser.new_lines, self.kps, self.pref_pack['tsa_radius'], self.pref_pack["tsa_resolution"], origami_size = [self.origami_length, self.origami_width], string_number=2, generation=self.limitation["mcts_epoch"])
        
        # if not self.pref_pack['debug_mode']:
        #     self.mcts_thread = MCTSThread(mcts, batch_size, [self.origami_length, self.origami_width], self.pref_pack, self.limitation, self.units, max_edge, input_units, max_size, total_bias, file_path)
        #     self.mcts_thread._emit.connect(self.drawProcess)
        #     self.mcts_thread.start()
        #     return
        # else:
        #     scores = []
        #     for i in range(self.limitation["mcts_epoch"]):
        #         methods, initial_method = mcts.ask(batch_size, i)

        #         try:
        #             with open(os.path.join(self.file_path, "current.json"), 'w', encoding="utf-8") as f:
        #                 json.dump(initial_method[0], f, indent=4)
        #         except:
        #             pass

        #         print("Debug mode, using 1 process")
        #         reward_list = [0.0 for _ in range(len(methods))]

        #         ori_sim = OrigamiSimulator(use_gui=False)

        #         ori_sim.string_total_information = methodToTotalInformation(methods[0], mcts.P_points, mcts.O_points)
        #         ori_sim.pref_pack = self.pref_pack
        #         ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
        #         ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
        #         ori_sim.initializeRunning()

        #         for j in range(len(methods)):
        #             reward_list[j] = (len(methods[j]["id"][0]) + len(methods[j]["id"][1])) / 18. + (methods[j]["id"][0][0] + methods[j]["id"][1][0]) / 48.

        #         mcts.tell(reward_list)

        #         maximum_reward = max(reward_list)
        #         maximum_reward_index = reward_list.index(maximum_reward)
        #         scores.append(maximum_reward)

        #         print("Epoch: " + str(i) + ", Max Value: " + str(maximum_reward) + ", Total batch size: " + str(len(reward_list)))

        #         best_method = methods[maximum_reward_index]
        #         total_string = deepcopy(best_method)
        #         total_string["score"] = maximum_reward
                
        #         try:
        #             with open(os.path.join(file_path, "result_epoch_" + str(i) + "_score_" + str(round(maximum_reward, 2))) + ".json", 'w', encoding="utf-8") as f:
        #                 json.dump(total_string, f, indent=4)
        #         except:
        #             pass
                
        #         score_list = {
        #             "score": scores
        #         }

        #         try:
        #             with open(os.path.join(file_path, "score.json"), 'w', encoding="utf-8") as f:
        #                 json.dump(score_list, f, indent=4)
        #         except:
        #             pass
            
        #     print("END TRAINING!")
        #     self.enable_mcts = False

    def _start_trainer(self, origami_name, settings):
        """
        Step 4: Launch trainer.py to search for threading strategies
        using the system features extracted by FOLD_SIM.

        Maps dialog settings (A-Q) to trainer.py CLI flags:
          A: origami_name               → -o
          B: min_tendon_count           → -s
          C: structure_height           → -height
          D: control_mode               → -ctm
          E: ground_enabled             → -ground
          F: gravity_flag               → -gravity  (flag 0-6, no vector conversion)
          G: simulation_time            → -sim-time / -sim-upper
          H: thread_count               → -t
          I: extra_tendon_length        → -add-len
          J: material_type              → -robot-type
          K: ground_friction            → -miu
          L: controller_type            → -ctrl-type
          M: search_mode                → -search-mode
          N: mask_crease_type           → -arbi
          O: ea_constraint_init         → -ea-init
          P: ea_constraint_final        → -ea-final
          Q: enable_work_criteria       → -work
        """
        # --- Search params ---
        tendon_count = settings.get("min_tendon_count", 1)
        thread_count = settings.get("thread_count", 4)
        search_mode = settings.get("search_mode", 0)
        mask_crease = settings.get("mask_crease_type", 0)
        ea_init = settings.get("ea_constraint_initial_segment", 1)
        ea_final = settings.get("ea_constraint_final_tip", 1)
        work_criteria = settings.get("enable_work_criteria", 1)

        # --- Simulation params ---
        height = settings.get("structure_height", 1.0)
        control_mode = settings.get("control_mode", 0)
        ground_enable = settings.get("ground_enabled", 0)
        gravity_flag = settings.get("gravity_flag", 0)
        sim_time = settings.get("simulation_time", 20.0)
        sim_upper_time = sim_time + 5.0  # G2 = G1 + 5
        extra_tendon_len = settings.get("extra_tendon_length", 0.0)
        robot_type = settings.get("material_type", 0)
        ground_friction = settings.get("ground_friction", 0.3)
        controller_type = settings.get("controller_type", 1)
        stroke_percent = settings.get("stroke_percent", 0.75)

        cmd_args = [
            "-o", origami_name,                    # A
            "-s", str(tendon_count),               # B
            "-t", str(thread_count),               # H
            "-print", "0",
            # Simulation config overrides
            "-height", str(height),                # C
            "-ctm", str(control_mode),             # D
            "-ground", str(ground_enable),         # E
            "-gravity", str(gravity_flag),         # F (flag only)
            "-sim-time", str(sim_time),            # G1
            "-sim-upper", str(sim_upper_time),     # G2
            "-add-len", str(extra_tendon_len),     # I
            "-robot-type", str(robot_type),        # J
            "-miu", str(ground_friction),          # K
            "-ctrl-type", str(controller_type),    # L
            # Search config
            "-search-mode", str(search_mode),      # M
            "-arbi", str(mask_crease),             # N
            "-ea-init", str(ea_init),              # O
            "-ea-final", str(ea_final),            # P
            "-work", str(work_criteria),           # Q
        ]
        if control_mode == 1:
            cmd_args += ["-stroke", str(stroke_percent)]   # S

        print(f"[Step 4] Launching trainer.py for '{origami_name}' "
              f"(tendons={tendon_count}, threads={thread_count}, search={search_mode}, "
              f"height={height}, ctm={control_mode}, ground={ground_enable}, "
              f"gravity={gravity_flag}, sim_t={sim_time}, sim_upper={sim_upper_time}, "
              f"add_len={extra_tendon_len}, robot_t={robot_type}, miu={ground_friction}, "
              f"ctrl_t={controller_type}, mask={mask_crease}, "
              f"ea_init={ea_init}, ea_final={ea_final}, work={work_criteria}, stroke={stroke_percent})")

        self.updateMessage(f"Searching threading strategies for '{origami_name}' via MCTS..."
                          " (Stop: Simulation → Stop Thread)")

        self.threading_design_process = QProcess(self)
        self.threading_design_process.setWorkingDirectory(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.threading_design_process.setProcessChannelMode(QProcess.MergedChannels)
        self.enable_threading_design = True

        def on_trainer_ready_read():
            proc = self.threading_design_process
            if proc is None:
                return
            while proc.bytesAvailable():
                chunk = bytes(proc.readAllStandardOutput()).decode('utf-8', errors='replace')
                if chunk:
                    print(chunk, end='', flush=True)

        def on_trainer_finished(exit_code, exit_status):
            proc = self.threading_design_process
            if proc is None:
                return
            # flush any remaining output
            while proc.bytesAvailable():
                chunk = bytes(proc.readAllStandardOutput()).decode('utf-8', errors='replace')
                if chunk:
                    print(chunk, end='', flush=True)
            print(f"[trainer] Exit code: {exit_code}, status: {exit_status}")
            if exit_status == QProcess.NormalExit and exit_code == 0:
                self.updateMessage(f"Threading search for '{origami_name}' completed → ./threadingResult/")
            elif exit_status == QProcess.NormalExit:
                self.updateMessage(f"Threading search for '{origami_name}' finished with errors (code: {exit_code}).")
            else:
                self.updateMessage(f"Threading search for '{origami_name}' was stopped or crashed.")
            self.threading_design_process = None
            self.enable_threading_design = False

        self.threading_design_process.readyReadStandardOutput.connect(on_trainer_ready_read)
        self.threading_design_process.finished.connect(on_trainer_finished)
        self.threading_design_process.start(sys.executable, ["-u", "trainer.py"] + cmd_args)

    def oneClickAddHoles(self):
        if self.add_hole_mode:
            for u in self.units:
                center = u.getCenter()
                self.addHoleToUnitUsingRealAxis(center[X], center[Y])
            self.updateMessage("Succeed in adding holes in the center of each unit...")
            self.enable_design = True
        else:
            self.updateMessage("Add-hole mode is not enabled, please enable add_hole mode...")

    def openFile(self):
        # self.enable_design = True
        
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a design pack", 
            ".", 
            "Json files (*.json);;Txt files (*.txt);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel opening file", self.state)
        else:
            try:
                
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.initialize()
                self.backup_open_file_path = path
                self.dictToStorage(input_json, self.PACKED_DATA)
                self.updateState("Succeeded to load packed-data file", self.DESIGN_FINISH)
                self.enable_design = True           
            except:
                self.updateState("Failed to open file, check if it is in correct format", self.state, "ERROR")

    def physicalDataCollecting(self):
        pass
        # # --Physical Sim module for simulate the folding process of origami-- #
        # if self.enable_output_stl:
        #     self.updateMessage("A stl file is being outputed, please wait...")
        #     return
        # if self.enable_cdf_curve_fitting:
        #     self.updateMessage("A cdf process is running, please wait...")
        #     return
        # if self.enable_phys_data_collecting:
        #     self.updateMessage("A physical simulation is running, please wait...")
        #     return
        # if self.enable_mcts:
        #     self.updateMessage("A MCTS Searching process is running, please wait...")
        #     return

        # file_path, _ = QFileDialog.getSaveFileName(
        #     self, 
        #     "Specify the filename of the outputed simulation result", 
        #     ".", 
        #     "json files (*.json)"
        # ) 
        # if file_path == '':
        #     self.updateMessage("Cancel exporting physical simulation result")
        # else:
        #     self.enable_phys_data_collecting = True

        #     max_edge = 0
        #     for ele in self.units:
        #         edge = len(ele.crease)
        #         if edge > max_edge:
        #             max_edge = edge

        #     if not self.full_description_mode:
        #         from phys_sim12 import OrigamiSimulator
        #         self.exportAsDxf("./dxfResult/phys_sim.dxf")

        #         # 获取点和线段信息
        #         dxfg = DxfDirectGrabber()
        #         dxfg.readFile("./dxfResult/phys_sim.dxf")

        #         # 获取折纸单元信息
        #         unit_parser = UnitPackParserReverse(
        #                         tsp=[0.0, 0.0],
        #                         kps=dxfg.kps,
        #                         lines=dxfg.lines,
        #                         lines_type=dxfg.lines_type
        #                     )

        #         unit_parser.setMaximumNumberOfEdgeInAllUnit(max_edge) #For every units, there exists at most 4 edges
        #         input_units = unit_parser.getUnits()

        #         # calculate max length of view
        #         max_size, max_x, max_y = unit_parser.getMaxDistance()
        #         total_bias = unit_parser.getTotalBias(units=input_units)

        #         ori_sim = OrigamiSimulator(use_gui=False)

        #         # ori_sim.string_total_information = deepcopy(self.string_total_information)
        #         ori_sim.pref_pack = self.pref_pack
        #         ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
        #         ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
        #         ori_sim.initializeRunning()
        #     else:
        #         from phys_sim18 import OrigamiSimulator
        #         self.exportDescriptionData('./descriptionData/phys_sim.json')
        #         ori_sim = OrigamiSimulator()

        #         # ori_sim.string_total_information = deepcopy(self.string_total_information)
        #         ori_sim.start("phys_sim", max_edge)
        #         ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
        #         ori_sim.initializeRunning()

            
        #     if not self.pref_pack["debug_mode"]:
        #         self.phys_data_collecting_thread = ThreadingMethodSearchingThread(file_path, ori_sim)
        #         self.phys_data_collecting_thread._emit.connect(self.drawProcess)
        #         self.phys_data_collecting_thread.start()
        #     else:
        #         step = 1
        #         while 1:
        #             ori_sim.step()
        #             if ori_sim.stop():
        #                 break
        #             step += 1

        #         all_dis = {
        #             "control_string_decrease": [],
        #             "string_decrease_each": [],
        #             "max_force": [],
        #             "folding_percent": [],
        #             "max_folding_percent": [],
        #             "min_folding_percent": [],
        #             "time": []
        #         }

        #         all_dis["control_string_decrease"] = ori_sim.recorded_string_decrease_length_control
        #         all_dis["string_decrease_each"] = ori_sim.recorded_string_decrease_length
        #         all_dis["folding_percent"] = ori_sim.recorded_folding_percent
        #         all_dis["max_folding_percent"] = ori_sim.recorded_maximum_folding_percent
        #         all_dis["min_folding_percent"] = ori_sim.recorded_minimum_folding_percent
        #         all_dis["max_force"] = ori_sim.recorded_max_force
        #         all_dis["time"] = ori_sim.recorded_t
        #         all_dis["deal"] = 1

        #         with open(self.file_path, 'w', encoding="utf-8") as f:
        #             json.dump(all_dis, f, indent=4)
        #         self.drawProcess(1.0)
        #         self.enable_phys_data_collecting = False

    def physicalSimulation(self):
        pass
        # max_edge = 0
        # for ele in self.units:
        #     edge = len(ele.crease)
        #     if edge > max_edge:
        #         max_edge = edge
        # if not self.full_description_mode:
        #     from phys_sim12 import OrigamiSimulator
        #     self.exportAsDxf("./dxfResult/phys_sim.dxf")

        #     ori_sim = OrigamiSimulator(use_gui=True, debug_mode=self.pref_pack['debug_mode'])
        #     # set string parameters
        #     # ori_sim.strings = deepcopy(self.strings)
        #     ori_sim.string_total_information = deepcopy(self.string_total_information)
            
        #     ori_sim.start("phys_sim", max_edge, ori_sim.FOLD_SIM)

        #     ori_sim.run()
        # else:
        from phys_sim25 import OrigamiSimulator
        self.exportDescriptionData('./descriptionData/phys_sim.json')
        ori_sim = OrigamiSimulator(use_gui=True, debug_mode=False, fast_simulation=self.pref_pack["fast_simulation_mode"], check_connection_matrix=False, g=[0., 0., 0.])
        # ori_sim.string_total_information = deepcopy(self.string_total_information)
        ori_sim.start("phys_sim", 4, ori_sim.FOLD_SIM)
        ori_sim.run(False, False)
        ori_sim.window.destroy()

    def physicalSimulationExplicit(self):
        pass

    def plotJsonReadFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 
            "Choose a score pack", 
            ".", 
            "All Files (*.*);;Json files (*.json);;Csv files (*.csv)"
        )
        if path == '':
            return False, {}
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                input_json["sim"] = 1
                input_json["real"] = 0
                return True, input_json
            except:
                #csv?
                import csv

                time_data = []
                displacement_data = []  # 对应第二列位移数据
                force_data = []         # 对应第三列力数据
                recorded_movement_x = []
                recorded_movement_y = []
                recorded_movement_z = []
                recorded_indices = []
            
                # 打开CSV文件
                try:
                    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        
                        for row in reader:
                            # 跳过空行
                            if not row:
                                continue
                            
                            if (not row) and len(time_data):
                                break
                                
                            # 确保行有至少3列数据
                            if len(row) >= 4:
                                # 提取三列数据
                                if row[1] == '':
                                    break
                                time = float(row[1])
                                displacement = float(row[2])
                                force = float(row[3])
                                
                                if (displacement >= (displacement_data[-1] if len(displacement_data) else 0)) or time < 1e-3:
                                    time_data.append(time)
                                    displacement_data.append(displacement)
                                    force_data.append(force)
                            elif len(row) == 3:
                                # 提取三列数据
                                if row[0] == '':
                                    break
                                time = float(row[0])
                                displacement = float(row[1])
                                force = float(row[2])
                                
                                if (displacement >= (displacement_data[-1] if len(displacement_data) else 0)) or time < 1e-3:
                                    time_data.append(time)
                                    displacement_data.append(displacement)
                                    force_data.append(force)
                            else:
                                break
                                    
                except Exception as e:
                    print(f"Error reading CSV file: {e} Maybe it is robot data format?")
                    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        
                        for row in reader:
                            # 跳过空行
                            if not row or ('nx' == row[3].lower()):
                                continue
                            
                            if (not row) and len(time_data):
                                break
                                
                            # 确保行有至少3列数据
                            if len(row) >= 4:
                                # 提取三列数据
                                if row[1] == '':
                                    break
                                time = float(row[0])
                                x = float(row[3])
                                y = float(row[4])
                                
                                time_data.append(time)
                                recorded_movement_x.append(x)
                                recorded_movement_y.append(y)

                            else:
                                break
                
                # 构建返回字典
                input_json = {
                    'time': time_data,
                    'control_string_decrease': [displacement_data],
                    'max_force': force_data,
                    "recorded_movement_x": recorded_movement_x,
                    "recorded_movement_y": recorded_movement_y,
                    "recorded_movement_z": recorded_movement_z,
                    "recorded_indices": recorded_indices,
                    "deal": 1
                }
                input_json["real"] = 1
                input_json["sim"] = 0
                return True, input_json

    def plotJson(self):
        plt.rcParams.update({'font.size': 16})
        plt.rcParams['font.sans-serif'] = 'Arial'
        
        data_number = 0
        real_data_number = 0
        maximum_force = 0.
        max_time = 0.0
        history_max_11 = 0
        
        x_sim_total = []
        x_exp_total = []
        
        rgb_sim = []
        rgb_real = []
        
        while 1:
            ok, data = self.plotJsonReadFile()
            if not ok:
                data_number += 1
                if data_number >= 3:
                    break
                continue
            else:
                if data["sim"] == 1:
                    data_number += 1
                else:
                    real_data_number += 1
                subfix = (("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")) if data["sim"] == 1 else ("Exp. " + str(real_data_number)))
                rgb = (np.random.random(), np.random.random(), np.random.random())
                
                if data["sim"] == 1:
                    if data_number == 1:
                        rgb = (0.2, 0.13, 0.5)
                    elif data_number == 2:
                        rgb = (0.75, 0.55, 0.88)
                    elif data_number == 3:
                        rgb = (0.27, 0.67, 0.6)
                else:
                    if real_data_number == 1:
                        rgb = (0., 0., 0.)
                    elif real_data_number == 2:
                        rgb = (0.7, 0.46, 0.16)
                    elif real_data_number == 3:
                        rgb = (0.05, 0.25, 0.52)
                
                if data['sim'] == 1:
                    x_sim_total.append(data)
                    rgb_sim.append(rgb)
                else:
                    x_exp_total.append(data)
                    rgb_real.append(rgb)
                    
                # # 0-0 folding-percent with current_time
                # ax[0][0].set_xmargin(0)
                # ax[0][0].set_ymargin(0)
                # ax[0][0].spines['top'].set_visible(False)
                # ax[0][0].spines['right'].set_visible(False)

                # current_max_x = max(data["time"])
                # if current_max_x > max_time:
                #     max_time = current_max_x
                   
                # if data["sim"] == 1:  
                #     error_mode = 1
                #     if not error_mode:
                #         x = data["time"]
                #         y = data["folding_percent"]
                #         y_upperbound = data["max_folding_percent"]
                #         y_lowerbound = data["min_folding_percent"]
                        
                #         ax[0][0].plot(x, y, color=rgb, linewidth=2, label='Average Error - Sim. ' + str(data_number))
                #         ax[0][0].fill_between(x, y_upperbound, y_lowerbound, alpha=0.25, facecolor=rgb, label='Bound of folding percent - ' + subfix)

                #         ax[0][0].set_ylim(-1, 1)
                #         ax[0][0].set_xlabel("Time (s)", fontsize=16)
                #         ax[0][0].set_ylabel("Folding percent", fontsize=16)
                #         ax[0][0].tick_params(labelsize=16)

                #         ax[0][0].set_title("Folding percent - Time", fontsize=20)
                #         ax[0][0].legend()
                #     else:
                #         x = data["time"]
                #         y = data["folding_error"]
                #         y_upperbound = data["max_folding_error"]
                #         y_lowerbound = data["min_folding_error"]

                #         ax[0][0].plot(x, y, color=rgb, linewidth=2, label='Average error - Sim. ' + str(data_number))
                #         ax[0][0].fill_between(x, y_upperbound, y_lowerbound, alpha=0.25, facecolor=rgb, label='Range of error - ' + subfix)

                        
                #         ax[0][0].set_xlim(0, max_time * 1.1)
                        
                #         ax[0][0].set_ylim(0, max(max(y) * 1.1, np.pi))
                #         ax[0][0].set_xlabel("Time (s)", fontsize=16)
                #         ax[0][0].set_ylabel("Folding angle error", fontsize=16)
                #         ax[0][0].tick_params(labelsize=16)

                #         ax[0][0].set_title("Folding angle error - Time", fontsize=20)
                #         ax[0][0].legend(loc='upper right')

                # # 0-1 max_force with current_time
                # ax[0][1].set_xmargin(0)
                # ax[0][1].set_ymargin(0)
                # ax[0][1].spines['top'].set_visible(False)
                # ax[0][1].spines['right'].set_visible(False)

                # x = data["time"]
                
                # if data["sim"] == 1:
                #     y = data["max_force"]
                #     y_each = data["string_force_each"]
                #     y_total = np.zeros_like(np.array(y))
                #     for i in range(len(y_each)):
                #         ele = y_each[i]
                #         new_rgb = [rgb[0] + 0.2 * i, rgb[1] + 0.1428 * i, rgb[2] + 0.33 * i]
                #         while new_rgb[0] > 1:
                #             new_rgb[0] -= 1
                #         while new_rgb[0] < 0:
                #             new_rgb[0] += 1
                #         while new_rgb[1] > 1:
                #             new_rgb[1] -= 1
                #         while new_rgb[1] < 0:
                #             new_rgb[1] += 1
                #         while new_rgb[2] > 1:
                #             new_rgb[2] -= 1
                #         while new_rgb[2] < 0:
                #             new_rgb[2] += 1
                #         if len(data["recorded_movement_x"]) == 0:
                #             ax[0][1].plot(x, ele, color=new_rgb, linewidth=2, label=subfix + ' String #' + str(i + 1))
                #         y_total += np.array(ele)
                # else:
                #     y_total = data["max_force"]
                    
                # if len(data["recorded_movement_x"]) == 0:
                #     # ax[0][1].plot(x, y, color=rgb, linewidth=2, label='Upperbound - Sim. ' + str(data_number))
                #     if data["sim"] == 1:
                #         if max(y) * 1.1 > maximum_force:
                #             ax[0][1].set_ylim(0, max(y) * 1.1)
                #             maximum_force = max(y) * 1.1

                #         ax[0][1].set_xlim(0, max_time * 1.1)
                #         ax[0][1].set_xlabel("Time (s)", fontsize=16)
                #         ax[0][1].set_ylabel("String tension (N)", fontsize=16)
                #         ax[0][1].tick_params(labelsize=16)

                #         ax[0][1].set_title("String tension - Time", fontsize=20)
                #         ax[0][1].legend()
                # else:
                #     ax[0][1].set_xmargin(0)
                #     ax[0][1].set_ymargin(0)
                #     ax[0][1].spines['top'].set_visible(False)
                #     ax[0][1].spines['right'].set_visible(False)
                #     ax[0][1].plot(x, y_total, color=rgb, linewidth=2, label=subfix)
                #     ax[0][1].set_xlabel("Time (s)", fontsize=16)
                #     ax[0][1].set_ylabel("Sum of string tension (N)", fontsize=16)
                #     ax[0][1].tick_params(labelsize=16)
                #     ax[0][1].set_xlim(0, max_time * 1.1)

                #     ax[0][1].set_title("Sum of string tension - Time", fontsize=20)
                #     ax[0][1].legend()

                # # 1-0 control_decrease with each_decrease
                # ax[1][0].set_xmargin(0)
                # ax[1][0].set_ymargin(0)
                # ax[1][0].spines['top'].set_visible(False)
                # ax[1][0].spines['right'].set_visible(False)

                # x = data["time"]
                # y_standard = data["control_string_decrease"]
                # # y_each = data["string_decrease_each"]

                # try:
                #     for i in range(len(y_standard)):
                #         ax[1][0].plot(x, y_standard[i], color=rgb, linewidth=2, label=('Control - ' + subfix) if i == 0 else None)
                #         # ax[1][0].plot(x, y_each[i], color=rgb, linewidth=1, linestyle='--', label='Real - Sim. ' + str(data_number) if i == 0 else None)
                #         ax[1][0].set_ylim(0, max(y_standard[i]) * 1.1)
                    
                #     ax[1][0].set_xlim(0, max_time * 1.1)
                #     ax[1][0].set_xlabel("Time (s)", fontsize=16)
                #     ax[1][0].set_ylabel("String length contraction (mm)", fontsize=16)
                #     ax[1][0].tick_params(labelsize=16)

                #     ax[1][0].set_title("String length contraction - Time", fontsize=20)
                #     ax[1][0].legend()
                # except:
                #     pass

                # # 1-1 Folding speed
                # if len(data["recorded_movement_x"]):
                #     x = data["recorded_movement_x"]
                #     y = data["recorded_movement_y"]
                #     ax[1][1].scatter(x, y, marker='o', label=subfix, color=rgb, s=4, alpha=0.2)
                #     # ax[1][1].set_xmargin(0)
                #     # ax[1][1].set_ymargin(0)
                #     # ax[1][1].spines['top'].set_visible(False)
                #     # ax[1][1].spines['right'].set_visible(False)

                #     # x = data["control_string_decrease"]
                #     # y = data["folding_percent"]

                #     # ax[1][1].plot(x, y, color=rgb, linewidth=2, label='Exp. ' + str(data_number))

                #     # ax[1][1].set_ylim(-1, 1)
                #     ax[1][1].set_xlabel("X (mm)", fontsize=12)
                #     ax[1][1].set_ylabel("Y (mm)", fontsize=12)
                #     ax[1][1].tick_params(labelsize=12)
                #     ax[1][1].set_aspect(1)

                #     ax[1][1].set_title("Tracking point movement", fontsize=16)
                #     # ax[1][1].legend()
                # else:
                #     ax[1][1].set_xmargin(0)
                #     ax[1][1].set_ymargin(0)
                #     ax[1][1].spines['top'].set_visible(False)
                #     ax[1][1].spines['right'].set_visible(False)
                    
                #     pointer_list = [0]
                #     for k in range(1, len(x)):
                #         if x[k] < 1e-3:
                #             pointer_list.append(k)
                #     pointer_list.append(len(x))
                    
                #     for k in range(len(pointer_list) - 1):
                #         start = pointer_list[k]
                #         end = pointer_list[k + 1]
                #         ax[1][1].plot(x[start: end], y_total[start: end], color=rgb, linewidth=2 if data["sim"] else 1, label=subfix if k == 0 else None, linestyle='--' if not data["sim"] else '-')
                        
                #     ax[1][1].set_xlabel("Time (s)", fontsize=16)
                #     ax[1][1].set_ylabel("Sum of string tension (N)", fontsize=16)
                #     ax[1][1].tick_params(labelsize=16)
                #     ax[1][1].set_xlim(0, max_time * 1.1)
                #     ax[1][1].set_ylim(0, max(max(y_total), history_max_11) * 1.1)

                #     ax[1][1].set_title("Sum of string tension - Time", fontsize=20)
                #     ax[1][1].legend()
                    
                #     if max(y_total) > history_max_11:
                #         history_max_11 = max(y_total)
            
            if data_number == 60:
                break

        if data_number == 0:
            return
        else:
            TITLE = 32
            NORMAL = 32
            LINE_WIDTH = 2
            PAD = 12
            LABELPAD = 12

            # 1
            fig, ax = plt.subplots(figsize=(10, 9))
            for i in range(len(x_sim_total)):
                data = x_sim_total[i]
                data_number = i + 1
                ax.set_xmargin(0)
                ax.set_ymargin(0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(LINE_WIDTH)

                current_max_x = max(data["time"])
                if current_max_x > max_time:
                    max_time = current_max_x
                    
                error_mode = 1
                if not error_mode:
                    x = data["time"]
                    y = data["folding_percent"]
                    y_upperbound = data["max_folding_percent"]
                    y_lowerbound = data["min_folding_percent"]
                    
                    ax.plot(x, y, color=rgb_sim[i], linewidth=2, label='Average Error - Sim. ' + (str(data_number) if len(x_sim_total) > 1 else ""))
                    ax.fill_between(x, y_upperbound, y_lowerbound, alpha=0.2, facecolor=rgb_sim[i], label='Bound of folding percent - Sim. ' + (str(data_number) if len(x_sim_total) > 1 else ""))

                    ax.set_ylim(-1, 1)
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Folding percent", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)

                    # ax.set_title("Folding percent - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                else:
                    x = data["time"]
                    y = data["folding_error"]
                    y_upperbound = data["max_folding_error"]
                    y_lowerbound = data["min_folding_error"]

                    ax.plot(x, y, color=rgb_sim[i], linewidth=2, label='Average error - Sim. ' + (str(data_number) if len(x_sim_total) > 1 else ""))
                    ax.fill_between(x, y_upperbound, y_lowerbound, alpha=0.2, facecolor=rgb_sim[i], label='Range of error - Sim. ' + (str(data_number) if len(x_sim_total) > 1 else ""))

                    
                    ax.set_xlim(0, max_time * 1.1)
                    
                    ax.set_ylim(0, max(max(y) * 1.1, np.pi))
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Folding angle error (rad)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    # ax.set_title("Folding angle error - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.99, 1.03), loc="upper right")
            # fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
            plt.tight_layout()
            plt.show()
            # 2
            fig, ax = plt.subplots(figsize=(10, 9))
            for i in range(len(x_sim_total)):
                data = x_sim_total[i]
                data_number = i + 1
                ax.set_xmargin(0)
                ax.set_ymargin(0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(LINE_WIDTH)

                x = data["time"]
                
                current_max_x = max(data["time"])
                if current_max_x > max_time:
                    max_time = current_max_x
                
                if data["sim"] == 1:
                    y = data["max_force"]
                    y_each = data["string_force_each"]
                    y_total = np.zeros_like(np.array(y))
                    for j in range(len(y_each)):
                        ele = y_each[j]
                        new_rgb = [rgb_sim[i][0] + 0.2 * j, rgb_sim[i][1] + 0.1428 * j, rgb_sim[i][2] + 0.5 * j]
                        while new_rgb[0] > 1:
                            new_rgb[0] -= 1
                        while new_rgb[0] < 0:
                            new_rgb[0] += 1
                        while new_rgb[1] > 1:
                            new_rgb[1] -= 1
                        while new_rgb[1] < 0:
                            new_rgb[1] += 1
                        while new_rgb[2] > 1:
                            new_rgb[2] -= 1
                        while new_rgb[2] < 0:
                            new_rgb[2] += 1
                        if len(data["recorded_movement_x"]) == 0:
                            ax.plot(x, ele, color=new_rgb, linewidth=2, label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")) + ' Tendon #' + str(j + 1))
                        y_total += np.array(ele)
                    x_sim_total[i]["max_force"] = y_total
                else:
                    y_total = data["max_force"]
                    
                if len(data["recorded_movement_x"]) == 0:
                    # ax.plot(x, y, color=rgb_sim[i], linewidth=2, label='Upperbound - Sim. ' + str(data_number))
                    if data["sim"] == 1:
                        if max(y) * 1.5 > maximum_force:
                            ax.set_ylim(0, max(y) * 1.5)
                            maximum_force = max(y) * 1.5

                        ax.set_xlim(0, max_time * 1.1)
                        ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                        ax.set_ylabel("Tendon tension (N)", fontsize=NORMAL, labelpad=LABELPAD)
                        ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                        # ax.set_title("String tension - Time", fontsize=TITLE, fontweight='bold')
                        ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                else:
                    ax.set_xmargin(0)
                    ax.set_ymargin(0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(LINE_WIDTH)
                    ax.plot(x, y_total, color=rgb_sim[i], linewidth=2, label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")))
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Sum of tendon tension (N)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                    ax.set_xlim(0, max_time * 1.1)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    # ax.set_title("Sum of string tension - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
            # fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
            plt.tight_layout()
            plt.show()
            
            # 3
            fig, ax = plt.subplots(figsize=(10, 9))
            for i in range(len(x_sim_total)):
                data = x_sim_total[i]
                data_number = i + 1
                ax.set_xmargin(0)
                ax.set_ymargin(0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(LINE_WIDTH)
                
                x = data["time"]
                y_standard = data["control_string_decrease"]
                # y_each = data["string_decrease_each"]
                current_max_x = max(data["time"])
                if current_max_x > max_time:
                    max_time = current_max_x

                try:
                    y_max = 0.0
                    for j in range(len(y_standard)):
                        ax.plot(x, y_standard[j], color=rgb_sim[i], linewidth=2, label=('Control - ' + ("Sim. " + (str(data_number) if len(x_sim_total) > 1 else ""))) if i == 0 else None)
                        # ax[1][0].plot(x, y_each[i], color=rgb_sim[i], linewidth=1, linestyle='--', label='Real - Sim. ' + str(data_number) if i == 0 else None)
                        if max(y_standard[j]) * 1.1 > y_max:
                            y_max = max(y_standard[j]) * 1.1
                            ax.set_ylim(0, y_max)
                    
                    ax.set_xlim(0, max_time * 1.1)
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Tendon length contraction (mm)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)

                    # ax.set_title("String length contraction - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                except:
                    pass
            # fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
            plt.tight_layout()
            plt.show()
            
            # 4
            op = 0
            fig, ax = plt.subplots(figsize=(10, 9))
            for spine in ax.spines.values():
                spine.set_linewidth(LINE_WIDTH)

            for i in range(len(x_sim_total)):
                data = x_sim_total[i]
                data_number = i + 1
                if len(data["recorded_movement_x"]):
                    if not op:      
                        ax.scatter(0, 0, marker='*', label="Origin", color='r', s=288, alpha=1)
                        ax.set_xticks(np.arange(-350, 350, 50))
                        ax.set_yticks(np.arange(-350, 350, 50))
                        op = 1
                    x = data["recorded_movement_x"]
                    y = data["recorded_movement_y"]
                    t = data["time"]
                    for idx in range(len(t)):
                        # if t[idx] > 11.063:
                        if t[idx] > 37.185:
                        # if t[idx] > 13.87:
                            break
                    x = x[:idx:10]
                    y = y[:idx:10]
                    # ax.scatter(x, y, marker='.', label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")), color=rgb_sim[i], s=64, alpha=0.85)
                    ax.plot(x, y, linewidth=2.5, label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")), color=rgb_sim[i], alpha=0.85)
                    ax.set_xlabel("X (mm)", fontsize=NORMAL-8, labelpad=LABELPAD)
                    ax.set_ylabel("Y (mm)", fontsize=NORMAL-8, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL-8, pad=PAD, width=LINE_WIDTH)
                    ax.set_aspect(1)
                    # ax.grid()
                    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
                    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)   
                    
                    # ax.set_title("Moving trajectory of the robot", fontsize=TITLE, fontweight='bold', pad=10)
                    ax.legend(fontsize=NORMAL - 12)
                else:
                    ax.set_xmargin(0)
                    ax.set_ymargin(0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(LINE_WIDTH)
                    
                    x = data["time"]
                    y_total = data["max_force"]
                    
                    current_max_x = max(data["time"])
                    if current_max_x > max_time:
                        max_time = current_max_x
                        
                    pointer_list = [0]
                    for k in range(1, len(x)):
                        if x[k] < 1e-3:
                            pointer_list.append(k)
                    pointer_list.append(len(x))
                    
                    for k in range(len(pointer_list) - 1):
                        start = pointer_list[k]
                        end = pointer_list[k + 1]
                        ax.plot(x[start: end], y_total[start: end], color=rgb_sim[i], linewidth=2 if data["sim"] else 1, label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")) if k == 0 else None, linestyle='--' if not data["sim"] else '-')
                        
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Sum of tendon tension (N)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                    ax.set_xlim(0, max_time * 1.1)
                    ax.set_ylim(0, max(max(y_total), history_max_11) * 1.1)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
                    # ax.set_title("Sum of string tension - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                    
                    if max(y_total) > history_max_11:
                        history_max_11 = max(y_total)
                        
            new_x = []
            new_y = []
            for i in range(len(x_exp_total)):
                data = x_exp_total[i]
                if not len(data["recorded_movement_x"]):
                    new_x.append(data["time"])
                    new_y.append(data["max_force"])
            
            # # 首先计算并绘制实验数据轨迹的置信区间（如果有多个轨迹）
            # exp_trajectories = []
            # exp_indices = []
            # for i in range(len(x_exp_total)):
            #     data = x_exp_total[i]
            #     if len(data["recorded_movement_x"]):
            #         x = data["recorded_movement_x"]
            #         y = data["recorded_movement_y"]
            #         exp_trajectories.append(np.column_stack([x, y]))
            #         exp_indices.append(i)
            
            # # 如果有多个实验轨迹，绘制平滑的置信区间（uniform-width tube）
            # if len(exp_trajectories) > 1:
            #     try:
            #         mean_traj, perp_std, avg_std, normals = self.compute_mean_and_stats(
            #             exp_trajectories, smooth_mean_sigma=10)
            #         upper, lower = self.compute_uniform_tube(mean_traj, avg_std, perp_std, normals)
                    
            #         rgb = rgb_real[exp_indices[0]]  # 使用第一个实验轨迹的颜色
                               
            #         # 绘制平滑的平均轨迹
            #         ax.plot(mean_traj[:, 0], mean_traj[:, 1],
            #                 linestyle="--", linewidth=1.5, color=rgb, alpha=0.8,
            #                 label="Exp. Avg." if len(x_exp_total) > 1 else None)
                
            #         ax.legend(fontsize=NORMAL - 8)
                    
            #         # 绘制置信区域：无边框描边，避免锯齿
            #         pts = np.vstack([upper, lower[::-1]])
            #         ax.fill(pts[:, 0], pts[:, 1],
            #                 color=rgb, alpha=0.2,
            #                 edgecolor='none', linewidth=0,
            #                 label="Exp. Boundary" if len(x_exp_total) > 1 else None)
                
            #         ax.legend(fontsize=NORMAL - 8)
            #     except:
            #         pass  # 如果计算失败，跳过置信区间绘制
            
            for i in range(len(x_exp_total)):
                data = x_exp_total[i]
                data_number = i + 1
                if len(data["recorded_movement_x"]):
                    x = data["recorded_movement_x"]
                    y_total = data["recorded_movement_y"]
                    # ax.scatter(x, y_total, marker='x', label=("Exp. " + str(data_number) if len(x_exp_total) >= 1 else ""), color=rgb_real[i], s=8, alpha=0.25)
                    ax.plot(x, y_total, linestyle="--", linewidth=1.5, label=("Exp. " + (str(data_number) if len(x_exp_total) > 1 else "")), color=rgb_real[i], alpha=0.55)
                    ax.set_xlabel("X (mm)", fontsize=NORMAL-8, labelpad=LABELPAD)
                    ax.set_ylabel("Y (mm)", fontsize=NORMAL-8, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL-8, pad=PAD, width=LINE_WIDTH)
                    ax.set_aspect(1)
                    ax.grid()

                    # ax.set_title("Moving trajectory of the robot", fontsize=TITLE, fontweight='bold', pad=10)
                    ax.legend(fontsize=NORMAL - 12, loc="upper right")
                else:
                    ax.set_xmargin(0)
                    ax.set_ymargin(0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(LINE_WIDTH)
                    
                    x = np.array(new_x).mean(axis=0)
                    y_total = np.array(new_y).mean(axis=0)
                    std_dev = np.array(new_y).std(axis=0)
                    
                    current_max_x = max(data["time"])
                    if current_max_x > max_time:
                        max_time = current_max_x
                    
                    pointer_list = [0]
                    for k in range(1, x.size):
                        if x[k] < 1e-3:
                            pointer_list.append(k)
                    pointer_list.append(x.size)
                    
                    for k in range(len(pointer_list) - 1):
                        start = pointer_list[k]
                        end = pointer_list[k + 1]
                        ax.plot(x[start: end], y_total[start: end], color=rgb_real[i], linewidth=2 if data["sim"] else 2, label=("Exp. ") if k == 0 else None, linestyle='--' if not data["sim"] else '-')
                        ax.fill_between(x[start: end], y_total[start: end] - std_dev[start: end], y_total[start: end] + std_dev[start: end], color=rgb_real[i], alpha=0.2)
                        
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Sum of tendon tension (N)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                    ax.set_xlim(0, max_time * 1.1)
                    ax.set_ylim(0, max(max(y_total), history_max_11) * 1.5)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    # ax.set_title("Sum of string tension - Time", fontsize=32, fontweight='bold', pad=10)
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                    
                    if max(y_total) > history_max_11:
                        history_max_11 = max(y_total)
                    break
            # fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
            
            plt.tight_layout()
            plt.show()

            # 5
            history_max_11 = 0.0
            fig, ax = plt.subplots(figsize=(10, 9))
            for i in range(len(x_sim_total)):
                data = x_sim_total[i]
                data_number = i + 1
                if not len(data["recorded_movement_x"]):
                    ax.set_xmargin(0)
                    ax.set_ymargin(0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(LINE_WIDTH)
                    
                    x = data["time"]
                    y_total = data["nodal_force"]
                    
                    current_max_x = max(data["time"])
                    if current_max_x > max_time:
                        max_time = current_max_x
                        
                    pointer_list = [0]
                    for k in range(1, len(x)):
                        if x[k] < 1e-3:
                            pointer_list.append(k)
                    pointer_list.append(len(x))
                    
                    for k in range(len(pointer_list) - 1):
                        start = pointer_list[k]
                        end = pointer_list[k + 1]
                        ax.plot(x[start: end], y_total[start: end], color=rgb_sim[i], linewidth=2 if data["sim"] else 1, label=("Sim. " + (str(data_number) if len(x_sim_total) > 1 else "")) if k == 0 else None, linestyle='--' if not data["sim"] else '-')
                        
                    ax.set_xlabel("Time (s)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.set_ylabel("Von-Mises stress (MPa)", fontsize=NORMAL, labelpad=LABELPAD)
                    ax.tick_params(labelsize=NORMAL, pad=PAD, width=LINE_WIDTH)
                    ax.set_xlim(0, max_time * 1.1)
                    ax.set_ylim(0, max(max(y_total), history_max_11) * 1.1)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    # ax.set_title("Maximum nodal stress (Von-Mises) - Time", fontsize=TITLE, fontweight='bold')
                    ax.legend(fontsize=NORMAL - 4, bbox_to_anchor=(0.01, 1.03), loc="upper left")
                    
                    if max(y_total) > history_max_11:
                        history_max_11 = max(y_total)
            # fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
            plt.tight_layout()
            plt.show()

    def plotEvolutionJsonReadFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 
            "Choose a score pack", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            return None, None, False
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                score_list = input_json["score"]
                list_length = len(score_list)
                x_list = [x for x in range(list_length)]
                return np.array(x_list), np.array(score_list), True
            except:
                pass

    def plotEvolutionJson(self):
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.figure()
        plt.title("Max Fitness Value - Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Value")
        length_list = []
        while 1:
            x, y, goon = self.plotEvolutionJsonReadFile()
            if not goon:
                break
            else:
                # plt.plot(x, 1.0 - y)
                plt.plot(x, y)
                length_list.append(len(x))
        if len(length_list) == 0:
            return
        max_length = max(length_list)
        step_number = int(max_length / 60) + 2
        labels = [str(i * 60) for i in range(step_number)]
        plt.xticks(range(0, step_number * 60, 60), labels=labels)
        plt.grid(axis='x')
        plt.ylim((0, 1.1 * max(y)))
        plt.show()

    def plotSimulationJson(self):
        plt.rcParams.update({'font.size': 22})
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.figure()

        plt.xlabel("Analytical reward")
        plt.ylabel("Simulation reward")

        path, _ = QFileDialog.getOpenFileName(
            None, 
            "Choose a score pack", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing simulation data", self.state)
            return

        total_x_list = []
        total_y_list = []
        total_c_list = []
        
        total_rf_list = []
        total_rm_list = []
        total_ra_list = []
        
        trend_x_data = []
        trend_y_data = []
        optimal_result_data = [0., False, False, 0] #reward, rf, rm, ra
        optimal_arg = 0

        buffer_step = 512

        file_name_list = path.split('.json')
        file_ahead = file_name_list[0][: -13]
        file_end = file_name_list[0][-7:]

        if file_end == "extract":
            plt.xlabel("Simulation reward (ideal)")
            plt.ylabel("Simulation reward (with perturbations)")
            total_x_hard_sim_list = []
            total_y_hard_sim_list = []
            total_mean_list = []
            total_std_list = []
            
            simulation_time = 0

            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                score_list = input_json["method"]
            except:
                pass

            for i in range(len(score_list)):
                reward_get = False
                try:
                    # reward = score_list[i]["reward_without_actuator"]
                    tension = score_list[i]["score"]
                    hard_sim_result = score_list[i]["hard_sim_result_without_actuator"]
                    simulation_time = len(hard_sim_result)
                    total_x_list.append(tension)
                    # total_y_list.append(reward)
                    for i in range(len(hard_sim_result)):
                        total_x_hard_sim_list.append(tension)
                        total_y_hard_sim_list.append(hard_sim_result[i])
                    array = np.array(hard_sim_result)
                    total_mean_list.append(array.mean())
                    total_std_list.append(array.std())
                    reward_get = True
                except:
                    pass
                if not reward_get:
                    break

            min_reward = 1.0
            for ele in total_x_list:
                if ele > -1 and ele < min_reward:
                    min_reward = ele
            for ele in total_x_list:
                total_c_list.append(ele if ele > -1 else min_reward)

            # plt.scatter(total_x_list, total_y_list, marker='o', c=total_c_list, cmap='coolwarm', label='Simulation without noise')

            total_c_list.clear()
            min_reward = 1.0
            for ele in total_y_hard_sim_list:
                if ele > -1 and ele < min_reward:
                    min_reward = ele
            for ele in total_y_hard_sim_list:
                total_c_list.append(ele if ele > -1 else min_reward)
            
            plt.scatter(total_x_hard_sim_list, total_y_hard_sim_list, marker='x', c=total_c_list, cmap='coolwarm', alpha=0.5, label='Simulation with noise')
            
            total_c_list.clear()
            min_reward = 1.0
            for ele in total_mean_list:
                if ele > -1 and ele < min_reward:
                    min_reward = ele
            for ele in total_mean_list:
                total_c_list.append(ele if ele > -1 else min_reward)
            cmap = mpl.colormaps['coolwarm']
            array_color_list = np.array(total_c_list)
            min_value = array_color_list.min()
            max_value = array_color_list.max()
            colors = cmap((array_color_list - min_value) / (max_value - min_value))

            for i in range(len(total_x_list)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                plt.errorbar(total_x_list[i], total_mean_list[i], total_std_list[i], fmt='.', color=colors[i], ecolor=colors[i], capsize=5)

            # plt.legend(loc="upper right")
            plt.title("Simulation with perturbations in creases " + f"({len(total_x_list)} Cases / {simulation_time} runs each)", pad=12)
            plt.ylim((-0.5, 2))
        else:
            step = 0

            if path != '':
                while 1:
                    file_open = False
                    file_total_name = file_ahead + str(step * buffer_step) + "_true_reward.json"
                    try:
                        with open(file_total_name, 'r', encoding='utf-8') as fw:
                            input_json = json.load(fw)
                        score_list = input_json["method"]
                        file_open = True
                    except:
                        pass

                    if not file_open:
                        step += 1
                        if step >= 60:
                            break
                        continue
                    try:
                        control_mode = input_json["control_mode"]
                    except:
                        control_mode = 0
                        
                    try:
                        phase = input_json["phase"]
                    except:
                        phase = 0
                        
                    for i in range(len(score_list)):
                        reward_get = False
                        try:
                            reward = score_list[i]["reward_without_actuator"]
                            tension = score_list[i]["score"]
                            rf = score_list[i]["rf"]
                            rm = score_list[i]["rm"]
                            ra = score_list[i]["ra"]
                            
                            if control_mode == 0:
                                if rf >= 1 - FOLDING_ERROR_MINIMUM / np.pi:
                                    trend_x_data.append(tension)
                                    trend_y_data.append(reward)
                            total_x_list.append(tension)
                            total_y_list.append(reward)
                            total_rf_list.append(rf)
                            total_rm_list.append(rm)
                            total_ra_list.append(ra)
                            reward_get = True
                            if control_mode == 0:
                                if reward > optimal_result_data[0]:
                                    optimal_result_data[0] = reward
                                    if rf >= 1 - FOLDING_ERROR_MINIMUM / np.pi:
                                        optimal_result_data[1] = True
                                    if rm >= 1 - FOLDING_ERROR_MINIMUM / np.pi:
                                        optimal_result_data[2] = True
                                    optimal_result_data[3] = ra
                        except:
                            pass
                        if not reward_get:
                            break
                    
                    step += 1
            if step >= 0:
                min_reward = 0.0
                if control_mode == 0:
                    find = 0
                    recommend_ID = 0
                    for order in range(len(total_y_list)):
                        rf = total_rf_list[order]
                        rm = total_rm_list[order]
                        ra = total_ra_list[order]
                        if ((rf >= 1 - FOLDING_ERROR_MINIMUM / np.pi and optimal_result_data[1]) or not optimal_result_data[1]) and \
                            ((rm >= 1 - FOLDING_ERROR_MINIMUM / np.pi and optimal_result_data[2]) or not optimal_result_data[2]):
                            recommend_ID = order
                            find = 1
                            if not optimal_result_data[1] and not optimal_result_data[2]:
                                find = 0
                            break
                    for order in range(-1, -1 - len(total_y_list), -1):
                        candidate_reward = total_y_list[order]
                        if candidate_reward == optimal_result_data[0]:
                            optimal_arg = order + len(total_y_list)
                            break
                    if not find:
                        recommend_ID = optimal_arg
                for ele in total_y_list:
                    total_c_list.append(ele if ele > -1 else min_reward)
                    
                # coeffs = np.polyfit(trend_x_data, trend_y_data, 1)
                # trend_line = np.poly1d(coeffs)
                # plt.plot(trend_x_data, trend_line(trend_x_data), color=(0.7, 0.04, 0.15), linewidth=1, linestyle='--', label='Trend of reward on solutions with low folding angle error')
                plt.scatter(total_x_list, total_y_list, marker='.', c=total_c_list, cmap='coolwarm')
                if control_mode == 0 and phase >= 0:
                    plt.annotate('#' + str(recommend_ID + 1), xy=(total_x_list[recommend_ID], total_y_list[recommend_ID]), xytext=(total_x_list[recommend_ID] + 0.01, total_y_list[recommend_ID] + 0.01), fontsize=22, color=(0.7, 0.04, 0.15))
                    plt.annotate('#' + str(optimal_arg + 1), xy=(total_x_list[optimal_arg], total_y_list[optimal_arg]), xytext=(total_x_list[optimal_arg] + 0.01, total_y_list[optimal_arg] + 0.01), fontsize=22, color=(0.7, 0.04, 0.15))
                    plt.scatter([total_x_list[recommend_ID]], [total_y_list[recommend_ID]], marker='x', color=(0.7, 0.04, 0.15), label='Optimal solution (minimal tendon tension)', s=100)
                    plt.scatter([total_x_list[optimal_arg]], [total_y_list[optimal_arg]], marker='*', color=(0.7, 0.04, 0.15), label='Optimal solution (maximum reward)', s=100)
                    plt.legend(loc="lower right", fontsize=14)
                
                if phase <= 0:  
                    correct_number = 0
                    for i in range(len(total_y_list)):
                        if total_y_list[i] > 0.0:
                            correct_number += 1
                        
                plt.title("Simulation reward - Analytical reward (" + (f"{correct_number} correct cases in " if phase <= 0 else "") + f"{len(total_y_list)} cases)", pad=12)
                plt.ylim((min(total_y_list + [-0.15]) * 1.2, 1.1 * max(total_y_list + [0.0])))
                
        plt.show()

    def pointInUnit(self, point):
        length = len(self.units)
        for unit_id in range(length):
            unit = self.units[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    def printOrigami(self):
        """
        @ function: print origami
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        printer = QPrinter()
        print_dialog = QPrintDialog(printer, self)
        if (QDialog.Accepted == print_dialog.exec_()):
            if self.show_square == 'A4':
                self.drawA4Pixmap()
                painter = QPainter(printer)
                rect = painter.viewport()
                size = self.A4_pixmap.size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
                painter.setWindow(self.A4_pixmap.rect())
                painter.drawImage(0, 0, self.A4_pixmap.toImage())
                painter.end()
                self.updateMessage("Succeed to print origami on A4-H...")
            elif self.show_square == None:
                painter = QPainter(printer)
                rect = painter.viewport()
                self.drawCreasePattern(self.output_pixmap, 36)
                size = self.output_pixmap.size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
                painter.setWindow(self.output_pixmap.rect())
                painter.drawImage(0, 0, self.output_pixmap.toImage())
                painter.end()
                self.updateMessage("Succeed to print origami for all screen...")   

    def recalculateContributions(self):
        self.changeUnitCenterContributeCoeff(not_dialog=True)
        self.enable_design = True
        self.updateMessage("Succeed to recalculate contributions for units...")  

    def resetView(self):
        self.pixel_bias = [25, 25]
        self.current_pixel_scale = 1.0
        self.pixel_scale_ranking = 14
        self.A4_length = 296
        self.A4_width = 210
        self.A4_half_length = 148
        self.A4_half_width = 105

    def saveResult(self):
        """
        @ function: Save packed result
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        ok = True
        if self.backup_open_file_path == None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save the design result", 
                ".", 
                "json files (*.json)"
            ) 
            if file_path == '':
                self.updateMessage("Cancel exporting design result")
                ok = False
        else:
            file_path = self.backup_open_file_path
        if ok:
            s = {
                "origami": [],
                "unit_bias_list": self.unit_bias_list,
                "util": {
                    "hole_axis": [ele for ele in self.hole_kps]
                },
                "setting": {
                    "unit_width": self.unit_width,
                    "copy_time": self.copy_time,
                    "entry_flag": self.entry_flag,
                    "add_hole_mode": self.add_hole_mode,
                    "hole_size": self.hole_size,
                    "hole_resolution": self.hole_resolution,
                    "enable_connection": self.enable_connection,
                    "con_left_length": self.con_left_length,
                    "con_right_length": self.con_right_length,
                    "con_radius": self.connection_radius,
                    "bias_val": self.bias_val
                },
                "line_features": [{
                    "type": line.getType(),
                    "level": line.level,
                    "coeff": line.coeff,
                    "recover_level": line.recover_level,
                    "recover_angle": line.recover_angle,
                    "hard": line.hard,
                    "hard_angle": line.folding_angle_upper_bound,
                    "hard_angle_down": line.folding_angle_lower_bound,
                    "thick_panel_height": line.thick_panel_height
                } for line in self.lines],
                "strings": {
                    "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                },
                "P_candidators": {
                    "points": self.P_candidate,
                    "connections": self.P_candidate_connection_index 
                },
                "contributions": self.unit_center_contribute_coeff
            }
            for i in range(self.origami_number):
                if type(self.storage[i][1]) == KinematicLine:
                    ori_type = "kl"
                elif type(self.storage[i][1]) == ModuleLeanMiura:
                    ori_type = "leanMiura"
                elif type(self.storage[i][1]) == DxfDirectGrabber:
                    ori_type = "dxf"
                dic = {
                    "type": ori_type,
                    "tsp": self.storage[i][0],
                    "data": self.storage[i][1].getData(),
                    "add_width": self.storage[i][2]
                }
                s["origami"].append(dic)
            if self.fixed_panel >= 0:
                s["fix"] = [self.fixed_panel]
            if len(self.crease_angle):
                s["crease_angle"] = self.crease_angle
            if len(self.crease_info):
                s["crease_info"] = self.crease_info
                
            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(s, f, indent=4)
            self.backup_open_file_path = file_path
            self.updateMessage("Succeed to save result at " + file_path)

    def setBiasAsDefault(self):
        self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = None
        self.doubleSpinBox_expert_mode.setVisible(False)

    def setBiasAsExpertModified(self):
        self.doubleSpinBox_expert_mode.setMinimum(self.pref_pack["print_accuracy"])
        self.doubleSpinBox_expert_mode.setMaximum(self.unit_width / 6.0)
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.doubleSpinBox_expert_mode.setValue(self.pref_pack["print_accuracy"])
            self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = self.pref_pack["print_accuracy"]
        else:
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])
        self.doubleSpinBox_expert_mode.setVisible(True)

    def setGlobalParameter(self, unit_width=None, copy_time=None, entry_flag=None, add_hole_mode=None, 
                           hole_size=None, hole_resolution=None, enable_connection=None, 
                           con_left_length=None, con_right_length=None, con_radius=None, bias_val=None):
        """
        @ function: Set global parameter of app
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220403
        @ spec: None
        """
        if unit_width != None:
            self.unit_width = unit_width
            self.paper_info['unit_width'] = unit_width
            self.spinbox_crease_width.setValue(unit_width)
        if copy_time != None:
            self.copy_time = copy_time
            self.design_info['copy_time'] = self.copy_time
            self.spinbox_copy_time.setValue(copy_time)
        if entry_flag != None:
            self.entry_flag = entry_flag
            self.slider_flag.setValue(entry_flag)
        if add_hole_mode != None:
            self.add_hole_mode = add_hole_mode
            if add_hole_mode:
                self.checkBox_add_hole_mode.setChecked(True)
            else:
                self.checkBox_add_hole_mode.setChecked(False)
        if hole_size != None:
            self.hole_size = hole_size
            self.spinbox_hole_size.setValue(hole_size)
        if hole_resolution != None:
            self.hole_resolution = hole_resolution
            self.spinbox_resolution.setValue(hole_resolution)
        if enable_connection != None:
            self.enable_connection = enable_connection
            if enable_connection:
                self.checkBox_connection.setChecked(True)
            else:
                self.checkBox_connection.setChecked(False)
        if con_left_length != None:
            self.con_left_length = con_left_length
            self.spinbox_con_left_length.setValue(con_left_length)
        if con_right_length != None:
            self.con_right_length = con_right_length
            self.spinbox_con_right_length.setValue(con_right_length)
        if con_radius != None:
            self.connection_radius = con_radius
            self.spinbox_connection_radius.setValue(con_radius)
        if bias_val != None:
            self.bias_val = bias_val
            self.doubleSpinBox_bias_val.setValue(self.bias_val)

    def setting(self):
        """
        @ function: Setting for application
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        self.pref_pack_window.readFile()
        self.pref_pack_window.show()
        if self.pref_pack_window.exec_():
            pass
        if self.pref_pack_window.ok:
            self.pref_pack = self.pref_pack_window.getPrefPack()
            self.limitation = self.pref_pack_window.getLimitation()
            self.updateMessage("Succeed to save settings...")
            self.enable_design = True
        else:
            self.updateMessage("Cancel saving settings...")

    def setupTimer(self):
        self.timer = QTimer(self)
        self.timer.start(33)
        self.timer.timeout.connect(self.updateWindow)

    def showA4Square(self):
        self.show_square = 'A4'

    def showCurve(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a curve json file", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing curve", self.state)
        else:
            try:
                self.curve_name = os.path.basename(path)
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.x_list = input_json["x"]
                self.y_list = input_json["y"]
                self.tm_window = TmWindow(res=self.limitation["discrete_resolution"], enable_3d=self.limitation["3D_match_enable"])
                self.tm_window.importXPoints(self.x_list)
                self.tm_window.importYPoints(self.y_list)
                self.z_list = [0 for _ in range(len(self.x_list))]
                if self.limitation["3D_match_enable"]:
                    self.z_list = input_json["z"]
                    self.tm_window.importZPoints(self.z_list)
                self.tm_window.plot()
                self.tm_window.show()
            except:
                pass

    def showDirection(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a curve json file", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing direction curve", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dir_list = input_json["dir"]
                # self.tm_window = TmWindow(self.limitation["discrete_resolution"])
                # self.tm_window.importXPoints(self.x_list)
                # self.tm_window.importYPoints(self.y_list)
                # self.tm_window.plot()
                # self.tm_window.show()
            except:
                pass

    def showIndex(self):
        if self.show_index:
            self.show_index = False
            self.actionShow_Index.setText("Show Index")
        else:
            self.show_index = True
            self.actionShow_Index.setText("Hide Index")


    def showNone(self):
        self.show_square = None
        
    def showTrajectory(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a simulation json file", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing trajectory", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.trajectory_x = input_json["recorded_movement_x"]
                self.trajectory_y = input_json["recorded_movement_y"]
                self.trajectory_z = input_json["recorded_movement_z"]
                self.tm_window = TmWindow(res=self.limitation["discrete_resolution"], enable_3d=self.limitation["3D_match_enable"])
                self.tm_window.importXPoints(self.x_list)
                self.tm_window.importYPoints(self.y_list)
                self.tm_window.importZPoints(self.z_list)
                self.tm_window.importTrajectory(self.trajectory_x, self.trajectory_y, self.trajectory_z)
                self.tm_window.plot()
                self.tm_window.show()
            except:
                pass

    def showTG(self):
        items = []
        real_index_mapper = []
        # List all Miura combo
        for i in range(self.origami_number):
            type_src = type(self.storage[i][1])
            if type_src == KinematicLine:
                items.append("Miura Combo[" + str(len(items)) + "] - storage[" + str(i) + "]")
                real_index_mapper.append(i)
        if len(items) == 0:
            self.updateMessage("No Miura Storages found.")
            return
        selected_item, ok = QInputDialog.getItem(self, "Select Miura Item", "Select a Miura combo:", items)
        # If press ok
        if ok:
            index = real_index_mapper[items.index(selected_item)]
            designer = deepcopy(self.designer)
            designer.setSource(self.storage[index][1])
            data_list = designer.getGeometryData()
            self.tm_window = TmWindow(res=self.limitation["discrete_resolution"], enable_3d=self.limitation["3D_match_enable"])
            self.tm_window.importXPoints(self.x_list)
            self.tm_window.importYPoints(self.y_list)
            self.tm_window.importZPoints(self.z_list)
            self.tm_window.importTrajectory(self.trajectory_x, self.trajectory_y, self.trajectory_z)
            self.tm_window.getTm().setSource(data_list)
            self.tm_window.getTm().enable_hypar_connection = self.limitation["hypar_enable"]
            self.tm_window.getTm().enable_5fold_connection = self.limitation["5fold_enable"]
            self.tm_window.startShow()

    def startAddString(self):
        self.actionAdd_TSA_A_point.setEnabled(True)
        self.exist_string_start = False
        self.string_start_point = []
        self.a_string = []
        self.string_type = BOTTOM
        self.updateMessage("Add-string mode enabled...")

    def stopThread(self):
        exist_thread = False
        if self.enable_cdf_curve_fitting and self.cdf_curve_fitting_thread:
            self.cdf_curve_fitting_thread.terminate()
            self.cdf_curve_fitting_thread = None
            self.enable_cdf_curve_fitting = False
            exist_thread = True
        if self.enable_output_stl and self.stl_output_thread:
            self.stl_output_thread.terminate()
            self.stl_output_thread = None
            self.enable_output_stl = False
            exist_thread = True
        if self.enable_phys_data_collecting and self.phys_data_collecting_thread:
            self.phys_data_collecting_thread.terminate()
            self.phys_data_collecting_thread = None
            self.enable_phys_data_collecting = False
            exist_thread = True
        if self.enable_mcts and self.mcts_thread:
            self.mcts_thread.terminate()
            self.mcts_thread = None
            self.enable_mcts = False
            exist_thread = True
        if self.enable_threading_design and self.threading_design_process:
            self.threading_design_process.kill()
            self.enable_threading_design = False
            exist_thread = True

        if exist_thread:
            self.drawProcess(1.0)
            self.updateMessage("Stop all threads running in application...")
        else:
            self.updateMessage("No threads are running in application...")

    def toPixel(self, x: list):
        normal_axis = [round(x[X] * self.current_pixel_scale + self.pixel_bias[X]), round(x[Y] * self.current_pixel_scale + self.pixel_bias[Y])]
        if self.mirror_x:
            normal_axis[X] = -normal_axis[X] + 2 * self.pixel_bias[X]
        if self.mirror_y:
            normal_axis[Y] = -normal_axis[Y] + 2 * self.pixel_bias[Y]
        if self.xy_rotation != 180:
            rad_rotation = (self.xy_rotation - 180.0) / 180.0 * math.pi
            new_axis = [int(math.cos(rad_rotation) * (normal_axis[X] - self.pixel_bias[X]) - math.sin(rad_rotation) * (normal_axis[Y] - self.pixel_bias[Y]) + self.pixel_bias[X]), \
                        int(math.sin(rad_rotation) * (normal_axis[X] - self.pixel_bias[X]) + math.cos(rad_rotation) * (normal_axis[Y] - self.pixel_bias[Y]) + self.pixel_bias[Y])]
            return new_axis
        else:
            return [int(normal_axis[X]), int(normal_axis[Y])]
    
    def updateMessage(self, message, message_type="MESSAGE"):
        self.label_message.setText(message_type + " :: " + message)
        if not self.enable_output_stl and not self.enable_cdf_curve_fitting and not self.enable_phys_data_collecting and not self.enable_mcts and not self.enable_threading_design:
            self.drawProcess(1.0)

    def updateState(self, state, type_state, message_type="INFO"):
        """
        @ function: Update state of application
        @ version: 0.1111
        @ developer: py
        @ progress: finish
        @ date: 20220302
        @ spec: add print origami, add adding holes
        """
        self.state = type_state
        if type_state == self.INITIAL_STATE:
            # No input, No design

            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(False)
            self.checkBox_add_string_mode.setEnabled(False)
            self.spinbox_hole_size.setEnabled(False)
            self.spinbox_resolution.setEnabled(False)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(False)
            self.spinbox_copy_time.setEnabled(False)
            self.slider_flag.setEnabled(False)
            self.horizontal_folding_slider.setEnabled(False)

            # Pack of design button
            self.button_design.setEnabled(False)
            self.button_threading_design.setEnabled(False)

            # Pack of actions
            self.actionSave_result.setEnabled(False)
            self.actionPrint_P.setEnabled(False)
            self.menuExport_E.setEnabled(False)
            self.actionLeanMiura.setEnabled(False)
            self.actionTransition_T.setEnabled(False)
            self.actionPhysical_Simulation_P.setEnabled(False)
            self.actionCollect_Physical_Data_C.setEnabled(False)
            self.actionAdd_Holes.setEnabled(False)
            self.actionAdd_TSA_A_point.setEnabled(False)
            self.actionExplicit_Simulation_E.setEnabled(False)
            self.actionExpert_Mode_E.setEnabled(False)
            self.actionEdit_kl_E.setEnabled(False)
            self.actionCalculate_Sequence.setEnabled(False)
            self.actionEdit_Sequence_S.setEnabled(False)
            self.actionImport_string_path.setEnabled(False)
            self.actionAdd_TSA_A_candidators.setEnabled(False)
            self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(False)
            self.spinbox_connection_radius.setEnabled(False)
            self.spinbox_con_left_length.setEnabled(False)
            self.spinbox_con_right_length.setEnabled(False)
            

            # Modify the state label
            self.label_state.setText(message_type + " :: " + state)
        elif type_state == self.IMPORT_SUCCESS:
            # User can design when in this state

            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(False)
            self.checkBox_add_string_mode.setEnabled(False)
            self.spinbox_hole_size.setEnabled(False)
            self.spinbox_resolution.setEnabled(False)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(True)
            self.spinbox_copy_time.setEnabled(True)
            self.slider_flag.setEnabled(True)
            self.horizontal_folding_slider.setEnabled(True)

            # Pack of design button
            self.button_design.setEnabled(True)
            self.button_threading_design.setEnabled(True)

            # Pack of actions
            self.actionSave_result.setEnabled(False)
            self.actionPrint_P.setEnabled(False)
            self.menuExport_E.setEnabled(False)
            self.actionLeanMiura.setEnabled(False)
            self.actionTransition_T.setEnabled(False)
            self.actionPhysical_Simulation_P.setEnabled(False)
            self.actionCollect_Physical_Data_C.setEnabled(False)
            self.actionAdd_Holes.setEnabled(False)
            self.actionExplicit_Simulation_E.setEnabled(False)
            self.actionExpert_Mode_E.setEnabled(False)
            self.actionEdit_kl_E.setEnabled(False)
            self.actionCalculate_Sequence.setEnabled(False)
            self.actionEdit_Sequence_S.setEnabled(False)
            self.actionImport_string_path.setEnabled(False)
            self.actionAdd_TSA_A_candidators.setEnabled(False)
            self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(False)
            self.spinbox_connection_radius.setEnabled(False)
            self.spinbox_con_left_length.setEnabled(False)
            self.spinbox_con_right_length.setEnabled(False)

            # Modify the state label
            self.label_state.setText(message_type + " :: " + state)
        elif type_state == self.DESIGN_FINISH:
            # User can design and add holes when in this state
            
            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(True)
            self.checkBox_add_string_mode.setEnabled(True)
            self.spinbox_hole_size.setEnabled(True)
            self.spinbox_resolution.setEnabled(True)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(True)
            self.spinbox_copy_time.setEnabled(True)
            self.slider_flag.setEnabled(True)
            self.horizontal_folding_slider.setEnabled(True)

            # Pack of design button
            self.button_design.setEnabled(True)
            self.button_threading_design.setEnabled(True)

            # Pack of actions
            self.actionSave_result.setEnabled(True)
            self.actionPrint_P.setEnabled(True)
            self.menuExport_E.setEnabled(True)
            self.actionLeanMiura.setEnabled(True)
            self.actionTransition_T.setEnabled(True)
            self.actionPhysical_Simulation_P.setEnabled(True)
            self.actionCollect_Physical_Data_C.setEnabled(True)
            self.actionAdd_Holes.setEnabled(True)
            self.actionExplicit_Simulation_E.setEnabled(True)
            self.actionExpert_Mode_E.setEnabled(True)
            self.actionEdit_kl_E.setEnabled(True)
            self.actionCalculate_Sequence.setEnabled(True)
            self.actionEdit_Sequence_S.setEnabled(True)
            self.actionImport_string_path.setEnabled(True)
            self.actionAdd_TSA_A_candidators.setEnabled(True)
            if len(self.P_candidate):
                self.actionDelete_TSA_A_Candidators.setEnabled(True)
            else:
                self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(True)
            self.spinbox_connection_radius.setEnabled(True)
            self.spinbox_con_left_length.setEnabled(True)
            self.spinbox_con_right_length.setEnabled(True)

            # Modify the state label
            self.label_state.setText(message_type + " :: " + state)
        elif type_state == self.DESIGN_ERROR:
            self.label_state.setText(message_type + " :: " + state)
        elif type_state == self.OUTPUT_FINISH:
            self.label_state.setText(message_type + " :: " + state)
        else:
            self.label_state.setText("ERROR :: Unknown state for application")
        self.updateStringPath()
        self.drawCreasePattern()

    def updateStringPath(self):
        s = {
            "strings": {
                "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
            },
            "P_candidators": {
                "points": self.P_candidate,
                "connections": self.P_candidate_connection_index 
            }
        }
        backup_exist_string_start = self.exist_string_start
        self.exist_string_start = False
        self.string_start_point = []
        self.string_type = BOTTOM

        self.strings = []  
        self.a_string = [] # one string
        self.string_total_information = [] # list of list

        self.P_candidate = s['P_candidators']["points"]
        self.P_candidate_connection_index = s['P_candidators']["connections"]

        string_type_list = s["strings"]['type']
        string_id_list = s["strings"]['id']
        string_reverse_list = s["strings"]['reverse']
        for i in range(len(string_type_list)):
            self.startAddString()
            if string_reverse_list[i][0] == -1:
                self.string_type = BOTTOM
            else:
                self.string_type = TOP
            for j in range(len(string_type_list[i])):
                if string_type_list[i][j] == 'A':
                    index = string_id_list[i][j]
                    if index >= len(self.P_candidate):
                        self.addStringPoint(0, 0, index)
                    else:
                        self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
                    # self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
                else:
                    # unit_axis = self.units[string_id_list[i][j]].getCenter()
                    unit_axis = self.calculateUnitCenterUsingContribution(string_id_list[i][j])
                    self.addStringPoint(unit_axis[X], unit_axis[Y], string_id_list[i][j], 'B', string_reverse_list[i][j])
            self.endAddString()
        self.exist_string_start = backup_exist_string_start

    def updateWindow(self):
        """
        @ function: Update window in 30Hz
        @ version: 0.111
        @ developer: py
        @ progress: on road
        @ date: 20230107
        @ spec: add zoom and axis bias show /0.111
        """
        try:
            self.label_zoom.setText("Zoom scale : " + str(self.current_pixel_scale))
            self.label_axis_bias.setText("(" + str(self.pixel_bias[0]) + ", " + str(self.pixel_bias[1]) + ")")
            self.bias_max = self.unit_width * 0.2
            self.checkHoleKpIsValid()
            if self.enable_moving:
                if self.pref_pack["cursor_axis_mode"] == "pixel_axis":
                    self.label_cursor_axis.setText("(" + str(self.cursor_x) + ", " + str(self.cursor_y) + ")")
                else:
                    self.label_cursor_axis.setText("(" + str(np.round(self.real_x, 1)) + ", " + str(np.round(self.real_y, 1)) + ")")
        except:
            pass
        if self.state:
            if self.enable_design:
                self.design()
                self.enable_design = False
            else:
                # paint result
                self.drawCreasePattern()

    def wheelEvent(self, event) -> None:
        """
        @ function: Zoom in/out
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230107
        @ spec: None
        """
        if (self.state == self.DESIGN_FINISH):
            focus = self.draw_panel.mapFromGlobal(QCursor.pos())
            x = focus.x()
            y = focus.y()
            if x > 0 and x < self.pixmap_length and y > 0 and y < self.pixmap_width:
                if event.angleDelta().y() > 0:
                    if self.pixel_scale_ranking > 0:
                        self.pixel_scale_ranking -= 1
                        old_scale = self.current_pixel_scale
                        self.current_pixel_scale = self.pixel_scale[self.pixel_scale_ranking]
                        self.pixel_bias[0] = int(x - (x - self.pixel_bias[0]) / old_scale * self.current_pixel_scale)
                        self.pixel_bias[1] = int(y - (y - self.pixel_bias[1]) / old_scale * self.current_pixel_scale)
                        self.A4_length = self.A4_length / old_scale * self.current_pixel_scale
                        self.A4_width = self.A4_width / old_scale * self.current_pixel_scale
                        self.A4_half_length = self.A4_half_length / old_scale * self.current_pixel_scale
                        self.A4_half_width = self.A4_half_width / old_scale * self.current_pixel_scale
                else:
                    if self.pixel_scale_ranking < self.pixel_scale_min_ranking - 1:
                        self.pixel_scale_ranking += 1
                        old_scale = self.current_pixel_scale
                        self.current_pixel_scale = self.pixel_scale[self.pixel_scale_ranking]
                        self.pixel_bias[0] = int(x - (x - self.pixel_bias[0]) / old_scale * self.current_pixel_scale)
                        self.pixel_bias[1] = int(y - (y - self.pixel_bias[1]) / old_scale * self.current_pixel_scale)
                        self.A4_length = self.A4_length / old_scale * self.current_pixel_scale
                        self.A4_width = self.A4_width / old_scale * self.current_pixel_scale
                        self.A4_half_length = self.A4_half_length / old_scale * self.current_pixel_scale
                        self.A4_half_width = self.A4_half_width / old_scale * self.current_pixel_scale

# def workerMultisim(mlist, method, pointer, pref_pack, max_edge, input_units, max_size, total_bias):
#     string_total_information = method

#     ori_sim = OrigamiSimulator(use_gui=False)

#     ori_sim.string_total_information = string_total_information
#     ori_sim.pref_pack = pref_pack

#     ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
#     ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
#     ori_sim.initializeRunning()
    
#     # while ori_sim.dead_count < 500:
#     #     ori_sim.step()
#     #     if ori_sim.folding_angle_reach_pi[0] or math.isnan(ori_sim.total_energy[0]) or (ori_sim.current_t > 4.0 and not ori_sim.can_rotate) or (ori_sim.current_t > 20.0 and ori_sim.can_rotate):
#     #         break
#     while 1:
#         ori_sim.step()
#         if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 500 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
#             break
        
#     if not ori_sim.can_rotate:
#         print("Batch: " + str(pointer) + ", Value: Error Actuation")
#         mlist[pointer] = 0.0
#     else:
#         folding_percent = ori_sim.recorded_folding_percent[-1]
#         folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])
#         value = folding_speed * folding_percent

#         print("Batch: " + str(pointer) + ", Value: " + str(value))
        
#         mlist[pointer] = value

# A thread for cdf-curve-fitting
class CdfCurveFittingThread(QThread):
    _emit = pyqtSignal(float)

    def __init__(self, pref_pack, curve_name: str, curve_x, curve_y, curve_z, curve_dir, algorithm='es') -> None:
        super().__init__()
        self.pref_pack = pref_pack
        self.curve_name = curve_name.split('.')[0]
        self.x = curve_x
        self.y = curve_y
        self.z = curve_z
        self.dir = curve_dir
        self.algorithm = algorithm
        self.storage_number = pref_pack["storage"]
        self.batch_size = pref_pack["batch_size"]
        self.process_num = pref_pack["process_number"]
        self.match_mode = pref_pack["match_mode"]
        self.discrete_resolution = pref_pack["discrete_resolution"]
        self.hypar_enable = pref_pack["hypar_enable"]
        self.direction_enable = pref_pack["direction_enable"]

        self.exo_angle1 = pref_pack["exo_angle1"]
        self.exo_angle2 = pref_pack["exo_angle2"]
        self.exo_X = pref_pack["exo_X"]
        self.exo_Y = pref_pack["exo_Y"]
        self.exo_theta = pref_pack["exo_theta"]

        self.BEST_MATCH = 0
        self.STRICT_MATCH = 1
        self.DISCRETE_MATCH = 2
        self.EXO_MATCH = 3
        self.ZERO_MATCH = 4
    
    def run(self):
        step_min = self.pref_pack["row_number"][0]
        step_max = self.pref_pack["row_number"][1]

        process_num = self.process_num

        cfh = CurveFittingHelper()
        if self.x != None:
            cfh.setGoalList([[self.x[i], self.y[i], self.z[i]] for i in range(len(self.x))])
        
        else:
            if self.match_mode == self.EXO_MATCH:
                cfh.setExoGoal(self.exo_X, self.exo_Y, self.exo_theta * math.pi / 180.0)

        if self.direction_enable:
            cfh.setDirectionGoalList(self.dir)

        tm = TransitionModel()
        tm.enable_hypar_connection = self.pref_pack["hypar_enable"]
        tm.enable_5fold_connection = self.pref_pack["5fold_enable"]
        tm.enable_3d = self.pref_pack["3D_match_enable"]
        
        total_step = (step_max - step_min + 1) * self.pref_pack["generation"] * 10
        update_scale_step = self.pref_pack["generation"] / 20
        output_best_flag  = self.pref_pack["generation"] / 20
        change_mode_flag  = self.pref_pack["generation"] / 2

        self.time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        folder_result = f"./cdfResult/{self.curve_name}-{self.time}/result"
        folder_score = f"./cdfResult/{self.curve_name}-{self.time}/score"
        folder_data = f"./cdfResult/{self.curve_name}-{self.time}/data"
        folder_picture = f"./cdfResult/{self.curve_name}-{self.time}/pictures/0"
        folder_picture2 = f"./cdfResult/{self.curve_name}-{self.time}/pictures/180"
        
        try:
            os.makedirs(folder_picture)
            os.makedirs(folder_picture2)
            os.makedirs(folder_result)
            os.makedirs(folder_score)
            os.makedirs(folder_data)
        except:
            pass
        
        s = {
            "origin": [
                [0.0, 0.0]
            ],
            "kl": [
            
            ],
            "add_width":[
                False
            ],
            "score": 0.0
        }

        score_dist = {
            "score": []
        }
        
        for i in range(int(step_min), int(step_max + 1)):
            for cycle in range(10):
                score_list = []
                current_best_sub_scores = []
                score = -2.0
                early_stop = 0

                if self.algorithm == 'es':
                    algo = ES(self.storage_number)
                    algo.mode("cmaes")
                    algo.initialize(self.batch_size * process_num, 2 * i)
                elif self.algorithm == 'ga':
                    algo = GA(self.storage_number)
                    algo.initialize(self.batch_size * process_num, 2 * i)
                mapper = []
                for j in range(i):
                    mapper.append([self.pref_pack["unit_length"][0], self.pref_pack["unit_length"][1]])
                    mapper.append([
                        [-self.pref_pack["miura_angle"][1] * math.pi / 180.0, -self.pref_pack["miura_angle"][0] * math.pi / 180.0],
                        [self.pref_pack["miura_angle"][0] * math.pi / 180.0, self.pref_pack["miura_angle"][1] * math.pi / 180.0]
                    ])
                algo.setMapper(mapper)
                
                while(algo.iteration < self.pref_pack["generation"]):
                    reward_list = []

                    self._emit.emit((((i - step_min) * 10 + cycle) * self.pref_pack["generation"] + algo.iteration) / total_step)
                    data = algo.ask()

                    if process_num == 1:
                        for ii in range(len(data)):
                            k_data = data[ii].data

 
                            tm.setSource([
                                [k_data[2 * k], 0 if k == -1 else abs(k_data[2 * k + 1]), 0 if k_data[2 * k + 1] < 0 and k > -1 else 1] for k in range(i)
                            ])

                            # tm_window = TmWindow()
                            # tm_window.tm = tm
                            # tm_window.printTm(folder_picture, folder_picture2, int(((i - step_min) * 10 + cycle) * self.pref_pack["generation"] + algo.iteration))

                            if (self.match_mode == self.BEST_MATCH or self.match_mode == self.STRICT_MATCH):
                                all_ef, all_ef_dir = tm.getAllEndEffector()
                                cfh.setOriginList(all_ef)
                                if self.match_mode == self.BEST_MATCH:
                                    p = cfh.bestMatch() 
                                else:
                                    p = cfh.strictMatch() * cfh.distanceMatch()
                                if self.direction_enable:
                                    cfh.setDirectionOriginList(all_ef_dir)
                                    p *= cfh.directionMatch(len(all_ef_dir))
                                reward_list.append(p)

                            elif (self.match_mode == self.EXO_MATCH):
                                all_ef = []
                                all_ef_dir = []
                                tm.main_folding_angle = self.exo_angle1 * math.pi / 180.0
                                tl, _, _ = tm.getTransitionLines()
                                all_ef.append(deepcopy(tm.end_ef))
                                all_ef_dir.append(deepcopy(tm.end_ef_dir))
                                intersect1 = calculateIntersection(tl)

                                tm.main_folding_angle = self.exo_angle2 * math.pi / 180.0
                                tl, _, _ = tm.getTransitionLines()
                                all_ef.append(deepcopy(tm.end_ef))
                                all_ef_dir.append(deepcopy(tm.end_ef_dir))
                                intersect2 = calculateIntersection(tl)

                                cfh.setOriginList(all_ef)
                                cfh.setDirectionOriginList(all_ef_dir)

                                p = cfh.exoMatch(intersect1, intersect2)
                                reward_list.append(p)

                            elif (self.match_mode == self.ZERO_MATCH):
                                part_ef, part_ef_dir = tm.getPartEndEffector(45, 45)
                                cfh.setOriginList(part_ef)
                                if self.direction_enable:
                                    cfh.setDirectionOriginList(part_ef_dir)
                                cfh.setIntersectionTime(tm.self_intersection_number)
                                
                                p, sub_scores = cfh.zeroMatch(direction_enable=self.direction_enable)
                                reward_list.append(p)
                                if p > score:
                                    current_best_sub_scores = sub_scores

                            else:
                                part_ef, part_ef_dir = tm.getPartEndEffector(45, self.discrete_resolution)
                                if self.pref_pack["3D_match_enable"]:
                                    part_ef = tm.all_end_ef_3d
                                cfh.setOriginList(part_ef)
                                cfh.setIntersectionTime(tm.self_intersection_number)
                                cfh.maximum_z = tm.maximum_z
  
                                if self.direction_enable:
                                    cfh.setDirectionOriginList(part_ef_dir)
                                    # p *= cfh.directionMatch(self.discrete_resolution)
                                p, sub_scores = cfh.partMatch(self.discrete_resolution, self.direction_enable)
                                reward_list.append(p)
                                if p > score:
                                    current_best_sub_scores = sub_scores

                        maximum_index = reward_list.index(max(reward_list))
                        k_data = data[maximum_index].data
                        
                        algo.evaluate(reward_list)
                    
                    new_score = algo.getCurrentBest()
                    
                    if new_score > score:
                        score = new_score
                        tm.setSource([
                            [k_data[2 * k], 0 if k == -1 else abs(k_data[2 * k + 1]), 0 if k_data[2 * k + 1] < 0 and k > -1 else 1] for k in range(i)
                        ])

                        tm_window = TmWindow(res=self.pref_pack["discrete_resolution"], enable_3d=self.pref_pack["3D_match_enable"])
                        tm_window.importXPoints(self.x)
                        tm_window.importYPoints(self.y)
                        tm_window.importZPoints(self.z)
                        tm_window.tm = tm
                        tm_window.enable_show_process = True
                        tm_window.printTm(folder_picture, folder_picture2, int(((i - step_min) * 10 + cycle) * self.pref_pack["generation"] + algo.iteration))
                        early_stop = 0
                    else:
                        early_stop += 1

                    score_list.append(score)

                    if early_stop >= 100 or algo.optimizer.should_stop():
                        print("Early stop at ITER: " + str(algo.iteration))
                        design_result = deepcopy(algo.storage[-1])
                        kl = []
                        data_result = []
                        alpha = 0.0
                        for k in range(i):
                            data = design_result.data[2*k: 2*k+2] #length, angle and up/down
                            length = data[0] * (self.pref_pack["unit_length"][1] - self.pref_pack["unit_length"][0]) + self.pref_pack["unit_length"][0]
                            if data[1] < 0.5:
                                angle = (2 * data[1] * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) - self.pref_pack["miura_angle"][1]) * math.pi / 180.0
                            else:
                                angle = ((2 * data[1] - 1) * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) + self.pref_pack["miura_angle"][0]) * math.pi / 180.0
                            if k == -1:
                                angle = 0.0
                            alpha += angle * 2
                            kl.append([length, alpha])
                            data_result.append([length, abs(angle), 0 if angle < 0 and k > -1 else 1])
                        s["kl"] = [kl]
                        s["score"] = design_result.score
                        score_dist["score"] = deepcopy(score_list)
                        
                        
                        result_filename = 'row' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'
                        with open(os.path.join(folder_result, result_filename), 'w', encoding="utf-8") as f:
                            json.dump(s, f, indent=4)

                        data_json = {
                            "data": data_result
                        }
                        
                        score_filename = 'score' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'
                        with open(os.path.join(folder_score, score_filename), 'w', encoding="utf-8") as f:
                            json.dump(score_dist, f, indent=4)

                        data_filename = 'data' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'
                        with open(os.path.join(folder_data, data_filename), 'w', encoding="utf-8") as f:
                            json.dump(data_json, f, indent=4)
                        break

                    print(str(cfh.intersection_time) + " ES Counter: " + str(early_stop) + " (" + algo.sample_mode + ")" + "ITER: " + str(algo.iteration) + "    Current Score / Best: " + str(np.round(new_score, 6)) + ' / ' + str(np.round(score, 6)) + ' ' + str(current_best_sub_scores))
                    if self.algorithm == "es":
                        if algo.iteration % update_scale_step == 0:
                            algo.updateScale(0.25 * (1 - algo.iteration / self.pref_pack["generation"]) ** 2)

                        if algo.iteration % output_best_flag == 0:
                            design_result = deepcopy(algo.storage[-1])
                            kl = []
                            data_result = []
                            alpha = 0.0
                            for k in range(i):
                                data = design_result.data[2*k: 2*k+2] #length, angle and up/down
                                length = data[0] * (self.pref_pack["unit_length"][1] - self.pref_pack["unit_length"][0]) + self.pref_pack["unit_length"][0]
                                if data[1] < 0.5:
                                    angle = (2 * data[1] * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) - self.pref_pack["miura_angle"][1]) * math.pi / 180.0
                                else:
                                    angle = ((2 * data[1] - 1) * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) + self.pref_pack["miura_angle"][0]) * math.pi / 180.0
                                if k == -1:
                                    angle = 0.0
                                alpha += angle * 2
                                kl.append([length, alpha])
                                data_result.append([length, abs(angle), 0 if angle < 0 and k > -1 else 1])
                            s["kl"] = [kl]
                            s["score"] = design_result.score
                            score_dist["score"] = deepcopy(score_list)

                            result_filename = 'row' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'
                            with open(os.path.join(folder_result, result_filename), 'w', encoding="utf-8") as f:
                                json.dump(s, f, indent=4)

                            data_json = {
                                "data": data_result
                            }
                            score_filename = 'score' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'

                            with open(os.path.join(folder_score, score_filename), 'w', encoding="utf-8") as f:
                                json.dump(score_dist, f, indent=4)
                            
                            data_filename = 'data' + str(i) + '-' + str(cycle) + '-' + str(algo.iteration) + '.json'
                            with open(os.path.join(folder_data, data_filename), 'w', encoding="utf-8") as f:
                                json.dump(data_json, f, indent=4)

        self._emit.emit(1.0)

# A thread for generating stl file
# The time is very long, so keep it inside the thread
class StlOutputThread(QThread):
    _emit = pyqtSignal(float)
 
    def __init__(
            self, 
            stl_writer: StlMaker, 
            show_process, 
            method,
            connection_enabled,
            board_enabled,
            bias,
            filepath,
            pref_pack):
        super().__init__()
        self.stl_writer = stl_writer
        self.show_process = show_process
        self.method = method
        self.connection_enabled = connection_enabled
        self.board_enabled = board_enabled
        self.bias = bias
        self.file_path = filepath
        self.pref_pack = pref_pack
 
    def run(self):
        show_process = self.show_process
        method = self.method
        connection_enabled = self.connection_enabled
        board_enabled = self.board_enabled
        bias = self.bias
        file_path = self.file_path
        pref_pack = self.pref_pack
        crease_file_path_dxf = file_path.split('.')[0] + '_crease.dxf'
        self.stl_writer.crease_file_path = crease_file_path_dxf

        try:
            # normal method
            if method == "upper_bias" or method == "both_bias":
                # generate unit
                if show_process:
                    unit_size = self.stl_writer.size()
                    if connection_enabled and board_enabled:
                        divide = 2.4
                    elif connection_enabled and not board_enabled:
                        divide = 1.2
                    elif not connection_enabled and board_enabled:
                        divide = 2
                    else:
                        divide = 1.0
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForUnit(i)
                        self._emit.emit(i / unit_size / divide * 2 / 3)
                else:
                    self.stl_writer.calculateTriPlaneForAllUnit()
                # output unit stl
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                    for i in range(unit_size):
                        tris = self.stl_writer.tri_list[i]
                        self.stl_writer.addInfoToStlFile(tris)
                        self._emit.emit(2 / divide / 3 + i / unit_size / divide / 3)
                    self.stl_writer.s += 'endsolid\n'
                    with open(file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                else:
                    self.stl_writer.outputAllStl()

                # generate crease
                connection_flag = ""
                if show_process:
                    crease_size = self.stl_writer.validCreaseSize()
                    if board_enabled:
                        base = 0.417
                    else:
                        base = 0.834
                if connection_enabled:
                    crease_file_path = file_path.split('.')[0] + '_crease.stl'
                    
                    if show_process:
                        for i in range(crease_size):
                            self.stl_writer.calculateTriPlaneForCrease(i)          
                            self._emit.emit(base + i / crease_size * 0.083)
                    else:
                        self.stl_writer.calculateTriPlaneForAllCrease()
                    # output crease stl
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                        for i in range(crease_size):
                            tris = self.stl_writer.crease_tri_list[i]
                            if tris != None:
                                self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(base + 0.083 + i / crease_size * 0.083)
                        self.stl_writer.s += 'endsolid\n'
                        with open(crease_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllCreaseStl(crease_file_path)
                    connection_flag = "(+ *_crease.stl)"

                # generate board
                if show_process:
                    if connection_flag:
                        head = 0.584
                    else:
                        head = 0.5
                if board_enabled:
                    board_file_path = file_path.split('.')[0] + '_board.stl'
                    if show_process:
                        for i in range(unit_size):
                            self.stl_writer.calculateTriPlaneForBoard(i)
                            self._emit.emit(head + i / unit_size * (1 - head))
                    else:
                        self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

            #symmetry method
            elif method == "symmetry":
                unit_size = self.stl_writer.size()               
                # set difference
                self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                if self.stl_writer.db_enable:
                    self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)
                    soft_file_path = file_path.split('.')[0] + '_S.stl'
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=True)
                            self._emit.emit(unit_id / unit_size * 0.133)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit(inner=True)
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.133 + i / unit_size * 0.067)
                        self.stl_writer.s += 'endsolid\n'
                        with open(soft_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(soft_file_path)
                    hard_file_path = file_path.split('.')[0] + '.stl'
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=False)
                            self._emit.emit(0.2 + unit_id / unit_size * 0.133)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit(inner=False)
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.333 + i / unit_size * 0.067)
                        self.stl_writer.s += 'endsolid\n'
                        with open(hard_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(hard_file_path)
                else:
                    self.stl_writer.setBoardHeight(3.0 * pref_pack["print_accuracy"])
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id)
                            self._emit.emit(unit_id / unit_size * 0.266)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit()
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.266 + i / unit_size * 0.133)
                        self.stl_writer.s += 'endsolid\n'
                        with open(file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(file_path)

                connection_flag = ""
 
                # set difference
                self.stl_writer.enable_difference = 0
                self.stl_writer.setBias(pref_pack["board_bias"])
                self.stl_writer.setHoleWidth(1e-5)
                self.stl_writer.setHoleLength(1e-5)
                self.stl_writer.getAdditionalLineForAllUnit()
                crease_size = self.stl_writer.validCreaseSize()
                
                crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self._emit.emit(0.5)
                else:
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                if show_process:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                    self._emit.emit(0.6)
                else:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                # if show_process:
                #     self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                #     for i in range(crease_size):
                #         self.stl_writer.calculateTriPlaneForCrease(i)
                #         self._emit.emit(0.4 + i / crease_size * 0.1)
                # else:
                #     self.stl_writer.calculateTriPlaneForAllCrease()
                # if show_process:
                #     for i in range(crease_size):
                #         tris = self.stl_writer.crease_tri_list[i]
                #         if tris != None:
                #             self.stl_writer.addInfoToStlFile(tris)
                #         self._emit.emit(0.5 + i / crease_size * 0.1)
                #     self.stl_writer.s += 'endsolid\n'
                #     with open(crease_file_path, 'w') as f:
                #         f.write(self.stl_writer.s)
                # else:
                #     self.stl_writer.outputAllCreaseStl(crease_file_path)
                connection_flag = "(+ *_midlayer_C.stl)"

                board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                if show_process:
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForBoard(i)
                        self._emit.emit(0.6 + i / unit_size * 0.4)
                else:
                    self.stl_writer.generateBoard()
                self.stl_writer.outputBoardStl(board_file_path)

                if (len(self.stl_writer.string_list)):
                    self.stl_writer.calculateTriPlaneForString()
                    string_file_path = file_path.split('.')[0] + '_string.stl'
                    self.stl_writer.outputStringStl(string_file_path)
            elif method == 'binding':
                unit_size = self.stl_writer.size()               

                self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)

                hard_file_path = file_path.split('.')[0] + '.stl'
                if show_process:
                    for unit_id in range(unit_size):
                        self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=False)
                        self._emit.emit(unit_id / unit_size * 0.333)
                else:
                    self.stl_writer.calculateTriPlaneForAllUnit(inner=False)
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                    for i in range(unit_size):
                        tris = self.stl_writer.tri_list[i]

                        if len(self.stl_writer.unit_rotation_matrix):
                            # calculate new norm and point
                            new_tris = []
                            for ele in tris:
                                new_points = []
                                for point in ele[1]:
                                    origin = np.array([point[X], point[Y], point[Z] - 0.5 * (self.stl_writer.height + self.stl_writer.board_height)])
                                    new_trans_point = self.stl_writer.unit_rotation_matrix[i] @ origin + self.stl_writer.unit_transformation_vector[i]
                                    new_points.append(new_trans_point.tolist())
                                new_tris.append(self.stl_writer.getTriangle(new_points[0], new_points[1], new_points[2]))
                            tris = new_tris

                        self.stl_writer.addInfoToStlFile(tris)
                        self._emit.emit(0.333 + i / unit_size * 0.067)
                    self.stl_writer.s += 'endsolid\n'
                    with open(hard_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                else:
                    self.stl_writer.outputAllStl(hard_file_path)

                connection_flag = ""

                # set difference
                self.stl_writer.enable_difference = 0
                self.stl_writer.setBias(pref_pack["board_bias"])
                self.stl_writer.setHoleWidth(1e-5)
                self.stl_writer.setHoleLength(1e-5)
                self.stl_writer.getAdditionalLineForAllUnit()
                crease_size = self.stl_writer.validCreaseSize()
                
                crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self._emit.emit(0.5)
                else:
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                if show_process:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                    self._emit.emit(0.6)
                else:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                connection_flag = "(+ *_midlayer_C.stl)"

                board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                if show_process:
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForBoard(i)
                        self._emit.emit(0.6 + i / unit_size * 0.4)
                    self.stl_writer.calculateTriPlaneForPillar()
                else:
                    self.stl_writer.generateBoard()
                self.stl_writer.outputBoardStl(board_file_path)

                if (len(self.stl_writer.string_list)):
                    self.stl_writer.calculateTriPlaneForString()
                    string_file_path = file_path.split('.')[0] + '_string.stl'
                    self.stl_writer.outputStringStl(string_file_path)

            self._emit.emit(1.0)
        except Exception as e:
            print("No")
            self._emit.emit(1.0)

# class MCTSThread(QThread):
#     _emit = pyqtSignal(float)

#     def __init__(self, mcts, batch_size, origami_size, pref_pack, limitation, units, max_edge, input_units, max_size, total_bias, file_path) -> None:
#         super().__init__()
#         self.mcts = mcts
#         self.batch_size = batch_size
#         self.origami_size = origami_size
#         self.pref_pack = pref_pack
#         self.limitation = limitation
#         self.units = units
#         self.max_edge = max_edge
#         self.input_units = input_units
#         self.max_size = max_size
#         self.total_bias = total_bias
#         self.file_path = file_path

#     def run(self):
#         mcts = self.mcts
#         batch_size = self.batch_size
#         origami_size = self.origami_size
#         max_edge = self.max_edge
#         input_units = self.input_units
#         max_size = self.max_size
#         total_bias = self.total_bias

#         scores = []
#         for i in range(self.limitation["mcts_epoch"]):
#             methods, initial_method = mcts.ask(batch_size, i)

#             try:
#                 with open(os.path.join(self.file_path, "current.json"), 'w', encoding="utf-8") as f:
#                     json.dump(initial_method[0], f, indent=4)
#             except:
#                 pass
            
#             self._emit.emit(i / self.limitation["mcts_epoch"])

#             valid_number = len(methods)
#             if valid_number >= 1: 
#                 print("There are " + str(valid_number) + " valid cases, using multi-process technology")
#                 initial_fitness_list = []
#                 for j in range(len(methods)):
#                     initial_fitness_list.append(0.0)

#                 mlist = multiprocessing.Manager().list(initial_fitness_list)

#                 p_list = []

#                 pointer = 0

#                 while pointer < len(methods):
#                     p = multiprocessing.Process(target=workerMultisim, args=(
#                             mlist, methodToTotalInformation(methods[pointer], mcts.P_points, mcts.O_points), pointer,
#                             self.pref_pack, max_edge, input_units, max_size, total_bias
#                         )
#                     )
#                     p_list.append(p)
#                     pointer += 1
                
#                 process_id = 0

#                 total_process_number = 4

#                 current_process_number = 0

#                 while process_id < len(methods):
#                     while current_process_number < total_process_number:
#                         p_list[process_id].start()
#                         current_process_number += 1
#                         process_id += 1
#                         if current_process_number == total_process_number or process_id == len(methods):
#                             break

#                     while current_process_number > 0:
#                         p_list[process_id - current_process_number].join()
#                         current_process_number -= 1

#                 reward_list = list(mlist)

#             elif valid_number == 1:
#                 print("Only 1 valid case, using 1 process")
#                 reward_list = [0.0]

#                 ori_sim = OrigamiSimulator(use_gui=False)

#                 ori_sim.string_total_information = methodToTotalInformation(methods[0], mcts.P_points, mcts.O_points)
#                 ori_sim.pref_pack = self.pref_pack

#                 ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
#                 ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
#                 ori_sim.initializeRunning()
                
#                 while 1:
#                     ori_sim.step()
#                     if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 500 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
#                         break
                    
#                 if not ori_sim.can_rotate:
#                     print("Epoch: " + str(i) + ", Batch: " + str(0) + ", Value: Error Actuation")
#                     reward_list[0] = 0.0
#                 else:
#                     folding_percent = ori_sim.recorded_folding_percent[-1]
#                     folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])

#                     value = folding_speed * folding_percent
#                     print("Epoch: " + str(i) + ", Batch: " + str(j) + ", Value: " + str(value))
                    
#                     reward_list[0] = value

#             mcts.tell(reward_list)

#             maximum_reward = max(reward_list)
#             maximum_reward_index = reward_list.index(maximum_reward)
#             scores.append(maximum_reward)

#             print("Epoch: " + str(i) + ", Max Value: " + str(maximum_reward) + ", Total batch size: " + str(len(reward_list)))

#             best_method = methods[maximum_reward_index]
#             total_string = deepcopy(best_method)
#             total_string["score"] = maximum_reward

#             try:
#                 with open(os.path.join(self.file_path, "result_epoch_" + str(i) + "_score_" + str(round(maximum_reward, 2))) + ".json", 'w', encoding="utf-8") as f:
#                     json.dump(total_string, f, indent=4)
#             except:
#                 pass
            
#             score_list = {
#                 "score": scores
#             }

#             try:
#                 with open(os.path.join(self.file_path, "score.json"), 'w', encoding="utf-8") as f:
#                     json.dump(score_list, f, indent=4)
#             except:
#                 pass
        
#         self._emit.emit(1.0)
#         print("END TRAINING!")

class ThreadingMethodSearchingThread(QThread):
    _emit = pyqtSignal(float)
 
    def __init__(
            self, file_path, ori_simulator):
        super().__init__()
        self.file_path = file_path
        self.ori_simulator = ori_simulator

    def run(self):
        self._emit.emit(0.0)
        ori_sim = self.ori_simulator

        while 1:
            ori_sim.step()
            if ori_sim.stop():
                break
            self._emit.emit(ori_sim.folding_percent)

        all_dis = {
            "control_string_decrease": [],
            "string_decrease_each": [],
            "max_force": [],
            "folding_percent": [],
            "max_folding_percent": [],
            "min_folding_percent": [],
            "time": []
        }

        all_dis["control_string_decrease"] = ori_sim.recorded_string_decrease_length_control
        all_dis["string_decrease_each"] = ori_sim.recorded_string_decrease_length
        all_dis["folding_percent"] = ori_sim.recorded_folding_percent
        all_dis["max_folding_percent"] = ori_sim.recorded_maximum_folding_percent
        all_dis["min_folding_percent"] = ori_sim.recorded_minimum_folding_percent
        all_dis["max_force"] = ori_sim.recorded_max_force
        all_dis["time"] = ori_sim.recorded_t
        all_dis["deal"] = 1

        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump(all_dis, f, indent=4)

        self._emit.emit(1.0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = Mainwindow()
    a.dataConv()