# import numpy as np
# import math
import re
import sys

sys.path.append('./gui')

from PyQt5.QtWidgets import QDialog
from gui.Ui_pref_window import Ui_Settings

class PreferencePackWindow(Ui_Settings, QDialog):
    def __init__(self, parent=None) -> None:
        QDialog.__init__(self, parent)
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.onAccepted)
        self.buttonBox.rejected.connect(self.onRejected)

        self.pref_pack = {
            "print_accuracy": 0.2,
            "split_distance": 1.0,
            "dxf_hole_enable": True,
            "output_bc": True,
            "board_bias" : 1.4,
            "connection_angle": 0,
            "middle_bias": 0.1,
            "enable_db": False,
            "enable_db_bind": False,
            "additional_line_option": 0,
            "stl_asymmetry": True,
            "cursor_axis_mode": "pixel_axis",
            "line_weight": 3,
            "show_keypoint": True,
            "theme": 0,
            "show_additional_lines": True,
            "tsa_radius": 100.00,
            "tsa_resolution": 72,
            "debug_mode": False,
            "thin_mode": False,
            "layer": 3,
            "layer_of_panel": 5,
            "asym": False,
            "only_two_sides": False,
            "disable_pillars": False,
            "fast_simulation_mode": False,
            "elastic_hinge_length": 10.0,
            "elastic_hinge_width": 10.0,
            "elastic_hinge_outer_length": 10.0,
            "elastic_hinge_outer_width": 10.0,
            "normal_hinge_length": 10.0,
            "normal_hinge_width": 10.0,
            "normal_hinge_outer_length": 10.0,
            "normal_hinge_outer_width": 10.0,
            "elastic_hinge_arrangement": 2,
            "normal_hinge_arrangement": 2,
            "elastic_hinge_hole_radius": 2.0,
            "normal_hinge_hole_radius": 2.0,
            "elastic_hinge_usage": 2
        }

        self.limitation = {
            "unit_length": [0, 200.0],
            "miura_angle": [15.0, 75.0],
            "row_number": [4, 5],
            "generation": 100000,
            "batch_size": 10,
            "process_number": 1,
            "storage": 100,
            "match_mode": "discrete",
            "discrete_resolution": 2,
            "hypar_enable": True,
            "5fold_enable": False,
            "direction_enable": False,
            "3D_match_enable": False,
            "mcts_epoch": 100,
            "exo_angle1": 90.0,
            "exo_angle2": 180.0,
            "exo_X": 0.0,
            "exo_Y": 0.0,
            "exo_theta": 0.0
        }

        self.ok = False
        self.default_path = "./setting/setting.log"

        self.readFile()
        self.setUiFromData()

    def getPrefPack(self):
        return self.pref_pack
    
    def getLimitation(self):
        return self.limitation
    
    def onAccepted(self):
        self.ok = True
        self.saveDataFromUi()
        self.saveFile()

    def onRejected(self):
        self.ok = False
    
    def setUiFromData(self):
        """
        @ function: carry data from inner space to Ui
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        #1
        self.doubleSpinBox_print_accuracy.setValue(self.pref_pack["print_accuracy"])
        #11
        self.doubleSpinBox_dxfsplit_dis.setValue(self.pref_pack["split_distance"])
        self.checkBox_output_holes_to_dxf.setChecked(self.pref_pack["dxf_hole_enable"])
        #2
        self.checkBox_output_bc.setChecked(self.pref_pack["output_bc"])
        #3
        self.doubleSpinBox_board_bias.setValue(self.pref_pack["board_bias"])
        #31
        self.doubleSpinBox_connection_angle.setValue(self.pref_pack["connection_angle"])
        #32
        self.checkBox_db.setChecked(self.pref_pack["enable_db"])
        #321
        self.checkBox_db_bind.setChecked(self.pref_pack["enable_db_bind"])
        #33
        if self.pref_pack["additional_line_option"] == 0:
            self.radioButton_nodiff.setChecked(True)
            self.radioButton_valleysmall.setChecked(False)
            self.radioButton_mountainsmall.setChecked(False)
        elif self.pref_pack["additional_line_option"] == 1:
            self.radioButton_nodiff.setChecked(False)
            self.radioButton_valleysmall.setChecked(True)
            self.radioButton_mountainsmall.setChecked(False)
        else:
            self.radioButton_nodiff.setChecked(False)
            self.radioButton_valleysmall.setChecked(False)
            self.radioButton_mountainsmall.setChecked(True)
        #34
        self.checkBox_stl_asymmetry.setChecked(self.pref_pack["stl_asymmetry"])
        #35
        self.checkBox_thin_mode.setChecked(self.pref_pack["thin_mode"])
        #36
        self.spinBox_layer.setValue(self.pref_pack["layer"])
        #362
        self.spinBox_layer_panel.setValue(self.pref_pack["layer_of_panel"])
        #37
        self.doubleSpinBox_middle_bias.setValue(self.pref_pack["middle_bias"])
        #38
        self.checkBox_asymmetric.setChecked(self.pref_pack["asym"])
        #39
        self.checkBox_onl_two_side.setChecked(self.pref_pack["only_two_sides"])
        #310
        self.checkBox_disable_pillars.setChecked(self.pref_pack["disable_pillars"])
        #4 We need to know which mode is
        cursor_axis_mode = self.pref_pack["cursor_axis_mode"]
        if cursor_axis_mode == "pixel_axis":
            self.radioButton_pixel_axis.setChecked(True)
            self.radioButton_real_axis.setChecked(False)
        elif cursor_axis_mode == "real_axis":
            self.radioButton_pixel_axis.setChecked(False)
            self.radioButton_real_axis.setChecked(True)
        #5
        self.spinBox_line_weight.setValue(self.pref_pack["line_weight"])
        #6
        self.checkBox_show_kp.setChecked(self.pref_pack["show_keypoint"])
        #7
        theme = self.pref_pack["theme"]
        if theme == 0:
            self.radioButton_light_theme.setChecked(True)
        else:
            self.radioButton_dark_theme.setChecked(True)
        #71
        self.radioButton_show_additional_lines.setChecked(self.pref_pack["show_additional_lines"])
        #8
        self.doubleSpinBox_tsa_radius.setValue(self.pref_pack["tsa_radius"])
        #9
        self.spinBox_tsa_resolution.setValue(self.pref_pack["tsa_resolution"])
        #10
        self.checkBox_debug_mode.setChecked(self.pref_pack["debug_mode"])
        #11
        self.radioButton_fast_simulation_mode.setChecked(self.pref_pack["fast_simulation_mode"])
        #21
        self.doubleSpinBox_elastic_hinge_length.setValue(self.pref_pack["elastic_hinge_length"])
        self.doubleSpinBox_elastic_hinge_width.setValue(self.pref_pack["elastic_hinge_width"])
        self.doubleSpinBox_elastic_hinge_outer_length.setValue(self.pref_pack["elastic_hinge_outer_length"])
        self.doubleSpinBox_elastic_hinge_outer_width.setValue(self.pref_pack["elastic_hinge_outer_width"])
        self.doubleSpinBox_elastic_hinge_arrangement.setValue(self.pref_pack["elastic_hinge_arrangement"])
        self.doubleSpinBox_elastic_hinge_hole_radius.setValue(self.pref_pack["elastic_hinge_hole_radius"])
        self.doubleSpinBox_elastic_hinge_usage_maximum.setValue(self.pref_pack["elastic_hinge_usage"])

        self.doubleSpinBox_normal_hinge_length.setValue(self.pref_pack["normal_hinge_length"])
        self.doubleSpinBox_normal_hinge_width.setValue(self.pref_pack["normal_hinge_width"])
        self.doubleSpinBox_normal_hinge_outer_length.setValue(self.pref_pack["normal_hinge_outer_length"])
        self.doubleSpinBox_normal_hinge_outer_width.setValue(self.pref_pack["normal_hinge_outer_width"])
        self.doubleSpinBox_normal_hinge_arrangement.setValue(self.pref_pack["normal_hinge_arrangement"])
        self.doubleSpinBox_normal_hinge_hole_radius.setValue(self.pref_pack["normal_hinge_hole_radius"])
        
        #LIMITATION
        #1
        self.doubleSpinBox_unit_length_min.setValue(self.limitation["unit_length"][0])
        self.doubleSpinBox_unit_length_max.setValue(self.limitation["unit_length"][1])
        #2
        self.doubleSpinBox_miura_angle_min.setValue(self.limitation["miura_angle"][0])
        self.doubleSpinBox_miura_angle_max.setValue(self.limitation["miura_angle"][1])
        #3
        self.spinBox_row_number_min.setValue(int(self.limitation["row_number"][0]))
        self.spinBox_row_number_max.setValue(int(self.limitation["row_number"][1]))
        #4
        self.spinBox_generation.setValue(self.limitation["generation"])
        #5
        self.spinBox_batch_size.setValue(self.limitation["batch_size"])
        #6
        self.spinBox_process_number.setValue(self.limitation["process_number"])
        #7
        self.spinBox_storage.setValue(self.limitation["storage"])
        #8
        if self.limitation["match_mode"] == 0:
            self.radioButton_bestmatch.setChecked(True)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(False)
            self.radioButton_exomatch.setChecked(False)
            self.radioButton_zeromatch.setChecked(False)
        elif self.limitation["match_mode"] == 1:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(True)
            self.radioButton_discretematch.setChecked(False)
            self.radioButton_exomatch.setChecked(False)
            self.radioButton_zeromatch.setChecked(False)
        elif self.limitation["match_mode"] == 2:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(True)
            self.radioButton_exomatch.setChecked(False)
            self.radioButton_zeromatch.setChecked(False)
        elif self.limitation["match_mode"] == 3:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(False)
            self.radioButton_exomatch.setChecked(True)
            self.radioButton_zeromatch.setChecked(False)
        else:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(False)
            self.radioButton_exomatch.setChecked(False)
            self.radioButton_zeromatch.setChecked(True)
        #9
        self.spinBox_discrete_resolution.setValue(self.limitation["discrete_resolution"])
        #10
        self.checkBox_hypar_enable.setChecked(self.limitation["hypar_enable"])
        self.checkBox_5fold_enable.setChecked(self.limitation["5fold_enable"])
        self.checkBox_3D_Match_enable.setChecked(self.limitation["3D_match_enable"])
        #11
        self.checkBox_direction_enable_2.setChecked(self.limitation["direction_enable"])
        #12
        self.spinBox_mcts_epoch.setValue(self.limitation["mcts_epoch"])
        #13
        self.doubleSpinBox_exo_angle_1.setValue(self.limitation["exo_angle1"])
        #14
        self.doubleSpinBox_exo_angle_2.setValue(self.limitation["exo_angle2"])
        #15
        self.doubleSpinBox_state2_X.setValue(self.limitation["exo_X"])
        #16
        self.doubleSpinBox_state2_Y.setValue(self.limitation["exo_Y"])
        #17
        self.doubleSpinBox_state2_dir.setValue(self.limitation["exo_theta"])

    def saveDataFromUi(self):
        """
        @ function: carry data from Ui to inner space
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        #1
        self.pref_pack["print_accuracy"] = self.doubleSpinBox_print_accuracy.value()
        #11
        self.pref_pack["split_distance"] = self.doubleSpinBox_dxfsplit_dis.value()
        self.pref_pack["dxf_hole_enable"] = self.checkBox_output_holes_to_dxf.isChecked()
        #2
        self.pref_pack["output_bc"] = self.checkBox_output_bc.isChecked()
        #3
        self.pref_pack["board_bias"] = self.doubleSpinBox_board_bias.value()
        #31
        self.pref_pack["connection_angle"] = self.doubleSpinBox_connection_angle.value()
        #32
        self.pref_pack["enable_db"] = self.checkBox_db.isChecked()
        #321
        self.pref_pack["enable_db_bind"] = self.checkBox_db_bind.isChecked()
        #33
        if self.radioButton_nodiff.isChecked():
            self.pref_pack["additional_line_option"] = 0
        elif self.radioButton_valleysmall.isChecked():
            self.pref_pack["additional_line_option"] = 1
        elif self.radioButton_mountainsmall.isChecked():
            self.pref_pack["additional_line_option"] = 2
        #34
        self.pref_pack["stl_asymmetry"] = self.checkBox_stl_asymmetry.isChecked()
        #35
        self.pref_pack["thin_mode"] = self.checkBox_thin_mode.isChecked()
        #36
        self.pref_pack["layer"] = self.spinBox_layer.value()
        #362
        self.pref_pack["layer_of_panel"] = self.spinBox_layer_panel.value()
        #37
        self.pref_pack["middle_bias"] = self.doubleSpinBox_middle_bias.value()
        #38
        self.pref_pack["asym"] = self.checkBox_asymmetric.isChecked()
        #39
        self.pref_pack["only_two_sides"] = self.checkBox_onl_two_side.isChecked()
        #310
        self.pref_pack["disable_pillars"] = self.checkBox_disable_pillars.isChecked()
        #4 We need to know which radio button is checked
        cursor_axis_mode = ""
        if self.radioButton_pixel_axis.isChecked():
            cursor_axis_mode = "pixel_axis"
        elif self.radioButton_real_axis.isChecked():
            cursor_axis_mode = "real_axis"
        self.pref_pack["cursor_axis_mode"] = cursor_axis_mode
        #5
        self.pref_pack["line_weight"] = self.spinBox_line_weight.value()
        #6
        self.pref_pack["show_keypoint"] = self.checkBox_show_kp.isChecked()
        #7
        theme = 0
        if self.radioButton_light_theme.isChecked():
            theme = 0
        else:
            theme = 1
        self.pref_pack["theme"] = theme
        #71
        self.pref_pack["show_additional_lines"] = self.radioButton_show_additional_lines.isChecked()
        #8
        self.pref_pack["tsa_radius"] = self.doubleSpinBox_tsa_radius.value()
        #9
        self.pref_pack["tsa_resolution"] = self.spinBox_tsa_resolution.value()
        #10
        self.pref_pack["debug_mode"] = self.checkBox_debug_mode.isChecked()
        #11
        self.pref_pack["fast_simulation_mode"] = self.radioButton_fast_simulation_mode.isChecked()
        #21
        self.pref_pack["elastic_hinge_length"] = self.doubleSpinBox_elastic_hinge_length.value()
        self.pref_pack["elastic_hinge_width"] = self.doubleSpinBox_elastic_hinge_width.value()
        self.pref_pack["elastic_hinge_outer_length"] = self.doubleSpinBox_elastic_hinge_outer_length.value()
        self.pref_pack["elastic_hinge_outer_width"] = self.doubleSpinBox_elastic_hinge_outer_width.value()
        self.pref_pack["elastic_hinge_arrangement"] = self.doubleSpinBox_elastic_hinge_arrangement.value()
        self.pref_pack["elastic_hinge_hole_radius"] = self.doubleSpinBox_elastic_hinge_hole_radius.value()
        self.pref_pack["elastic_hinge_usage"] = self.doubleSpinBox_elastic_hinge_usage_maximum.value()

        self.pref_pack["normal_hinge_length"] = self.doubleSpinBox_normal_hinge_length.value()
        self.pref_pack["normal_hinge_width"] = self.doubleSpinBox_normal_hinge_width.value()
        self.pref_pack["normal_hinge_outer_length"] = self.doubleSpinBox_normal_hinge_outer_length.value()
        self.pref_pack["normal_hinge_outer_width"] = self.doubleSpinBox_normal_hinge_outer_width.value()
        self.pref_pack["normal_hinge_arrangement"] = self.doubleSpinBox_normal_hinge_arrangement.value()
        self.pref_pack["normal_hinge_hole_radius"] = self.doubleSpinBox_normal_hinge_hole_radius.value()

        #LIMITATION
        #1
        self.limitation["unit_length"] = [
            self.doubleSpinBox_unit_length_min.value(), self.doubleSpinBox_unit_length_max.value()
        ]
        #2
        self.limitation["miura_angle"] = [
            self.doubleSpinBox_miura_angle_min.value(), self.doubleSpinBox_miura_angle_max.value()
        ]
        #3
        self.limitation["row_number"] = [
            self.spinBox_row_number_min.value(), self.spinBox_row_number_max.value()
        ]
        #4
        self.limitation["generation"] = self.spinBox_generation.value()
        #5
        self.limitation["batch_size"] = self.spinBox_batch_size.value()
        #6
        self.limitation["process_number"] = self.spinBox_process_number.value()
        #7
        self.limitation["storage"] = self.spinBox_storage.value()
        #8
        if self.radioButton_bestmatch.isChecked():
            self.limitation["match_mode"] = 0
        elif self.radioButton_strict_match.isChecked():
            self.limitation["match_mode"] = 1
        elif self.radioButton_discretematch.isChecked():
            self.limitation["match_mode"] = 2
        elif self.radioButton_exomatch.isChecked():
            self.limitation["match_mode"] = 3
        elif self.radioButton_zeromatch.isChecked():
            self.limitation["match_mode"] = 4
        #9
        self.limitation["discrete_resolution"] = self.spinBox_discrete_resolution.value()
        #10
        self.limitation["hypar_enable"] = self.checkBox_hypar_enable.isChecked()
        self.limitation["5fold_enable"] = self.checkBox_5fold_enable.isChecked()
        self.limitation["3D_match_enable"] = self.checkBox_3D_Match_enable.isChecked()
        #11
        self.limitation["direction_enable"] = self.checkBox_direction_enable_2.isChecked()
        #12
        self.limitation["mcts_epoch"] = self.spinBox_mcts_epoch.value()
        #13
        self.limitation["exo_angle1"] = self.doubleSpinBox_exo_angle_1.value()
        #14
        self.limitation["exo_angle2"] = self.doubleSpinBox_exo_angle_2.value()
        #15
        self.limitation["exo_X"] = self.doubleSpinBox_state2_X.value()
        #16
        self.limitation["exo_Y"] = self.doubleSpinBox_state2_Y.value()
        #17
        self.limitation["exo_theta"] = self.doubleSpinBox_state2_dir.value()

    def saveFile(self):
        """
        @ function: save the setting file
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        file_path = self.default_path
        s = ""
        for key in self.pref_pack:
            s += "[" + key + "] = " + str(self.pref_pack[key]) + '\n'
        
        for key in self.limitation:
            if key == "generation":
                break
            s += "[" + key + "] = [" + str(self.limitation[key][0]) + ', ' + str(self.limitation[key][1]) + ']\n'   
        for ele in ["generation", "batch_size", "process_number", "storage", "match_mode", "discrete_resolution", "3D_match_enable",
                    "hypar_enable", "5fold_enable", "direction_enable", "mcts_epoch", "exo_angle1", "exo_angle2", "exo_X", "exo_Y", "exo_theta"]:
            s += "[" + ele + "] = " + str(self.limitation[ele]) + '\n'

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(s)

    def readFile(self):
        """
        @ function: read the setting file
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        path = self.default_path
        try:
            with open(path, 'r', encoding='utf-8') as fw:
                s = fw.readlines()
                string_len = len(s)
                for i in range(string_len):
                    segment = s[i].split(' ', 2)
                    head = segment[0][1: -1]
                    content = segment[-1][0: -1]
                    if head not in [
                        "print_accuracy", "split_distance", "output_bc", "board_bias", "asym", "only_two_sides", "disable_pillars", "dxf_hole_enable",
                        "connection_angle", "enable_db", "enable_db_bind", "additional_line_option", "stl_asymmetry", "thin_mode", "layer", "layer_of_panel", "middle_bias", 
                        "cursor_axis_mode", "line_weight", "show_keypoint", "theme", "show_additional_lines", "tsa_radius", "tsa_resolution", "fast_simulation_mode", 
                        "unit_length", "miura_angle", "row_number", "generation", 
                        "batch_size", "process_number", "storage", "match_mode", "discrete_resolution", "5fold_enable", "3D_match_enable",
                        "hypar_enable", "direction_enable", "debug_mode", "mcts_epoch", "exo_angle1", "exo_angle2", "exo_X", "exo_Y", "exo_theta",
                        "elastic_hinge_length", "elastic_hinge_width", "elastic_hinge_outer_length", "elastic_hinge_outer_width", "elastic_hinge_usage",
                        "elastic_hinge_arrangement", "elastic_hinge_hole_radius", 
                        "normal_hinge_length", "normal_hinge_width", "normal_hinge_outer_length", "normal_hinge_outer_width",
                        "normal_hinge_arrangement", "normal_hinge_hole_radius"
                    ]:
                        raise TypeError
                    #1
                    if head in ["print_accuracy", "split_distance", "board_bias", "connection_angle", "middle_bias", "tsa_radius", 
                                "elastic_hinge_length", "elastic_hinge_width", "elastic_hinge_outer_length", "elastic_hinge_outer_width",
                                "elastic_hinge_hole_radius", 
                                "normal_hinge_length", "normal_hinge_width", "normal_hinge_outer_length", "normal_hinge_outer_width",
                                "normal_hinge_hole_radius", "elastic_hinge_usage", "elastic_hinge_arrangement", "normal_hinge_arrangement"]:
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #2
                    elif head in ["output_bc", "enable_db", "enable_db_bind", "stl_asymmetry", "thin_mode", "show_keypoint", "show_additional_lines", "debug_mode", "asym", "only_two_sides", "disable_pillars", "fast_simulation_mode", "dxf_hole_enable"]:
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #33
                    elif head in ["additional_line_option", "layer", "layer_of_panel", "line_weight", "theme", "tsa_resolution"]:
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #4
                    elif head == "cursor_axis_mode":
                        self.pref_pack[head] = content

                    #LIMITATION
                    #1
                    elif head in ["unit_length", "miura_angle", "row_number"]:
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 2:
                            self.limitation[head] = [float(ans[0]), float(ans[1])]
                            continue
                        else:
                            raise TypeError
                    #4
                    elif head in ["exo_angle1", "exo_angle2", "exo_X", "exo_Y", "exo_theta"]:
                        ans = re.findall(r"-?\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #4
                    elif head in ["generation", "batch_size", "process_number", "storage", "match_mode", "discrete_resolution", "mcts_epoch"]:
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #10
                    elif head in ["hypar_enable", "direction_enable", "5fold_enable", "3D_match_enable"]:
                        if content == "True":
                            self.limitation[head] = True
                            continue
                        elif content == "False":
                            self.limitation[head] = False
                            continue
                        else:
                            raise TypeError

        except:
            # No log file exists, we will create one
            self.saveFile()
        # Set Ui data
        self.setUiFromData()
