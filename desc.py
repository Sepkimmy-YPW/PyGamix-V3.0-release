import math
import numpy as np
from new_module_dialog import NewModuleDialog, LEFT_HALF, RIGHT_HALF, NO_HALF
from PyQt5.QtWidgets import QInputDialog
import re

from utils import MOUNTAIN, VALLEY, BORDER, FACET
import dxfgrabber

ORIGIN = 0
DATA = 1
ADD_WIDTH = 2

class KinematicLine:
    """
    运动学线段列表类。
    中文：
        用于存储和管理一系列运动学线段数据。
        支持列表式访问（索引和长度）。

    Kinematic line list class.
    English:
        Used to store and manage a series of kinematic line data.
        Supports list-style access (indexing and length).
    """
    def __init__(self) -> None:
        """
        初始化空的运动学线段列表。

        Initialize an empty kinematic line list.
        """
        self.lines = []

    def append(self, data):
        """
        添加线段数据到列表。

        Append line data to the list.

        :param data: 线段数据 / Line data
        """
        self.lines.append(data)

    def getData(self):
        """
        获取所有线段数据。

        Get all line data.

        :return: 线段列表 / List of lines
        """
        return self.lines
    
    def __getitem__(self, index):
        """
        通过索引获取线段（支持列表式访问）。

        Get line by index (supports list-style access).

        :param index: 索引 / Index
        :return: 线段数据 / Line data
        """
        return self.lines[index]
    
    def __len__(self):
        """
        返回线段数量。

        Return the number of lines.

        :return: 线段数量 / Number of lines
        """
        return len(self.lines)

class UnitPack:
    """
    单元包类，用于管理自定义折纸单元的点组。
    中文：
        存储多个单元的点组列表，支持通过对话框添加新的点组。
        每个点组定义一个自定义折纸单元的边界。

    Unit pack class for managing custom origami unit point groups.
    English:
        Stores multiple unit point groups, supports adding new groups via dialog.
        Each point group defines the boundary of a custom origami unit.
    """
    def __init__(self) -> None:
        """
        初始化空的单元包。

        Initialize an empty unit pack.
        """
        self.unit_point_list = []
        self.tsp = []

    def setTsp(self):
        """
        设置平移起始点（取最后一个点组的第一个点）。

        Set translation start point (first point of last point group).
        """
        self.tsp = self.unit_point_list[-1][0]

    def addPointGroup(self, unit):
        """
        添加点组到单元包。

        Add a point group to the unit pack.

        :param unit: 点组 [[x, y], ...] / Point group [[x, y], ...]
        """
        self.unit_point_list.append(unit)
    
    def getLength(self):
        """
        获取单元包中的点组数量。

        Get the number of point groups in the unit pack.

        :return: 点组数量 / Number of point groups
        """
        return len(self.unit_point_list)
    
    def raiseDialog(self, parent):
        """
        弹出对话框让用户输入点组坐标。
        中文：通过Qt输入对话框循环接收用户输入的2D坐标点，
              当输入点数>=3时保存为新的点组。

        Raise dialog for user to input point group coordinates.
        English: Uses Qt input dialog to loop receiving 2D coordinates from user.
                 Saves as new point group when >=3 points are entered.

        :param parent: 父窗口对象 / Parent window object
        """
        ok = True
        unit = []
        while ok:
            text, ok = QInputDialog.getText(parent, "Enter a 2D point axis", "2D point axis splited with ','")
            if not ok:
                break
            ans = re.findall(r"\d+\.?\d*", text)
            if len(ans) == 2:
                x = ans[0]
                y = ans[1]
                unit.append([x, y])
        if len(unit) >= 3:
            self.unit_point_list.append(unit)
            self.setTsp()


class ModuleLeanMiura:
    """
    斜三浦折纸模块类。
    中文：
        用于定义和管理斜三浦折纸（Lean Miura）模块的参数。
        包含单元宽度、复制次数、拉伸长度、连接参数等属性。
        支持通过对话框进行参数配置。

    Lean Miura origami module class.
    English:
        Defines and manages parameters for Lean Miura origami modules.
        Includes unit width, copy times, stretch length, connection parameters, etc.
        Supports parameter configuration via dialog.
    """
    def __init__(self, data=[]) -> None:
        """
        初始化斜三浦模块（所有参数为None，需调用initialize设置）。

        Initialize Lean Miura module (all params are None, call initialize to set).
        """
        self.unit_width = None
        self.entry_flag = None
        self.copy_time = None
        self.half_flag = None
        self.stretch_length = None
        self.connection_flag = None
        self.con_left_length = None
        self.con_right_length = None
        self.connection_hole_size = None
        self.modify_stretch_flag = None

        self.tsp = [0, 0]

        self.enable_global_modify = None

        self.initial_flag = False

    def initialize(self, unit_width, entry_flag, copy_time, stretch_length, connection_flag, con_left_length, con_right_length, con_radius, half_flag, tsp, enabled):
        """
        初始化模块的所有参数。

        Initialize all module parameters.

        :param unit_width: 单元宽度 / Unit width
        :param entry_flag: 入射标志 / Entry flag
        :param copy_time: 复制次数 / Copy times
        :param stretch_length: 拉伸长度 / Stretch length
        :param connection_flag: 连接标志 / Connection flag
        :param con_left_length: 左侧连接长度 / Left connection length
        :param con_right_length: 右侧连接长度 / Right connection length
        :param con_radius: 连接孔半径 / Connection hole radius
        :param half_flag: 半单元标志 / Half unit flag
        :param tsp: 平移起始点 [x, y] / Translation start point [x, y]
        :param enabled: 是否启用全局修改 / Enable global modification
        """
        self.initial_flag = True
        self.setUnitWidth(unit_width)
        self.setEntryFlag(entry_flag)
        self.setCopyTime(copy_time)
        self.setStretchLength(stretch_length)
        self.setConnectionFlag(connection_flag)
        self.setConnectionLeftLength(con_left_length)
        self.setConnectionRightLength(con_right_length)
        self.setConnectionHoleSize(con_radius)
        self.setHalfFlag(half_flag)
        self.setEnableModify(enabled)
        self.setTsp(tsp)
        
    def getData(self):
        """
        获取模块数据字典。

        Get module data dictionary.

        :return: 包含half_type和stretch_length的字典 / Dict with half_type and stretch_length
        """
        return {
            "half_type": self.half_flag,
            "stretch_length": self.stretch_length
        }
    
    def setUnitWidth(self, unit_width):
        """设置单元宽度 / Set unit width"""
        self.unit_width = unit_width

    def setEntryFlag(self, entry_flag):
        """设置入射标志 / Set entry flag"""
        self.entry_flag = entry_flag

    def setCopyTime(self, copy_time):
        """设置复制次数 / Set copy times"""
        self.copy_time = copy_time

    def setStretchLength(self, stretch_length):
        """设置拉伸长度 / Set stretch length"""
        self.stretch_length = stretch_length

    def setConnectionFlag(self, connection_flag):
        """设置连接标志 / Set connection flag"""
        self.connection_flag = connection_flag

    def setConnectionLeftLength(self, con_left_length):
        """设置左侧连接长度 / Set left connection length"""
        self.con_left_length = con_left_length

    def setConnectionRightLength(self, con_right_length):
        """设置右侧连接长度 / Set right connection length"""
        self.con_right_length = con_right_length

    def setConnectionHoleSize(self, con_radius):
        """设置连接孔大小 / Set connection hole size"""
        self.connection_hole_size = con_radius

    def setHalfFlag(self, half_flag):
        """设置半单元标志 / Set half unit flag"""
        self.half_flag = half_flag

    def setEnableModify(self, enabled: bool):
        """设置是否启用全局修改 / Set global modification enabled"""
        self.enable_global_modify = enabled
    
    def setTsp(self, tsp):
        """设置平移起始点 / Set translation start point"""
        self.tsp = tsp

    def raiseDialog(self, parent):
        """
        弹出模块配置对话框。
        中文：显示NewModuleDialog对话框，加载当前参数，
              用户确认后更新模块参数。

        Raise module configuration dialog.
        English: Displays NewModuleDialog with current parameters,
                 updates module parameters after user confirmation.

        :param parent: 父窗口对象 / Parent window object
        """
        dialog = NewModuleDialog(parent=parent)
        if self.initial_flag:
            dialog.doubleSpinBox_unit_width.setValue(self.unit_width)
            dialog.spinBox_copy_time.setValue(self.copy_time)
            dialog.horizontalSlider_entry_flag.setValue(self.entry_flag)
            dialog.checkBox_enable_connection.setChecked(self.connection_flag)
            dialog.doubleSpinBox_con_left.setValue(self.con_left_length)
            dialog.doubleSpinBox_con_right.setValue(self.con_right_length)
            dialog.doubleSpinBox_con_radius.setValue(self.connection_hole_size)
            if self.half_flag == LEFT_HALF:
                dialog.radioButton_left_half.setChecked(True)
                dialog.radioButton_right_half.setChecked(False)
                dialog.radioButton_all.setChecked(False)
            if self.half_flag == RIGHT_HALF:
                dialog.radioButton_left_half.setChecked(False)
                dialog.radioButton_right_half.setChecked(True)
                dialog.radioButton_all.setChecked(False)
            if self.half_flag == NO_HALF:
                dialog.radioButton_left_half.setChecked(False)
                dialog.radioButton_right_half.setChecked(False)
                dialog.radioButton_all.setChecked(True)
            dialog.doubleSpinBox_stretch_length.setValue(self.stretch_length)
            dialog.doubleSpinBox_tspx.setValue(self.tsp[0])
            dialog.doubleSpinBox_tspy.setValue(self.tsp[1])
        if dialog.exec_():
            pass
        if not dialog.getOK():
            return
        else:
            self.initialize(
                unit_width          =dialog.getUnitWidth(),
                copy_time           =dialog.getCopyTime(),
                entry_flag          =dialog.getEntryFlag(),
                stretch_length      =dialog.getStretchLength(),
                connection_flag     =dialog.getEnableConnection(),
                con_left_length     =dialog.getConnectionLeftLength(),
                con_right_length    =dialog.getConnectionRightLength(),
                con_radius          =dialog.getConnectionRadius(),
                half_flag           =dialog.getHalfFlag(),
                tsp                 =dialog.getTsp(),
                enabled             =dialog.getUsingGlobalData()
            )
        dialog.destroy()

class DxfDirectGrabber:
    """
    DXF文件直接读取器类。
    中文：
        用于直接读取DXF文件并提取折纸图案的关键点和折痕信息。
        支持两种模式：
        - 'kl'模式：读取运动学线段（LWPOLYLINE/POLYLINE）
        - 'normal_dxf'模式：读取普通DXF线段（LINE），根据图层或颜色识别折痕类型

    DXF file direct reader class.
    English:
        Reads DXF files and extracts key points and crease information.
        Supports two modes:
        - 'kl' mode: reads kinematic lines (LWPOLYLINE/POLYLINE)
        - 'normal_dxf' mode: reads regular DXF lines, identifies crease types by layer/color
    """
    def __init__(self) -> None:
        """
        初始化DXF读取器。

        Initialize the DXF reader.
        """
        self.lines = []
        self.lines_type = []
        self.maximum_folding_angle = []
        self.minimum_folding_angle = []
        self.kps = []
        self.units = []
        self.mode = 'normal_dxf'

        self.kl = []
        self.origin = []

    def readFile(self, path):
        """
        读取DXF文件并解析内容。
        中文：
            根据文件中的实体类型自动选择解析模式：
            - LWPOLYLINE/POLYLINE：解析为运动学线段（kl模式）
            - LINE：解析为普通折痕，根据图层名或颜色识别类型
              （MOUNTAIN/VALLEY/BORDER/FACET）

        Read and parse DXF file.
        English:
            Automatically selects parsing mode based on entity type:
            - LWPOLYLINE/POLYLINE: parses as kinematic lines (kl mode)
            - LINE: parses as regular creases, identifies type by layer name or color
              (MOUNTAIN/VALLEY/BORDER/FACET)

        :param path: DXF文件路径 / DXF file path
        """
        # 读取dxf文件
        self.kps.clear()
        self.lines.clear()
        self.lines_type.clear()
        self.kl.clear()
        self.origin.clear()
        dxf = dxfgrabber.readfile(path)
        print(dxf.entities[0].dxftype)
        if dxf.entities[0].dxftype in ['LWPOLYLINE', 'POLYLINE']:
            self.mode = 'kl'
            for entity in dxf.entities:
                sub_kl = []
                points = entity.points
                previous_angle = 0.0
                self.origin.append([points[0][0], points[0][1]])
                for i in range(1, len(points)):
                    current_point = points[i]
                    previous_point = points[i - 1]
                    length = math.sqrt((current_point[0] - previous_point[0]) ** 2 + (current_point[1] - previous_point[1]) ** 2)
                    angle = math.atan2(current_point[1] - previous_point[1], current_point[0] - previous_point[0])
                    while angle - previous_angle >= math.pi:
                        angle -= 2 * math.pi
                    while angle - previous_angle <= -math.pi:
                        angle += 2 * math.pi
                    sub_kl.append([length, angle])
                    previous_angle = angle
                self.kl.append(sub_kl)
        else:
            self.mode = 'normal_dxf'
            # 遍历所有实体
            for entity in dxf.entities:
                # 如果是线段
                if entity.dxftype == 'LINE':
                    # 获取线段的两个端点
                    start_point = entity.start
                    end_point = entity.end
                    # 获取线段所在的图层
                    layer = entity.layer
                    # 获取线段的颜色
                    color = entity.color

                    duplicate = False
                    for i in range(len(self.kps)):
                        if (start_point[0] - self.kps[i][0]) ** 2 + (start_point[1] - self.kps[i][1]) ** 2 < 1e-10:
                            duplicate = True
                            break
                    
                    if duplicate:
                        start_point = self.kps[i]
                    else:
                        self.kps.append(start_point)

                    duplicate = False
                    for i in range(len(self.kps)):
                        if (end_point[0] - self.kps[i][0]) ** 2 + (end_point[1] - self.kps[i][1]) ** 2 < 1e-10:
                            duplicate = True
                            break
                    
                    if duplicate:
                        end_point = self.kps[i]
                    else:
                        self.kps.append(end_point)

                    self.lines.append([start_point, end_point])
                    if layer.upper() == 'MOUNTAIN' or color in [1, 10]:
                        self.lines_type.append(MOUNTAIN)
                    elif layer.upper() == 'VALLEY' or color in [5, 170]:
                        self.lines_type.append(VALLEY)
                    elif layer.upper() == 'BORDER' or color in [7, 250, 90, 256]:
                        self.lines_type.append(BORDER)
                    else:
                        self.lines_type.append(FACET)
    
    def getData(self):
        """
        获取读取的数据。

        Get the read data.

        :return: 包含kps、lines、lines_type的字典 / Dict with kps, lines, lines_type
        """
        ret = {
            "kps": self.kps,
            "lines": self.lines,
            "lines_type": self.lines_type,
        }
        if len(self.units) > 0:
            ret["units"] = self.units
        return ret