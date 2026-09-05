# import numpy as np
import math
from utils import *
from copy import deepcopy
# import heapq

class Miura:
    """
    三浦折（Miura-ori）折纸单元类。
    中文：
        表示一个三浦折单元，分为 ACTIVE_MIURA（主动三浦）和 PASSIVE_MIURA（被动三浦）两种类型。
        该类存储折纸单元的几何参数，并提供获取关键点、折痕线段的方法，
        用于生成完整的三浦折纸图案。

    Miura-ori origami unit class.
    English:
        Represents a Miura-ori unit, with ACTIVE_MIURA and PASSIVE_MIURA subtypes.
        Stores geometric parameters of the unit and provides methods to obtain
        key points and crease line segments for generating a complete Miura-ori pattern.
    """
    def __init__(self, origin, main_length, width, alpha, function_type, entry_flag, direction, trans_origin, trans_length, border_start, data=None) -> None:
        """
        初始化三浦折单元。
        中文：
            可以直接提供各参数，也可以通过 data 列表批量初始化。
            计算折痕在x和y方向的分量长度，以及边界起终点位置。

        Initialize a Miura-ori unit.
        English:
            Can be initialized with individual parameters or via a data list.
            Computes crease x/y length components and boundary start/end positions.

        :param origin: 单元的原点坐标 [x, y] / Origin point [x, y] of the unit
        :param main_length: 主体折痕的水平长度 / Horizontal length of the main crease
        :param width: 折纸单元的宽度 / Width of the origami unit
        :param alpha: 折痕的扇形角（弧度） / Sector angle of the crease (radians)
        :param function_type: 类型（ACTIVE_MIURA=1 或 PASSIVE_MIURA=0） / Type (ACTIVE_MIURA=1 or PASSIVE_MIURA=0)
        :param entry_flag: 入射标志，决定折痕的山谷方向（V=True） / Entry flag, determines mountain/valley direction (V=True)
        :param direction: 折痕方向 / Crease direction
        :param trans_origin: 过渡折叠的原点 / Origin for transition fold
        :param trans_length: 过渡折叠的长度 / Length of transition fold
        :param border_start: 边界起始x坐标 / Starting x-coordinate of the border
        :param data: 可选的批量初始化数据列表（按上述参数顺序） / Optional batch init data list (in above parameter order)
        """
        if data == None:
            self.origin = origin
            self.main_length = main_length
            self.width = width
            self.alpha = alpha
            self.function_type = function_type
            self.entry_flag = entry_flag
            self.direction = direction
            self.trans_origin = trans_origin
            self.trans_length = trans_length
            self.border_start = border_start
        else:
            self.origin = data[0]
            self.main_length = data[1]
            self.width = data[2]
            self.alpha = data[3]
            self.function_type = data[4]
            self.entry_flag = data[5]
            self.direction = data[6]
            self.trans_origin = data[7]
            self.trans_length = data[8]
            self.border_start = data[9]
        self.up_line = EMPTY
        self.down_line = EMPTY
        self.entity_line = SHOW_COLOR
        self.kp = []
        self.line = []
        self.line_connect_to_body = []
        self.left_percent = 0.0
        self.right_percent = 1.0
        self.border_left_percent = 0.0
        self.border_right_percent = 1.0
        self.main_line_left_percent = 0.0
        self.main_line_right_percent = 1.0
        self.most_left = origin[0]
        self.crease_y_length = width / 2
        self.crease_x_length = self.crease_y_length / math.tan(self.alpha)
        self.modify_flag = False
        self.enable_connect_to_body = False
        self.connect_direction = None
        self.same_flag = [True, True, True] #01 123 46/57
        if function_type == ACTIVE_MIURA:
            self.border_end = self.origin[0] + self.main_length - self.crease_x_length
            if self.border_start > self.border_end:
                self.border_start = self.border_end
        else:
            self.border_end = self.origin[0] + self.main_length + self.crease_x_length
            if self.border_start > self.border_end:
                self.border_start = self.border_end

    def getFunctionType(self):
        """
        获取三浦折单元的功能类型。
        中文：返回 ACTIVE_MIURA 或 PASSIVE_MIURA。

        Get the function type of the Miura unit.
        English: Returns ACTIVE_MIURA or PASSIVE_MIURA.

        :return: 功能类型常量 / Function type constant
        """
        return self.function_type
        
    def setMainLength(self, main_length):
        """
        设置主体折痕的水平长度。

        Set the horizontal length of the main crease.

        :param main_length: 主体折痕长度 / Main crease length
        """
        self.main_length = main_length

    def setUpBorder(self, up_line):
        """
        设置上边界类型（EMPTY/HAVE_BORDER/HAVE_CONNECTION）。
        中文：决定单元上边是否生成边界线或连接线。

        Set the upper border type (EMPTY/HAVE_BORDER/HAVE_CONNECTION).
        English: Determines whether a border or connection line is generated on the top.

        :param up_line: 上边界类型常量 / Upper border type constant
        """
        self.up_line = up_line

    def setDownBorder(self, down_line):
        """
        设置下边界类型（EMPTY/HAVE_BORDER/HAVE_CONNECTION）。
        中文：决定单元下边是否生成边界线或连接线。

        Set the lower border type (EMPTY/HAVE_BORDER/HAVE_CONNECTION).
        English: Determines whether a border or connection line is generated on the bottom.

        :param down_line: 下边界类型常量 / Lower border type constant
        """
        self.down_line = down_line

    def setBorder(self, up_line, down_line):
        """
        同时设置上下边界类型。

        Set both upper and lower border types simultaneously.

        :param up_line: 上边界类型 / Upper border type
        :param down_line: 下边界类型 / Lower border type
        """
        self.up_line = up_line
        self.down_line = down_line
    
    def setEntityColor(self, color):
        """
        设置折痕的颜色显示模式（SHOW_COLOR 或 BLACK_COLOR）。
        中文：控制折痕是否以颜色区分类型（山折/谷折）显示。

        Set the color display mode of the crease (SHOW_COLOR or BLACK_COLOR).
        English: Controls whether creases are displayed with type-distinguishing colors.

        :param color: 颜色常量（SHOW_COLOR=1 或 BLACK_COLOR=0） / Color constant (SHOW_COLOR=1 or BLACK_COLOR=0)
        """
        self.entity_line = color

    def setLeftAndRightPercent(self, left_percent=None, right_percent=None):
        """
        设置折痕左右端的位置百分比（裁剪折痕范围）。
        中文：通过调整左右百分比，可以裁剪折痕的显示范围（0=最左，1=最右）。

        Set the left/right position percentages of the crease (for clipping crease extent).
        English: Adjusts left/right display bounds of the crease (0=leftmost, 1=rightmost).

        :param left_percent: 左端百分比（可选） / Left end percentage (optional)
        :param right_percent: 右端百分比（可选） / Right end percentage (optional)
        """
        if left_percent != None:
            self.left_percent = left_percent
        if right_percent != None:
            self.right_percent = right_percent
    
    def setLeftAndRightPercentOfMainLine(self, left_percent=None, right_percent=None):
        """
        设置主折痕（主体线）左右端的位置百分比。
        中文：调整主体中间折痕的左右显示范围。

        Set left/right position percentages of the main line.
        English: Adjusts the left/right extent of the central main crease.

        :param left_percent: 左端百分比（可选） / Left end percentage (optional)
        :param right_percent: 右端百分比（可选） / Right end percentage (optional)
        """
        if left_percent != None:
            self.main_line_left_percent = left_percent
        if right_percent != None:
            self.main_line_right_percent = right_percent

    def setLeftAndRightPercentOfBorder(self, left_percent, right_percent):
        """
        设置边界折痕左右端的位置百分比。
        中文：调整上下边界折痕的左右显示范围。

        Set left/right position percentages of the border crease.
        English: Adjusts the left/right extent of the top/bottom border creases.

        :param left_percent: 左端百分比 / Left end percentage
        :param right_percent: 右端百分比 / Right end percentage
        """
        self.border_left_percent = left_percent
        self.border_right_percent = right_percent

    def getMainLine(self):
        """
        获取三浦折单元顶部（最右端）的两条主体折痕线段。
        中文：
            根据类型（ACTIVE/PASSIVE），返回单元最右端的两条 BORDER 类型折痕，
            用于与相邻单元的连接。

        Get the two main crease lines at the rightmost end of the Miura unit.
        English:
            Returns two BORDER-type creases at the unit's right end (varies by ACTIVE/PASSIVE type),
            used for connecting to adjacent units.

        :return: 两条折痕对象列表 [Crease, Crease] / List of two Crease objects
        """
        crease_y_length = self.crease_y_length
        crease_x_length = self.crease_x_length
        if crease_x_length < 1e-8:
            crease_x_length = 0.0
        if self.function_type == ACTIVE_MIURA:
            return [
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length - crease_x_length, self.origin[1]),
                    BORDER,
                    False
                ), 
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length - crease_x_length, self.origin[1] + self.width),
                    BORDER,
                    False
                )
            ]
        else:
            return [
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length + crease_x_length, self.origin[1]),
                    BORDER,
                    False
                ), 
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length + crease_x_length, self.origin[1] + self.width),
                    BORDER,
                    False
                )
            ]

    def getBorderXPoint(self):
        """
        获取边界折痕的结束x坐标。
        中文：返回折纸单元右侧边界的x坐标终点。

        Get the ending x-coordinate of the border crease.
        English: Returns the rightmost x-coordinate of the unit's border crease.

        :return: 边界结束x坐标（浮点数） / Border end x-coordinate (float)
        """
        return self.border_end

    def getNextBorderStartXPoint(self):
        """
        获取下一个单元的边界起始x坐标。
        中文：返回 border_start 和 border_end 中较大的值，作为下一单元的起点。

        Get the starting x-coordinate for the next unit's border.
        English: Returns the larger of border_start and border_end as the next unit's start.

        :return: 下一单元边界起始x坐标 / Next unit's border start x-coordinate
        """
        if self.border_start > self.border_end:
            return self.border_start
        else:
            return self.border_end

    def setBorderXPoint(self, x):
        """
        修改边界折痕的结束x坐标（仅当x值小于当前border_end时有效）。
        中文：若给定x小于当前border_end，则更新并设置修改标志。

        Modify the border end x-coordinate (only if x is smaller than current border_end).
        English: Updates border_end and sets modify flag if x < current border_end.

        :param x: 新的边界结束x坐标 / New border end x-coordinate
        """
        if x < self.border_end:
            self.modify_flag = True
            self.border_end = x
        
    def getEndPoint(self):
        """
        获取折纸单元的结束点x坐标。
        中文：若单元已被修改（modify_flag=True），返回修改后的border_end；否则返回border_start。

        Get the end point x-coordinate of the origami unit.
        English: Returns the modified border_end if modify_flag is True, otherwise border_start.

        :return: 结束x坐标 / End x-coordinate
        """
        if self.modify_flag:
            return self.border_end
        else:
            return self.border_start

    def setEnableConnectionToBody(self, flag: bool, direction):
        """
        设置该单元是否允许连接到主体（Body），并指定连接方向。
        中文：启用后，对应方向的边界折痕会被替换为连接折痕（山折或谷折）。

        Set whether this unit is allowed to connect to the body, with a specified direction.
        English: When enabled, border creases in the given direction are replaced with connection creases.

        :param flag: 是否启用连接 / Whether to enable body connection
        :param direction: 连接方向（UP或DOWN） / Connection direction (UP or DOWN)
        """
        self.enable_connect_to_body = flag
        self.connect_direction = direction

    def getKeypoint(self):
        """
        计算并返回三浦折单元的所有关键点坐标。
        中文：
            根据功能类型（ACTIVE/PASSIVE）和各百分比参数，
            计算折纸单元的8-9个关键节点坐标，存储到 self.kp 并返回。
            关键点用于后续生成折痕线段。

        Calculate and return all key point coordinates of the Miura unit.
        English:
            Based on function type (ACTIVE/PASSIVE) and percentage parameters,
            computes 8-9 key vertex coordinates, stores in self.kp, and returns them.
            Key points are used for generating crease line segments.

        :return: 关键点坐标列表 [(x, y), ...] / List of key point coordinates [(x, y), ...]
        """
        self.kp.clear()
        # 1
        self.kp.append((self.origin[0], self.origin[1] + self.width / 2))
        crease_y_length = self.crease_y_length
        crease_x_length = self.crease_x_length
        if crease_x_length < 1e-8:
            crease_x_length = 0.0
        if self.function_type == ACTIVE_MIURA:
            # 3
            flag = self.main_length - (1 - self.main_line_right_percent) * crease_x_length
            if flag <= 0.0:
                self.kp.append((self.origin[0], self.origin[1] + crease_y_length))
            else:
                self.kp.append((self.origin[0] + flag, self.origin[1] + crease_y_length))
                self.same_flag[0] = False
            self.kp.append((self.origin[0] + self.main_length - (1 - self.right_percent) * crease_x_length, self.origin[1] + self.right_percent * crease_y_length))
            self.kp.append((self.origin[0] + self.main_length - (1 - self.right_percent) * crease_x_length, self.origin[1] + (2 - self.right_percent) * crease_y_length))
            if self.right_percent != 1:
                self.same_flag[1] = False
            # 2/2
            self.kp.append((self.origin[0] + self.main_length - crease_x_length * (1 - self.left_percent), self.origin[1] + crease_y_length * self.left_percent))
            self.kp.append((self.origin[0] + self.main_length - crease_x_length * (1 - self.left_percent), self.origin[1] + crease_y_length * (2 - self.left_percent)))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1]))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1] + self.width))
            if self.left_percent != 0:
                self.same_flag[2] = False
        else:
            # 3
            flag = self.main_length + self.main_line_left_percent * crease_x_length
            if flag <= 1e-5:
                self.kp.append((self.origin[0], self.origin[1] + crease_y_length))
            else:
                self.kp.append((self.origin[0] + flag, self.origin[1] + crease_y_length))
                self.same_flag[0] = False
            self.kp.append((self.origin[0] + self.main_length + self.left_percent * crease_x_length, self.origin[1] + (1 - self.left_percent) * crease_y_length))
            self.kp.append((self.origin[0] + self.main_length + self.left_percent * crease_x_length, self.origin[1] + (1 + self.left_percent) * crease_y_length))
            if self.left_percent != 0:
                self.same_flag[1] = False
            # 2/2
            self.kp.append((self.origin[0] + self.main_length + crease_x_length * self.right_percent, self.origin[1] + crease_y_length * (1 - self.right_percent)))
            self.kp.append((self.origin[0] + self.main_length + crease_x_length * self.right_percent, self.origin[1] + crease_y_length * (1 + self.right_percent)))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1]))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1] + self.width))
            if self.right_percent != 1:
                self.same_flag[2] = False

        return self.kp

    def getLine(self):
        """
        计算并返回三浦折单元的所有折痕线段（含边界、山折、谷折）。
        中文：
            先调用 getKeypoint() 获取关键点，
            再根据 entry_flag 决定山折/谷折方向，
            生成所有折痕（含连接到主体的折痕），并处理上下边界。

        Calculate and return all crease line segments of the Miura unit (boundary, mountain, valley).
        English:
            Calls getKeypoint() first, then uses entry_flag to determine mountain/valley direction,
            generates all creases (including body-connection creases) and handles top/bottom borders.

        :return: 折痕对象列表 [Crease, ...] / List of Crease objects
        """
        self.line.clear()
        self.line_connect_to_body.clear()
        kp = self.getKeypoint()
        show_color = True
        if (self.entity_line == BLACK_COLOR):
            show_color = False
        if self.entry_flag == V:
            flag = VALLEY
            op_flag = MOUNTAIN
        else:
            flag = MOUNTAIN
            op_flag = VALLEY
        # if (self.main_length != 0):
        self.line.append(Crease(kp[0], kp[1], flag))
        self.line.append(Crease(kp[1], kp[2], BORDER, show_color))
        self.line.append(Crease(kp[1], kp[3], BORDER, show_color))
        if (self.function_type == ACTIVE_MIURA):
            self.line.append(Crease(kp[2], kp[4], op_flag, show_color))
            self.line.append(Crease(kp[3], kp[5], op_flag, show_color))
        else:
            self.line.append(Crease(kp[2], kp[4], flag, show_color))
            self.line.append(Crease(kp[3], kp[5], flag, show_color))
        self.line.append(Crease(kp[4], kp[-2], BORDER, show_color))
        self.line.append(Crease(kp[5], kp[-1], BORDER, show_color))
        
        if (self.up_line == HAVE_BORDER):
            if self.border_end > self.border_start:
                if (self.enable_connect_to_body and self.connect_direction == UP):
                    line = Crease(
                        [self.border_end, kp[-1][1]], 
                        [self.border_start, kp[-1][1]], 
                        op_flag
                    )
                    self.line.append(line)
                    self.line_connect_to_body.append(line)
                else:
                    self.line.append(Crease(
                        [self.border_end, kp[-1][1]],
                        [self.border_start, kp[-1][1]], 
                        BORDER
                    ))
        elif (self.up_line == HAVE_CONNECTION):
            if self.border_end > self.border_start:
                self.line.append(Crease(
                    [self.border_end, kp[-1][1]],
                    [self.border_start, kp[-1][1]], 
                    op_flag
                ))
        if (self.down_line == HAVE_BORDER):
            if self.border_end > self.border_start:
                if (self.enable_connect_to_body and self.connect_direction == DOWN):
                    line = Crease(
                        [self.border_end, kp[-2][1]], 
                        [self.border_start, kp[-2][1]], 
                        op_flag
                    )
                    self.line.append(line)
                    self.line_connect_to_body.append(line)
                else:
                    self.line.append(Crease(
                        [self.border_end, kp[-2][1]], 
                        [self.border_start, kp[-2][1]], 
                        BORDER
                    ))
        elif (self.down_line == HAVE_CONNECTION):
            if self.border_end > self.border_start:
                self.line.append(Crease(
                    [self.border_end, kp[-2][1]], 
                    [self.border_start, kp[-2][1]], 
                    op_flag
                ))

        return self.line
    
    def getLineConnectToBody(self):
        """
        获取连接到主体（Body）的折痕列表。
        中文：返回当前单元中类型为"连接到主体"的折痕（山折/谷折），
        用于标识哪些折痕连接到相邻的主体结构。

        Get the list of creases connecting to the body.
        English: Returns creases of type "connect to body" (mountain/valley),
                 identifying which creases link to an adjacent body structure.

        :return: 连接折痕列表 / List of body-connection creases
        """
        return self.line_connect_to_body

    def getSameFlagList(self):
        """
        获取折痕同类标志列表（用于连接相邻单元时判断是否需要添加额外折痕）。
        中文：返回 same_flag 列表，每个元素表示对应段折痕是否与原始位置完全重合。

        Get the same-flag list (used for determining whether extra creases are needed between adjacent units).
        English: Returns the same_flag list, where each element indicates whether
                 the corresponding crease segment exactly coincides with its original position.

        :return: [bool, bool, bool] 三个标志 / Three boolean flags [bool, bool, bool]
        """
        return self.same_flag

    def getTransition(self):
        """
        获取从当前三浦折单元到下一个单元的过渡参数。
        中文：
            返回下一个单元的原点、过渡原点和入射标志（取反），
            用于生成相邻三浦折单元的连接信息。

        Get the transition parameters from the current Miura unit to the next unit.
        English:
            Returns the next unit's origin, transition origin, and inverted entry flag,
            used for linking adjacent Miura units.

        :return: (新原点, 新过渡原点, 取反的入射标志) / (new_origin, new_trans_origin, flipped_entry_flag)
        """
        tmp_origin = deepcopy(self.origin)
        if self.main_length > 0.0:
            tmp_origin[0] += self.main_length
        tmp_trans_origin = deepcopy(self.trans_origin)
        tmp_trans_origin[0] += self.trans_length
        tmp_entry_flag = not deepcopy(self.entry_flag)
        return tmp_origin, tmp_trans_origin, tmp_entry_flag

class LeanMiura:
    """
    倾斜三浦折（Lean-Miura）折纸单元类。
    中文：
        表示一种改进的三浦折单元，由左部分、右部分和中间连接部分组成。
        支持全单元（NO_HALF）、仅左半（LEFT_HALF）或仅右半（RIGHT_HALF）三种模式，
        并可选带连接结构（connection_flag）。
        常用于TSA驱动的折纸结构设计。

    Lean-Miura origami unit class (modified Miura unit with lean configuration).
    English:
        Represents a modified Miura unit consisting of left, right, and middle sections.
        Supports full-unit (NO_HALF), left-half-only (LEFT_HALF), or right-half-only (RIGHT_HALF) modes,
        with optional connection structure (connection_flag).
        Commonly used in TSA-driven origami structure design.
    """
    def __init__(self, unit_width, entry_flag, tsp, copy_time, half_flag, 
                 stretch_length, connection_flag, con_left_length, con_right_length, connection_hole_size) -> None:
        """
        初始化倾斜三浦折单元。
        中文：
            根据给定参数计算折纸的几何布局，包括全局偏移量、折纸长度和宽度，
            初始化左右关键点、折痕列表及面片单元列表。

        Initialize a Lean-Miura origami unit.
        English:
            Computes the geometric layout from given parameters, including global biases,
            origami length and width, and initializes left/right key points, crease lists, and unit lists.

        :param unit_width: 折纸单元宽度（mm） / Unit width (mm)
        :param entry_flag: 入射标志（V=谷折优先，其他=山折优先） / Entry flag (V=valley first, other=mountain first)
        :param tsp: 过渡起始点坐标 [x, y] / Transition start point [x, y]
        :param copy_time: 单元重复次数（行数） / Number of unit repetitions (rows)
        :param half_flag: 半片标志（NO_HALF/LEFT_HALF/RIGHT_HALF） / Half-unit flag (NO_HALF/LEFT_HALF/RIGHT_HALF)
        :param stretch_length: 拉伸（中间直板）段长度 / Stretch (middle flat section) length
        :param connection_flag: 是否有连接结构 / Whether a connection structure exists
        :param con_left_length: 左侧连接长度 / Left connection length
        :param con_right_length: 右侧连接长度 / Right connection length
        :param connection_hole_size: 连接孔的尺寸 / Connection hole size
        """
        self.unit_width = unit_width
        self.entry_flag = entry_flag
        self.tsp = tsp
        self.copy_time = copy_time
        self.half_flag = half_flag
        self.stretch_length = stretch_length
        self.connection_flag = connection_flag
        self.con_left_length = con_left_length
        self.con_right_length = con_right_length
        self.connection_hole_size = connection_hole_size

        self.enable_global_modify = False

        self.global_left_bias = 0
        self.global_right_bias = 0

        self.origami_length = 0
        self.origami_width = 0

        # Left part of lean-Miura
        self.left_kp = []
        self.left_line = []

        # Right part of lean-Miura
        self.right_kp = []
        self.right_line = []

        self.middle_line = []

        self.left_unit = []
        self.right_unit = []
        self.middle_unit = []

        self.connection_left_unit = []
        self.connection_right_unit = []

        half_unit_width = self.unit_width / 2

        # define bias and get length of origami
        if self.half_flag == NO_HALF:
            if self.connection_flag:
                self.global_left_bias = self.con_left_length + half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            else:
                self.global_left_bias = half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            self.origami_length = self.stretch_length + 2 * half_unit_width * self.copy_time
        elif self.half_flag == LEFT_HALF:
            if self.connection_flag:
                self.global_left_bias = self.con_left_length + half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            else:
                self.global_left_bias = half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            self.origami_length = self.stretch_length + half_unit_width * self.copy_time
        elif self.half_flag == RIGHT_HALF:
            self.global_left_bias = 0
            self.global_right_bias = self.stretch_length
            self.origami_length = self.stretch_length + half_unit_width * self.copy_time
        
        self.origami_width = self.unit_width * self.copy_time

    def clearUnit(self):
        """
        清空所有面片单元列表（左、右、中间及连接单元）。
        中文：在重新生成单元前调用，避免重复添加。

        Clear all unit lists (left, right, middle, and connection units).
        English: Called before regenerating units to avoid duplication.
        """
        self.left_unit.clear()
        self.right_unit.clear()
        self.middle_unit.clear()
        self.connection_left_unit.clear()
        self.connection_right_unit.clear()

    def setUnitWidth(self, unit_width):
        """设置折纸单元宽度。/ Set the origami unit width."""
        self.unit_width = unit_width

    def setEntryFlag(self, entry_flag):
        """设置入射标志（决定山折/谷折）。/ Set the entry flag (determines mountain/valley)."""
        self.entry_flag = entry_flag

    def setTransitionStartPoint(self, tsp):
        """设置过渡起始点坐标。/ Set the transition start point coordinates."""
        self.tsp = tsp

    def setCopyTime(self, copy_time):
        """设置单元重复次数（行数）。/ Set the number of unit repetitions (rows)."""
        self.copy_time = copy_time

    def setStretchLength(self, stretch_length):
        """设置中间拉伸段长度。/ Set the middle stretch section length."""
        self.stretch_length = stretch_length

    def setConnectionFlag(self, connection_flag):
        """设置是否有连接结构标志。/ Set the connection structure flag."""
        self.connection_flag = connection_flag

    def setConnectionLeftLength(self, con_left_length):
        """设置左侧连接段长度。/ Set the left connection length."""
        self.con_left_length = con_left_length

    def setConnectionRightLength(self, con_right_length):
        """设置右侧连接段长度。/ Set the right connection length."""
        self.con_right_length = con_right_length

    def setHalfFlag(self, half_flag):
        """设置半片标志（NO_HALF/LEFT_HALF/RIGHT_HALF）。/ Set the half-unit flag (NO_HALF/LEFT_HALF/RIGHT_HALF)."""
        self.half_flag = half_flag

    def setEnableModify(self, enabled: bool):
        """设置是否启用全局修改模式。/ Set whether global modification mode is enabled."""
        self.enable_global_modify = enabled

    def getOrigamiInfo(self):
        """
        获取折纸的总长度和总宽度。
        中文：返回根据参数计算得到的折纸展开尺寸。

        Get the total length and width of the origami.
        English: Returns the unfolded dimensions computed from the unit parameters.

        :return: (origami_length, origami_width) / (total_length, total_width)
        """
        return self.origami_length, self.origami_width
    
    def getHalfFlag(self):
        """
        获取当前半片标志。

        Get the current half-unit flag.

        :return: half_flag 常量（NO_HALF/LEFT_HALF/RIGHT_HALF） / Half-unit flag constant
        """
        return self.half_flag
    
    def addLeftKp(self, kp):
        """
        添加左侧关键点（自动加上过渡起始点和全局左偏移）。
        中文：将给定坐标加上 tsp 和 global_left_bias 后追加到左侧关键点列表。

        Add a left-side key point (with tsp offset and global left bias applied).
        English: Appends the key point shifted by tsp and global_left_bias to the left key point list.

        :param kp: 相对坐标 [x, y] / Relative coordinates [x, y]
        """
        self.left_kp.append([kp[X] + self.tsp[X] + self.global_left_bias, kp[Y] + self.tsp[Y]])

    def addRightKp(self, kp):
        """
        添加右侧关键点（自动加上过渡起始点和全局右偏移）。
        中文：将给定坐标加上 tsp 和 global_right_bias 后追加到右侧关键点列表。

        Add a right-side key point (with tsp offset and global right bias applied).
        English: Appends the key point shifted by tsp and global_right_bias to the right key point list.

        :param kp: 相对坐标 [x, y] / Relative coordinates [x, y]
        """
        self.right_kp.append([kp[X] + self.tsp[X] + self.global_right_bias, kp[Y] + self.tsp[Y]])

    def addLeftLine(self, kp1, kp2, type_crease, show_color=True):
        """
        在左侧折痕列表中添加一条折痕。

        Add a crease to the left-side crease list.

        :param kp1: 起点坐标 / Start point coordinates
        :param kp2: 终点坐标 / End point coordinates
        :param type_crease: 折痕类型（MOUNTAIN/VALLEY/BORDER等） / Crease type (MOUNTAIN/VALLEY/BORDER, etc.)
        :param show_color: 是否显示折痕颜色 / Whether to display crease color
        """
        self.left_line.append(Crease(kp1, kp2, type_crease, show_color))

    def addRightLine(self, kp1, kp2, type_crease, show_color=True):
        """
        在右侧折痕列表中添加一条折痕。

        Add a crease to the right-side crease list.

        :param kp1: 起点坐标 / Start point coordinates
        :param kp2: 终点坐标 / End point coordinates
        :param type_crease: 折痕类型 / Crease type
        :param show_color: 是否显示折痕颜色 / Whether to display crease color
        """
        self.right_line.append(Crease(kp1, kp2, type_crease, show_color))

    def getKeyPoint(self):
        """
        计算并返回倾斜三浦折单元的所有关键点坐标。
        中文：
            按照斜向折叠的规律，计算左右两侧各行的关键点坐标，
            同时包含边界点和连接结构的关键点（若启用）。
            根据 half_flag 决定返回左侧、右侧还是全部关键点。

        Calculate and return all key point coordinates of the Lean-Miura unit.
        English:
            Computes key point coordinates for all rows on both left and right sides
            following the lean-fold geometry, including boundary and connection key points (if enabled).
            Returns left, right, or combined key points based on half_flag.

        :return: 关键点坐标列表 [[x, y], ...] / Key point coordinate list [[x, y], ...]
        """
        half_unit_width = self.unit_width / 2
 
        self.origami_width = self.unit_width * self.copy_time
        # add kps
        #1 lean direction
        for i in range(self.copy_time):
            self.addLeftKp([0 - half_unit_width * i, 0 + half_unit_width * i])
            self.addLeftKp([0 - half_unit_width * i, self.origami_width - half_unit_width * i])
            self.addRightKp([0 + half_unit_width * i, 0 + half_unit_width * i])
            self.addRightKp([0 + half_unit_width * i, self.origami_width - half_unit_width * i])
        left_x = -half_unit_width * self.copy_time
        right_x = half_unit_width * self.copy_time
        y = half_unit_width * self.copy_time
        self.addLeftKp([left_x, y])
        self.addRightKp([right_x, y])
        #2 outborder
        for i in range(1, self.copy_time + 1):
            self.addLeftKp([left_x, y - half_unit_width * i])
            self.addLeftKp([left_x, y + half_unit_width * i])
            self.addRightKp([right_x, y - half_unit_width * i])
            self.addRightKp([right_x, y + half_unit_width * i])
        #3 connection
        if self.connection_flag:
            left_x = -half_unit_width * self.copy_time - self.con_left_length
            right_x = half_unit_width * self.copy_time + self.con_right_length
            self.addLeftKp([left_x, y])
            self.addRightKp([right_x, y])
            for i in range(1, self.copy_time + 1):
                self.addLeftKp([left_x, y - half_unit_width * i])
                self.addLeftKp([left_x, y + half_unit_width * i])
                self.addRightKp([right_x, y - half_unit_width * i])
                self.addRightKp([right_x, y + half_unit_width * i])
        
        # LEFT_HALF
        if self.half_flag == LEFT_HALF:
            return self.left_kp
            
        # RIGHT_HALF
        if self.half_flag == RIGHT_HALF:
            return self.right_kp
        
        # NO_HALF
        return self.left_kp + self.right_kp

    def getLine(self):
        """
        计算并返回倾斜三浦折单元的所有折痕线段。
        中文：
            包括竖向折痕、45度斜向折痕、内部水平折痕、边界折痕和连接折痕（若启用），
            以及中间拉伸段的边界折痕。
            根据 half_flag 返回左、右或全部折痕列表。

        Calculate and return all crease line segments of the Lean-Miura unit.
        English:
            Includes vertical creases, 45° diagonal creases, inner horizontal creases,
            border creases, connection creases (if enabled), and middle stretch border creases.
            Returns left, right, or all creases based on half_flag.

        :return: 折痕对象列表 [Crease, ...] / List of Crease objects
        """
        # get flag
        if self.entry_flag == V:
            flag = VALLEY
            op_flag = MOUNTAIN
        else:
            flag = MOUNTAIN
            op_flag = VALLEY

        #1 vertical
        for i in range(self.copy_time):
            id = self.copy_time - i
            if id % 2:
                f = op_flag
            else:
                f = flag
            self.addLeftLine(self.left_kp[2 * i], self.left_kp[2 * i + 1], f)
            self.addRightLine(self.right_kp[2 * i], self.right_kp[2 * i + 1], f)

        #2 45deg
        for i in range(self.copy_time):
            id = self.copy_time - i
            if id % 2:
                f = flag
            else:
                f = op_flag
            self.addLeftLine(self.left_kp[2 * i], self.left_kp[2 * i + 2], f)
            self.addLeftLine(self.left_kp[2 * i + 1], self.left_kp[2 * i + 3 if i < self.copy_time - 1 else 2 * i + 2], f)
            self.addRightLine(self.right_kp[2 * i], self.right_kp[2 * i + 2], f)
            self.addRightLine(self.right_kp[2 * i + 1], self.right_kp[2 * i + 3 if i < self.copy_time - 1 else 2 * i + 2], f)

        #inner horizontal
        if self.connection_flag:
            show_color = True
        else:
            show_color = False
        
        center_id = 2 * self.copy_time
        for i in range(1, self.copy_time + 1):
            if i == self.copy_time:
                f = BORDER
            elif i % 2:
                f = op_flag
            else:
                f = flag
            self.addLeftLine(self.left_kp[center_id - 2 * i], self.left_kp[center_id - 1 + 2 * i], f)
            self.addLeftLine(self.left_kp[center_id - 2 * i + 1], self.left_kp[center_id + 2 * i], f)
            self.addRightLine(self.right_kp[center_id - 2 * i], self.right_kp[center_id - 1 + 2 * i], f)
            self.addRightLine(self.right_kp[center_id - 2 * i + 1], self.right_kp[center_id + 2 * i], f)
        
        #border
        self.addLeftLine(self.left_kp[center_id], self.left_kp[center_id + 1], op_flag, show_color)
        self.addLeftLine(self.left_kp[center_id], self.left_kp[center_id + 2], op_flag, show_color)
        self.addRightLine(self.right_kp[center_id], self.right_kp[center_id + 1], op_flag, show_color)
        self.addRightLine(self.right_kp[center_id], self.right_kp[center_id + 2], op_flag, show_color)
        
        for i in range(1, self.copy_time):
            if i % 2:
                f = flag
            else:
                f = op_flag
            self.addLeftLine( self.left_kp [center_id - 1 + 2 * i], self.left_kp [center_id + 1 + 2 * i], f, show_color)
            self.addLeftLine( self.left_kp [center_id + 2 * i],     self.left_kp [center_id + 2 + 2 * i], f, show_color)
            self.addRightLine(self.right_kp[center_id - 1 + 2 * i], self.right_kp[center_id + 1 + 2 * i], f, show_color)
            self.addRightLine(self.right_kp[center_id + 2 * i],     self.right_kp[center_id + 2 + 2 * i], f, show_color)
    
        #connection
        if self.connection_flag:
            connection_center_id = 4 * self.copy_time + 1
            self.addLeftLine(self.left_kp[center_id], self.left_kp[connection_center_id], flag, show_color)
            self.addRightLine(self.right_kp[center_id], self.right_kp[connection_center_id], flag, show_color)
            for i in range(1, self.copy_time + 1):
                if i == self.copy_time:
                    f = BORDER
                elif i % 2:
                    f = op_flag
                else:
                    f = flag
                self.addLeftLine(self.left_kp[center_id - 1 + 2 * i], self.left_kp[connection_center_id - 1 + 2 * i], f)
                self.addLeftLine(self.left_kp[center_id + 2 * i], self.left_kp[connection_center_id + 2 * i], f)
                self.addRightLine(self.right_kp[center_id - 1 + 2 * i], self.right_kp[connection_center_id - 1 + 2 * i], f)
                self.addRightLine(self.right_kp[center_id + 2 * i], self.right_kp[connection_center_id + 2 * i], f)
            #connection border
            self.addLeftLine(self.left_kp[connection_center_id], self.left_kp[connection_center_id + 1], BORDER)
            self.addLeftLine(self.left_kp[connection_center_id], self.left_kp[connection_center_id + 2], BORDER)
            self.addRightLine(self.right_kp[connection_center_id], self.right_kp[connection_center_id + 1], BORDER)
            self.addRightLine(self.right_kp[connection_center_id], self.right_kp[connection_center_id + 2], BORDER)
            for i in range(1, self.copy_time):
                self.addLeftLine( self.left_kp [connection_center_id - 1 + 2 * i], self.left_kp [connection_center_id + 1 + 2 * i], BORDER)
                self.addLeftLine( self.left_kp [connection_center_id + 2 * i],     self.left_kp [connection_center_id + 2 + 2 * i], BORDER)
                self.addRightLine(self.right_kp[connection_center_id - 1 + 2 * i], self.right_kp[connection_center_id + 1 + 2 * i], BORDER)
                self.addRightLine(self.right_kp[connection_center_id + 2 * i],     self.right_kp[connection_center_id + 2 + 2 * i], BORDER)
        #middle
        if self.stretch_length > 0:
            self.middle_line.append(Crease(self.left_kp[0], self.right_kp[0], BORDER))
            self.middle_line.append(Crease(self.left_kp[1], self.right_kp[1], BORDER))
            
            if self.half_flag == LEFT_HALF:
                self.middle_line.append(Crease(self.right_kp[0], self.right_kp[1], BORDER))
            
            if self.half_flag == RIGHT_HALF:
                self.middle_line.append(Crease(self.left_kp[0], self.left_kp[1], BORDER))

        else:
            if self.half_flag == LEFT_HALF:
                self.left_line[0].crease_type = BORDER
            
            if self.half_flag == RIGHT_HALF:
                self.right_line[0].crease_type = BORDER

        if self.half_flag == LEFT_HALF:
            return self.left_line + self.middle_line
        
        if self.half_flag == RIGHT_HALF:
            return self.right_line + self.middle_line
        
        return self.left_line + self.right_line + self.middle_line
    
    def makeUnits(self, kp_id: list, ret=False):
        """
        根据给定的关键点索引列表生成左右两个面片单元。
        中文：
            以 kp_id 中的索引在左右关键点列表中取对应坐标，
            根据折痕类型构建两个 Unit 对象（左侧和右侧）。
            若 ret=True，返回这两个 Unit；否则追加到内部列表。

        Generate left and right panel units from a list of key point indices.
        English:
            Uses kp_id indices to look up coordinates in left/right key point lists,
            builds two Unit objects (left and right) with the appropriate crease types.
            Returns the units if ret=True; otherwise appends to internal lists.

        :param kp_id: 关键点索引列表（顺序决定面片形状） / Key point index list (order determines panel shape)
        :param ret: True则返回Unit对象，False则追加到内部列表 / True to return units, False to append to lists
        :return: ret=True时返回 (left_unit, right_unit)，否则返回 None / (left_unit, right_unit) if ret=True, else None
        """
        ul = Unit()
        ur = Unit()
        kp_len = len(kp_id)

        i = 0
        j = 0
        while i < kp_len:
            next_i = (i + 1) % kp_len
            next_j = (j - 1 + kp_len) % kp_len

            # left kp
            lstart = self.left_kp[kp_id[i]]
            lend = self.left_kp[kp_id[next_i]]
            # right kp
            rstart = self.right_kp[kp_id[j]]
            rend = self.right_kp[kp_id[next_j]]

            left_crease_type = BORDER
            for line in self.left_line:
                if line.pointIsStartAndEnd(lstart, lend):
                    left_crease_type = line.getType()
                    break

            right_crease_type = BORDER
            for line in self.right_line:
                if line.pointIsStartAndEnd(rstart, rend):
                    right_crease_type = line.getType()
                    break

            ul.addCrease(Crease(lstart, lend, left_crease_type))
            ur.addCrease(Crease(rstart, rend, right_crease_type))

            i += 1
            j -= 1

        if not ret:
            self.left_unit.append(ul)
            self.right_unit.append(ur)
        else:
            return ul, ur

    def addConnectionUnit(self):
        """
        生成并添加连接段面片单元（需要 connection_flag=True）。
        中文：
            基于连接区域的关键点索引，生成连接左右两侧的面片单元，
            分别追加到 connection_left_unit 和 connection_right_unit 列表。

        Generate and add connection-section panel units (requires connection_flag=True).
        English:
            Based on connection area key point indices, generates panel units connecting
            both left and right sides, appending to connection_left_unit and connection_right_unit.
        """
        if self.connection_flag:
            part_horizontal = 2 * self.copy_time
            part_connection = 4 * self.copy_time + 1

            ul, ur = self.makeUnits([part_connection, part_horizontal, part_horizontal + 1, part_connection + 1], True)
            self.connection_left_unit.append(ul)
            self.connection_right_unit.append(ur)
            ul, ur = self.makeUnits([part_connection, part_connection + 2, part_horizontal + 2, part_horizontal], True)
            self.connection_left_unit.append(ul)
            self.connection_right_unit.append(ur)
            for i in range(1, self.copy_time):
                ul, ur = self.makeUnits([part_connection + 2 * i - 1, part_horizontal + 2 * i - 1, part_horizontal + 2 * i + 1, part_connection + 2 * i + 1], True)
                self.connection_left_unit.append(ul)
                self.connection_right_unit.append(ur)
                ul, ur = self.makeUnits([part_connection + 2 * i, part_connection + 2 * i + 2, part_horizontal + 2 * i + 2, part_horizontal + 2 * i], True)
                self.connection_left_unit.append(ul)
                self.connection_right_unit.append(ur)

    def getUnits(self, connection=False):
        """
        生成并返回倾斜三浦折单元的所有面片单元列表。
        中文：
            清空现有单元列表，逐步生成竖向、水平及中间拉伸段的面片单元。
            若 connection=True，同时生成连接段面片。
            根据 half_flag 返回左侧、右侧或全部面片列表。

        Generate and return all panel units of the Lean-Miura unit.
        English:
            Clears existing unit lists, generates vertical, horizontal, and middle-stretch panel units.
            If connection=True, also generates connection-section panels.
            Returns left, right, or all panels based on half_flag.

        :param connection: True则同时生成连接段面片 / True to also generate connection panels
        :return: Unit对象列表 / List of Unit objects
        """
        self.clearUnit()

        part_horizontal = 2 * self.copy_time
        # part_connection = 4 * self.copy_time + 1

        #1 vertical
        for i in range(self.copy_time - 1):
            self.makeUnits([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])

        self.makeUnits([part_horizontal - 2, part_horizontal, part_horizontal - 1])

        #2 horizontal
        self.makeUnits([part_horizontal - 2, part_horizontal + 1, part_horizontal])
        self.makeUnits([part_horizontal - 1, part_horizontal, part_horizontal + 2])

        for i in range(1, self.copy_time):
            self.makeUnits([part_horizontal - 2 * i, part_horizontal - 2 * i - 2, part_horizontal + 2 * i + 1, part_horizontal + 2 * i - 1])
            self.makeUnits([part_horizontal - 2 * i - 1, part_horizontal - 2 * i + 1, part_horizontal + 2 * i, part_horizontal + 2 * i + 2])

        if self.stretch_length > 0:
            mid_unit = Unit()
            if self.entry_flag == V:
                flag = MOUNTAIN
            else:
                flag = VALLEY

            mid_unit.addCrease(Crease(self.right_kp[0], self.left_kp[0], BORDER))
            if self.half_flag == RIGHT_HALF:
                mid_unit.addCrease(Crease(self.left_kp[0], self.left_kp[1], BORDER))
            else:
                mid_unit.addCrease(Crease(self.left_kp[0], self.left_kp[1], flag))
            mid_unit.addCrease(Crease(self.left_kp[1], self.right_kp[1], BORDER))
            if self.half_flag == LEFT_HALF:
                mid_unit.addCrease(Crease(self.right_kp[1], self.right_kp[0], BORDER))
            else:
                mid_unit.addCrease(Crease(self.right_kp[1], self.right_kp[0], flag))
            self.middle_unit.append(mid_unit)

        if connection:
            self.addConnectionUnit()

        if self.half_flag == LEFT_HALF:
            return self.left_unit + self.middle_unit + self.connection_left_unit
    
        if self.half_flag == RIGHT_HALF:
            return self.right_unit + self.middle_unit + self.connection_right_unit
        
        return self.left_unit + self.right_unit + self.middle_unit + self.connection_left_unit + self.connection_right_unit

def getUnitWithinMiura(miura1: Miura, miura2: Miura):
    """
    计算两个相邻三浦折单元之间的过渡面片单元对。
    中文：
        根据两个相邻 Miura 单元的类型（ACTIVE/PASSIVE）和 same_flag，
        构建连接它们之间的两个面片单元（unit1 和 unit2），
        用于填充两个单元之间的几何空隙。

    Calculate the transition panel unit pair between two adjacent Miura units.
    English:
        Based on the types (ACTIVE/PASSIVE) and same_flag of two adjacent Miura units,
        constructs two panel units (unit1 and unit2) to bridge the geometric gap between them.

    :param miura1: 第一个三浦折单元（左侧） / First Miura unit (left side)
    :param miura2: 第二个三浦折单元（右侧） / Second Miura unit (right side)
    :return: (unit1, unit2) 两个面片单元对 / Pair of panel units (unit1, unit2)
    """
    type1 = miura1.getFunctionType()
    type2 = miura2.getFunctionType()
    same_flag1 = miura1.getSameFlagList()
    same_flag2 = miura2.getSameFlagList()
    kp1 = miura1.getKeypoint()
    kp2 = miura2.getKeypoint()
    line1 = miura1.getLine()
    line2 = miura2.getLine()
    unit1 = Unit()
    unit2 = Unit()
    if type2 == ACTIVE_MIURA:
        if type1 == PASSIVE_MIURA:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            if not same_flag2[1]:
                unit1.addCrease(line2[1])
                unit2.addCrease(Crease(
                    line2[2][END],
                    line2[2][START],
                    line2[2].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                    line2[6][END],
                    line2[6][START],
                    line2[6].getType()
                ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
            if not same_flag1[1]:
                unit1.addCrease(Crease(
                    line1[1][1],
                    line1[1][0],
                    line1[1].getType()
                ))
                unit2.addCrease(line1[2])
        else:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            if not same_flag2[1]:
                unit1.addCrease(line2[1])
                unit2.addCrease(Crease(
                    line2[2][END],
                    line2[2][START],
                    line2[2].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
    else:
        if type1 == PASSIVE_MIURA:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
        else:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
    unit2_modified = Unit()
    creases = unit2.getCrease()
    unit2_length = len(creases)
    for i in range(0, -unit2_length, -1):
        unit2_modified.addCrease(creases[i])
    return unit1, unit2_modified

class UnitPackParser:
    """
    自定义折纸单元包解析器（正向版）。
    中文：
        接受关键点、折痕列表及折痕类型，
        将其平移到指定起始点（tsp），
        并使用 PolygonIdentifier 识别出折纸图案中的所有多边形面片单元。

    Custom origami unit pack parser (forward version).
    English:
        Accepts key points, crease list, and crease types,
        translates them to the specified start point (tsp),
        and uses PolygonIdentifier to identify all polygon panel units in the origami pattern.
    """
    def __init__(self, tsp, kps, lines, lines_type, units=[]) -> None:
        """
        初始化解析器，将关键点和折痕平移到指定起始点。

        Initialize the parser, translating key points and creases to the specified start point.

        :param tsp: 平移起始点 [x, y] / Translation start point [x, y]
        :param kps: 关键点坐标列表 / Key point coordinate list
        :param lines: 折痕起终点列表 / Crease start/end point list
        :param lines_type: 对应折痕类型列表 / Corresponding crease type list
        :param units: 预定义的单元列表 / Predefined unit list
        """
        self.tsp = tsp
        self.maximum_finding_level = 12

        self.new_kps = []
        self.new_lines = []
        self.new_units = []

        for ele in kps:
            self.new_kps.append([ele[X] + tsp[X], ele[Y] + tsp[Y]])

        for i in range(len(lines)):
            ele = lines[i]
            self.new_lines.append(Crease(
                [ele[START][X] + tsp[X], ele[START][Y] + tsp[Y]],
                [ele[END][X] + tsp[X], ele[END][Y] + tsp[Y]],
                lines_type[i]
            ))
        
        self.operation = 1

        if len(units):
            self.operation = 0
            self.packUnitsToNewUnits(units)
    
    def packUnitsToNewUnits(self, units):
        for i in range(len(units)):
            unit = units[i]
            new_unit = Unit()
            for j in range(len(unit)):
                start_point = [unit[j][X] + self.tsp[X], unit[j][Y] + self.tsp[Y]]
                next_point = [unit[(j + 1) % len(unit)][X] + self.tsp[X], unit[(j + 1) % len(unit)][Y] + self.tsp[Y]]
                new_crease = Crease(start_point, next_point, crease_type=BORDER)
                for line in self.new_lines:
                    if sameCrease(new_crease, line) or sameCrease(new_crease, line.getReverse()):
                        new_crease.crease_type = line.getType()
                        break
                new_unit.addCrease(new_crease)
            self.new_units.append(new_unit)

    def setMaximumNumberOfEdgeInAllUnit(self, n):
        """
        设置识别多边形时的最大边数限制。

        Set the maximum number of edges when identifying polygons.

        :param n: 最大边数 / Maximum number of edges
        """
        self.maximum_finding_level = n

    def getMaxDistance(self):
        """
        计算关键点集在x和y方向的最大跨度。

        Calculate the maximum span of the key point set in x and y directions.

        :return: x和y方向的最大跨度 / Maximum span in x and y directions
        """
        max_x = max([self.new_kps[i][X] for i in range(len(self.new_kps))])
        min_x = min([self.new_kps[i][X] for i in range(len(self.new_kps))])
        max_y = max([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        min_y = min([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        return max(max_x - min_x, max_y - min_y)
    
    def getTotalBias(self):
        """
        计算所有面片单元顶点的几何重心（偏移量）。

        Calculate the geometric centroid (bias) of all panel unit vertices.

        :return: 重心坐标 [mean_x, mean_y] / Centroid coordinates [mean_x, mean_y]
        """
        units = self.getUnits()
        total_x = 0.0
        total_y = 0.0
        count = 0
        for unit in units:
            seq_points = unit.getSeqPoint()
            for p in seq_points:
                total_x += p[X]
                total_y += p[Y]
                count += 1
        return [total_x / count, total_y / count]
    
    def getKeyPoint(self):
        """
        获取平移后的关键点列表。

        Get the translated key point list.

        :return: 关键点坐标列表 [[x, y], ...] / Key point coordinate list [[x, y], ...]
        """
        return self.new_kps

    def getLine(self):
        """
        获取平移后的折痕列表。

        Get the translated crease list.

        :return: 折痕对象列表 [Crease, ...] / Crease object list [Crease, ...]
        """
        return self.new_lines

    def getUnits(self):
        """
        识别并返回所有多边形面片单元。
        中文：
            使用深度优先搜索（DFS）遍历折痕网络，识别由折痕围成的多边形面片。
            从3边形（三角形）开始，逐步搜索到 maximum_finding_level 边形。
            排除纯边界（BORDER）组成的多边形，确保面片方向一致（逆时针）。

        Identify and return all polygonal panel units.
        English:
            Uses DFS traversal of the crease network to identify polygonal panels
            bounded by creases. Searches from triangles (3 edges) up to
            maximum_finding_level edges. Excludes polygons composed solely of
            BORDER creases and ensures consistent panel orientation (counter-clockwise).

        :return: 面片单元列表 [Unit, ...] / List of panel units [Unit, ...]
        """
        if not self.operation:
            return self.new_units
        
        self.operation = 0
        index_list = []
        use_time_list = [(2 if ele.getType() == BORDER else 2) for ele in self.new_lines]
        not_consider_list = []
        self.new_units.clear()

        for edge_max in range(3, self.maximum_finding_level + 1):
            exit_condition = True
            for i in range(len(use_time_list)):
                if use_time_list[i] > 0 and self.new_lines[i].getType() != BORDER:
                    exit_condition = False
                    break
            if exit_condition:
                break
            for i in range(len(self.new_lines)):
                index = [[i]]
                queue = [(i, 1, START)]
                start_point = self.new_lines[i][START]

                while len(queue) > 0:
                    id = queue[-1][0]
                    level = queue[-1][1]
                    before = queue[-1][2]
                    sub_index = index[-1]

                    del(queue[-1])
                    del(index[-1])

                    if before == START:
                        end = self.new_lines[id][END]
                    else:
                        end = self.new_lines[id][START]

                    if level >= edge_max:
                        if distance(start_point, end) < 1e-2:
                            temp = []
                            for existing_index in index_list:
                                temp_index = sorted(deepcopy(existing_index))
                                temp.append(temp_index)
                            current_index = sorted(deepcopy(sub_index))
                            if current_index not in temp:
                                # not include other creases
                                u = Unit()
                                crease = self.new_lines[sub_index[0]]

                                next_start = crease[END]
                                u.addCrease(crease)

                                number_of_crease = len(sub_index)
                                
                                for i in range(1, number_of_crease):
                                    crease = self.new_lines[sub_index[i]]
                                    if distance(crease[START], next_start) < 1e-2:
                                        next_start = crease[END]
                                        u.addCrease(crease)
                                    else:
                                        next_start = crease[START]
                                        u.addCrease(crease.getReverse())

                                seq_point = u.getSeqPoint()
                                kp_in_polygon = False
                                for line in self.new_lines:
                                    if line.getMidPoint()[X] > 187 and line.getMidPoint()[X] < 193 and line.getMidPoint()[Y] > 862 and line.getMidPoint()[Y] < 863:
                                        a = 1
                                    if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-2:
                                        kp_in_polygon = True    

                                if not kp_in_polygon:
                                    for ele in current_index:
                                        if use_time_list[ele] >= 2:
                                            use_time_list[ele] -= 1
                                        elif use_time_list[ele] == 1:
                                            use_time_list[ele] -= 1
                                    index_list.append(deepcopy(sub_index))

                                    creases = u.getCrease()
                                    all_border = True
                                    for crease in creases:
                                        if crease.getType() != BORDER:
                                            all_border = False
                                            break
                                    
                                    if not all_border:
                                        if crossProduct(creases[0], creases[1]) < 0:
                                            self.new_units.append(u)

                                        else:
                                            unit_modified = Unit()
                                            unit_length = len(creases)
                                            for ii in range(0, -unit_length, -1):
                                                unit_modified.addCrease(creases[ii].getReverse())
                                            self.new_units.append(unit_modified)
                                    else:
                                        a = 1
                        continue

                    for j in range(len(self.new_lines)):
                        if (j not in sub_index) and (j not in not_consider_list):
                            if distance(self.new_lines[j][START], end) < 1e-2:
                                queue.insert(0, (j, level + 1, START))
                                index.insert(0, sub_index + [j])
                            elif distance(self.new_lines[j][END], end) < 1e-2:
                                queue.insert(0, (j, level + 1, END))
                                index.insert(0, sub_index + [j])
            
            for i in range(len(use_time_list)):
                if use_time_list[i] <= 0:
                    not_consider_list.append(i)

        return self.new_units

    def _angle(self, o, a, b):
        """
        计算从向量OA到向量OB的有向夹角（弧度）。

        Calculate the signed angle from vector OA to vector OB (in radians).

        :param o: 原点坐标 [x, y] / Origin point [x, y]
        :param a: 点A坐标 [x, y] / Point A coordinates [x, y]
        :param b: 点B坐标 [x, y] / Point B coordinates [x, y]
        :return: 有向夹角（弧度）/ Signed angle in radians
        """
        return math.atan2(b[1] - o[1], b[0] - o[0]) - math.atan2(a[1] - o[1], a[0] - o[0])
    
    # def getUnits(self):
    #     if not self.operation:
    #         return self.new_units
        
    #     self.operation = 0
    #     poly = PolygonIdentifier(self.new_lines)
    #     res_idx = poly.identify_polygons()

    #         # # ===================== 以下完全是你的原有代码（无任何改动）=====================
    #         # u = Unit()
    #         # crease = lines[ring[0]]
    #         # next_start = crease[END]
    #         # u.addCrease(crease)

    #         # number_of_crease = len(ring)
    #         # for i in range(1, number_of_crease):
    #         #     crease = lines[ring[i]]
    #         #     if distance(crease[START], next_start) < 1e-2:
    #         #         next_start = crease[END]
    #         #         u.addCrease(crease)
    #         #     else:
    #         #         next_start = crease[START]
    #         #         u.addCrease(crease.getReverse())

    #         # seq_point = u.getSeqPoint()
    #         # kp_in_polygon = False
    #         # for line in lines:
    #         #     if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-2:
    #         #         kp_in_polygon = True
    #         # if not kp_in_polygon:
    #         #     creases = u.getCrease()
    #         #     all_border = True
    #         #     for crease in creases:
    #         #         if crease.getType() != BORDER:
    #         #             all_border = False
    #         #             break
    #         #     if not all_border:
    #         #         if crossProduct(creases[0], creases[1]) > 0:
    #         #             self.new_units.append(u)
    #         #         else:
    #         #             unit_modified = Unit()
    #         #             unit_length = len(creases)
    #         #             for i in range(0, -unit_length, -1):
    #         #                 unit_modified.addCrease(creases[i].getReverse())
    #         #             self.new_units.append(unit_modified)

    #     return self.new_units

class UnitPackParserReverse:
    """
    自定义折纸单元包解析器（反向/通用版）。
    中文：
        与 UnitPackParser 类似，接受关键点、折痕列表及折痕类型，
        将其平移到指定起始点（tsp）。
        使用深度优先搜索（DFS）遍历方式识别折痕围成的所有多边形面片，
        与 UnitPackParser 的区别在于识别算法不同（基于DFS而非 PolygonIdentifier）。

    Custom origami unit pack parser (reverse/generic version).
    English:
        Similar to UnitPackParser: accepts key points, crease list, and crease types,
        translates them to the specified start point (tsp).
        Uses DFS traversal to identify all polygonal panel units formed by creases;
        differs from UnitPackParser in using DFS rather than PolygonIdentifier.
    """
    def __init__(self, tsp, kps, lines, lines_type, units=[]) -> None:
        """
        初始化反向解析器，将关键点和折痕平移到指定起始点。

        Initialize the reverse parser, translating key points and creases to the specified start point.

        :param tsp: 平移起始点 [x, y] / Translation start point [x, y]
        :param kps: 关键点坐标列表 / Key point coordinate list
        :param lines: 折痕起终点列表 / Crease start/end point list
        :param lines_type: 对应折痕类型列表 / Corresponding crease type list
        :param units: 可选的面片单元列表 / Optional panel unit list
        """
        self.tsp = tsp
        self.maximum_finding_level = 12

        self.new_kps = []
        self.new_lines = []
        self.new_units = []

        for ele in kps:
            self.new_kps.append([ele[X] + tsp[X], ele[Y] + tsp[Y]])

        for i in range(len(lines)):
            ele = lines[i]
            self.new_lines.append(Crease(
                [ele[START][X] + tsp[X], ele[START][Y] + tsp[Y]],
                [ele[END][X] + tsp[X], ele[END][Y] + tsp[Y]],
                lines_type[i]
            ))
        
        self.operation = 1
        if len(units):
            self.operation = 0
            self.packUnitsToNewUnits(units)
    
    def packUnitsToNewUnits(self, units):
        for i in range(len(units)):
            unit = units[i]
            new_unit = Unit()
            for j in range(len(unit)):
                start_point = [unit[j][X] + self.tsp[X], unit[j][Y] + self.tsp[Y]]
                next_point = [unit[(j + 1) % len(unit)][X] + self.tsp[X], unit[(j + 1) % len(unit)][Y] + self.tsp[Y]]
                new_crease = Crease(start_point, next_point, crease_type=BORDER)
                for line in self.new_lines:
                    if sameCrease(new_crease, line) or sameCrease(new_crease, line.getReverse()):
                        new_crease.crease_type = line.getType()
                        break
                new_unit.addCrease(new_crease)
            self.new_units.append(new_unit)

    def getKeyPoint(self):
        """
        获取平移后的关键点列表。

        Get the translated key point list.

        :return: 关键点坐标列表 [[x, y], ...] / Key point coordinate list [[x, y], ...]
        """
        return self.new_kps
    
    def getMaxDistance(self):
        """
        计算关键点集的最大包围尺寸及x、y方向的跨度。

        Calculate the maximum bounding size and x/y spans of the key point set.

        :return: (最大跨度, x跨度, y跨度) / (max_span, x_span, y_span)
        """
        max_x = max([self.new_kps[i][X] for i in range(len(self.new_kps))])
        min_x = min([self.new_kps[i][X] for i in range(len(self.new_kps))])
        max_y = max([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        min_y = min([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        return max(max_x - min_x, max_y - min_y), max_x - min_x, max_y - min_y
    
    def getTotalBias(self, units=None):
        """
        计算所有面片单元顶点的几何重心（偏移量）。
        中文：若未传入 units 参数，则先调用 getUnits() 获取面片；
              否则用传入的 units 计算重心坐标。

        Calculate the geometric centroid (bias) of all panel unit vertices.
        English: If units is not provided, calls getUnits() first;
                 otherwise computes centroid from the provided units.

        :param units: 可选的面片单元列表 / Optional panel unit list
        :return: 重心坐标 [mean_x, mean_y] / Centroid coordinates [mean_x, mean_y]
        """
        if units == None:
            unit = self.getUnits()
        else:
            unit = units
        total_x = 0.0
        total_y = 0.0
        count = 0
        for unit in units:
            seq_points = unit.getSeqPoint()
            for p in seq_points:
                total_x += p[X]
                total_y += p[Y]
                count += 1
        return [total_x / count, total_y / count]

    def getLine(self):
        """
        获取平移后的折痕列表。

        Get the translated crease list.

        :return: 折痕对象列表 [Crease, ...] / Crease object list [Crease, ...]
        """
        return self.new_lines

    def setMaximumNumberOfEdgeInAllUnit(self, n):
        """
        设置识别多边形时的最大边数限制。

        Set the maximum number of edges when identifying polygons.

        :param n: 最大边数 / Maximum number of edges
        """
        self.maximum_finding_level = n
        
    def getUnits(self):
        """
        识别并返回所有多边形面片单元（反向/通用版）。
        中文：
            使用深度优先搜索（DFS）遍历折痕网络，识别由折痕围成的多边形面片。
            与 UnitPackParser.getUnits 类似，但使用不同的算法实现。
            从3边形（三角形）开始，逐步搜索到 maximum_finding_level 边形。
            排除纯边界（BORDER）组成的多边形，确保面片方向一致。

        Identify and return all polygonal panel units (reverse/generic version).
        English:
            Uses DFS traversal of the crease network to identify polygonal panels.
            Similar to UnitPackParser.getUnits but with a different algorithm.
            Searches from triangles (3 edges) up to maximum_finding_level edges.
            Excludes polygons composed solely of BORDER creases.

        :return: 面片单元列表 [Unit, ...] / List of panel units [Unit, ...]
        """
        # 设计方案未改动时，不需重新计算多边形单元
        if not self.operation:
            return self.new_units
        
        self.operation = 0

        index_list = []
        # 记录每条边的使用次数，若多边形的所有边都是BORDER，则视为剪裁局域，虽然识别它但是不把它纳入单元中
        use_time_list = [2 for ele in self.new_lines]
        # 记录当前不需考虑的折痕
        not_consider_list = []
        self.new_units = []

        # 从3边形开始（三角形开始）依次搜索多边形，最后搜索出的多边形按照边数从小到大排序
        for edge_max in range(3, self.maximum_finding_level + 1):
            exit_condition = True
            for i in range(len(use_time_list)):
                if use_time_list[i] > 0 and self.new_lines[i].getType() != BORDER:
                    exit_condition = False
                    break
            if exit_condition:
                break
            for i in range(len(self.new_lines)):
                index = [[i]]
                queue = [(i, 1, START)]
                start_point = self.new_lines[i][START]

                while len(queue) > 0:
                    id = queue[-1][0]
                    level = queue[-1][1]
                    before = queue[-1][2]
                    sub_index = index[-1]

                    del(queue[-1])
                    del(index[-1])

                    if before == START:
                        end = self.new_lines[id][END]
                    else:
                        end = self.new_lines[id][START]

                    if level >= edge_max:
                        if distance(start_point, end) < 1e-2:
                            temp = []
                            for existing_index in index_list:
                                temp_index = sorted(deepcopy(existing_index))
                                temp.append(temp_index)
                            current_index = sorted(deepcopy(sub_index))
                            if current_index not in temp:
                                # no other creases
                                u = Unit()
                                crease = self.new_lines[sub_index[0]]
                                next_start = crease[END]
                                u.addCrease(crease)

                                number_of_crease = len(sub_index)
                                
                                for i in range(1, number_of_crease):
                                    crease = self.new_lines[sub_index[i]]
                                    if distance(crease[START], next_start) < 1e-2:
                                        next_start = crease[END]
                                        u.addCrease(crease)
                                    else:
                                        next_start = crease[START]
                                        u.addCrease(crease.getReverse())

                                seq_point = u.getSeqPoint()
                                kp_in_polygon = False
                                for line in self.new_lines:
                                    if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-2:
                                        kp_in_polygon = True
                                if not kp_in_polygon:
                                    for ele in current_index:
                                        if use_time_list[ele] >= 2:
                                            use_time_list[ele] -= 1
                                        elif use_time_list[ele] == 1:
                                            use_time_list[ele] -= 1
                                    index_list.append(deepcopy(sub_index))

                                    creases = u.getCrease()
                                    all_border = True
                                    for crease in creases:
                                        if crease.getType() != BORDER:
                                            all_border = False
                                            break
                                    
                                    if not all_border:
                                        if crossProduct(creases[0], creases[1]) > 0:
                                            self.new_units.append(u)

                                        else:
                                            unit_modified = Unit()
                                            unit_length = len(creases)
                                            for i in range(0, -unit_length, -1):
                                                unit_modified.addCrease(creases[i].getReverse())
                                            self.new_units.append(unit_modified)
                        continue

                    for j in range(len(self.new_lines)):
                        if (j not in sub_index) and (j not in not_consider_list):
                            if distance(self.new_lines[j][START], end) < 1e-2:
                                queue.insert(0, (j, level + 1, START))
                                index.insert(0, sub_index + [j])
                            elif distance(self.new_lines[j][END], end) < 1e-2:
                                queue.insert(0, (j, level + 1, END))
                                index.insert(0, sub_index + [j])
            for i in range(len(use_time_list)):
                if use_time_list[i] <= 0:
                    not_consider_list.append(i)

        return self.new_units
    
    def _angle(self, o, a, b):
        """
        计算从向量OA到向量OB的有向夹角（弧度）。

        Calculate the signed angle from vector OA to vector OB (in radians).

        :param o: 原点坐标 [x, y] / Origin point [x, y]
        :param a: 点A坐标 [x, y] / Point A coordinates [x, y]
        :param b: 点B坐标 [x, y] / Point B coordinates [x, y]
        :return: 有向夹角（弧度）/ Signed angle in radians
        """
        return math.atan2(b[1] - o[1], b[0] - o[0]) - math.atan2(a[1] - o[1], a[0] - o[0])
    
    # def getUnits(self):
    #     # 设计方案未改动时，不需重新计算多边形单元
    #     if not self.operation:
    #         return self.new_units
        
    #     self.operation = 0
    #     self.new_units = []

    #     # 1. 构建半边网格（O(N)时间）
    #     half_edges = {}
    #     edge_map = {}
    #     for idx, line in enumerate(self.new_lines):
    #         s, e = tuple(line[START]), tuple(line[END])
    #         he1 = HalfEdge(s, e, idx)
    #         he2 = HalfEdge(e, s, idx)
    #         he1.twin = he2
    #         he2.twin = he1
    #         half_edges[(s,e)] = he1
    #         half_edges[(e,s)] = he2
    #         edge_map[idx] = (he1, he2)

    #     # 2. 构建半边的next指针（拓扑关系）
    #     for he in half_edges.values():
    #         # 找以he.end为起点的半边，按顺时针排序（核心）
    #         candidates = [h for h in half_edges.values() if h.start == he.end and h != he.twin]
    #         # 按角度排序（确定顺时针方向）
    #         candidates.sort(key=lambda h: self._angle(he.end, he.start, h.end))
    #         if candidates:
    #             he.next = candidates[0]

    #     # 3. 遍历所有半边，提取多边形环（O(N)时间）
    #     for he in half_edges.values():
    #         if he.visited:
    #             continue
            
    #         # 遍历环：半边链 -> 多边形
    #         ring = []
    #         curr = he
    #         while True:
    #             curr.visited = True
    #             ring.append(curr.line_idx)
    #             curr = curr.next
    #             if curr == he:
    #                 break

    #             if len(ring) < 3:
    #                 continue

    #             u = Unit()
    #             crease = self.new_lines[ring[0]]
    #             next_start = crease[END]
    #             u.addCrease(crease)

    #             number_of_crease = len(ring)
                
    #             for i in range(1, number_of_crease):
    #                 crease = self.new_lines[ring[i]]
    #                 if distance(crease[START], next_start) < 1e-2:
    #                     next_start = crease[END]
    #                     u.addCrease(crease)
    #                 else:
    #                     next_start = crease[START]
    #                     u.addCrease(crease.getReverse())

    #             seq_point = u.getSeqPoint()
    #             kp_in_polygon = False
    #             for line in self.new_lines:
    #                 if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-2:
    #                     kp_in_polygon = True
    #             if not kp_in_polygon:
    #                 creases = u.getCrease()
    #                 all_border = True
    #                 for crease in creases:
    #                     if crease.getType() != BORDER:
    #                         all_border = False
    #                         break
                    
    #                 if not all_border:
    #                     if crossProduct(creases[0], creases[1]) > 0:
    #                         self.new_units.append(u)
    #                     else:
    #                         unit_modified = Unit()
    #                         unit_length = len(creases)
    #                         for i in range(0, -unit_length, -1):
    #                             unit_modified.addCrease(creases[i].getReverse())
    #                         self.new_units.append(unit_modified)

    #     return self.new_units

class Body:
    """
    主体区域类，表示折纸图案的外边界。
    中文：
        用于定义折纸设计的主区域边界，由一系列关键点组成。
        可根据关键点自动生成边界折痕（BORDER类型）。

    Body region class representing the outer boundary of an origami pattern.
    English:
        Used to define the main region boundary of an origami design,
        composed of a series of key points.
        Can automatically generate boundary creases (BORDER type) from key points.
    """
    def __init__(self, main_area_point_list) -> None:
        """
        初始化主体区域。

        Initialize the body region.

        :param main_area_point_list: 边界关键点列表 [[x, y], ...] / Boundary key point list [[x, y], ...]
        """
        self.kp = main_area_point_list
        self.line = []

    def getKeyPoint(self):
        """
        获取边界关键点列表。

        Get the boundary key point list.

        :return: 关键点列表 [[x, y], ...] / Key point list [[x, y], ...]
        """
        return self.kp
    
    def getLine(self):
        """
        根据关键点生成边界折痕列表。
        中文：将关键点依次连接形成闭合多边形边界，所有折痕类型为 BORDER。

        Generate boundary crease list from key points.
        English: Connects key points sequentially to form a closed polygon boundary;
                 all creases are of type BORDER.

        :return: 边界折痕列表 [Crease, ...] / Boundary crease list [Crease, ...]
        """
        self.line.clear()
        kp_length = len(self.kp)
        for i in range(kp_length - 1):
            self.line.append(Crease(self.kp[i], self.kp[i + 1], BORDER))
        self.line.append(Crease(self.kp[kp_length - 1], self.kp[0], BORDER))
        return self.line

class TreeBasedOrigamiGraph:
    """
    基于树的折纸图结构类。
    中文：
        用于分析和构建折纸图案的图结构表示。
        将关键点视为图的顶点，折痕视为图的边，
        并计算每个顶点的连接度和折痕的层级关系。
        支持分析折纸的层次结构（level）和系数（coeff）。

    Tree-based origami graph structure class.
    English:
        Used to analyze and build a graph representation of origami patterns.
        Treats key points as graph vertices and creases as graph edges,
        computing vertex connectivity and crease hierarchical relationships.
        Supports analysis of origami level structure and coefficients.
    """
    def __init__(self, kps, lines) -> None:
        """
        初始化树状折纸图。

        Initialize the tree-based origami graph.

        :param kps: 关键点列表 [[x, y], ...] / Key point list [[x, y], ...]
        :param lines: 折痕列表 [Crease, ...] / Crease list [Crease, ...]
        """
        self.kps = kps
        self.lines = lines

        self.vertices = []
        self.edges = []
        self.units = []

    def calculateTreeBasedGraph(self):
        """
        计算树状折纸图的层级结构和系数。
        中文：
            分析折纸图案中每个顶点（Vertex）的连接关系，
            计算每个折痕的层级（level）和系数（coeff）。
            该方法识别折纸的层次结构，用于后续的折叠序列规划。
            算法流程：
            1. 构建顶点和边的关联关系
            2. 计算每个内部顶点的角度分布
            3. 根据角度关系确定折痕层级和系数
            4. 传播层级信息到所有相关折痕

        Calculate the level structure and coefficients of the tree-based origami graph.
        English:
            Analyzes connectivity at each vertex, computes level and coeff for each crease.
            Identifies hierarchical structure for fold sequence planning.
            Algorithm steps:
            1. Build vertex-edge relationships
            2. Compute angle distribution at each interior vertex
            3. Determine crease levels and coefficients from angle relationships
            4. Propagate level information to all related creases
        """
        self.vertices.clear()
        self.edges.clear()
        self.units.clear()

        for kp in self.kps:
            self.vertices.append(Vertex(kp))

        for line_i in range(len(self.lines)):
            line = self.lines[line_i]
            line.visited = False
            start_point = line[START]
            end_point = line[END]
            crease_type = line.getType()
            match_start = False
            match_end = False
            for i in range(len(self.kps)):
                kp = self.kps[i]
                if not match_start and samePoint(start_point, kp, 2):
                    match_start = True
                    line.start_index = i
                    self.vertices[i].dn += 1
                    self.vertices[i].connection_index.append(line_i)
                    if crease_type == BORDER:
                        self.vertices[i].is_border_node = True
                if not match_end and samePoint(end_point, kp, 2):
                    match_end = True
                    line.end_index = i
                    self.vertices[i].dn += 1
                    self.vertices[i].connection_index.append(line_i)
                    if crease_type == BORDER:
                        self.vertices[i].is_border_node = True
                if match_start and match_end:
                    break

        for vertex in self.vertices:
            # vertex = Vertex()
            if not vertex.is_border_node and vertex.dn == 4:
                creases = [self.lines[index] for index in vertex.connection_index]
                crease_types = [creases[i].getType() for i in range(4)]
                same_type = False
                diff_type = False
                for i in range(1, 4):
                    if crease_types[0] == crease_types[i]:
                        same_type = True
                        if diff_type:
                            i -= 1
                            break
                    else:
                        diff_type = True
                        if same_type:
                            break
                if same_type:
                    only_crease_index = vertex.connection_index[i]
                else:
                    only_crease_index = vertex.connection_index[0]

                angles = []
                betas = [0, .0, .0, .0]
                for i in range(4):
                    if vertex.connection_index[i] != only_crease_index:
                        angle = angleBetweenCreases(self.lines[only_crease_index], creases[i])
                        angles.append([angle, vertex.connection_index[i]])
                
                angles = sorted(deepcopy(angles), key=lambda x: x[0])
                if angles[1][0] >= 0:
                    betas[1] = angles[1][0]
                    betas[0] = -angles[0][0]
                    betas[3] = angles[2][0] - angles[1][0]
                    betas[2] = 2 * math.pi - betas[0] - betas[1] - betas[3]
                else:
                    betas[1] = angles[2][0]
                    betas[0] = -angles[1][0]
                    betas[2] = angles[1][0] - angles[0][0]
                    betas[3] = 2 * math.pi - betas[0] - betas[1] - betas[2]
                
                # kind
                level = 0.
                coeff = 1.

                up_level = False

                if betas[0] + betas[1] - math.pi < -1e-2:
                    new_coeff = (math.sin(betas[0]) + math.sin(betas[1])) / math.sin(betas[0] + betas[1])
                elif betas[0] + betas[1] - math.pi < 1e-2:
                    new_coeff = 1.
                    up_level = True
                else:
                    new_coeff = 1.
                
                # C180
                if abs(betas[0] + betas[3] - math.pi) < 1e-2 and abs(betas[1] + betas[2] - math.pi) < 1e-2:
                    if not up_level:
                        # give information to vertex
                        if angles[1][0] >= 0:
                            index_second_crease_1 = angles[0][1]
                            index_second_crease_2 = angles[1][1]
                        else:
                            index_second_crease_1 = angles[1][1]
                            index_second_crease_2 = angles[2][1]
                        vertex.level_list = [level for i in range(4)]
                        vertex.coeff_list = [coeff for i in range(4)]
                        vertex.coeff_list[vertex.connection_index.index(index_second_crease_1)] = new_coeff * coeff
                        vertex.coeff_list[vertex.connection_index.index(index_second_crease_2)] = new_coeff * coeff
                    else:
                        # give information to vertex
                        if angles[1][0] >= 0:
                            index_second_crease_1 = angles[0][1]
                            index_second_crease_2 = angles[1][1]
                        else:
                            index_second_crease_1 = angles[1][1]
                            index_second_crease_2 = angles[2][1]
                        vertex.level_list = [level for i in range(4)]
                        vertex.coeff_list = [coeff for i in range(4)]
                        vertex.level_list[vertex.connection_index.index(index_second_crease_1)] += 1
                        vertex.level_list[vertex.connection_index.index(index_second_crease_2)] += 1
                else:
                    vertex.level_list = [1 for i in range(4)]
                    vertex.coeff_list = [1. for i in range(4)]

        # give information to crease
        visited = len(self.lines)

        # remove borders
        for i in range(len(self.lines)):
            line = self.lines[i]
            if line.getType() == BORDER:
                visited -= 1

        backup_visited = visited
        backup_initial_crease_id = []

        # calculate level and coeff for creases
        while visited:
            crease_id = []

            # randomly select a crease
            for i in range(len(self.lines)):
                line = self.lines[i]
                if line.getType() != BORDER and not line.visited:
                    line.visited = True
                    line.level = 0
                    line.coeff = 1.
                    line.undefined = False
                    visited -= 1
                    crease_id.append(i)
                    break
            
            previous_min_level = 0
            while len(crease_id):
                index = 0
                min_level = self.lines[crease_id[0]].level
                min_index = 0
                min_problem = False

                for index in range(1, len(crease_id)):
                    if self.lines[crease_id[index]].level < min_level:
                        min_index = index
                        min_level = self.lines[crease_id[index]].level
                        if min_level < previous_min_level:
                            previous_min_level = min_level
                            backup_initial_crease_id.append(crease_id[index])
                            crease_id = deepcopy(backup_initial_crease_id)
                            min_problem = True
                            break
                
                previous_min_level = min_level
                if min_problem:
                    for line_i in self.lines:
                        line_i.visited = False
                    visited = backup_visited
                    continue
                    
                crease_first_id = crease_id[min_index]
                del(crease_id[min_index])
                kp1_id = self.lines[crease_first_id].start_index
                kp2_id = self.lines[crease_first_id].end_index
                kp_list = [kp1_id, kp2_id]
                for ele in kp_list:
                    vertex = self.vertices[ele]
                    if not vertex.is_border_node and vertex.dn == 4:
                        # find itself
                        position = vertex.connection_index.index(crease_first_id)
                        for i in range(4):
                            if i != position and not self.lines[vertex.connection_index[i]].visited:
                                divider = vertex.coeff_list[i] / vertex.coeff_list[position]
                                adder = vertex.level_list[i] - vertex.level_list[position]
                                new_crease_id = vertex.connection_index[i]
                                if adder != 0:
                                    self.lines[new_crease_id].level = int(self.lines[crease_first_id].level + adder)
                                    # self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                    self.lines[new_crease_id].coeff = 1.
                                    self.lines[new_crease_id].visited = True
                                    self.lines[new_crease_id].undefined = True
                                else:
                                    self.lines[new_crease_id].level = int(self.lines[crease_first_id].level)
                                    # self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                    self.lines[new_crease_id].coeff = self.lines[crease_first_id].coeff * divider
                                    self.lines[new_crease_id].visited = True
                                    self.lines[new_crease_id].undefined = self.lines[crease_first_id].undefined
                                visited -= 1
                                crease_id.append(new_crease_id)
                            elif i != position and self.lines[vertex.connection_index[i]].visited:
                                #validate
                                divider = vertex.coeff_list[i] / vertex.coeff_list[position]
                                adder = vertex.level_list[i] - vertex.level_list[position]
                                new_crease_id = vertex.connection_index[i]

                                # judge level
                                new_level = int(self.lines[crease_first_id].level + adder)
                                # initial
                                if new_level != self.lines[new_crease_id].level:
                                    # record conflict
                                    if new_level < self.lines[new_crease_id].level:
                                        crease_id.append(vertex.connection_index[i])
                                        break
                                    else:
                                        if adder >= 0:
                                            self.lines[new_crease_id].level = new_level
                                            # self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                            crease_id.append(vertex.connection_index[i])

                                new_coeff = self.lines[crease_first_id].coeff * divider
                                # initial
                                if new_coeff != self.lines[new_crease_id].coeff:
                                    # record conflict
                                    pass
                                self.lines[new_crease_id].undefined = False
        

                        # for i in range(4):
                        #     self.lines[vertex.connection_index[i]].level = vertex.level_list[i]
                        #     self.lines[vertex.connection_index[i]].coeff = vertex.coeff_list[i]
                        #     self.lines[vertex.connection_index[i]].visited = True

        self.sequence_max_level = self.lines[0].level
        self.sequence_min_level = self.lines[0].level
        for i in range(len(self.lines)):
            level = self.lines[i].level
            if level > self.sequence_max_level:
                self.sequence_max_level = level
            elif level < self.sequence_min_level:
                self.sequence_min_level = level
        
        for level in range(int(self.sequence_min_level), int(self.sequence_max_level + 1)):
            for i in range(len(self.lines)):
                if self.lines[i].level == level:
                    break
            self.min_coeff = self.lines[i].coeff
            # get min coeff
            for j in range(i, len(self.lines)):
                if self.lines[j].level == level and self.lines[j].coeff < self.min_coeff:
                    self.min_coeff = self.lines[j].coeff
            for k in range(len(self.lines)):
                if self.lines[k].level == level:
                    self.lines[k].coeff /= self.min_coeff

                
                





    

    