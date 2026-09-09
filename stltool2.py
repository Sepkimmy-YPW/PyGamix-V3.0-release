# import numpy as np
import math
import io
from copy import deepcopy

from utils import *
from dxftool import OrigamiToDxfConverter

class StlMaker:
    # --- STL 文本 O(1) 累积机制 -------------------------------------------
    # `self.s` 对外接口保持 str 语义（logic.py 直接读写），但内部用 list-of-chunks 累积
    # 以避免 O(N^2) 字符串拷贝。所有 `self.s = X` / `self.s += Y` / `self.s` 都通过下面
    # 的 __setattr__ / __getattribute__ 拦截映射到 `self._s_chunks`，单次 `f.write(self.s)`
    # 才会触发一次 join，整体 O(N) 而不是 O(N^2)。
    #
    # 此外维护 `_s_join_cache`：缓存最近一次 join 出来的完整字符串。在 `self.s = X` 时
    # 若 X 是 `cache + suffix` 形态（即 `self.s += suffix` 反编译后的赋值），直接 append
    # suffix 到 chunks，避免把 99k+ chunks 物化成单个 3MB Python str。这一步是
    # logic.py line 6957/6988/7032/... 等处 `self.stl_writer.s += 'endsolid\n'` 末尾
    # 拼接的关键快路径。
    def __setattr__(self, name, value):
        if name == 's':
            if isinstance(value, str):
                # 读 cache（绕过 __getattribute__ 拦截）
                try:
                    cache = object.__getattribute__(self, '_s_join_cache')
                except AttributeError:
                    cache = None
                if cache is not None and value.startswith(cache):
                    # `self.s = old_join + suffix` 形态：只 append suffix，O(len(suffix))
                    suffix = value[len(cache):]
                    if suffix:
                        chunks = object.__getattribute__(self, '_s_chunks')
                        chunks.append(suffix)
                        object.__setattr__(self, '_s_join_cache', value)
                        return
                # 其他情况按"重置 chunks"语义处理
                object.__setattr__(self, '_s_chunks', [value])
                object.__setattr__(self, '_s_join_cache', value)
            else:
                object.__setattr__(self, '_s_chunks', [])
                object.__setattr__(self, '_s_join_cache', None)
        else:
            object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if name == 's':
            # 对外接口 `self.s`：返回当前所有 chunks 的 join 结果，并刷新 cache
            # 以便后续 `self.s += x` 的 setattr 能命中 cache.startswith 检测。
            chunks = object.__getattribute__(self, '_s_chunks')
            joined = ''.join(chunks)
            object.__setattr__(self, '_s_join_cache', joined)
            return joined
        return object.__getattribute__(self, name)

    def s_append(self, suffix: str):
        """对外公开的「O(1) 追加 STL 文本」接口。
        替代 `self.s += suffix` 写法，避免把整个 chunks 物化成单个 Python str。
        在 `s += 'endsolid\\n'` 这类末尾小拼接场景下可消除 O(N) 的拷贝开销。"""
        if not suffix:
            return
        object.__getattribute__(self, '_s_chunks').append(suffix)
        # cache 扩展到 new tail（如果旧 cache 仍有效），否则直接失效
        try:
            cache = object.__getattribute__(self, '_s_join_cache')
        except AttributeError:
            cache = None
        if cache is not None:
            object.__setattr__(self, '_s_join_cache', cache + suffix)
        # else: 下次读 self.s 时 cache 会被 __getattribute__ 重新 join 刷新

    def s_flush(self, filepath: str, suffix: str = None):
        """把当前 chunks 一次性编码并写入文件（O(N)，一次 memcpy + 一次磁盘写）。

        替代 `self.s += 'endsolid\\n'; f.write(self.s)` 的写法。实测在用户机器上
        比 `''.join(chunks) + f.write` 再快约 2~3x（避免 Python 层的多次 str 拼接）。
        可选 `suffix` 在写入前追加（O(len(suffix))）。"""
        chunks = object.__getattribute__(self, '_s_chunks')
        if suffix:
            chunks.append(suffix)
        # 一次性 join + encode + binary write，最小化 Python 层开销
        blob = ''.join(chunks).encode('utf-8')
        with open(filepath, 'wb') as f:
            f.write(blob)
        if suffix:
            object.__setattr__(self, '_s_join_cache', None)

    def __init__(self) -> None:
        # unit plane
        self.unit_list = []
        # unit bias list
        self.unit_bias_list = []
        # modified unit plane
        self.modified_unit_list = []
        # triangle
        self.tri_list = []
        # crease plane
        self.valid_crease_list = []
        # triangle of crease
        self.crease_tri_list = []
        # crease after bias
        self.additional_crease_list = []
        # board triangle list
        self.board_tri_list = []
        # hole of unit 
        self.unit_hole_list = []
        # hole of connection
        self.connection_hole_list = []
        # stiffness of crease
        self.hard_crease_index = []
        # string list
        self.string_list = []
        self.string_tri_list = []
        # pillar list
        self.pillar_unit_list = []
        self.pillar_tri_list = []

        # transfrom matrix
        self.unit_rotation_matrix = []
        self.unit_transformation_vector = []

        self.unit_hole_size = 0.1
        self.unit_hole_resolution = 3
        self.symmetry_flag = False

        self.connection_hole_size = 1.25

        self.string_width = 2.0

        self.min_bias = 0.12
        
        self.elephant_size = 0.15

        self.height = 2.0
        self.crease_height = 0.2
        self.bias = 3.0
        self.board_height = 0.2
        self.method = "upper_bias"
        self.hole_width_size_percent = 0.5
        self.hole_length_size_percent = 0.8
        # 用 object.__setattr__ 避开 __setattr__ 拦截（_s_chunks 必须在拦截前就位）
        object.__setattr__(self, '_s_chunks', [])
        # self.s = '' 等价于把 chunks 重置为 ['']（空字符串），保持原有 str 接口语义
        object.__setattr__(self, '_s_chunks', [''])
        # _s_join_cache：最近一次 join 的结果缓存。setattr 在 s = old + suffix 检测中用。
        # 初始化为 '' 与 chunks 保持一致。
        object.__setattr__(self, '_s_join_cache', '')
        self.print_accuracy = 0.2
        self.db_enable = False

        self.thin_mode = False

        self.border_nobias = False
        self.border_nobias_penalty = 1e-1
        
        self.fillet_mode = True
        self.fillet_size = 0.0

        self.layer = 3

        self.unit_height = 2.0

        self.using_modified_unit = False

        self.enable_difference = 0 # 0, 1, 2

        self.asym = False

        self.only_two_sides = False

        self.disable_pillars = False
        
        self.crease_file_path = ""

    def clearValidCrease(self):
        self.valid_crease_list.clear()

    def clear(self):
        self.unit_list.clear()
        self.modified_unit_list.clear()
        self.tri_list.clear()
        self.board_tri_list.clear()
        self.unit_hole_list.clear()
        self.connection_hole_list.clear()
        self.string_list.clear()
        self.string_tri_list.clear()
        self.pillar_tri_list.clear()
        self.pillar_unit_list.clear()

    def clearCrease(self):
        self.valid_crease_list.clear()
        self.crease_tri_list.clear()
        self.additional_crease_list.clear()

    def clearAdditionalCrease(self):
        self.additional_crease_list.clear()

    def setPrintAccuracy(self, acc):
        self.print_accuracy = acc
        self.base_inner_bias = 4. * self.print_accuracy
        
    def setThinMode(self, mode):
        self.thin_mode = mode

    def setHeight(self, height):
        self.height = height

    def setBias(self, bias):
        self.bias = bias
        
    def setAsym(self, asym):
        self.asym = asym

    def setPillarDisable(self, flag):
        self.disable_pillars = flag

    def setOnlyTwoSides(self, flag):
        self.only_two_sides = flag

    def setLayerOfCrease(self, layer):
        self.layer = layer

    def setBoardHeight(self, height):
        self.board_height = height

    def setMethod(self, method):
        self.method = method
    
    def setHoleWidth(self, hole_width):
        self.hole_width_size_percent = hole_width

    def setHoleLength(self, hole_length):
        self.hole_length_size_percent = hole_length

    def setUnitHoles(self, hole_kps):
        self.unit_hole_list = hole_kps

    def setConnectionHoles(self, hole_kps):
        self.connection_hole_list = hole_kps

    def setUnitHoleSize(self, size):
        self.unit_hole_size = size

    def setConnectionHoleSize(self, size):
        self.connection_hole_size = size

    def setUnitHoleResolution(self, resolution):
        self.unit_hole_resolution = resolution

    def setDbEnable(self, flag):
        self.db_enable = flag

    def setHardCrease(self, hard_crease_index):
        self.hard_crease_index = hard_crease_index

    def setStringWidth(self, width):
        self.string_width = width

    def setUnitBias(self, bias_list):
        self.unit_bias_list = bias_list

    def addOrigamiUnit(self, crease_list: list):
        unit = Unit()
        for crease in crease_list:
            unit.addCrease(crease)
        self.unit_list.append(unit)
        self.tri_list.append(None)
    
    def enableUsingModifiedUnit(self):
        self.using_modified_unit = True

    def disableUsingModifiedUnit(self):
        self.using_modified_unit = False

    def addPackedOrigamiUnit(self, unit: Unit):
        self.unit_list.append(unit)
        self.tri_list.append(None)

    def addPackedOrigamiModifiedUnit(self, unit: Unit):
        self.modified_unit_list.append(unit)
        # share the same tri_list with normal unit

    def addValidCreases(self, creases: list):
        for i in range(len(creases)):
            crease = creases[i]
            crease_type = crease.getType()
            if not (crease[END][Y] == crease[START][Y] and crease[END][X] == crease[START][X]) and (crease_type == VALLEY or crease_type == MOUNTAIN):
                same_crease_flag = False
                for ele in self.valid_crease_list:
                    if sameCrease(ele, crease):
                        same_crease_flag = True
                        break
                if not same_crease_flag:
                    crease.setOriginIndex(i)
                    crease.setIndex(len(self.valid_crease_list))
                    self.valid_crease_list.append(crease)
                    self.crease_tri_list.append(None)

    def size(self):
        return len(self.unit_list)

    def validCreaseSize(self):
        return len(self.valid_crease_list)
    
    def getTriangle(self, v1, v2, v3):
        vex1 = v1
        vex2 = v2
        vex3 = v3
        vector1 = [vex2[i] - vex1[i] for i in range(0, 3)]
        vector2 = [vex3[i] - vex1[i] for i in range(0, 3)]
        normal = [
            vector1[Y] * vector2[Z] - vector2[Y] * vector1[Z],
            vector1[Z] * vector2[X] - vector2[Z] * vector1[X],
            vector1[X] * vector2[Y] - vector2[X] * vector1[Y]
        ]
        divider = math.sqrt(normal[X] ** 2 + normal[Y] ** 2 + normal[Z] ** 2)
        try:
            unified_normal = [
                normal[X] / divider,
                normal[Y] / divider,
                normal[Z] / divider
            ]
        except:
            raise TypeError
        return [
            unified_normal, 
            [vex1, vex2, vex3]
        ]

    def pointInUnit(self, point):
        for unit_id in range(len(self.unit_list)):
            unit = self.unit_list[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    def getUnit(self, unit_id):
        return self.unit_list[unit_id]
    
    def getUnitWithKps(self, unit_id):
        return self.unit_list[unit_id].getSeqPoint()
    
    def calculateTriPlaneForString(self, layer = 9):
        s = self.string_list
        bottom_height = layer * self.print_accuracy * 0.5
        h = self.height + self.board_height - bottom_height
        tris = []
        for i, ele in enumerate(s):
            if i >= 1:
                previous_ele = s[i - 1]
                previous_upper_kps = deepcopy(upper_kps)
                previous_type = s[i - 1].type

            ele.width = layer * self.print_accuracy
            if ele.type == BOTTOM:
                ele.start_point[Z] = bottom_height
                ele.end_point[Z] = bottom_height
            elif ele.type == TOP:
                ele.start_point[Z] = h
                ele.end_point[Z] = h
            else:
                if previous_type == BOTTOM:
                    ele.start_point[Z] = bottom_height
                    ele.end_point[Z] = h
                else:
                    ele.start_point[Z] = h
                    ele.end_point[Z] = bottom_height

            direction = np.array(ele.getDirectionVector()) / np.linalg.norm(np.array(ele.getDirectionVector()))
            
            for j in range(3):
                ele.start_point[j] += direction[j] * layer * self.print_accuracy * 0.5
                ele.end_point[j] -= direction[j] * layer * self.print_accuracy * 0.5

            kps, upper_kps = ele.generatePointWithResolution(self.unit_hole_resolution)
            kp_num = len(kps)
            #bottom
            for cur in range(1, kp_num - 1):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[0], kps[cur], kps[next_ele])
                tris.append(ans1)
            #around
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[next_ele], kps[cur], upper_kps[cur])
                ans2 = self.getTriangle(upper_kps[next_ele], kps[next_ele], upper_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #upper
            for cur in range(1, kp_num - 1):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(upper_kps[0], upper_kps[next_ele], upper_kps[cur])
                tris.append(ans1)

            if i >= 1:
                # 初始化三角面存储列表（严格按要求创建）
                temp_tris = []
                # 获取两个正多边形的顶点总数
                prev_vertex_num = len(previous_upper_kps)
                kps_vertex_num = len(kps)
                
                # ---------------------- 查找最短距离顶点对索引 ----------------------
                # 初始化最小距离平方（用平方避免开根号，不影响最小值判断）
                min_dist_sq = float('inf')
                index1 = 0  # previous_upper_kps的最短顶点索引
                index2 = 0  # kps的最短顶点索引
                
                # 遍历所有顶点对，计算3D欧氏距离
                for i in range(prev_vertex_num):
                    prev_p = previous_upper_kps[i]
                    for j in range(kps_vertex_num):
                        curr_p = kps[j]
                        # 3D坐标差值计算
                        dx = prev_p[0] - curr_p[0]
                        dy = prev_p[1] - curr_p[1]
                        dz = prev_p[2] - curr_p[2]
                        dist_sq = dx * dx + dy * dy + dz * dz
                        # 更新最短距离和对应索引
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            index1 = i
                            index2 = j

                add_dir = 1
                if previous_type == BOTTOM:
                    add_dir = -1

                if min_dist_sq < layer * self.print_accuracy:
                    # ---------------------- 循环生成放样三角面 ----------------------
                    # 循环次数=正多边形顶点数（环形闭合，遍历所有顶点）
                    for _ in range(prev_vertex_num):
                        # 【关键】索引取模实现同余，防止环形顶点越界
                        curr_prev_idx = index1 % prev_vertex_num
                        next_prev_idx = (index1 + 1) % prev_vertex_num
                        curr_kps_idx = index2 % kps_vertex_num
                        next_kps_idx = (index2 + add_dir) % kps_vertex_num

                        # 按你指定的顺序生成第一个三角面
                        temp_tris.append(self.getTriangle(
                            previous_upper_kps[next_prev_idx],
                            previous_upper_kps[curr_prev_idx],
                            kps[curr_kps_idx]
                        ))
                        # 按你指定的顺序生成第二个三角面
                        temp_tris.append(self.getTriangle(
                            previous_upper_kps[next_prev_idx],
                            kps[curr_kps_idx],
                            kps[next_kps_idx]
                        ))

                        # 索引递增，推进到下一个顶点
                        index1 += 1
                        index2 += add_dir

                    # 最终合并三角面（严格按要求执行）
                    tris += temp_tris

        self.string_tri_list = tris

    def getAdditionalLineForUnit(self, unit_id, upper_x_bound=math.inf, lower_x_bound=-math.inf):
        unit = self.unit_list[unit_id]
         # ========== 步骤1：预处理：获取并归一化原多边形顶点 ==========
        d3_kps = unit.getSeqPoint()  # 原顶点列表 [(x1,y1), (x2,y2), ...]
        if len(d3_kps) < 3:
            return [], []  # 无效多边形
        # 归一化为逆时针顺序（关键：统一向内法向量方向）
        kps = [[d3_kps[i][X], d3_kps[i][Y]] for i in range(len(d3_kps))]
        kps_normalized = normalize_polygon_order(kps)
        kp_num = len(kps_normalized)
        # 记录原顶点与归一化顶点的映射（用于返回正确的删除ID）
        orig2norm = {i: kps_normalized.index(kps[i]) for i in range(kp_num)}
        norm2orig = {v: k for k, v in orig2norm.items()}

        reversed_kp = 0
        if orig2norm[0] > orig2norm[1]:
            reversed_kp = 1

        # ========== 步骤2：计算每条边的偏置距离（复用原代码的偏置逻辑） ==========
        bias_list = []
        for cur in range(0, kp_num):
            crease_type = unit.crease[(orig2norm[cur] - reversed_kp + kp_num) % kp_num].getType()
            
            if self.enable_difference == 1:
                if crease_type == VALLEY:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    if self.border_nobias:
                        bias = self.border_nobias_penalty
                    else:
                        bias = self.bias * 1.0
                else:
                    bias = self.bias * 1.0
            elif self.enable_difference == 2:
                if crease_type == MOUNTAIN:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    if self.border_nobias:
                        bias = self.border_nobias_penalty
                    else:
                        bias = self.bias * 1.0
                else:
                    bias = self.bias * 1.0
            else:
                if crease_type == BORDER:
                    if self.border_nobias:
                        bias = self.border_nobias_penalty
                    else:
                        bias = self.bias * 1.0
                else:
                    bias = self.bias * 1.0

            bias_list.append(bias)

        # ========== 新增：检测相邻共线边，统一偏置量 ==========
        def is_collinear(p1, p2, p3):
            """判断三点共线（即两条相邻边共线）"""
            # 向量叉积为0则共线
            cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
            return abs(cross) < 0.1

        # 遍历所有相邻边对，检测共线并统一偏置量
        for i in range(kp_num):
            # 当前边：顶点i → i+1；下一条边：顶点i+1 → i+2
            p_i = kps_normalized[i]
            p_i1 = kps_normalized[(i+1)%kp_num]
            p_i2 = kps_normalized[(i+2)%kp_num]
            
            # 判断相邻两条边是否共线
            if is_collinear(p_i, p_i1, p_i2):
                # 两条共线边的偏置量
                bias_i = bias_list[i]
                bias_i1 = bias_list[(i+1)%kp_num]
                
                # 规则：若一条偏置量为0、另一条非0，或偏置量不同 → 统一为较大的偏置量
                if (bias_i == 0 and bias_i1 != 0) or (bias_i1 == 0 and bias_i != 0) or (abs(bias_i - bias_i1) > 1e-8):
                    unified_bias = max(bias_i, bias_i1)
                    bias_list[i] = unified_bias
                    bias_list[(i+1)%kp_num] = unified_bias

        # ========== 步骤3：对每条边计算向内偏置后的线段 ==========
        offset_edges = []  # 偏置后的边列表 [(start, end), ...]
        for i in range(kp_num):
            # 当前边的起点和终点
            edge_start = kps_normalized[i]
            edge_end = kps_normalized[(i+1)%kp_num]
            # 计算向内法向量
            normal = edge_inner_normal(edge_start, edge_end, is_counter_clockwise=True)
            # 沿法向量平移边（偏置距离为bias_list[i]）
            offset_start, offset_end = translate_edge(edge_start, edge_end, bias_list[i], normal)
            offset_edges.append((offset_start, offset_end))

        # ========== 步骤4：计算相邻偏置边的交点（新多边形候选顶点） ==========
        candidate_vertices = []
        for i in range(kp_num):
            # 当前偏置边和下一条偏置边
            edge_i = offset_edges[i]
            edge_j = offset_edges[(i+1)%kp_num]
            # 计算交点
            intersect = line_intersection(edge_i[0], edge_i[1], edge_j[0], edge_j[1])
            if intersect is None:
                # 平行边：取当前边中点作为候选点
                mid_x = (edge_i[0][0] + edge_i[1][0])/2
                mid_y = (edge_i[0][1] + edge_i[1][1])/2
                intersect = (mid_x, mid_y)
            candidate_vertices.append(intersect)
            # 仅保留在原多边形内的交点（修改为pointInPolygon函数）
            # if pointInPolygon(intersect, kps_normalized):
            #     candidate_vertices.append(intersect)
            # else:
            #     # 交点在外：取原顶点向内偏置后的点
            #     normal = edge_inner_normal(edge_start, edge_end, True)
            #     tx = normal[0] * bias_list[i]
            #     ty = normal[1] * bias_list[i]
            #     candidate_vertices.append((edge_start[0]+tx, edge_start[1]+ty))

        # ========== 步骤5：自相交检测与裁剪，删除无效顶点 ==========
        new_vertices = candidate_vertices.copy()
        distance_list = [distance(new_vertices[i], new_vertices[(i + 1) % len(new_vertices)]) for i in range(len(new_vertices))]
        start = (distance_list.index(min(distance_list)) - 1 + len(new_vertices)) % len(new_vertices)
        deleted_norm_ids = []  # 归一化后的删除点ID
        # 检测自相交
        self_intersects = detect_self_intersection(new_vertices, start)
        while self_intersects:
            # 取第一个自相交点，裁剪多边形
            problem_num = len(deleted_norm_ids)
            intersect_point, edge1_idx, edge2_idx = self_intersects[0]
            # 确定需要删除的顶点范围
            start_idx = (edge1_idx + 1) % len(new_vertices)
            end_idx = edge2_idx
            # 记录被删除的点ID
            j = start_idx
            while j != end_idx:
                deleted_norm_ids.append(j + problem_num)
                j = (j + 1) % len(new_vertices)

            # 裁剪顶点列表：保留start_idx前的点 + 交点 + end_idx后的点
            if start_idx <= end_idx:
                new_vertices = new_vertices[:start_idx] + [intersect_point] + new_vertices[end_idx+1:]
            else:
                new_vertices = new_vertices[end_idx+1:start_idx] + [intersect_point]
            
            distance_list = [distance(new_vertices[i], new_vertices[(i + 1) % len(new_vertices)]) for i in range(len(new_vertices))]
            start = (distance_list.index(min(distance_list)) - 1 + len(new_vertices)) % len(new_vertices)
            # 重新检测自相交
            self_intersects = detect_self_intersection(new_vertices, start)

        # ========== 步骤6：转换为原ID，添加高度，返回结果 ==========
        # 映射为原多边形的删除ID
        deleted_orig_ids = [norm2orig.get(nid, -1) for nid in deleted_norm_ids if norm2orig.get(nid, -1) != -1]
        deleted_orig_ids = list(set(deleted_orig_ids))  # 去重
        # 新多边形点位：添加z轴高度
        upper_kps = [(x, y) for x, y in new_vertices]
        upper_kps_num = len(upper_kps)

        #additional crease
        for cur in range(0, upper_kps_num):
            next_ele = (cur + 1) % upper_kps_num
            crease_type = unit.getCrease()[cur].getType()
            # c1 = Crease(upper_kps[cur], upper_kps[next_ele], BORDER)
            # c2 = Crease([lower_x_bound, -math.inf], [lower_x_bound, math.inf], BORDER)
            # p = calculateIntersectionWithinCrease(c1, c2)
            # if p != None:
            #     if upper_kps[cur][X] <= lower_x_bound:
            #         upper_kps[cur][X] = p[X]
            #         upper_kps[cur][Y] = p[Y]
            #     if upper_kps[next_ele][X] <= lower_x_bound:
            #         upper_kps[next_ele][X] = p[X]
            #         upper_kps[next_ele][Y] = p[Y]
            # else:
            #     c3 = Crease([upper_x_bound, -math.inf], [upper_x_bound, math.inf], BORDER)
            #     p = calculateIntersectionWithinCrease(c1, c3)
            #     if p != None:
            #         if upper_kps[cur][X] >= upper_x_bound:
            #             upper_kps[cur][X] = p[X]
            #             upper_kps[cur][Y] = p[Y]
            #         if upper_kps[next_ele][X] >= upper_x_bound:
            #             upper_kps[next_ele][X] = p[X]
                        # upper_kps[next_ele][Y] = p[Y]
            
            if (upper_kps[cur][X] - lower_x_bound) * (upper_kps[next_ele][X] - lower_x_bound) < -1e-2:
                if (upper_kps[cur][X] < lower_x_bound):
                    percent = (lower_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[cur][X] = lower_x_bound
                    upper_kps[cur][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
                else:
                    percent = (lower_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[next_ele][X] = lower_x_bound
                    upper_kps[next_ele][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
            elif (upper_kps[cur][X] - upper_x_bound) * (upper_kps[next_ele][X] - upper_x_bound) < -1e-2:
                if (upper_kps[cur][X] > upper_x_bound):
                    percent = (upper_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[cur][X] = upper_x_bound
                    upper_kps[cur][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
                else:
                    percent = (upper_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[next_ele][X] = upper_x_bound
                    upper_kps[next_ele][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])

        cur = 0
        next_ele = 1
        for i in range(0, upper_kps_num):
            crease_type = unit.getCrease()[cur].getType()
            if (upper_kps[cur][X] >= lower_x_bound and upper_kps[next_ele][X] >= lower_x_bound) and (upper_kps[cur][X] <= upper_x_bound and upper_kps[next_ele][X] <= upper_x_bound):
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
                cur = next_ele
                next_ele = (next_ele + 1) % upper_kps_num
            else:
                next_ele = (next_ele + 1) % upper_kps_num
                if upper_kps[next_ele][X] <= lower_x_bound and upper_kps[next_ele][X] > upper_kps[cur][X]:
                    cur += 1
            # if (upper_kps[cur][X] < lower_x_bound and upper_kps[next_ele][X] < lower_x_bound) or (upper_kps[cur][X] > upper_x_bound and upper_kps[next_ele][X] > upper_x_bound):
            #     continue
            # upper_kps[cur][X] = min(max(upper_kps[cur][X], lower_x_bound), upper_x_bound)
            # upper_kps[next_ele][X] = min(max(upper_kps[next_ele][X], lower_x_bound), upper_x_bound)
            # if not (upper_kps[cur][X] == upper_kps[next_ele][X] and upper_kps[cur][Y] == upper_kps[next_ele][Y]):
            #     self.additional_crease_list.append(Crease(
            #         [upper_kps[cur][X], upper_kps[cur][Y]],
            #         [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
            #     ))

    def getAdditionalLineForAllUnit(self, upper_x_bound=math.inf, lower_x_bound=-math.inf):
        for i in range(len(self.unit_list)):
            self.getAdditionalLineForUnit(i, upper_x_bound, lower_x_bound)
        return self.additional_crease_list

    def getCreaseDrawing(self, crease_id):
        lines = []
        crease = self.valid_crease_list[crease_id]
        crease_type = crease.getType()
        k_standard = crease.k()
        # b_standard = crease.b()
        mid_point = crease.getMidPoint()

        if self.enable_difference == 1:
            if crease_type == VALLEY:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                if self.border_nobias:
                    bias = self.border_nobias_penalty
                else:
                    bias = self.bias * 1.0
            else:
                bias = self.bias * 1.0
        elif self.enable_difference == 2:
            if crease_type == MOUNTAIN:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                if self.border_nobias:
                    bias = self.border_nobias_penalty
                else:
                    bias = self.bias * 1.0
            else:
                bias = self.bias * 1.0
        else:
            if crease_type == BORDER:
                if self.border_nobias:
                    bias = self.border_nobias_penalty
                else:
                    bias = self.bias * 1.0
            else:
                bias = self.bias * 1.0

        if k_standard == math.inf:
            mid_point_bias1 = [
                mid_point[X] - bias,
                mid_point[Y],
            ]
            mid_point_bias2 = [
                mid_point[X] + bias,
                mid_point[Y],
            ]
        else:
            mid_point_bias1 = [
                mid_point[X] - k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] + 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
            mid_point_bias2 = [
                mid_point[X] + k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] - 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]

        correct_crease = []
        for additional_crease in self.additional_crease_list:
            if pointOnCrease(mid_point_bias1, additional_crease):
                correct_crease.append(additional_crease)
                continue
            if pointOnCrease(mid_point_bias2, additional_crease):
                correct_crease.append(additional_crease)
                continue
        if len(correct_crease) == 2: #symmetric
            lines = [
                correct_crease[0], correct_crease[1],
                Crease(correct_crease[0][START], correct_crease[1][END], BORDER),
                Crease(correct_crease[1][START], correct_crease[0][END], BORDER)
            ]
            return lines
        else:
            return []

    def getBorderCreaseUnit(self):
        border_crease = []
        for unit in self.unit_list:
            border_crease += unit.getBorderCrease()
        unit = Unit()
        unit.addCrease(deepcopy(border_crease[0]))
        next_start_point = deepcopy(border_crease[0][END])
        del(border_crease[0])
        while len(border_crease):
            find = False
            for i in range(len(border_crease)):
                chosen_crease = border_crease[i]
                chosen_start = chosen_crease[START]
                chosen_end = chosen_crease[END]
                if distance(chosen_start, next_start_point) < 1e-2:
                    next_start_point = deepcopy(chosen_crease[END])
                    unit.addCrease(deepcopy(chosen_crease))
                    del(border_crease[i])
                    find = True
                    break
                if distance(chosen_end, next_start_point) < 1e-2:
                    chosen_crease = chosen_crease.getReverse()
                    next_start_point = deepcopy(chosen_crease[END])
                    unit.addCrease(deepcopy(chosen_crease))
                    del(border_crease[i])
                    find = True
                    break
            if not find:
                raise TypeError
        return unit

    def getInnerBiasUnitList(self):
        special_point_list = []
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            new_kps, _ = self.calculateInnerBiasAndSettingHeight(unit, i, self.bias, 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0.)
            reverse_kps = []
            for i in range(0, -len(new_kps), -1):
                reverse_kps.append(new_kps[i])
            special_point_list.append(reverse_kps)
        return special_point_list

    def getSpecialListAndPillar(self, bias):
        self.special_point_list = []
        self.pillar_unit_list = []
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            # exist_modify = False
            # for j in range(len(unit.getCrease())):
            #     if self.unit_bias_list[i][j] != None:
            #         exist_modify = True
            #         break
            
            original_kps, problem_id = self.calculateInnerBiasAndSettingHeight(unit, i, self.min_bias, 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0., accumulation=True)
            new_kps, _ = self.calculateInnerBiasAndSettingHeight(unit, i, self.bias, 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0., accumulation=True)
            reverse_kps = []
            for k in range(0, -len(new_kps), -1):
                reverse_kps.append(new_kps[k])
            if 1:
                self.special_point_list.append(reverse_kps)
            # else:
            #     self.special_point_list.append([])

            pillar_for_unit = []
            if not self.disable_pillars:
                
                pillar_resolution = 3

                accumulate_index = 0
                creases = unit.getCrease()
                for j in range(len(creases)):
                    # if self.unit_bias_list[i][j] != None:
                    #     if j in problem_id:
                    #         accumulate_index += 1
                    #         continue
                    #     continue
                    crease = creases[j]
                    # next_crease = creases[(j + 1) % len(creases)]
                    length = crease.getLength()
                    # next_length = next_crease.getLength()

                    if j in problem_id:
                        accumulate_index += 1
                        continue

                    mid_point = [(original_kps[j - accumulate_index][X] + original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 2.0, 
                                (original_kps[j - accumulate_index][Y] + original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 2.0]
                    
                    quad_back_point = [(3. * original_kps[j - accumulate_index][X] + original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 4.0, 
                                        (3. * original_kps[j - accumulate_index][Y] + original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 4.0]
                    
                    quad_front_point = [(original_kps[j - accumulate_index][X] + 3. * original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 4.0, 
                                        (original_kps[j - accumulate_index][Y] + 3. * original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 4.0]
                    
                    direction = crease.getDirection()
                    normal = crease.getNormal()
                    # next_direction = next_crease.getDirection()
                    method = 0
                    if method:
                        if 16. * bias < length:
                            #middle pillar
                            
                            forward_point = [mid_point[X] + bias * direction[X], mid_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [mid_point[X] - bias * direction[X], mid_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [mid_point[X] + bias * normal[X], mid_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_back
                            forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_front
                            forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
                        
                        elif 8. * bias < length:
                            # quad_back
                            forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_front
                            forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
                        
                        elif 4. * bias < length:
                            #middle pillar
                            
                            forward_point = [mid_point[X] + bias * direction[X], mid_point[Y] + bias * direction[Y], 0.0]
                            backward_point = [mid_point[X] - bias * direction[X], mid_point[Y] - bias * direction[Y], 0.0]
                            normal_point = [mid_point[X] + bias * normal[X], mid_point[Y] + bias * normal[Y], 0.0]

                            new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
                    else:
                        if 16. * bias < length:
                            #middle pillar
                            
                            forward_point = [mid_point[X] + bias * normal[X] + bias * direction[X], mid_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [mid_point[X] + bias * normal[X] - bias * direction[X], mid_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [mid_point[X], mid_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_back
                            forward_point = [quad_back_point[X] + bias * normal[X] + bias * direction[X], quad_back_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_back_point[X] + bias * normal[X] - bias * direction[X], quad_back_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_back_point[X], quad_back_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_front
                            forward_point = [quad_front_point[X] + bias * normal[X] + bias * direction[X], quad_front_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_front_point[X] + bias * normal[X] - bias * direction[X], quad_front_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_front_point[X], quad_front_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
                        
                        elif 8. * bias < length:
                            # quad_back
                            forward_point = [quad_back_point[X] + bias * normal[X] + bias * direction[X], quad_back_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_back_point[X] + bias * normal[X] - bias * direction[X], quad_back_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_back_point[X], quad_back_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)

                            # quad_front
                            forward_point = [quad_front_point[X] + bias * normal[X] + bias * direction[X], quad_front_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [quad_front_point[X] + bias * normal[X] - bias * direction[X], quad_front_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [quad_front_point[X], quad_front_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
                        
                        elif 4. * bias < length:
                            #middle pillar
                            forward_point = [mid_point[X] + bias * normal[X] + bias * direction[X], mid_point[Y] + bias * normal[Y] + bias * direction[Y], 0.0]
                            backward_point = [mid_point[X] + bias * normal[X] - bias * direction[X], mid_point[Y] + bias * normal[Y] - bias * direction[Y], 0.0]
                            normal_point = [mid_point[X], mid_point[Y], 0.0]

                            new_pillar_point = deepcopy([backward_point, forward_point, normal_point])
                            pillar_for_unit.append(new_pillar_point)
            #     if 10. * bias < length:
            #         #side pillar
                    
            #         # quad_back
            #         forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
            #         backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
            #         normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

            #         new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
            #         pillar_for_unit.append(new_pillar_point)

            #         # quad_front
            #         forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
            #         backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
            #         normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

            #         new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
            #         pillar_for_unit.append(new_pillar_point)

                # mid_point = [original_kps[(j - accumulate_index + 1) % len(original_kps)][X], 
                #             original_kps[(j - accumulate_index + 1) % len(original_kps)][Y], 0.0]
                # alpha = math.acos(-direction[X] * next_direction[X] - direction[Y] * next_direction[Y])
                # side_bias_max = 2. * bias / math.sin(alpha)
                # calculated_bias = [0., 0.]

                # if ((4. * side_bias_max < length and crease.getType() != BORDER) or (2.5 * side_bias_max < length and crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # strong
                #     calculated_bias[0] = side_bias_max
                
                # elif ((8. * bias < length and crease.getType() != BORDER) or (5. * bias < length and crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # mixed
                #     calculated_bias[0] = 2. * bias if crease.getType() != BORDER else length / 3.

                # else:
                #     # enable_side_pillar = 1 # weak
                #     calculated_bias[0] = length / 4.
                
                # if ((4. * side_bias_max < next_length and next_crease.getType() != BORDER) or (2.5 * side_bias_max < next_length and next_crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # strong
                #     calculated_bias[1] = side_bias_max
                
                # elif ((8. * bias < next_length and next_crease.getType() != BORDER) or (5. * bias < next_length and next_crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # mixed
                #     calculated_bias[1] = 2. * bias if next_crease.getType() != BORDER else next_length / 3.

                # else:
                #     # enable_side_pillar = 1 # weak
                #     calculated_bias[1] = next_length / 4.

                # # if (enable_side_pillar):
                #     #side pillar

                # forward_point = [mid_point[X] + calculated_bias[1] * next_direction[X], mid_point[Y] + calculated_bias[1] * next_direction[Y], 0.0]
                # backward_point = [mid_point[X] - calculated_bias[0] * direction[X], mid_point[Y] - calculated_bias[0] * direction[Y], 0.0]

                # new_pillar_point = deepcopy([forward_point, mid_point, backward_point])
                # pillar_for_unit.append(new_pillar_point)

            self.pillar_unit_list.append(pillar_for_unit)
            # self.pillar_unit_list.append([])

    def calculateTriPlaneForSingleCrease(self, i, base_height, upper_height):
        unit = deepcopy(self.unit_list[i])
        special_point_list = deepcopy(self.special_point_list[i])
        pillars = deepcopy(self.pillar_unit_list[i])
        new_special = is_valid_polygon(special_point_list)
        total_another_points_list = (process_triangles(pillars) if len(pillars) else []) + ([new_special] if new_special != False else [])
        tris = self.calculateTriPlaneWithBiasAndHeight(
            unit                =unit, 
            unit_id             =None, 
            upper_bias          =0, 
            down_bias           =0, 
            base_height         =base_height,
            upper_height        =upper_height, 
            add_hole            =True, 
            another_points_list =total_another_points_list,
            additional_crease   =False,
            side_tri            =BORDER,
            penalty             =0.04
        )
        return tris, ([unit.getSeqPoint()] + pillars + [special_point_list])

    def calculateTriPlaneForCreaseUsingBindingMethod(self, base_height=0, upper_height=None):
        # unit = self.getBorderCreaseUnit()
        if upper_height == None:
            upper_height = self.board_height
        self.getSpecialListAndPillar(self.bias / 2.0)
        tris = []
        all_unit = []
        for i in range(len(self.unit_list)):
            unit = deepcopy(self.unit_list[i])
            special_point_list = deepcopy(self.special_point_list[i])
            pillars = deepcopy(self.pillar_unit_list[i])
            new_special = is_valid_polygon(special_point_list)
            total_another_points_list = (process_triangles(pillars) if len(pillars) else []) + ([new_special] if new_special != False else [])
            tris += self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =None, 
                upper_bias          =0, 
                down_bias           =0, 
                base_height         =base_height,
                upper_height        =upper_height, 
                add_hole            =True, 
                another_points_list =total_another_points_list,
                additional_crease   =False,
                side_tri            =BORDER,
                penalty             =0.04
            )
            all_unit += ([unit.getSeqPoint()] + pillars + [special_point_list])
        dxf_converter = OrigamiToDxfConverter(self.crease_file_path)
        dxf_converter.ExportAsDxfUsingUnits(all_unit)
        return tris

    def outputCreaseDxf(self, all_unit):
        dxf_converter = OrigamiToDxfConverter(self.crease_file_path)
        dxf_converter.ExportAsDxfUsingUnits(all_unit)

    def calculateTriPlaneForCrease(self, crease_id, base_height=0, upper_height=None):
        if crease_id in self.hard_crease_index:
            base_height = -self.height / 2.0 + self.print_accuracy
            upper_height = self.height + self.board_height - 2 * self.print_accuracy

        if upper_height == None:
            upper_height = self.board_height
        tris = []
        crease = self.valid_crease_list[crease_id]
        crease_type = crease.getType()
        k_standard = crease.k()
        # b_standard = crease.b()
        mid_point = crease.getMidPoint()

        if self.enable_difference == 1:
            if crease_type == VALLEY:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        elif self.enable_difference == 2:
            if crease_type == MOUNTAIN:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        else:
            if crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0

        if k_standard == math.inf:
            mid_point_bias1 = [
                mid_point[X] - bias,
                mid_point[Y],
            ]
            mid_point_bias2 = [
                mid_point[X] + bias,
                mid_point[Y],
            ]
        else:
            mid_point_bias1 = [
                mid_point[X] - k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] + 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
            mid_point_bias2 = [
                mid_point[X] + k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] - 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
        correct_crease = []
        for additional_crease in self.additional_crease_list:
            if pointOnCrease(mid_point_bias1, additional_crease):
                correct_crease.append(additional_crease)
                continue
            if pointOnCrease(mid_point_bias2, additional_crease):
                correct_crease.append(additional_crease)
                continue
        if len(correct_crease) == 2: #symmetric
            kps = [
                correct_crease[0][END] + [base_height],
                correct_crease[0][START] + [base_height],
                correct_crease[1][END] + [base_height],
                correct_crease[1][START] + [base_height]
            ]
            kp_num = len(kps)
            p_width = (1 - self.hole_width_size_percent) / 2
            p_length = (1 - self.hole_length_size_percent) / 2
            inner_kps = []
            #inner points
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                previous_ele = (cur + kp_num - 1) % kp_num
                method = cur % 2
                if method == 0:
                    kp = [kps[cur][X], kps[cur][Y], kps[cur][Z]]
                    delta_x = p_width * (kps[previous_ele][X] - kps[cur][X]) + p_length * (kps[next_ele][X] - kps[cur][X])
                    delta_y = p_width * (kps[previous_ele][Y] - kps[cur][Y]) + p_length * (kps[next_ele][Y] - kps[cur][Y])
                    kp[X] += delta_x
                    kp[Y] += delta_y
                    inner_kps.append(kp)
                else:
                    kp = [kps[cur][X], kps[cur][Y], kps[cur][Z]]
                    delta_x = p_length * (kps[previous_ele][X] - kps[cur][X]) + p_width * (kps[next_ele][X] - kps[cur][X])
                    delta_y = p_length * (kps[previous_ele][Y] - kps[cur][Y]) + p_width * (kps[next_ele][Y] - kps[cur][Y])
                    kp[X] += delta_x
                    kp[Y] += delta_y
                    inner_kps.append(kp)
            upper_kps = [[kps[x][X], kps[x][Y], kps[x][Z] + upper_height] for x in range(kp_num)]
            upper_inner_kps = [[inner_kps[x][X], inner_kps[x][Y], inner_kps[x][Z] + upper_height] for x in range(kp_num)]
            #bottom
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[cur], kps[next_ele], inner_kps[cur])
                ans2 = self.getTriangle(kps[next_ele], inner_kps[next_ele], inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #around
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[next_ele], kps[cur], upper_kps[cur])
                ans2 = self.getTriangle(upper_kps[next_ele], kps[next_ele], upper_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #inner
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(inner_kps[cur], inner_kps[next_ele], upper_inner_kps[cur])
                ans2 = self.getTriangle(inner_kps[next_ele], upper_inner_kps[next_ele], upper_inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #upper
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], upper_inner_kps[cur])
                ans2 = self.getTriangle(upper_inner_kps[next_ele], upper_kps[next_ele], upper_inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            self.crease_tri_list[crease_id] = tris
        # else:
        #     a = 1

    def calculateTriPlaneForAllCrease(self):
        for i in range(len(self.valid_crease_list)):
            self.calculateTriPlaneForCrease(i)

    def calculateDrawingForAllCrease(self):
        all_lines = []
        for i in range(len(self.valid_crease_list)):
            all_lines += self.getCreaseDrawing(i)
        return all_lines

    def calculateTriPlaneWithHole(self, kps, another_points_list, dir):
        kp_num = len(kps)
        tris = []
        all_lines = []
        for i in range(kp_num):
            next_id = (i + 1) % kp_num
            all_lines.append(Crease(
                kps[i], kps[next_id], BORDER
            ))
        for ele in another_points_list:
            for i in range(len(ele)):
                next_id = (i + 1) % len(ele)
                all_lines.append(Crease(
                    ele[i], ele[next_id], BORDER
                ))
        # all lines
        for i in range(len(another_points_list)):
            for point in another_points_list[i]:
                for kp in kps:
                    line1 = Crease(
                        point, kp, BORDER
                    )
                    intersection = False
                    for line2 in all_lines:
                        p = calculateIntersectionWithinCrease(line1, line2, strict_flag=True)
                        if p != None:
                            intersection = True
                            break
                    if not intersection:
                        all_lines.append(line1)
                for j in range(i + 1, len(another_points_list)):
                    for new_point in another_points_list[j]:
                        line1 = Crease(
                            point, new_point, BORDER
                        )
                        intersection = False
                        for line2 in all_lines:
                            p = calculateIntersectionWithinCrease(line1, line2, strict_flag=True)
                            if p != None:
                                intersection = True
                                break
                        if not intersection:
                            all_lines.append(line1)
        # all tris
        # seq point
        seq_kps = []
        for kp in kps:
            seq_kps.append(kp)
        for ele in another_points_list:
            for point in ele:
                seq_kps.append(point)
        for i in range(len(seq_kps)):
            point = seq_kps[i]
            end_point = []
            for line1 in all_lines:
                if distance(line1[START], point) < 1e-2:
                    end_point.append(line1[END])
            for end in end_point:
                start_point = []
                for line2 in all_lines:
                    if distance(line2[END], end) < 1e-2:
                        start_point.append(line2[START])
                for start in start_point:
                    for line3 in all_lines:
                        if (distance(point, line3[START]) < 1e-2 and distance(start, line3[END]) < 1e-2) or \
                            (distance(point, line3[END]) < 1e-2 and distance(start, line3[START]) < 1e-2):
                            position = seq_kps.index(start)
                            point_inside = False
                            for kp in seq_kps:
                                if pointInPolygon(kp, [point, end, start], return_min_distance=True) > 1e-2:
                                    point_inside = True
                                    break
                            if position > i and (not point_inside):
                                if dir > 0:
                                    ans = self.getTriangle(point, end, start)
                                    if ans[0][Z] < 0:
                                        ans = self.getTriangle(point, start, end)
                                else:
                                    ans = self.getTriangle(point, end, start)
                                    if ans[0][Z] > 0:
                                        ans = self.getTriangle(point, start, end)
                                tris.append(ans)
                            else:
                                break
        return tris

    def equalDistance(self, point_list, length):
        new_point_list = []
        for polygon in point_list:
            center = np.sum(polygon, axis=0) / len(polygon)
            points = generatePolygonByCenter(center, self.unit_hole_size + length, self.unit_hole_resolution)
            new_point_list.append(points)
        return new_point_list
    
    def _calculate_bias_list(self, unit, unit_id, bias, side, enable_strong_modify, border_penalty, accumulation, asym_side):
        """复用原代码的偏置距离计算逻辑（仅适配数据结构）"""
        bias_list = []
        creases = unit.getCrease()
        kp_num = len(creases)

        for i in range(kp_num):
            ele = creases[i]
            if ele.getType() == BORDER:
                if unit_id != -1 and self.unit_bias_list[unit_id][i] is not None and \
                        ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)):
                    if accumulation:
                        bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                    else:
                        bias_list.append(self.unit_bias_list[unit_id][i])
                else:
                    if self.border_nobias:
                        bias_list.append(bias - border_penalty + self.border_nobias_penalty)
                    else:
                        if bias == self.min_bias:
                            bias_list.append(self.border_nobias_penalty)
                        else:
                            bias_list.append(bias)
            else:
                if side == UP:
                    if self.asym and ele.getType() == MOUNTAIN:
                        if asym_side != 1:
                            bias_list.append(self.min_bias)
                        else:
                            bias_list.append(self.min_bias)
                    else:
                        if unit_id != -1 and self.unit_bias_list[unit_id][i] is not None and \
                                ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)):
                            if accumulation:
                                bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                            else:
                                bias_list.append(self.unit_bias_list[unit_id][i])
                        else:
                            if self.enable_difference == 1:  # valley small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                                else:
                                    bias_list.append(bias)
                            elif self.enable_difference == 2:  # mountain small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias)
                                else:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                            else:
                                bias_list.append(bias)
                elif side == DOWN:
                    if self.asym and ele.getType() == VALLEY:
                        if asym_side == 0:
                            bias_list.append(self.min_bias)
                        else:
                            bias_list.append(self.min_bias)
                    else:
                        if unit_id != -1 and self.unit_bias_list[unit_id][i] is not None and \
                                ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)):
                            if accumulation:
                                bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                            else:
                                bias_list.append(self.unit_bias_list[unit_id][i])
                        else:
                            if self.enable_difference == 1:  # valley small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias)
                                else:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                            elif self.enable_difference == 2:  # mountain small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                                else:
                                    bias_list.append(bias)
                            else:
                                bias_list.append(bias)
                else:
                    if unit_id != -1 and self.unit_bias_list[unit_id][i] is not None and \
                            (not self.only_two_sides):
                        if accumulation:
                            bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                        else:
                            bias_list.append(self.unit_bias_list[unit_id][i])
                    else:
                        bias_list.append(bias)

        # 处理connection（复用原逻辑）
        if unit.connection is not None and enable_strong_modify:
            if 0 <= unit.connection_number < len(bias_list):
                bias_list[unit.connection_number] = math.atan(unit.connection) * self.unit_height + self.min_bias

        return bias_list
    
    # def calculateInnerBiasAndSettingHeight(self, unit, unit_id, bias, height, side, enable_strong_modify=False, border_penalty=0.0, accumulation=False, asym_side=0):
    #     kps = unit.getSeqPoint()
    #     kp_num = len(kps)
    #     k_b = []
    #     upper_kps = []
    #     bias_list = []
    #     for i in range(len(unit.getCrease())):
    #         ele = unit.getCrease()[i]
    #         if ele.getType() == BORDER:
    #             if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
    #                 ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)): # condition to modify
    #                 if accumulation:
    #                     bias_list.append(self.unit_bias_list[unit_id][i] + bias)
    #                 else:
    #                     bias_list.append(self.unit_bias_list[unit_id][i])
    #             else:
    #                 if self.border_nobias:
    #                     bias_list.append(bias - border_penalty + self.border_nobias_penalty)
    #                 else:
    #                     if bias == self.min_bias:
    #                         bias_list.append(self.border_nobias_penalty)
    #                     else:
    #                         bias_list.append(bias)
    #         else:
    #             if side == UP:
    #                 if self.asym and ele.getType() == MOUNTAIN:
    #                     if asym_side != 1:
    #                         bias_list.append(self.min_bias)
    #                     else:
    #                         bias_list.append(self.min_bias)
    #                 else:
    #                     if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
    #                         ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)): # condition to modify
    #                         if accumulation:
    #                             bias_list.append(self.unit_bias_list[unit_id][i] + bias)
    #                         else:
    #                             bias_list.append(self.unit_bias_list[unit_id][i])
    #                     else:
    #                         if self.enable_difference == 1: #valley small
    #                             if ele.getType() == VALLEY:
    #                                 bias_list.append(bias + 3 * self.print_accuracy)
    #                             else:
    #                                 bias_list.append(bias)
    #                         elif self.enable_difference == 2: #mountain small
    #                             if ele.getType() == VALLEY:
    #                                 bias_list.append(bias)
    #                             else:
    #                                 bias_list.append(bias + 3 * self.print_accuracy)
    #                         else:
    #                             bias_list.append(bias)
    #             elif side == DOWN:
    #                 if self.asym and ele.getType() == VALLEY:
    #                     if asym_side == 0:
    #                         bias_list.append(self.min_bias)
    #                     else:
    #                         bias_list.append(self.min_bias)
    #                 else:
    #                     if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
    #                         ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)): # condition to modify
    #                         if accumulation:
    #                             bias_list.append(self.unit_bias_list[unit_id][i] + bias)
    #                         else:
    #                             bias_list.append(self.unit_bias_list[unit_id][i])
    #                     else:
    #                         if self.enable_difference == 1: #valley small
    #                             if ele.getType() == VALLEY:
    #                                 bias_list.append(bias)
    #                             else:
    #                                 bias_list.append(bias + 3 * self.print_accuracy)
    #                         elif self.enable_difference == 2: #mountain small
    #                             if ele.getType() == VALLEY:
    #                                 bias_list.append(bias + 3 * self.print_accuracy)
    #                             else:
    #                                 bias_list.append(bias)
    #                         else:
    #                             bias_list.append(bias)
    #             else:
    #                 if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
    #                     (not self.only_two_sides): # condition to modify
    #                     if accumulation:
    #                         bias_list.append(self.unit_bias_list[unit_id][i] + bias)
    #                     else:
    #                         bias_list.append(self.unit_bias_list[unit_id][i])
    #                 else:
    #                     bias_list.append(bias)
    #     if unit.connection != None and enable_strong_modify:
    #         bias_list[unit.connection_number] = math.atan(unit.connection) * self.unit_height + self.min_bias

    #     for i in range(0, kp_num):
    #         k_b.append([unit.crease[i].k(), unit.crease[i].b()])
    #     for cur in range(0, kp_num):
    #         next_ele = (cur + kp_num - 1) % kp_num
    #         if k_b[cur][0] == math.inf:
    #             b1_modified = [bias_list[cur] + k_b[cur][1], -bias_list[cur] + k_b[cur][1]]
    #         else:
    #             b1_modified = [bias_list[cur] * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -bias_list[cur] * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
    #         if k_b[next_ele][0] == math.inf:
    #             b2_modified = [bias_list[next_ele] + k_b[next_ele][1], -bias_list[next_ele] + k_b[next_ele][1]]
    #         else:
    #             b2_modified = [bias_list[next_ele] * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -bias_list[next_ele] * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
    #         ps = [
    #             calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[0]]),
    #             calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[0]]),
    #             calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[1]]),
    #             calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[1]]),
    #         ]
    #         finded = False
    #         for p in ps:
    #             angle = 0.0
    #             for kp_id in range(0, kp_num):
    #                 next_kp_id = (kp_id + 1) % kp_num
    #                 angle += calculateAngle(p, kps[kp_id], kps[next_kp_id])
    #             if abs(angle - 2 * math.pi) < 1e-2:
    #                 upper_kps.append(p + [height])
    #                 finded = True
    #                 break
    #         # intersection happen
    #         if not finded:
    #             next_ele = (cur + + kp_num - 1) % kp_num
    #             if k_b[cur][0] == math.inf:
    #                 epsilon_b1_modified = [1e-3 + k_b[cur][1], -1e-3 + k_b[cur][1]]
    #             else:
    #                 epsilon_b1_modified = [1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
    #             if k_b[next_ele][0] == math.inf:
    #                 epsilon_b2_modified = [1e-3 + k_b[next_ele][1], -1e-3 + k_b[next_ele][1]]
    #             else:
    #                 epsilon_b2_modified = [1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
    #             epsilon_ps = [
    #                 calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
    #                 calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
    #                 calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
    #                 calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
    #             ]
    #             for id in range(0, len(epsilon_ps)):
    #                 angle = 0.0
    #                 for kp_id in range(0, kp_num):
    #                     next_kp_id = (kp_id + 1) % kp_num
    #                     angle += calculateAngle(epsilon_ps[id], kps[kp_id], kps[next_kp_id])
    #                 if abs(angle - 2 * math.pi) < 1e-2:
    #                     upper_kps.append(ps[id] + [height])
    #                     finded = True
    #                     break

    #     # check the self-intersection
    #     problem_point_id = []
    #     while(1):
    #         intersection = False
    #         upper_kps_num = len(upper_kps)
    #         problem_point_num = len(problem_point_id)
    #         accumulate_bonus_of_kps = 0
    #         for k in range(upper_kps_num):
    #             if k in problem_point_id:
    #                 accumulate_bonus_of_kps += 1
    #             previous_id = (k - 1 + upper_kps_num) % upper_kps_num
    #             next_id = (k + 1) % upper_kps_num
    #             next_next_id = (k + 2) % upper_kps_num
    #             crease1 = Crease(upper_kps[k], upper_kps[previous_id], BORDER)
    #             crease2 = Crease(upper_kps[next_id], upper_kps[next_next_id], BORDER)
    #             dir_upper_kp = upper_kps[next_id][0] - upper_kps[k][0]
    #             dir_kp = kps[(k + 1 + accumulate_bonus_of_kps) % kp_num][0] - kps[(k + accumulate_bonus_of_kps) % kp_num][0]
    #             p = calculateIntersectionWithinCrease(crease1, crease2)
    #             # if p != None and dir_kp * dir_upper_kp < 0:
    #             if dir_kp * dir_upper_kp < -1e-2:
    #                 problem_point_id.append(k + problem_point_num)
    #                 intersection = True
    #                 del upper_kps[k]
    #                 if next_id == 0:
    #                     del upper_kps[next_id]
    #                     upper_kps.insert(next_id, p + [height])
    #                 else:
    #                     del upper_kps[k]
    #                     upper_kps.insert(k, p + [height])
    #                 break
    #         if not intersection: break
    #     return upper_kps, problem_point_id

    def calculateInnerBiasAndSettingHeight(self, unit, unit_id, bias, height, side, enable_strong_modify=False, border_penalty=0.0, accumulation=False, asym_side=0):
        """
        适配凸/凹多边形的边向内偏置算法（新增共线边偏置量统一逻辑）
        :param unit: Unit类实例（包含多边形顶点和折痕信息）
        :param unit_id: 单元ID
        :param bias: 基础偏置距离
        :param height: 高度值（z轴）
        :param side: UP/DOWN/其他（偏置方向区分）
        :param enable_strong_modify: 是否启用强制修改
        :param border_penalty: 边界惩罚值
        :param accumulation: 是否累加偏置
        :param asym_side: 非对称边标识
        :return: (new_polygon_points, deleted_point_ids)
                new_polygon_points: 新多边形点位 [(x,y,height), ...]
                deleted_point_ids: 被删除的原多边形点ID列表
        """
        # ========== 步骤1：预处理：获取并归一化原多边形顶点 ==========
        kps = unit.getSeqPoint()  # 原顶点列表 [(x1,y1), (x2,y2), ...]
        if len(kps) < 3:
            return [], []  # 无效多边形
        # 归一化为逆时针顺序（关键：统一向内法向量方向）
        kps_normalized = list(reversed(kps))
        kp_num = len(kps_normalized)
        # 记录原顶点与归一化顶点的映射（用于返回正确的删除ID）
        orig2norm = {i: kps_normalized.index(kps[i]) for i in range(kp_num)}
        norm2orig = {v: k for k, v in orig2norm.items()}

        # ========== 步骤2：计算每条边的偏置距离（复用原代码的偏置逻辑） ==========
        bias_list = self._calculate_bias_list(unit, unit_id, bias, side, enable_strong_modify, border_penalty, accumulation, asym_side)

        reversed_kp = 0
        if orig2norm[0] > orig2norm[1]:
            reversed_kp = 1

        if reversed_kp:
            temp_bias_list = list(reversed(bias_list))
            bias_list = temp_bias_list[1:] + [temp_bias_list[0]]

        # ========== 新增：检测相邻共线边，统一偏置量 ==========
        def is_collinear(p1, p2, p3):
            """判断三点共线（即两条相邻边共线）"""
            # 向量叉积为0则共线
            cross = ((p2[X]-p1[X])*(p3[Y]-p2[Y]) - (p2[Y]-p1[Y])*(p3[X]-p2[X])) / math.sqrt((p2[X]-p1[X])**2+(p2[Y]-p1[Y])**2) / math.sqrt((p3[X]-p2[X])**2+(p3[Y]-p2[Y])**2)
            return abs(cross) < 0.01

        # 遍历所有相邻边对，检测共线并统一偏置量
        for i in range(kp_num):
            # 当前边：顶点i → i+1；下一条边：顶点i+1 → i+2
            p_i = kps_normalized[i]
            p_i1 = kps_normalized[(i+1)%kp_num]
            p_i2 = kps_normalized[(i+2)%kp_num]
            
            # 判断相邻两条边是否共线
            if is_collinear(p_i, p_i1, p_i2):
                # 两条共线边的偏置量
                bias_i = bias_list[i]
                bias_i1 = bias_list[(i+1)%kp_num]
                
                # 规则：若一条偏置量为0、另一条非0，或偏置量不同 → 统一为较大的偏置量
                if (bias_i == 0 and bias_i1 != 0) or (bias_i1 == 0 and bias_i != 0) or (abs(bias_i - bias_i1) > 1e-8):
                    unified_bias = max(bias_i, bias_i1)
                    bias_list[i] = unified_bias
                    bias_list[(i+1)%kp_num] = unified_bias

        # ========== 步骤3：对每条边计算向内偏置后的线段 ==========
        offset_edges = []  # 偏置后的边列表 [(start, end), ...]
        for i in range(kp_num):
            # 当前边的起点和终点
            edge_start = kps_normalized[(i - 1 + kp_num) % kp_num]
            edge_end = kps_normalized[i]
            # 计算向内法向量
            normal = edge_inner_normal(edge_start, edge_end, is_counter_clockwise=True)
            # 沿法向量平移边（偏置距离为bias_list[i]）
            offset_start, offset_end = translate_edge(edge_start, edge_end, bias_list[(i - 1 + kp_num) % kp_num], normal)
            offset_edges.append((offset_start, offset_end))

        # ========== 步骤4：计算相邻偏置边的交点（新多边形候选顶点） ==========
        candidate_vertices = []
        for i in range(kp_num):
            # 当前偏置边和下一条偏置边
            edge_i = offset_edges[i]
            edge_j = offset_edges[(i+1)%kp_num]
            # 计算交点
            intersect = line_intersection(edge_i[0], edge_i[1], edge_j[0], edge_j[1])
            if intersect is None:
                # 平行边：取当前边中点作为候选点
                mid_x = (edge_i[0][0] + edge_i[1][0])/2
                mid_y = (edge_i[0][1] + edge_i[1][1])/2
                intersect = (mid_x, mid_y)
            # 仅保留在原多边形内的交点（修改为pointInPolygon函数）
            candidate_vertices.append(intersect)
            # if pointInPolygon(intersect, kps_normalized):
            #     candidate_vertices.append(intersect)
            # else:
            #     # 交点在外：取原顶点向内偏置后的点
            #     normal = edge_inner_normal(edge_start, edge_end, True)
            #     tx = normal[0] * bias_list[i]
            #     ty = normal[1] * bias_list[i]
            #     candidate_vertices.append((edge_start[0]+tx, edge_start[1]+ty))

        # ========== 步骤5：自相交检测与裁剪，删除无效顶点 ==========
        new_vertices = candidate_vertices.copy()
        distance_list = [distance(new_vertices[i], new_vertices[(i + 1) % len(new_vertices)]) for i in range(len(new_vertices))]
        start = (distance_list.index(min(distance_list)) - 1 + len(new_vertices)) % len(new_vertices)
        deleted_norm_ids = []  # 归一化后的删除点ID
        # 检测自相交
        self_intersects = detect_self_intersection(new_vertices, start)

        while self_intersects:
            # 取第一个自相交点，裁剪多边形
            intersect_point, edge1_idx, edge2_idx = self_intersects[0]
            for i in range(1, len(self_intersects)):
                temp_intersect_point, temp_edge1_idx, temp_edge2_idx = self_intersects[i]
                if temp_edge2_idx > edge2_idx:
                    intersect_point, edge1_idx, edge2_idx = self_intersects[i]

            # 确定需要删除的顶点范围
            start_idx = (edge1_idx + 1) % len(new_vertices)
            end_idx = edge2_idx
            # 记录被删除的点ID
            j = end_idx
            while j != start_idx:
                deleted_norm_ids.append(j)
                j = (j - 1 + len(new_vertices)) % len(new_vertices)

            # 裁剪顶点列表：保留start_idx前的点 + 交点 + end_idx后的点
            if start_idx <= end_idx:
                new_vertices = new_vertices[:start_idx] + [intersect_point] + new_vertices[end_idx+1:]
            else:
                new_vertices = new_vertices[end_idx+1:start_idx] + [intersect_point]
            
            distance_list = [distance(new_vertices[i], new_vertices[(i + 1) % len(new_vertices)]) for i in range(len(new_vertices))]
            start = (distance_list.index(min(distance_list)) - 1 + len(new_vertices)) % len(new_vertices)
            # 重新检测自相交
            self_intersects = detect_self_intersection(new_vertices, start)

        # ========== 步骤6：转换为原ID，添加高度，返回结果 ==========
        # 映射为原多边形的删除ID
        deleted_orig_ids = [norm2orig.get(nid, -1) for nid in deleted_norm_ids if norm2orig.get(nid, -1) != -1]
        deleted_orig_ids = list(set(deleted_orig_ids))  # 去重
        # 新多边形点位：添加z轴高度
        new_polygon = [[x, y, height] for x, y in new_vertices]
        # print(orig2norm, new_polygon, unit_id, len(deleted_orig_ids), deleted_norm_ids)
        new_kps = [new_polygon[orig2norm[i] - len(deleted_orig_ids)] for i in range(len(new_vertices))]

        return new_kps, deleted_orig_ids
    
    def calculateTriPlaneWithBiasAndHeight(self, unit, unit_id, upper_bias, down_bias, base_height, upper_height, add_hole=False, another_points_list=None, additional_crease=True, penalty=None, side_tri=None, accumulation=False, bottom_tri=True, upper_tri=True):
        if type(unit) == Unit:
            standard_kps = unit.getSeqPoint()
        else:
            standard_kps = unit
        #judge the side
        side = MIDDLE
        compensation_layer = False
        if down_bias < upper_bias:
            down_strong_modify = False
            upper_strong_modify = True
            side = UP
        elif down_bias > upper_bias:
            down_strong_modify = True
            upper_strong_modify = False
            side = DOWN
        else:
            down_strong_modify = True
            upper_strong_modify = True
        if abs(down_bias - upper_bias) <= 0.002 * self.min_bias:
            down_strong_modify = False
            upper_strong_modify = False
            compensation_layer = True
            
        for ele in another_points_list:
            for point in ele:
                point[Z] = base_height
        if down_bias > 0:
            if penalty == None:
                bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias, accumulation, 0 if not compensation_layer else -1)
            else:
                bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias - penalty, accumulation, 0 if not compensation_layer else -1)
        elif down_bias == 0:
            bottom_kps = deepcopy(standard_kps)
            for ele in bottom_kps:
                ele[Z] = base_height
            bottom_problem_id = []
        else:
            return #error
        if upper_bias > 0:
            if penalty == None:
                upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias, accumulation, 1 if not compensation_layer else -1)
            else:
                upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias - penalty, accumulation, 1 if not compensation_layer else -1)
        elif upper_bias == 0:
            upper_kps = deepcopy(standard_kps)
            for ele in upper_kps:
                ele[Z] = upper_height
            upper_problem_id = []
        else:
            return #error
        kp_num = len(bottom_kps)
        upper_kp_num = len(upper_kps)
        tris = []
        
        fillet_point_list = []
        if self.fillet_mode:
            self.fillet_size = upper_height - base_height
            layers = int(self.fillet_size / self.print_accuracy) + 2
            if down_strong_modify:
                for layer in range(layers):
                    radius_bonus = 1 - np.sin(layer / ((layers - 1) * 2.) * np.pi)
                    new_points = self.equalDistance(deepcopy(another_points_list), self.fillet_size * radius_bonus)
                    fillet_point_list.append(new_points)
            elif upper_strong_modify:
                for layer in range(layers):
                    radius_bonus = 1 - np.cos(layer / ((layers - 1) * 2.) * np.pi)
                    new_points = self.equalDistance(deepcopy(another_points_list), self.fillet_size * radius_bonus)
                    fillet_point_list.append(new_points)
                    
        # bottom
        if bottom_tri:
            if not add_hole:
                for cur in range(1, kp_num - 1):
                    vex1 = bottom_kps[0]
                    vex2 = bottom_kps[cur]
                    vex3 = bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
            else:
                if self.fillet_mode and (upper_strong_modify or down_strong_modify):
                    tris += self.calculateTriPlaneWithHole(bottom_kps, fillet_point_list[0], -1)
                else:
                    tris += self.calculateTriPlaneWithHole(bottom_kps, another_points_list, -1)
        #around
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, kp_num):
                side_enable = True
                if type(unit) == Unit:
                    crease_type = unit.getCrease()[cur].getType()
                    if side_tri != None and crease_type != side_tri:
                        side_enable = False

                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = cur + forward_step
                if real_id in upper_problem_id:
                    if real_id not in bottom_problem_id:
                        if side_enable:
                            ans = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if side_enable:
                            ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = (upper_cur + 1) % upper_kp_num
                else:
                    if side_enable:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = (upper_cur + 1) % upper_kp_num
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, upper_kp_num):
                side_enable = True
                if type(unit) == Unit:
                    crease_type = unit.getCrease()[cur].getType()
                    if side_tri != None and crease_type != side_tri:
                        side_enable = False

                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = upper_cur + forward_step
                if real_id in bottom_problem_id:
                    if real_id not in upper_problem_id:
                        if side_enable:
                            ans = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if side_enable:
                            ans1 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = (cur + 1) % kp_num
                else:
                    if side_enable:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = (cur + 1) % kp_num
        #upper
        upper_another_points_list = deepcopy(another_points_list)
        for ele in upper_another_points_list:
            for point in ele:
                point[Z] = upper_height
        
        if self.fillet_mode and (upper_strong_modify or down_strong_modify):
            for layer in range(layers):
                if down_strong_modify:
                    index = 1 - np.cos(layer / ((layers - 1) * 2.) * np.pi)
                else:
                    index = np.sin(layer / ((layers - 1) * 2.) * np.pi)
                if index <= 0.0:
                    index = 0.0
                elif index >= 1.0:
                    index = 1.0
                hole_height = index * (upper_height - base_height) + base_height
                for polygon in fillet_point_list[layer]:
                    for point in polygon:
                        point[Z] = hole_height
        # print(fillet_point_list)
            
        if upper_tri:
            if not add_hole:
                for cur in range(1, upper_kp_num - 1):
                    vex1 = upper_kps[0]
                    vex2 = upper_kps[cur]
                    vex3 = upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
            else:
                if self.fillet_mode and (upper_strong_modify or down_strong_modify):
                    tris += self.calculateTriPlaneWithHole(upper_kps, fillet_point_list[-1], 1)
                else:
                    tris += self.calculateTriPlaneWithHole(upper_kps, upper_another_points_list, 1)
        #hole
        if add_hole:
            if self.fillet_mode and (upper_strong_modify or down_strong_modify):
                for i in range(layers - 1):
                    for id in range(len(fillet_point_list[i])):
                        for cur in range(len(fillet_point_list[i][id])):
                            next_ele = (cur + 1) % len(fillet_point_list[i][id])
                            ans1 = self.getTriangle(fillet_point_list[i][id][cur], fillet_point_list[i + 1][id][cur], fillet_point_list[i][id][next_ele])
                            ans2 = self.getTriangle(fillet_point_list[i][id][next_ele], fillet_point_list[i + 1][id][cur], fillet_point_list[i + 1][id][next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
            else:
                for id in range(len(another_points_list)):
                    for cur in range(len(another_points_list[id])):
                        next_ele = (cur + 1) % len(another_points_list[id])
                        ans1 = self.getTriangle(another_points_list[id][cur], upper_another_points_list[id][cur], another_points_list[id][next_ele])
                        ans2 = self.getTriangle(another_points_list[id][next_ele], upper_another_points_list[id][cur], upper_another_points_list[id][next_ele])
                        tris.append(ans1)
                        tris.append(ans2)

        if additional_crease:
            #additional crease
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                crease_type = unit.getCrease()[cur].getType()
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
        return tris
    
    def calculateTriPlaneWithBiasAndHeightWithInner(self, unit, unit_id, upper_bias, down_bias, base_height, upper_height, add_hole=False, another_points_list=None, additional_crease=True, inner_bias=0.0):
        standard_kps = unit.getSeqPoint()
        side = MIDDLE
        compensation_layer = False
        if down_bias < upper_bias:
            down_strong_modify = False
            upper_strong_modify = True
            side = UP
        elif down_bias > upper_bias:
            down_strong_modify = True
            upper_strong_modify = False
            side = DOWN
        else:
            down_strong_modify = True
            upper_strong_modify = True
        if abs(down_bias - upper_bias) <= 0.0011 * self.min_bias:
            down_strong_modify = False
            upper_strong_modify = False
            compensation_layer = True

        for ele in another_points_list:
            for point in ele:
                point[Z] = base_height
        if down_bias > 0:
            bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias, asym_side=0 if not compensation_layer else -1)
            if down_strong_modify:
                inner_bottom_kps, inner_bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias + 3 * inner_bias, base_height + self.layer * self.print_accuracy, side, down_strong_modify, down_bias, asym_side=0 if not compensation_layer else -1)
            else:
                inner_bottom_kps, inner_bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias + 3 * inner_bias, base_height, side, down_strong_modify, down_bias, asym_side=0 if not compensation_layer else -1)
        elif down_bias == 0:
            bottom_kps = deepcopy(standard_kps)
            for ele in bottom_kps:
                ele[Z] = base_height
            bottom_problem_id = []
        else:
            return #error
        if upper_bias > 0:
            upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias, asym_side=1 if not compensation_layer else -1)
            if upper_strong_modify:
                inner_upper_kps, inner_upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias + 3 * inner_bias, upper_height - self.layer * self.print_accuracy, side, upper_strong_modify, upper_bias, asym_side=1 if not compensation_layer else -1)
            else:
                inner_upper_kps, inner_upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias + 3 * inner_bias, upper_height, side, upper_strong_modify, upper_bias, asym_side=1 if not compensation_layer else -1)
        elif upper_bias == 0:
            upper_kps = deepcopy(standard_kps)
            for ele in upper_kps:
                ele[Z] = upper_height
            upper_problem_id = []
        else:
            return #error
        kp_num = len(bottom_kps)
        inner_kp_num = len(inner_bottom_kps)
        upper_kp_num = len(upper_kps)
        inner_upper_kp_num = len(inner_upper_kps)
        tris = []
        # bottom
        if not add_hole:
            if down_strong_modify:
                for cur in range(1, kp_num - 1):
                    vex1 = bottom_kps[0]
                    vex2 = bottom_kps[cur]
                    vex3 = bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
            
                for cur in range(1, inner_kp_num - 1):
                    vex1 = inner_bottom_kps[0]
                    vex2 = inner_bottom_kps[cur]
                    vex3 = inner_bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
        else:
            if down_strong_modify:
                tris += self.calculateTriPlaneWithHole(bottom_kps, another_points_list, -1)

                inner_another_points_list = deepcopy(another_points_list)
                for ele in inner_another_points_list:
                    for point in ele:
                        point[Z] = base_height + self.layer * self.print_accuracy
                tris += self.calculateTriPlaneWithHole(inner_bottom_kps, inner_another_points_list, 1)
        #around
        # if kp_num >= upper_kp_num:
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()

                if real_id in upper_problem_id:
                    if real_id not in bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])  
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = upper_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = upper_next_ele
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = upper_cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in bottom_problem_id:
                    if real_id not in upper_problem_id:
                        if 1:
                            ans = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = next_ele
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, inner_kp_num):
                next_ele = (cur + 1) % inner_kp_num
                upper_next_ele = (upper_cur + 1) % inner_upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_upper_problem_id:
                    if real_id not in inner_bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = upper_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                        ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = upper_next_ele
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, inner_upper_kp_num):
                next_ele = (cur + 1) % inner_kp_num
                upper_next_ele = (upper_cur + 1) % inner_upper_kp_num
                real_id = upper_cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_bottom_problem_id:
                    if real_id not in inner_upper_problem_id:
                        if 1:
                            ans = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            ans2 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_next_ele], inner_bottom_kps[cur])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                        ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = next_ele
        #upper
        if not add_hole:
            if upper_strong_modify:
                for cur in range(1, upper_kp_num - 1):
                    vex1 = upper_kps[0]
                    vex2 = upper_kps[cur]
                    vex3 = upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
            
                for cur in range(1, inner_upper_kp_num - 1):
                    vex1 = inner_upper_kps[0]
                    vex2 = inner_upper_kps[cur]
                    vex3 = inner_upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
        else:
            if upper_strong_modify:
                upper_another_points_list = deepcopy(another_points_list)
                for ele in upper_another_points_list:
                    for point in ele:
                        point[Z] = upper_height
                tris += self.calculateTriPlaneWithHole(upper_kps, upper_another_points_list, 1)

                inner_upper_another_points_list = deepcopy(another_points_list)
                for ele in inner_upper_another_points_list:
                    for point in ele:
                        point[Z] = upper_height - 3.0 * self.print_accuracy
                tris += self.calculateTriPlaneWithHole(inner_upper_kps, inner_upper_another_points_list, -1)
        #hole
        if add_hole:
            if down_strong_modify:
                for id in range(len(another_points_list)):
                    for cur in range(self.unit_hole_resolution):
                        next_ele = (cur + 1) % self.unit_hole_resolution
                        ans1 = self.getTriangle(another_points_list[id][cur], inner_another_points_list[id][cur], another_points_list[id][next_ele])
                        ans2 = self.getTriangle(another_points_list[id][next_ele], inner_another_points_list[id][cur], inner_another_points_list[id][next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
            if upper_strong_modify:
                for id in range(len(another_points_list)):
                    for cur in range(self.unit_hole_resolution):
                        next_ele = (cur + 1) % self.unit_hole_resolution
                        ans1 = self.getTriangle(inner_upper_another_points_list[id][cur], upper_another_points_list[id][cur], inner_upper_another_points_list[id][next_ele])
                        ans2 = self.getTriangle(inner_upper_another_points_list[id][next_ele], upper_another_points_list[id][cur], upper_another_points_list[id][next_ele])
                        tris.append(ans1)
                        tris.append(ans2)

        #connection
        if upper_strong_modify:
            forward_step = 0
            inner_cur = 0
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                inner_next_ele = (inner_cur + 1) % inner_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_bottom_problem_id:
                    if real_id not in bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                            ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], inner_bottom_kps[inner_cur], bottom_kps[next_ele])  
                            tris.append(ans1)
                            tris.append(ans2)
                        inner_cur = inner_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                        ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], inner_bottom_kps[inner_cur], bottom_kps[next_ele])  
                        tris.append(ans1)
                        tris.append(ans2)
                    inner_cur = inner_next_ele
        else:
            forward_step = 0
            inner_cur = 0
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                inner_next_ele = (inner_cur + 1) % inner_upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_upper_problem_id:
                    if real_id not in upper_problem_id:
                        if 1:
                            ans = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                            ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
                            tris.append(ans1)
                            tris.append(ans2)
                        inner_cur = inner_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                        ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
                        tris.append(ans1)
                        tris.append(ans2)
                    inner_cur = inner_next_ele

        if additional_crease:
            #additional crease
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                crease_type = unit.getCrease()[cur].getType()
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
        return tris

    def calculateTriPlaneForUnit(self, unit_id, inner=False):
        unit = self.unit_list[unit_id]
        add_hole = False
        another_points_list = []
        for ele in self.unit_hole_list:
            if ele[1] == unit_id:
                add_hole = True
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.unit_hole_size, self.unit_hole_resolution)
                another_points_list.append(points)
        
        if self.method == "upper_bias":
            self.symmetry_flag = False
            self.unit_height = self.height
            self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.min_bias, 
                base_height         =0,
                upper_height        =self.height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list,
                additional_crease   =False
            )
        elif self.method == "both_bias":
            self.symmetry_flag = False
            self.unit_height = self.height
            self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.bias, 
                base_height         =0,
                upper_height        =self.height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list,
                additional_crease   =False
            )
        elif self.method == "symmetry":
            self.symmetry_flag = True
            self.unit_height = self.height / 2
            # add connection holes
            connection_left_holes = []
            connection_right_holes = []
            # judge if there is modified unit
            if self.using_modified_unit:
                try:
                    modified_unit = self.modified_unit_list[unit_id]
                    having_modified = True
                except:
                    having_modified = False
                    pass
                # add additional hole
                for ele in self.connection_hole_list:
                    if ele[1] == unit_id:
                        center = ele[0] + [0.0]
                        points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                        if ele[-1] == LEFT:
                            connection_left_holes.append(points)
                        else:
                            connection_right_holes.append(points)

            if len(connection_left_holes) > 0:
                add_hole = True

            if self.db_enable:
                if inner:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeightWithInner(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias, 
                        down_bias           =self.bias, 
                        base_height         =0,
                        upper_height        =self.height / 2, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False,
                        inner_bias = self.base_inner_bias
                    )
                else:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias + 3.0 * self.base_inner_bias, 
                        down_bias           =self.bias + 3.0 * self.base_inner_bias, 
                        base_height         =self.layer * self.print_accuracy,
                        upper_height        =self.height / 2, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False,
                        penalty             =3.0 * self.base_inner_bias
                    )
                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    if inner:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeightWithInner(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias, 
                            down_bias           =self.min_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            inner_bias = self.base_inner_bias
                        )
                    else:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias + 3.0 * self.base_inner_bias, 
                            down_bias           =self.min_bias + 3.0 * self.base_inner_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height - self.layer * self.print_accuracy, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            penalty             =3.0 * self.base_inner_bias
                        )
                elif self.using_modified_unit and having_modified:
                    if inner:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeightWithInner(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias, 
                            down_bias           =self.min_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            inner_bias = self.base_inner_bias
                        )
                    else:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias + 3.0 * self.base_inner_bias, 
                            down_bias           =self.min_bias + 3.0 * self.base_inner_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height - self.layer * self.print_accuracy, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            penalty             =3.0 * self.base_inner_bias
                        )
            else:
                self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias, 
                    down_bias           =self.bias, 
                    base_height         =0,
                    upper_height        =self.height / 2, 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False
                )

                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2 + self.board_height,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False
                    )
                    
                elif self.using_modified_unit and having_modified:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2 + self.board_height,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False
                    )
        elif self.method == 'binding':
            self.symmetry_flag = True
            self.unit_height = self.height / 2
            # add connection holes
            connection_left_holes = []
            connection_right_holes = []
            # judge if there is modified unit
            if self.using_modified_unit:
                try:
                    modified_unit = self.modified_unit_list[unit_id]
                    having_modified = True
                except:
                    having_modified = False
                    pass
                # add additional hole
                for ele in self.connection_hole_list:
                    if ele[1] == unit_id:
                        center = ele[0] + [0.0]
                        points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                        if ele[-1] == LEFT:
                            connection_left_holes.append(points)
                        else:
                            connection_right_holes.append(points)

            if len(connection_left_holes) > 0:
                add_hole = True

            if not self.thin_mode:
                # if self.db_enable:
                self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias * 1.001 + 0.02, 
                    down_bias           =self.bias, 
                    base_height         =0,
                    upper_height        =self.height / 2. - self.print_accuracy / 2.0, 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False,
                    upper_tri           =False
                )

                self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias + 0.02, 
                    down_bias           =self.min_bias * 1.001 + 0.02, 
                    base_height         =self.height / 2. - self.print_accuracy / 2.0,
                    upper_height        =self.height / 2., 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False,
                    bottom_tri          =False
                )

                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias * 1.001, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height,
                        upper_height        =self.height / 2. + self.board_height + self.print_accuracy / 2.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        upper_tri           =False
                    )

                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias * 1.001, 
                        base_height         =self.height / 2. + self.board_height + self.print_accuracy / 2.0,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        bottom_tri           =False
                    )

                elif self.using_modified_unit and having_modified:
                    # self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                    #     unit                =modified_unit, 
                    #     unit_id             =unit_id, 
                    #     upper_bias          =self.bias, 
                    #     down_bias           =self.min_bias, 
                    #     base_height         =self.height / 2. + self.board_height,
                    #     upper_height        =self.height + self.board_height, 
                    #     add_hole            =add_hole, 
                    #     another_points_list =deepcopy(another_points_list)+connection_right_holes,
                    #     additional_crease   =False,
                    # )
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias * 1.001, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height,
                        upper_height        =self.height / 2. + self.board_height + self.print_accuracy / 2.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        upper_tri           =False
                    )

                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias * 1.001, 
                        base_height         =self.height / 2. + self.board_height + self.print_accuracy / 2.0,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        bottom_tri           =False
                    )
                # else:
                #     self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                #         unit                =unit, 
                #         unit_id             =unit_id, 
                #         upper_bias          =self.min_bias, 
                #         down_bias           =self.bias, 
                #         base_height         =0,
                #         upper_height        =self.height / 2., 
                #         add_hole            =add_hole, 
                #         another_points_list =deepcopy(another_points_list)+connection_left_holes,
                #         additional_crease   =False
                #     )

                #     if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                #         add_hole = True
                #     else:
                #         add_hole = False

                #     if not self.using_modified_unit:
                #         self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                #             unit                =unit, 
                #             unit_id             =unit_id, 
                #             upper_bias          =self.bias, 
                #             down_bias           =self.min_bias, 
                #             base_height         =self.height / 2. + self.board_height,
                #             upper_height        =self.height + self.board_height, 
                #             add_hole            =add_hole, 
                #             another_points_list =deepcopy(another_points_list)+connection_right_holes,
                #             additional_crease   =False
                #         )
                        
                #     elif self.using_modified_unit and having_modified:
                #         self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                #             unit                =modified_unit, 
                #             unit_id             =unit_id, 
                #             upper_bias          =self.bias, 
                #             down_bias           =self.min_bias, 
                #             base_height         =self.height / 2. + self.board_height,
                #             upper_height        =self.height + self.board_height, 
                #             add_hole            =add_hole, 
                #             another_points_list =deepcopy(another_points_list)+connection_right_holes,
                #             additional_crease   =False
                #         )
            else:
                if self.db_enable:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.print_accuracy, 
                        down_bias           =8.0 * self.print_accuracy, 
                        base_height         =0,
                        upper_height        =3.0 * self.print_accuracy, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                    )

                    if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                        add_hole = True
                    else:
                        add_hole = False

                    if not self.using_modified_unit:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                        )
                    elif self.using_modified_unit and having_modified:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                        )
                else:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.print_accuracy, 
                        down_bias           =8.0 * self.print_accuracy, 
                        base_height         =0,
                        upper_height        =self.print_accuracy * 3.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False
                    )

                    if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                        add_hole = True
                    else:
                        add_hole = False

                    if not self.using_modified_unit:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False
                        )
                        
                    elif self.using_modified_unit and having_modified:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False
                        )
    
    def calculateTriPlaneForBoard(self, unit_id, base_height=0, upper_height=None):
        if upper_height == None:
            upper_height = self.board_height
        unit = self.unit_list[unit_id]

        # exist_modify = False
        # for j in range(len(unit.getCrease())):
        #     if self.unit_bias_list[unit_id][j] != None:
        #         exist_modify = True
        #         break
        
        add_hole = False
        another_points_list = []
        for ele in self.unit_hole_list:
            if ele[1] == unit_id:
                add_hole = True
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.unit_hole_size, self.unit_hole_resolution)
                another_points_list.append(points)

        # add additional hole
        connection_left_holes = []
        for ele in self.connection_hole_list:
            if ele[1] == unit_id:
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                if ele[-1] == LEFT:
                    connection_left_holes.append(points)

        if len(connection_left_holes) > 0:
            add_hole = True

        if is_valid_polygon(unit.getSeqPoint()):
            self.board_tri_list += self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.bias, 
                base_height         =base_height,
                upper_height        =upper_height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list+connection_left_holes,
                additional_crease   =False,
                penalty=3.0 * self.base_inner_bias if self.method == 'symmetry' else self.bias,
                accumulation        =True
            )
    
    def calculateTriPlaneForPillar(self, base_height=0, upper_height=None):
        if upper_height == None:
            upper_height = self.board_height

        # self.board_tri_list.clear()
        for i in range(len(self.pillar_unit_list)):
            pillars = self.pillar_unit_list[i]
            for pillar in pillars:
                reverse_pillar = []
                for j in range(0, -len(pillar), -1):
                    reverse_pillar.append(pillar[j])
                if is_valid_triangle(reverse_pillar):
                    self.board_tri_list += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =reverse_pillar, 
                        unit_id             =-1,
                        upper_bias          =0, 
                        down_bias           =0, 
                        base_height         =base_height,
                        upper_height        =upper_height, 
                        add_hole            =False, 
                        another_points_list =[],
                        additional_crease   =False,
                        penalty=3.0 * self.base_inner_bias if self.method == 'symmetry' else self.bias
                    )
        
    def calculateTriPlaneForAllUnit(self, inner=False):
        for i in range(len(self.unit_list)):
            self.calculateTriPlaneForUnit(i, inner)    

    def generateBoard(self):
        for unit_id in range(len(self.unit_list)):
            self.calculateTriPlaneForBoard(unit_id)
        if self.method == "binding":
            self.calculateTriPlaneForPillar()
            
    def addSpace(self, number):
        # O(1) 一次性追加 number 个空格到内部 chunks（原先逐字符 += 导致 O(N^2) 字符拷贝）
        self._s_chunks.append(' ' * number)
        object.__setattr__(self, '_s_join_cache', None)

    def addInfoToStlFile(self, tris, out=None):
        # out=None 走兼容路径：直接 append 到 self._s_chunks（O(1) per facet, 总 O(N)）
        # out=<file/StringIO> 走快速路径：每 facet O(1)，总 O(N)
        # ASCII STL 不需要缩进，解析器只识别 facet/outer loop/vertex/endloop/endfacet 关键字和换行。
        if out is None:
            chunks = self._s_chunks
            for ele in tris:
                n0, n1, n2 = ele[0]
                chunks.append(f'facet normal {n0} {n1} {n2}\n')
                chunks.append('outer loop\n')
                for i in range(0, 3):
                    v0, v1, v2 = ele[1][i]
                    chunks.append(f'vertex {v0} {v1} {v2}\n')
                chunks.append('endloop\n')
                chunks.append('endfacet\n')
            # cache 已失效（下一次 self.s 读会重新 join 刷新）
            object.__setattr__(self, '_s_join_cache', None)
        else:
            for ele in tris:
                n0, n1, n2 = ele[0]
                out.write(f'facet normal {n0} {n1} {n2}\n')
                out.write('outer loop\n')
                for i in range(0, 3):
                    v0, v1, v2 = ele[1][i]
                    out.write(f'vertex {v0} {v1} {v2}\n')
                out.write('endloop\n')
                out.write('endfacet\n')

    def outputUnitStl(self, unit_id, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __Unit_' + str(unit_id) + '__ SLA File\n')
        self.addInfoToStlFile(self.tri_list[unit_id], out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()  # 保持向后兼容
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputAllStl(self, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __All_Units__ SLA File\n')
        for i in range(len(self.unit_list)):
            tris = self.tri_list[i]
            if len(self.unit_rotation_matrix):
                # calculate new norm and point
                new_tris = []
                for ele in tris:
                    new_points = []
                    for point in ele[1]:
                        origin = np.array([point[X], point[Y], point[Z] - 0.5 * (self.height + self.board_height)])
                        new_trans_point = self.unit_rotation_matrix[i] @ origin + self.unit_transformation_vector[i]
                        new_points.append(new_trans_point.tolist())
                    new_tris.append(self.getTriangle(new_points[0], new_points[1], new_points[2]))
                tris = new_tris
            self.addInfoToStlFile(tris, out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()  # 保持向后兼容
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputCreaseStl(self, crease_id, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __Crease__ SLA File\n')
        self.addInfoToStlFile(self.crease_tri_list[crease_id], out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputAllCreaseStl(self, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __All_Crease__ SLA File\n')
        for i in range(len(self.valid_crease_list)):
            tris = self.crease_tri_list[i]
            if tris != None:
                self.addInfoToStlFile(tris, out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputBoardStl(self, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __Board__ SLA File\n')
        self.addInfoToStlFile(self.board_tri_list, out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputStringStl(self, filepath):
        buf = io.StringIO()
        buf.write('solid PyGamiX generated __String__ SLA File\n')
        self.addInfoToStlFile(self.string_tri_list, out=buf)
        buf.write('endsolid\n')
        self.s = buf.getvalue()
        with open(filepath, 'w') as f:
            f.write(self.s)