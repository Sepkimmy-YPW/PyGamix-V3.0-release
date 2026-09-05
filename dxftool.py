import ezdxf
import numpy as np
import math
from utils import *

class OrigamiToDxfConverter:
    """
    折纸到DXF转换器类。
    中文：
        将折纸图案数据导出为AutoCAD DXF格式文件。
        支持导出折痕线（山折/谷折/边界/切割）和圆形孔洞。

    Origami to DXF converter class.
    English:
        Exports origami pattern data to AutoCAD DXF format.
        Supports exporting crease lines (mountain/valley/border/cutting) and circular holes.
    """
    def __init__(self, filename=None) -> None:
        """
        初始化DXF转换器。

        Initialize the DXF converter.

        :param filename: 输出文件名 / Output filename
        """
        self.filename = filename
        self.circle = []
    
    def setFileName(self, filename):
        """设置输出文件名 / Set output filename"""
        self.filename = filename
    
    def addCircle(self, center, radius, tp):
        """
        添加圆形孔洞。

        Add a circular hole.

        :param center: 圆心坐标 [x, y] / Center coordinates [x, y]
        :param radius: 半径 / Radius
        :param tp: 类型（HOLE/HINGE_HOLE）/ Type (HOLE/HINGE_HOLE)
        """
        self.circle.append([center, radius, tp])

    def ExportAsDxf(self, lines: list):
        """
        将折痕线列表导出为DXF文件。
        中文：根据折痕类型（MOUNTAIN/VALLEY/BORDER等）创建对应的DXF图层，
              并将折痕线导出到相应图层。

        Export crease lines to DXF file.
        English: Creates DXF layers based on crease types and exports lines to corresponding layers.

        :param lines: 折痕线列表 [Crease, ...] / List of crease lines
        """
        doc = ezdxf.new(setup=True, units=4)
        msp = doc.modelspace()
        doc.layers.new(name='Mountain', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 25,
                'color': 1
            }
        )
        doc.layers.new(name='Valley', 
            dxfattribs={
                'linetype': 'DASHEDX2', 
                'lineweight': 25,
                'color': 5
            }
        )
        doc.layers.new(name='Cutting', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 25,
                'color': 3
            }
        )
        doc.layers.new(name='Border', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 30,
                'color': 7
            }
        )
        doc.layers.new(name='Hole', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 30,
                'color': 3
            }
        )
        doc.layers.new(name='Hinge_hole', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 30,
                'color': 8
            }
        )
        for i in range(len(lines)):
            line = lines[i]
            type_crease = line.getType()
            if distance(line[START], line[END]) < 1e-5:
                continue
            not_duplicate = True
            for j in range(i):
                other_line = lines[j]
                if (distance(line[START], other_line[START]) < 1e-5 and distance(line[END], other_line[END]) < 1e-5) or (distance(line[START], other_line[END]) < 1e-5 and distance(line[END], other_line[START]) < 1e-5):
                    not_duplicate = False
                    break
            if not not_duplicate:
                continue
            if type_crease == MOUNTAIN:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Mountain'})
            elif type_crease == VALLEY:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Valley'})
            elif type_crease == CUTTING:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Cutting'})
            elif type_crease == HOLE:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Hole'})
            elif type_crease == HINGE_HOLE:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Hinge_hole'})
            else:
                msp.add_line((line[0][0], line[0][1]), (line[1][0], line[1][1]), dxfattribs={'layer': 'Border'})
        
        if len(self.circle):
            for ele in self.circle:
                if ele[2] == HOLE:
                    msp.add_circle(ele[0], ele[1], dxfattribs={'layer': 'Hole'})
                elif ele[2] == HINGE_HOLE:
                    msp.add_circle(ele[0], ele[1], dxfattribs={'layer': 'Hinge_hole'})
        doc.saveas(self.filename)
    
    def ExportAsDxfUsingUnits(self, unit: list):
        """
        使用面片单元列表导出DXF文件。
        中文：将面片单元的多边形顶点导出为DXF多段线（LWPOLYLINE）。

        Export DXF file using panel units.
        English: Exports panel unit polygons as DXF polylines (LWPOLYLINE).

        :param unit: 面片单元顶点列表 / List of panel unit vertices
        """
        doc = ezdxf.new(setup=True, units=4)
        msp = doc.modelspace()
        doc.layers.new(name='Common_Polyline', 
            dxfattribs={
                'linetype': 'BYLAYER', 
                'lineweight': 30,
                'color': 7
            }
        )
        for sub_unit in unit:
            sub_unit.append(sub_unit[0])
            msp.add_lwpolyline(points=sub_unit, dxfattribs={'layer': 'Common_Polyline'})
        
        doc.saveas(self.filename)

            
