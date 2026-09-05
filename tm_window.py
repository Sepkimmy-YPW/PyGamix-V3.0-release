# import sys
import matplotlib
from matplotlib import cm
from matplotlib.patches import Rectangle
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from cdftool import *

class TmWindow(QMainWindow):
    def __init__(self, res=2, timer_enable=False, enable_3d=False):
        super().__init__()

        self.tm = TransitionModel(enable_3d=enable_3d)
        self.folding_angle = 1
        self.discrete_resolution = res
        self.enable_update_canvas = False
        self.enable_3d = enable_3d
        self.setWindowTitle("Transition angle")

        # 创建一个垂直布局和一个QWidget小部件
        self.lay_out = QVBoxLayout()
        self.widget = QWidget(self)
        self.widget.setLayout(self.lay_out)
        self.setCentralWidget(self.widget)

        self.dynamic_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        if enable_3d:
            self.dynamic_ax = self.dynamic_canvas.figure.add_subplot(111, projection='3d')
        else:
            self.dynamic_ax = self.dynamic_canvas.figure.subplots()
        self.lay_out.addWidget(self.dynamic_canvas)

        self.horizontal_folding_slider = QtWidgets.QSlider()
        self.horizontal_folding_slider.setGeometry(QtCore.QRect(165, 480, 751, 22))
        self.horizontal_folding_slider.setMaximum(180)
        self.horizontal_folding_slider.setProperty("value", 0)
        self.horizontal_folding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontal_folding_slider.setObjectName("horizontal_folding_slider")
        self.lay_out.addWidget(self.horizontal_folding_slider)

        self.horizontal_folding_slider.valueChanged.connect(self.changeFoldingAngle)

        self.enable_show_process = False

        self.timer_enable = timer_enable
        if timer_enable:
            self._timer = self.dynamic_canvas.new_timer(33, [
                (self.updateCanvas, (), {})
            ])
            self._timer.start()

        self.x_axis = []
        self.y_axis = []
        self.z_axis = []

        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []

    def importXPoints(self, xpoints):
        self.x_axis = xpoints

    def importYPoints(self, ypoints):
        self.y_axis = ypoints

    def importZPoints(self, zpoints):
        self.z_axis = zpoints
    
    def importTrajectory(self, x, y, z):
        self.trajectory_x = x
        self.trajectory_y = y
        self.trajectory_z = z

    def enable3d(self):
        return self.enable_3d

    def plot(self):
        res = self.discrete_resolution
        self.dynamic_ax.clear()
        # Get a AABB box
        max_x = max(self.x_axis)
        max_y = max(self.y_axis)
        if not self.enable3d():
            self.dynamic_ax.plot(
                self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                marker='*', c='r', markersize=12
            )
            self.dynamic_ax.plot([0], [0], marker='^', c=cm.OrRd(1.0), markersize=10)
        else:
            self.dynamic_ax.plot(
                self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                self.z_axis[0: len(self.z_axis): len(self.z_axis) // (res - 1)] + [self.z_axis[-1]], 
                marker='*', c='r', markersize=12
            )
            self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10)
            if 1:
                x = 100
                y = 40
                z = 0
                dx = max_x - x
                dy = max_y - y
                dz = 100
                xx = np.linspace(x, x + dx, 2)
                yy = np.linspace(y, y + dy, 2)
                zz = np.linspace(z, z + dz, 2)

                xx1, yy1 = np.meshgrid(xx, yy)
                self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z), cmap=cm.hot, alpha=0.5)
                self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z + dz), cmap=cm.hot, alpha=0.5)

                yy2, zz2 = np.meshgrid(yy, zz)
                self.dynamic_ax.plot_surface(np.full_like(yy2, x), yy2, zz2, cmap=cm.hot, alpha=0.5)
                self.dynamic_ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2, cmap=cm.hot, alpha=0.5)

                xx3, zz3 = np.meshgrid(xx, zz)
                self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y), zz3, cmap=cm.hot, alpha=0.5)
                self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y + dy), zz3, cmap=cm.hot, alpha=0.5)

                if len(self.trajectory_x) > 0:
                    self.dynamic_ax.scatter(self.trajectory_x, self.trajectory_y, self.trajectory_z, color=(0.5625, 0.94, 0.5625), linewidth=0.5, label="simulation")

        # rect1 = Rectangle((140.0, 30.0), 110, 110, color='orange')
        # rect2 = Rectangle((150.0, 40.0), 100, 100, color='grey')
        # self.dynamic_ax.add_patch(rect1)
        # self.dynamic_ax.add_patch(rect2)

        # self.dynamic_ax.set_xlim(-50.0, 300.0)
        # self.dynamic_ax.set_ylim(-30.0, 170.0)

        self.dynamic_ax.figure.canvas.draw()
        if not self.enable3d():
            self.dynamic_ax.set_aspect('equal')
        else:
            self.dynamic_ax.set_xlabel("X")
            self.dynamic_ax.set_ylabel("Y")
            self.dynamic_ax.set_zlabel("Z")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Up:
            self.folding_angle += 1
            if self.folding_angle >= 180:
                self.folding_angle = 180
        elif event.key() == Qt.Key_Down:
            self.folding_angle -= 1
            if self.folding_angle <= 0:
                self.folding_angle = 0
        elif event.key() == Qt.Key_C:
            self.enable_show_process = not self.enable_show_process
        self.updateCanvas()
    
    def getTm(self):
        return self.tm
    
    def updateCanvas(self):
        res = self.discrete_resolution
        if self.enable_update_canvas:
            self.tm.setMainFoldingAngle(self.folding_angle * math.pi / 180.0)
            self.tl, _, _ = self.tm.getTransitionLines()
            if self.enable3d():
                self.tl_3d, _, _ = self.tm.calculate3DLines()
            self.dynamic_ax.clear()
            # Get a AABB box
            min_x, max_x, min_y, max_y, min_z, max_z = self.tm.getAABB()

            if len(self.x_axis) > 0:
                # self.dynamic_ax.plot(self.x_axis, self.y_axis, marker='x', c='g')
                if not self.enable3d():
                    self.dynamic_ax.plot(
                        self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                        self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                        marker='*', c='r', markersize=10
                    )
                    self.dynamic_ax.plot([0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=1)
                else:
                    self.dynamic_ax.plot(
                        self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                        self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                        self.z_axis[0: len(self.z_axis): len(self.z_axis) // (res - 1)] + [self.z_axis[-1]], 
                        marker='*', c='r', markersize=10, zorder=1
                    )
                    self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=1)
                    if 1:
                        x = 100
                        y = 40
                        z = 0
                        dx = max_x - x
                        dy = max_y - y
                        dz = 100
                        xx = np.linspace(x, x + dx, 2)
                        yy = np.linspace(y, y + dy, 2)
                        zz = np.linspace(z, z + dz, 2)

                        xx1, yy1 = np.meshgrid(xx, yy)
                        self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z), cmap=cm.hot, alpha=0.5, zorder=1)
                        self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z + dz), cmap=cm.hot, alpha=0.5, zorder=1)

                        yy2, zz2 = np.meshgrid(yy, zz)
                        self.dynamic_ax.plot_surface(np.full_like(yy2, x), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)
                        self.dynamic_ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)

                        xx3, zz3 = np.meshgrid(xx, zz)
                        self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
                        self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y + dy), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
                
                # rect1 = Rectangle((140.0, 30.0), 1000, 1000, color='orange')
                # rect2 = Rectangle((150.0, 40.0), 1000, 1000, color='grey')
                # self.dynamic_ax.add_patch(rect1)
                # self.dynamic_ax.add_patch(rect2)

            if len(self.trajectory_x) > 0:
                self.dynamic_ax.scatter(self.trajectory_x, self.trajectory_y, self.trajectory_z, color=(0.5625, 0.94, 0.5625), linewidth=0.5, label="simulation")
                
            # 绘制所有线段
            if not self.enable3d():
                for i in range(len(self.tl)):
                    start, end = self.tl[i].getData()
                    line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl)), zorder=100)      
                
                if self.enable_show_process:
                    self.dynamic_ax.plot(
                        [self.tm.all_end_ef[i][X] for i in range(1, len(self.tm.all_end_ef))], 
                        [self.tm.all_end_ef[i][Y] for i in range(1, len(self.tm.all_end_ef))], c='purple', zorder=1
                    )

                self.dynamic_ax.plot([0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=2)
                self.dynamic_ax.plot([self.tm.end_ef[X]], [self.tm.end_ef[Y]], marker='o', c=cm.OrRd(0.5), zorder=1)

            else:
                for i in range(len(self.tl_3d)):
                    start, end = self.tl_3d[i].getData()
                    order = 1
                    if not (end[X] <= 100 and end[Y] >= 40 and end[Z] <= 100):
                        order = 200
                    line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], [start[Z], end[Z]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl_3d)), zorder=order)
                
                if self.enable_show_process:
                    xs = [self.tm.all_end_ef_3d[i][X] for i in range(1, len(self.tm.all_end_ef_3d))]
                    ys = [self.tm.all_end_ef_3d[i][Y] for i in range(1, len(self.tm.all_end_ef_3d))]
                    zs = [self.tm.all_end_ef_3d[i][Z] for i in range(1, len(self.tm.all_end_ef_3d))]

                    high_order_xs = []
                    high_order_ys = []
                    high_order_zs = []
                    low_order_xs = []
                    low_order_ys = []
                    low_order_zs = []

                    for i in range(0, len(self.tm.all_end_ef_3d) - 1):
                        if not (xs[i] <= 100 and ys[i] >= 40 and zs[i] <= 100):
                            high_order_xs.append(xs[i])
                            high_order_ys.append(ys[i])
                            high_order_zs.append(zs[i])
                        else:
                            low_order_xs.append(xs[i])
                            low_order_ys.append(ys[i])
                            low_order_zs.append(zs[i])
                    self.dynamic_ax.scatter(
                        high_order_xs, 
                        high_order_ys,
                        high_order_zs, c='purple', zorder=100, linewidths=0.5, label="kinematic model"
                    )
                    self.dynamic_ax.scatter(
                        low_order_xs, 
                        low_order_ys,
                        low_order_zs, c='purple', zorder=0, linewidths=0.5, label="kinematic model"
                    )

                self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=100)
                self.dynamic_ax.plot([self.tm.end_ef_3d[X]], [self.tm.end_ef_3d[Y]], [self.tm.end_ef_3d[Z]], marker='o', c=cm.OrRd(0.5), zorder=1)


            self.dynamic_ax.set_xlim(min_x, max_x)
            self.dynamic_ax.set_ylim(min_y, max_y)
            self.dynamic_ax.set_xlabel("X")
            self.dynamic_ax.set_ylabel("Z")
            if self.enable3d():
                self.dynamic_ax.set_zlim(min_z, max_z)
                self.dynamic_ax.set_xlabel("X")
                self.dynamic_ax.set_ylabel("Y")
                self.dynamic_ax.set_zlabel("Z")

            self.dynamic_ax.figure.canvas.draw()
            if not self.enable3d():
                self.dynamic_ax.set_aspect('equal')
            # self.dynamic_ax.figure.canvas.print_figure("111.png")
        
    def changeFoldingAngle(self):
        self.folding_angle = self.horizontal_folding_slider.value()
        if self.folding_angle <= 1:
            self.folding_angle = 1
        self.updateCanvas()

    def startShow(self):
        self.tm.setMainFoldingAngle(0.001)
        self.tl, _, _ = self.tm.getTransitionLines()
        self.enable_update_canvas = True
        # self.tm.plotUi()
        self.updateCanvas()
        self.show()
    
    def printTm(self, file_path, file_path2, time):
        res = self.discrete_resolution
        self.tm.setMainFoldingAngle(0.001)
        self.tl, _, _ = self.tm.getTransitionLines()
        if self.enable3d():
            self.tl_3d, _, _ = self.tm.calculate3DLines()
        self.dynamic_ax.clear()
        # Get a AABB box
        min_x, max_x, min_y, max_y, min_z, max_z = self.tm.getAABB()

        if self.x_axis != None and len(self.x_axis) > 0:
            # self.dynamic_ax.plot(self.x_axis, self.y_axis, marker='x', c='g')
            if not self.enable3d():
                self.dynamic_ax.plot(
                    self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]],               
                    self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                    marker='*', c='r', markersize=10
                )
            else:
                self.dynamic_ax.plot(
                    self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                    self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                    self.z_axis[0: len(self.z_axis): len(self.z_axis) // (res - 1)] + [self.z_axis[-1]], 
                    marker='*', c='r', markersize=10, zorder=1
                )
                self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=1)
                if 1:
                    x = 100
                    y = 40
                    z = 0
                    dx = max_x - x
                    dy = max_y - y
                    dz = 100
                    xx = np.linspace(x, x + dx, 2)
                    yy = np.linspace(y, y + dy, 2)
                    zz = np.linspace(z, z + dz, 2)

                    xx1, yy1 = np.meshgrid(xx, yy)
                    self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z), cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z + dz), cmap=cm.hot, alpha=0.5, zorder=1)

                    yy2, zz2 = np.meshgrid(yy, zz)
                    self.dynamic_ax.plot_surface(np.full_like(yy2, x), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)

                    xx3, zz3 = np.meshgrid(xx, zz)
                    self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y + dy), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
            # rect1 = Rectangle((140.0, 30.0), 1000, 1000, color='orange')
            # rect2 = Rectangle((150.0, 40.0), 1000, 1000, color='grey')
            # self.dynamic_ax.add_patch(rect1)
            # self.dynamic_ax.add_patch(rect2)

        # 绘制所有线段
        if not self.enable3d():
            for i in range(len(self.tl)):
                start, end = self.tl[i].getData()
                line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl)))
                    
            if self.enable_show_process:
                self.dynamic_ax.plot(
                    [self.tm.all_end_ef[i][X] for i in range(1, len(self.tm.all_end_ef))], 
                    [self.tm.all_end_ef[i][Y] for i in range(1, len(self.tm.all_end_ef))], c='purple'
                )

            self.dynamic_ax.plot([0], [0], marker='^', c=cm.OrRd(1.0), markersize=10)
            self.dynamic_ax.plot([self.tm.end_ef[X]], [self.tm.end_ef[Y]], marker='o', c=cm.OrRd(0.5))
        else:
            for i in range(len(self.tl_3d)):
                start, end = self.tl_3d[i].getData()
                order = 1
                if not (end[X] <= 100 and end[Y] >= 40 and end[Z] <= 100):
                    order = 200
                line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], [start[Z], end[Z]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl_3d)), zorder=order)
            
            if self.enable_show_process:
                xs = [self.tm.all_end_ef_3d[i][X] for i in range(1, len(self.tm.all_end_ef_3d))]
                ys = [self.tm.all_end_ef_3d[i][Y] for i in range(1, len(self.tm.all_end_ef_3d))]
                zs = [self.tm.all_end_ef_3d[i][Z] for i in range(1, len(self.tm.all_end_ef_3d))]

                high_order_xs = []
                high_order_ys = []
                high_order_zs = []
                low_order_xs = []
                low_order_ys = []
                low_order_zs = []

                for i in range(0, len(self.tm.all_end_ef_3d) - 1):
                    if not (xs[i] <= 100 and ys[i] >= 40 and zs[i] <= 100):
                        high_order_xs.append(xs[i])
                        high_order_ys.append(ys[i])
                        high_order_zs.append(zs[i])
                    else:
                        low_order_xs.append(xs[i])
                        low_order_ys.append(ys[i])
                        low_order_zs.append(zs[i])
                self.dynamic_ax.scatter(
                    high_order_xs, 
                    high_order_ys,
                    high_order_zs, c='purple', zorder=100, linewidths=0.5
                )
                self.dynamic_ax.scatter(
                    low_order_xs, 
                    low_order_ys,
                    low_order_zs, c='purple', zorder=0, linewidths=0.5
                )

            self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=100)
            self.dynamic_ax.plot([self.tm.end_ef_3d[X]], [self.tm.end_ef_3d[Y]], [self.tm.end_ef_3d[Z]], marker='o', c=cm.OrRd(0.5), zorder=1)

        self.dynamic_ax.set_xlim(min_x, max_x)
        self.dynamic_ax.set_ylim(min_y, max_y)
        self.dynamic_ax.set_xlabel("X")
        self.dynamic_ax.set_ylabel("Z")
        if self.enable3d():
            self.dynamic_ax.set_xlabel("X")
            self.dynamic_ax.set_ylabel("Y")
            self.dynamic_ax.set_zlabel("Z")
            self.dynamic_ax.set_zlim(min_z, max_z)

        if not self.enable3d():
            self.dynamic_ax.set_aspect('equal')

        self.dynamic_ax.figure.canvas.print_figure(file_path + '/' + str(time).zfill(8) + '.png')
        
        # ---
        self.tm.setMainFoldingAngle(math.pi)
        self.tl, _, _ = self.tm.getTransitionLines()
        if self.enable3d():
            self.tl_3d, _, _ = self.tm.calculate3DLines()
        self.dynamic_ax.clear()
        # Get a AABB box
        min_x, max_x, min_y, max_y, min_z, max_z = self.tm.getAABB()

        if self.x_axis != None and len(self.x_axis) > 0:
            # self.dynamic_ax.plot(self.x_axis, self.y_axis, marker='x', c='g')
            if not self.enable3d():
                self.dynamic_ax.plot(
                    self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]],               
                    self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                    marker='*', c='r', markersize=12
                )
            else:
                self.dynamic_ax.plot(
                    self.x_axis[0: len(self.x_axis): len(self.x_axis) // (res - 1)] + [self.x_axis[-1]], 
                    self.y_axis[0: len(self.y_axis): len(self.y_axis) // (res - 1)] + [self.y_axis[-1]], 
                    self.z_axis[0: len(self.z_axis): len(self.z_axis) // (res - 1)] + [self.z_axis[-1]], 
                    marker='*', c='r', markersize=10, zorder=1
                )
                self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=1)
                if 1:
                    x = 100
                    y = 40
                    z = 0
                    dx = max_x - x
                    dy = max_y - y
                    dz = 100
                    xx = np.linspace(x, x + dx, 2)
                    yy = np.linspace(y, y + dy, 2)
                    zz = np.linspace(z, z + dz, 2)

                    xx1, yy1 = np.meshgrid(xx, yy)
                    self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z), cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(xx1, yy1, np.full_like(xx1, z + dz), cmap=cm.hot, alpha=0.5, zorder=1)

                    yy2, zz2 = np.meshgrid(yy, zz)
                    self.dynamic_ax.plot_surface(np.full_like(yy2, x), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2, cmap=cm.hot, alpha=0.5, zorder=1)

                    xx3, zz3 = np.meshgrid(xx, zz)
                    self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
                    self.dynamic_ax.plot_surface(xx3, np.full_like(yy2, y + dy), zz3, cmap=cm.hot, alpha=0.5, zorder=1)
            # rect1 = Rectangle((140.0, 30.0), 1000, 1000, color='orange')
            # rect2 = Rectangle((150.0, 40.0), 1000, 1000, color='grey')
            # self.dynamic_ax.add_patch(rect1)
            # self.dynamic_ax.add_patch(rect2)

        # 绘制所有线段
        if not self.enable3d():
            for i in range(len(self.tl)):
                start, end = self.tl[i].getData()
                line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl)))
            
            if self.enable_show_process:
                self.dynamic_ax.plot(
                    [self.tm.all_end_ef[i][X] for i in range(1, len(self.tm.all_end_ef))], 
                    [self.tm.all_end_ef[i][Y] for i in range(1, len(self.tm.all_end_ef))], c='purple'
                )

            self.dynamic_ax.plot([0], [0], marker='^', c=cm.OrRd(1.0), markersize=10)
            self.dynamic_ax.plot([self.tm.end_ef[X]], [self.tm.end_ef[Y]], marker='o', c=cm.OrRd(0.5))

        else:
            for i in range(len(self.tl_3d)):
                start, end = self.tl_3d[i].getData()
                order = 1
                if not (end[X] <= 100 and end[Y] >= 40 and end[Z] <= 100):
                    order = 200
                line, = self.dynamic_ax.plot([start[X], end[X]], [start[Y], end[Y]], [start[Z], end[Z]], linewidth=4, c=cm.OrRd(1.0 - 0.5 * i / len(self.tl_3d)), zorder=order)
            
            if self.enable_show_process:
                xs = [self.tm.all_end_ef_3d[i][X] for i in range(1, len(self.tm.all_end_ef_3d))]
                ys = [self.tm.all_end_ef_3d[i][Y] for i in range(1, len(self.tm.all_end_ef_3d))]
                zs = [self.tm.all_end_ef_3d[i][Z] for i in range(1, len(self.tm.all_end_ef_3d))]

                high_order_xs = []
                high_order_ys = []
                high_order_zs = []
                low_order_xs = []
                low_order_ys = []
                low_order_zs = []

                for i in range(0, len(self.tm.all_end_ef_3d) - 1):
                    if not (xs[i] <= 100 and ys[i] >= 40 and zs[i] <= 100):
                        high_order_xs.append(xs[i])
                        high_order_ys.append(ys[i])
                        high_order_zs.append(zs[i])
                    else:
                        low_order_xs.append(xs[i])
                        low_order_ys.append(ys[i])
                        low_order_zs.append(zs[i])
                self.dynamic_ax.scatter(
                    high_order_xs, 
                    high_order_ys,
                    high_order_zs, c='purple', zorder=100, linewidths=0.5
                )
                self.dynamic_ax.scatter(
                    low_order_xs, 
                    low_order_ys,
                    low_order_zs, c='purple', zorder=0, linewidths=0.5
                )

            self.dynamic_ax.plot([0], [0], [0], marker='^', c=cm.OrRd(1.0), markersize=10, zorder=100)
            self.dynamic_ax.plot([self.tm.end_ef_3d[X]], [self.tm.end_ef_3d[Y]], [self.tm.end_ef_3d[Z]], marker='o', c=cm.OrRd(0.5), zorder=1)

        self.dynamic_ax.set_xlim(min_x, max_x)
        self.dynamic_ax.set_ylim(min_y, max_y)
        self.dynamic_ax.set_xlabel("X")
        self.dynamic_ax.set_ylabel("Z")
        if self.enable3d():
            self.dynamic_ax.set_xlabel("X")
            self.dynamic_ax.set_ylabel("Y")
            self.dynamic_ax.set_zlabel("Z")
            self.dynamic_ax.set_zlim(min_z, max_z)

        if not self.enable3d():
            self.dynamic_ax.set_aspect('equal')

        self.dynamic_ax.figure.canvas.print_figure(file_path2 + '/' + str(time).zfill(8) + '.png')

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            self.folding_angle += 1
            if self.folding_angle >= 180:
                self.folding_angle = 180
        else:
            self.folding_angle -= 1
            if self.folding_angle <= 1:
                self.folding_angle = 1
        self.horizontal_folding_slider.setValue(self.folding_angle)
    
    def closeEvent(self, event):
        if self.timer_enable:
            self._timer.stop()
        self.deleteLater()
