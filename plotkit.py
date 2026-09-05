import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QWidget

import matplotlib.pyplot as plt
from matplotlib import ticker
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.sans-serif'] = 'Arial'

class PlotCSVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plotter")
        self.setGeometry(100, 100, 600, 400)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
                
        self.open_button = QPushButton("Open CSV File")
        self.open_button.setFixedSize(600, 200)  # Set button size to 200x50
        self.open_button.clicked.connect(self.open_file)
        
        self.open_button_xlsx = QPushButton("Open Xlsx File")
        self.open_button_xlsx.setFixedSize(600, 200)  # Set button size to 200x50
        self.open_button_xlsx.clicked.connect(self.open_file_xlsx)
        
        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.open_button_xlsx)
        
    def open_file_xlsx(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open XLSX File", "", "XLSX Files (*.xlsx);;All Files (*)")
        if file_path:
            self.plot_xlsx(file_path)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.plot_csv(file_path)
    
    def regulate_data(self, data_num, x, y, min=0, max=160):
        step = (max - min) * 1.0 / data_num
        new_x = np.zeros(data_num)
        new_y = np.zeros(data_num)
        for i in range(data_num):
            cur_val = min + i * step
            new_x[i] = cur_val
            for j in range(x.size - 1):
                if i == 0 or (j == 0 and cur_val <= x[0]) or (cur_val > x[j] and cur_val <= x[j + 1]) or x[j + 3] < x[j]:
                    break
            new_y[i] = y[j]
        return new_x, new_y
        
        
    def plot_xlsx(self, file_path):
        data = pd.read_excel(file_path, "Sheet1")
        plt.rcParams.update({'font.size': 20})
        plt.rcParams['font.sans-serif'] = 'Arial'
        formatter = ticker.ScalarFormatter(useMathText=True, useOffset=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        leg = [
            r'$\omega$ = 3 deg/s',
            r'$\omega$ = 6 deg/s',
            r'$\omega$ = 12 deg/s',
            r'$\omega$ = 24 deg/s',
            r'$\omega$ = 48 deg/s',
            r'$\omega$ = 72 deg/s',
        ]
        col = [
            (0.1800, 0.2800, 0.5500),  # 冷调深雾蓝（阶1）
            (0.7500, 0.4000, 0.3500),  # 暖调砖红（阶2）
            (0.2200, 0.5500, 0.3500),  # 冷调灰绿（阶3）
            (0.7000, 0.5000, 0.2000),  # 暖调橙棕（阶4）
            (0.3000, 0.4000, 0.6500),  # 冷调浅紫蓝（阶5）
            (0.6000, 0.3000, 0.5500),  # 暖调暗紫红（阶6）
        ]
        low_speed_average = 0
        high_speed_average = 0
        
        for i in range(0, 6):
            try:
                a1 = data['Angle (deg)' if i == 0 else f'Angle (deg).{3 * i}'].to_numpy()
                a2 = data[f'Angle (deg).{3 * i + 1}'].to_numpy()
                a3 = data[f'Angle (deg).{3 * i + 2}'].to_numpy()
                
                y1 = data['Torque (N·m)' if i == 0 else f'Torque (N·m).{3 * i}'].to_numpy() * 1000.0
                y2 = data[f'Torque (N·m).{3 * i + 1}'].to_numpy() * 1000.0
                y3 = data[f'Torque (N·m).{3 * i + 2}'].to_numpy() * 1000.0
                
                x, new_y1 = self.regulate_data(320, a1, y1)
                _, new_y2 = self.regulate_data(320, a2, y2)
                _, new_y3 = self.regulate_data(320, a3, y3)
                
                y_total = np.array([
                    new_y1, new_y2, new_y3
                ])

                mean = np.mean(y_total, axis=0)
                std_dev = np.std(y_total, axis=0)
                
                if i == 0:
                    low_speed_average = np.mean(mean)
                else:
                    high_speed_average = np.mean(mean)
                
                print(np.mean(mean))
                
                ax.plot(x, mean, label=leg[i], color=col[i], linewidth=1.5)
                ax.fill_between(x, mean - std_dev, mean + std_dev, color=col[i], alpha=0.2)

            except:
                break
          
        print(high_speed_average - low_speed_average)  
        ax.set_xlabel("Folding angle (deg)", fontsize=20, labelpad=7)
        ax.set_ylabel("Torque (N·mm)", fontsize=20, labelpad=7)
        ax.tick_params(labelsize=20)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(
            axis='both',        # 同时设置x和y轴
            which='major',      # 只针对主刻度（minor是次刻度，一般不用改）
            width=2,            # 刻度线宽度（和边框匹配）
            length=6,           # 刻度线长度（默认4，建议6-8）
            pad=9
        )
        # ax.set_title("Torque - Folding angle", fontsize=32, pad=10, fontweight='bold')
        # ax.legend(loc='lower right', fontsize=20)
        ax.legend(
            loc='upper center',          # 图例对齐基准：上中
            bbox_to_anchor=(0.5, 1.45),  # 关键：(x=0.5居中, y=1.15在绘图区上方)
            ncol=3,                      # 6列横向排列（适配6条曲线，不占竖向空间）
            frameon=False,               # 去掉图例边框（科研审美）
            fontsize=20,                 # 图例字体大小
            # 调整pad参数（核心）
            handletextpad=0.1,             # 图例线（手柄）到文字的间距（默认1.5，适配fontsize=20建议3-5）
            columnspacing=0.5,             # 图例列之间的间距（默认2，适配fontsize=20建议4-6）
            labelspacing=0.5               # 图例行之间的间距（默认0.5，适配fontsize=20建议2-3，6条曲线分2行时生效）  
        )
        ax.set_xlim(0, 180)
        ax.grid(True,
                color='gray',       # 网格颜色设为灰色
                alpha=0.3,          # 透明度（0-1，0.3既可见又不突兀）
                linewidth=0.75         # 网格线宽度（比边框细，突出曲线）)
        )
        ax.set_ylim(0, int(max(mean) + 2))
        x_step = 30  # X轴步长
        y_step = 1  # Y轴步长
        ax.set_xticks(np.arange(0, 210, x_step))  # 生成X轴刻度
        ax.set_yticks(np.arange(0, int(max(mean) + 3), y_step))       # 生成Y轴刻度
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        plt.tight_layout()

        plt.show()
        
    def plot_csv(self, file_path):
        try:
            data = pd.read_csv(file_path)
            plt.rcParams.update({'font.size': 24})
            plt.rcParams['font.sans-serif'] = 'Arial'
            formatter = ticker.ScalarFormatter(useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 5))
            fig, ax = plt.subplots(figsize=(11, 9))

            x = data['step'].to_numpy()
            y1 = data["maximum_reward"].to_numpy()
            y2 = data["tree_policy_maximum_reward"].to_numpy()
            
            ax.plot(x, y1, color=(1.0, 0.4, 0.4), linewidth=2, label='Maximum reward')
            ax.plot(x, y2, color=(0.4, 0.4, 1.0), linewidth=2, label='Maximum reward (tree policy)')

            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Step", fontsize=24, labelpad=10)
            ax.set_ylabel("Simulation reward", fontsize=24, labelpad=10)
            ax.tick_params(labelsize=24)

            # if x[-1] > 1e5:
            #     plt.xticks(rotation=30)

            ax.set_title("Simulation reward - Step", fontsize=32, pad=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=24)
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, max(y1) * 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.tick_params(pad=7)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(11, 9))
            cut_nodes = data["cut"].to_numpy()
            y1 = cut_nodes
            
            ax.plot(x, y1, color=(0.2, 0.2, 0.8), linewidth=2)

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlabel("Step", fontsize=24, labelpad=10)
            ax.set_ylabel("The number of remaining MCTS nodes", fontsize=24, labelpad=10)
            ax.tick_params(labelsize=24)

            # if x[-1] > 1e5:
            #     plt.xticks(rotation=30)

            ax.set_title("The number of remaining MCTS nodes - Step", fontsize=32, pad=10, fontweight='bold')
            ax.set_xlim(0, x[-1] + 25)
            ax.set_ylim(0, max(y1) * 1.01)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.tick_params(pad=7)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(11, 9))
            valid_num = data["valid"].to_numpy()
            rollout_valid_num = data["rollout_valid"].to_numpy()
            sum_valid_num = valid_num + rollout_valid_num
            
            simulated_number = data["simulated_number"].to_numpy()
            
            ax.plot(x, valid_num, color=(0.125, 0.625, 0.75), linewidth=1, label='Valid - Tree')
            ax.plot(x, rollout_valid_num, color=(1.0, 0.74, 0.02), linewidth=1, label='Valid - Rollout')
            ax.plot(x, sum_valid_num, color=(0.01, 0.1875, 0.29), linewidth=2, label='Valid cases')
            
            ax.plot(x, simulated_number, color=(0.5, 0.1, 0.5), linewidth=2, label='All simulated cases')

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel("Step", fontsize=24, labelpad=10)
            ax.set_ylabel("The number of simulated cases", fontsize=24, labelpad=10)
            ax.tick_params(labelsize=24)

            # if x[-1] > 1e5:
            #     plt.xticks(rotation=30)

            ax.set_title("The number of simulated cases - Step", fontsize=32, pad=10, fontweight='bold')
            ax.legend(loc='upper left', fontsize=24)
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, max(simulated_number) * 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.tick_params(pad=7)
            plt.show()
            
            fig, ax = plt.subplots(figsize=(11, 9))

            x = data['step'].to_numpy()
            y = data["time"].to_numpy() / 3600.0
            
            ax.plot(x, y, color=(0.0, 0.0, 0.0), linewidth=2, label='time')

            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Step", fontsize=24, labelpad=10)
            ax.set_ylabel("Training time (h)", fontsize=24, labelpad=10)
            ax.tick_params(labelsize=24)

            # if x[-1] > 1e5:
            #     plt.xticks(rotation=30)

            ax.set_title("Training time - Step", fontsize=32, pad=10, fontweight='bold')
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, max(y) * 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.tick_params(pad=7)
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotCSVApp()
    window.show()
    sys.exit(app.exec_())