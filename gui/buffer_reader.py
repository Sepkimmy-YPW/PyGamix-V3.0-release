import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import re

class FolderParserApp:
    def __init__(self, path):
        self.root = tk.Tk()
        self.root.title("Physical parameters setting tool")
        self.root.geometry("900x450")
        
        self.path = path
        self.root.deiconify()
        
        # create main UI
        self.create_main_ui()
        
    def create_main_ui(self):
        """create the main user interface"""
        self.clear_window()
        
        tk.Label(self.root, text="Physical parameters setting tool", font=("Arial", 16)).pack(pady=20)
        self.select_and_parse_folder()
        
        self.root.mainloop()
    
    def clear_window(self):
        """clear all widgets in the window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def select_and_parse_folder(self):
        """select a folder and parse its contents"""
        
        # scan the folder
        file_count = self.scan_files(self.path)
        if file_count is None:
            return
        
        # 保存结果
        self.folder_path = self.path
        self.file_count = file_count
        
        # 显示参数输入界面
        self.show_parameter_input()
    
    def scan_files(self, folder_path):
        """Scan files in the given folder and count valid files"""
        file_pattern = r'^result_train_time_1_candidate_buf_(\d+)\.json$'
        file_count = 0
        
        for filename in os.listdir(folder_path):
            match = re.match(file_pattern, filename)
            if match:
                # check if the number is a multiple of 512
                number = int(match.group(1))
                if number % 512 == 0:
                    file_count += 1
        
        print(f"find {file_count} valid files in {folder_path}")
        return file_count
    
    def show_parameter_input(self):
        """show the parameter input interface"""
        self.clear_window()
        self.root.title("Set Physical Parameters")
        
        # create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # add title label
        ttk.Label(
            main_frame, 
            text=f"Set Physical Parameters",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # add file count label
        ttk.Label(
            main_frame, 
            text=f"Detected {self.file_count} valid files in {self.folder_path}",
            foreground="blue"
        ).grid(row=1, column=0, columnspan=3, pady=(0, 15))
        
        # create input fields
        row = 2
        
        # Origami height
        ttk.Label(main_frame, text="Origami height:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.height_var = tk.IntVar(value=33)
        height_spin = ttk.Spinbox(
            main_frame, 
            from_=1, 
            to=500, 
            textvariable=self.height_var,
            width=10
        )
        height_spin.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Gravity mode
        ttk.Label(main_frame, text="Gravity mode:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.gravity_var = tk.IntVar(value=0)
        gravity_spin = ttk.Spinbox(
            main_frame, 
            from_=0, 
            to=6, 
            textvariable=self.gravity_var,
            width=10
        )
        gravity_spin.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Ground friction coeff
        ttk.Label(main_frame, text="Ground friction coeff:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.miu_var = tk.DoubleVar(value=0.5)
        miu_entry = ttk.Entry(main_frame, textvariable=self.miu_var, width=13)
        miu_entry.grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        
        # Control mode
        ttk.Label(main_frame, text="Control mode:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.control_mode_var = tk.IntVar(value=0)
        ttk.Radiobutton(
            main_frame, 
            text="single", 
            variable=self.control_mode_var, 
            value=0
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            main_frame, 
            text="cycle", 
            variable=self.control_mode_var, 
            value=1
        ).grid(row=row, column=2, sticky=tk.W, pady=5)
        row += 1
        
        # Friction enable
        ttk.Label(main_frame, text="Friction enable:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.friction_mode_var = tk.IntVar(value=0)
        friction_check = ttk.Checkbutton(
            main_frame, 
            text="Friction enabled", 
            variable=self.friction_mode_var,
            onvalue=1, 
            offvalue=0
        )
        friction_check.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Buffer range
        ttk.Label(main_frame, text="Buffer range:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        
        # buffer lower limit
        self.buf_low_var = tk.IntVar(value=0)
        buf_low_spin = ttk.Spinbox(
            main_frame, 
            from_=0, 
            to=self.file_count-1, 
            textvariable=self.buf_low_var,
            width=5
        )
        buf_low_spin.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(main_frame, text="to").grid(row=row, column=1, padx=50)
        
        # buffer upper limit
        self.buf_high_var = tk.IntVar(value=self.file_count)
        buf_high_spin = ttk.Spinbox(
            main_frame, 
            from_=1, 
            to=self.file_count, 
            textvariable=self.buf_high_var,
            width=5
        )
        buf_high_spin.grid(row=row, column=2, sticky=tk.W, pady=5)
        row += 1
        
        # Thread number
        ttk.Label(main_frame, text="Thread number:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.thread_var = tk.IntVar(value=4)
        thread_spin = ttk.Spinbox(
            main_frame, 
            from_=0, 
            to=8, 
            textvariable=self.thread_var,
            width=10
        )
        thread_spin.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # simulation case
        ttk.Label(main_frame, text="Simulation batch number:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.batch_var = tk.IntVar(value=16)
        batch_spin = ttk.Spinbox(
            main_frame, 
            from_=1, 
            to=32, 
            textvariable=self.batch_var,
            width=10
        )
        batch_spin.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # simulation mode
        ttk.Label(main_frame, text="Simulation mode:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.simulation_mode_var = tk.IntVar(value=0)
        ttk.Radiobutton(
            main_frame, 
            text="fast", 
            variable=self.simulation_mode_var, 
            value=-1
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            main_frame, 
            text="normal", 
            variable=self.simulation_mode_var, 
            value=0
        ).grid(row=row, column=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            main_frame, 
            text="noise", 
            variable=self.simulation_mode_var, 
            value=2
        ).grid(row=row, column=3, sticky=tk.W, pady=5)
        row += 1
        
        # submit buttons
        submit_frame = ttk.Frame(main_frame)
        submit_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(
            submit_frame, 
            text="ok", 
            command=self.submit_parameters
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            submit_frame, 
            text="back", 
            command=self.create_main_ui
        ).pack(side=tk.LEFT, padx=10)
        
        self.root.update()
     
    def submit_parameters(self):
        """submit the parameters and display the result"""
        # 获取所有输入值
        self.height = self.height_var.get()
        self.gravity_flag = self.gravity_var.get()
        self.miu = self.miu_var.get()
        self.control_mode = self.control_mode_var.get()
        self.friction_mode = self.friction_mode_var.get()
        self.buf_low = self.buf_low_var.get()
        self.buf_high = self.buf_high_var.get()
        self.thread_number = self.thread_var.get()
        self.batch_number = self.batch_var.get()
        self.extract_mode = self.simulation_mode_var.get()
        
        # check if all inputs are valid
        if self.buf_low >= self.buf_high:
            messagebox.showerror("input error", "the buffer range is invalid")
            self.create_main_ui()
        
        # exit
        self.root.destroy()
        self.root.quit()

if __name__ == "__main__":
    app = FolderParserApp('./threadingResult/train-SBS-mountain-big-fix-new-1sim-2string-18episodes-cutnodes-uplimit_6-A1-LB0.80-GB0.56')