import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import queue
from datetime import datetime
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.modules['torchaudio'] = None

try:
    from detect import DefectDetector
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detect import DefectDetector

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_timer = None
        
    def write(self, string):
        self.queue.put(string)
        if self.update_timer is None:
            self.update_timer = self.text_widget.after(100, self.update_text_widget)
    
    def update_text_widget(self):
        try:
            while True:
                text = self.queue.get_nowait()
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')
                self.queue.task_done()
        except queue.Empty:
            pass
        
        self.update_timer = None
        
    def flush(self):
        pass

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("以心医疗瓣叶缺陷检测")
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass
        
        self.detector = None
        self.current_image = None
        self.current_image_path = None
        self.current_video_path = None
        self.video_output_path = None
        self.detection_results = None
        self.processing_thread = None
        self.stop_video_processing = False
        self.pause_video_processing = False
        self.video_speed = 1.0
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_logs.txt")
        
        self.setup_styles()
        
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_control_panel()
        self.create_display_panel()
        self.create_log_panel()
        self.create_video_controls()
        
        self.redirect_stdout()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.load_config()
        print("应用程序初始化完成，正在尝试加载上次使用的模型...")
        self.try_load_last_model()
    
    def set_app_background(self):
        background_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "back.jpg")
        if os.path.exists(background_path):
            try:
                # 加载背景图
                bg_image = Image.open(background_path)
                self.bg_image = bg_image  # 保存引用防止被垃圾回收
                
                # 创建背景标签
                self.bg_label = tk.Label(self.root)
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                
                # 延迟加载背景，等待窗口初始化完成
                self.root.update_idletasks()
                self.root.after(100, self.update_background)
                
                print(f"程序背景图已加载: {background_path}")
                print(f"图像尺寸: {bg_image.width}x{bg_image.height}")
            except Exception as e:
                print(f"加载背景图失败: {e}")
        else:
            print(f"背景图不存在: {background_path}")
    
    def update_background(self):
        # 窗口完全初始化后更新背景图大小
        try:
            # 获取当前窗口尺寸
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            # 调整图像大小
            bg_image_resized = self.bg_image.resize((window_width, window_height), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image_resized)
            
            # 更新标签图像
            self.bg_label.configure(image=self.bg_photo)
            
            # 确保背景在最底层
            self.bg_label.lower()
            
            print(f"窗口尺寸: {window_width}x{window_height}")
            print("背景图已更新")
        except Exception as e:
            print(f"更新背景图失败: {e}")
    
    def setup_styles(self):
        self.style = ttk.Style()
        
        self.style.configure("Primary.TButton", font=("微软雅黑", 10, "bold"))
        self.style.configure("Secondary.TButton", font=("微软雅黑", 10))
        
        self.style.configure("Title.TLabel", font=("微软雅黑", 12, "bold"))
        self.style.configure("Subtitle.TLabel", font=("微软雅黑", 10))
        
        self.style.configure("Panel.TFrame", relief=tk.RIDGE, borderwidth=2)
    
    def create_control_panel(self):
        control_frame = ttk.Frame(self.main_frame, style="Panel.TFrame", padding=5)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        model_frame = ttk.LabelFrame(control_frame, text="模型设置", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
        
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_path_frame, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="浏览", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(model_frame, text="推理设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.device_var = tk.StringVar(value="auto")
        device_choices = ["auto", "cpu"]
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_choices.append(f"cuda:{i}")
        
        ttk.Combobox(model_frame, textvariable=self.device_var, values=device_choices).grid(row=1, column=1, sticky=tk.EW, pady=2)
        
        ttk.Label(model_frame, text="置信度阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.conf_thres_var = tk.DoubleVar(value=0.25)
        ttk.Scale(model_frame, from_=0.01, to=1.0, variable=self.conf_thres_var, 
                orient=tk.HORIZONTAL, command=self.update_conf_label).grid(row=2, column=1, sticky=tk.EW, pady=2)
        
        self.conf_label = ttk.Label(model_frame, text="0.25")
        self.conf_label.grid(row=2, column=2, padx=(5, 0))
        
        ttk.Label(model_frame, text="批处理大小:").grid(row=3, column=0, sticky=tk.W, pady=2)
        
        self.batch_size_var = tk.IntVar(value=1)
        ttk.Scale(model_frame, from_=1, to=32, variable=self.batch_size_var, 
                orient=tk.HORIZONTAL, command=self.update_batch_label).grid(row=3, column=1, sticky=tk.EW, pady=2)
        
        self.batch_label = ttk.Label(model_frame, text="1")
        self.batch_label.grid(row=3, column=2, padx=(5, 0))
        
        ttk.Button(model_frame, text="加载模型", style="Primary.TButton", 
                 command=self.load_model).grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        operation_frame = ttk.LabelFrame(control_frame, text="操作", padding=5)
        operation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(operation_frame, text="从图像中检测", 
                 command=self.detect_from_image).pack(fill=tk.X, pady=2)
        
        ttk.Button(operation_frame, text="从视频中检测", 
                 command=self.detect_from_video).pack(fill=tk.X, pady=2)
        
        ttk.Button(operation_frame, text="从摄像头检测", 
                 command=self.detect_from_camera).pack(fill=tk.X, pady=2)
        
        ttk.Button(operation_frame, text="批量处理图像", 
                 command=self.batch_process_images).pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(operation_frame, text="停止处理", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.save_result_button = ttk.Button(operation_frame, text="保存当前结果", 
                                          command=self.save_results, state=tk.DISABLED)
        self.save_result_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(operation_frame, text="导出日志", 
                 command=self.export_log).pack(fill=tk.X, pady=2)
        
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = ScrolledText(result_frame, width=30, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        
        ttk.Button(result_frame, text="导出结果详情", command=self.export_result_details).pack(fill=tk.X, pady=2)

    def create_display_panel(self):
        display_frame = ttk.Frame(self.main_frame, style="Panel.TFrame", padding=5)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        status_frame = ttk.Frame(display_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.file_info_label = ttk.Label(status_frame, text="未加载任何文件")
        self.file_info_label.pack(side=tk.LEFT)
        
        self.time_info_label = ttk.Label(status_frame, text="")
        self.time_info_label.pack(side=tk.RIGHT)
    
    def create_log_panel(self):
        log_frame = ttk.LabelFrame(self.main_frame, text="日志", padding=5)
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(log_control_frame, text="搜索:").pack(side=tk.LEFT)
        self.log_search_var = tk.StringVar()
        ttk.Entry(log_control_frame, textvariable=self.log_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(log_control_frame, text="查找", command=self.search_log).pack(side=tk.LEFT)
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT, padx=5)
        
        self.log_text = ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
    
    def create_video_controls(self):
        self.video_controls_frame = ttk.LabelFrame(self.main_frame, text="视频控制", padding=5)
        
        controls_inner_frame = ttk.Frame(self.video_controls_frame)
        controls_inner_frame.pack(fill=tk.X)
        
        self.pause_button = ttk.Button(controls_inner_frame, text="暂停", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_inner_frame, text="前一帧", command=self.previous_frame, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_inner_frame, text="后一帧", command=self.next_frame, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls_inner_frame, text="速度:").pack(side=tk.LEFT, padx=(15, 5))
        
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(controls_inner_frame, from_=0.1, to=2.0, variable=self.speed_var, 
                orient=tk.HORIZONTAL, command=self.update_speed_label).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.speed_label = ttk.Label(controls_inner_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        self.current_frame_label = ttk.Label(controls_inner_frame, text="0/0")
        self.current_frame_label.pack(side=tk.RIGHT, padx=5)
        
        self.frame_progress = ttk.Scale(self.video_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_progress.pack(fill=tk.X, pady=5)
        self.frame_progress.state(['disabled'])
    
    def toggle_video_controls(self, show=False):
        if show:
            self.video_controls_frame.pack(fill=tk.X, pady=(10, 0), before=self.log_text.master)
            self.pause_button.config(state=tk.NORMAL)
            self.frame_progress.state(['!disabled'])
        else:
            self.video_controls_frame.pack_forget()
            self.pause_button.config(state=tk.DISABLED)
            self.frame_progress.state(['disabled'])
    
    def toggle_pause(self):
        self.pause_video_processing = not self.pause_video_processing
        self.pause_button.config(text="继续" if self.pause_video_processing else "暂停")
    
    def previous_frame(self):
        pass
    
    def next_frame(self):
        pass
    
    def seek_frame(self, event):
        pass
    
    def update_speed_label(self, event=None):
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f}x")
        self.video_speed = speed
    
    def redirect_stdout(self):
        self.stdout_redirector = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirector
    
    def update_conf_label(self, event=None):
        self.conf_label.config(text=f"{self.conf_thres_var.get():.2f}")
    
    def update_batch_label(self, event=None):
        self.batch_label.config(text=f"{int(self.batch_size_var.get())}")
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Model Files", "*.pt *.pth *.onnx"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                    if 'model_path' in config and os.path.exists(config['model_path']):
                        self.model_path_var.set(config['model_path'])
                        
                    if 'device' in config:
                        self.device_var.set(config['device'])
                        
                    if 'conf_thres' in config:
                        self.conf_thres_var.set(config['conf_thres'])
                        self.update_conf_label()
                        
                    if 'batch_size' in config:
                        self.batch_size_var.set(config['batch_size'])
                        self.update_batch_label()
                        
                    print(f"已加载配置: {self.config_file}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def save_config(self):
        config = {
            'model_path': self.model_path_var.get(),
            'device': self.device_var.get(),
            'conf_thres': self.conf_thres_var.get(),
            'batch_size': int(self.batch_size_var.get()),
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
            print(f"配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def try_load_last_model(self):
        model_path = self.model_path_var.get().strip()
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def log_detection_result(self, source_type, source_path, results):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"[{timestamp}] 来源: {source_type}, 文件: {os.path.basename(source_path)}\n"
        log_entry += f"检测到 {results['num_detections']} 个缺陷, 推理时间: {results['inference_time']:.3f}秒\n"
        
        class_counts = {}
        for det in results['detections']:
            cls_name = det['class_name']
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
        
        log_entry += "类别统计:\n"
        for cls_name, count in class_counts.items():
            log_entry += f"  - {cls_name}: {count}个\n"
            
        log_entry += "-" * 50 + "\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def load_model(self):
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            possible_paths = [
                './runs/detect/defect_detection_v8n/weights/best.pt',
                './weights/best.pt',
                './model/best.pt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    self.model_path_var.set(path)
                    break
            
            if not model_path:
                messagebox.showerror("错误", "请指定有效的模型文件路径")
                return
        
        device = self.device_var.get()
        if device == "auto":
            device = None
            
        conf_thres = self.conf_thres_var.get()
        batch_size = int(self.batch_size_var.get())
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("加载模型")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f'+{x}+{y}')
        
        ttk.Label(progress_window, text=f"正在加载模型: {os.path.basename(model_path)}").pack(pady=(10, 5))
        progress = ttk.Progressbar(progress_window, mode="indeterminate", length=200)
        progress.pack(pady=5, padx=10)
        status_label = ttk.Label(progress_window, text="初始化中...")
        status_label.pack(pady=5)
        
        progress.start()
        
        def load_model_thread():
            try:
                progress_window.after(0, lambda: status_label.config(text="正在加载模型..."))
                self.detector = DefectDetector(
                    model_path=model_path,
                    conf_thres=conf_thres,
                    device=device,
                    batch_size=batch_size
                )
                
                print(f"模型加载成功: {model_path}")
                self.save_config()
                
                self.root.after(0, lambda: [
                    progress_window.destroy(),
                    messagebox.showinfo("成功", "模型加载成功!")
                ])
                
            except Exception as e:
                print(f"模型加载失败: {e}")
                
                self.root.after(0, lambda: [
                    progress_window.destroy(),
                    messagebox.showerror("错误", f"模型加载失败: {e}")
                ])
                self.detector = None
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def detect_from_image(self):
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.current_image_path = file_path
        self.current_video_path = None
        
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            print(f"正在检测图像: {file_path}")
            
            self.detector.conf_thres = self.conf_thres_var.get()
            self.detector.batch_size = int(self.batch_size_var.get())
            
            self.detection_results = self.detector.detect(file_path)
            
            self.current_image = cv2.imread(file_path)
            
            result_image = self.detector.draw_results(self.current_image.copy(), self.detection_results)
            
            self.show_image(result_image)
            
            file_name = os.path.basename(file_path)
            img_h, img_w = self.current_image.shape[:2]
            self.file_info_label.config(text=f"文件: {file_name} | 尺寸: {img_w}x{img_h}")
            
            self.time_info_label.config(text=f"检测时间: {self.detection_results['inference_time']:.3f}秒")
            
            self.update_result_text()
            self.log_detection_result("图像", file_path, self.detection_results)
            
            self.save_result_button.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"图像检测失败: {e}")
            messagebox.showerror("错误", f"图像检测失败: {e}")
        finally:
            self.root.config(cursor="")
    
    def batch_process_images(self):
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if not folder_path:
            return
        
        output_folder = filedialog.askdirectory(title="选择输出文件夹")
        if not output_folder:
            return
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        if not image_files:
            messagebox.showinfo("提示", "选定文件夹中没有找到图像文件")
            return
        
        self.detector.conf_thres = self.conf_thres_var.get()
        self.detector.batch_size = int(self.batch_size_var.get())
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("批量处理图像")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f'+{x}+{y}')
        
        ttk.Label(progress_window, text=f"正在处理 {len(image_files)} 个图像文件...").pack(pady=(10, 5))
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=300)
        progress_bar.pack(pady=10, padx=10)
        
        status_label = ttk.Label(progress_window, text="0/0")
        status_label.pack(pady=5)
        
        cancel_button = ttk.Button(progress_window, text="取消", command=lambda: setattr(self, 'stop_batch_processing', True))
        cancel_button.pack(pady=5)
        
        self.stop_batch_processing = False
        
        def process_batch_thread():
            try:
                total_files = len(image_files)
                processed = 0
                successful = 0
                
                summary_file = os.path.join(output_folder, f"批量处理结果_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                with open(summary_file, 'w', encoding='utf-8') as sf:
                    sf.write("文件名,检测到的缺陷数量,推理时间(秒),结果路径\n")
                
                for i, img_path in enumerate(image_files):
                    if self.stop_batch_processing:
                        break
                    
                    try:
                        file_name = os.path.basename(img_path)
                        progress_window.after(0, lambda: status_label.config(text=f"正在处理: {file_name} ({i+1}/{total_files})"))
                        
                        results = self.detector.detect(img_path)
                        
                        img = cv2.imread(img_path)
                        result_img = self.detector.draw_results(img, results)
                        
                        output_img_path = os.path.join(output_folder, f"processed_{file_name}")
                        cv2.imwrite(output_img_path, result_img)
                        
                        json_path = os.path.splitext(output_img_path)[0] + '.json'
                        self.detector.save_results_to_json(results, json_path)
                        
                        self.log_detection_result("批处理", img_path, results)
                        
                        with open(summary_file, 'a', encoding='utf-8') as sf:
                            sf.write(f"{file_name},{results['num_detections']},{results['inference_time']:.3f},{output_img_path}\n")
                        
                        successful += 1
                        
                    except Exception as e:
                        print(f"处理图像 {img_path} 失败: {e}")
                        with open(summary_file, 'a', encoding='utf-8') as sf:
                            sf.write(f"{os.path.basename(img_path)},处理失败,0,错误: {str(e)}\n")
                    
                    processed += 1
                    progress = (processed / total_files) * 100
                    progress_window.after(0, lambda p=progress: progress_var.set(p))
                
                self.root.after(0, lambda: [
                    progress_window.destroy(),
                    messagebox.showinfo("批处理完成", f"成功处理 {successful}/{total_files} 个图像文件\n结果保存在: {output_folder}\n摘要文件: {summary_file}")
                ])
                
            except Exception as e:
                print(f"批处理失败: {e}")
                
                self.root.after(0, lambda: [
                    progress_window.destroy(),
                    messagebox.showerror("错误", f"批处理失败: {e}")
                ])
        
        threading.Thread(target=process_batch_thread, daemon=True).start()
    
    def detect_from_video(self):
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        save_video = messagebox.askyesno("保存视频", "是否要保存处理后的视频?")
        
        if save_video:
            output_path = filedialog.asksaveasfilename(
                title="选择保存路径",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
                initialfile=f"processed_{os.path.basename(file_path)}"
            )
            
            if not output_path:
                return
                
            self.video_output_path = output_path
        else:
            self.video_output_path = None
        
        self.current_video_path = file_path
        self.current_image_path = None
        
        self.stop_video_processing = False
        self.pause_video_processing = False
        
        self.toggle_video_controls(True)
        self.stop_button.config(state=tk.NORMAL)
        
        self.disable_operation_buttons()
        
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_video(self):
        progress_frame = None
        percent_label = None
        
        try:
            self.detector.conf_thres = self.conf_thres_var.get()
            self.detector.batch_size = int(self.batch_size_var.get())
            
            print(f"正在处理视频: {self.current_video_path}")
            
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {self.current_video_path}")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
            
            file_name = os.path.basename(self.current_video_path)
            
            def update_file_info():
                self.file_info_label.config(text=f"文件: {file_name} | 尺寸: {width}x{height} | FPS: {fps:.1f}")
            
            self.root.after(0, update_file_info)
            
            if self.video_output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.video_output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processing_times = []
            start_time = time.time()
            
            progress_var = tk.DoubleVar()
            
            def create_progress_bar():
                nonlocal progress_frame, percent_label
                progress_frame = ttk.Frame(self.main_frame)
                progress_frame.pack(fill=tk.X, pady=5)
                
                progress_label = ttk.Label(progress_frame, text="处理进度:")
                progress_label.pack(side=tk.LEFT, padx=5)
                
                progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100, length=200)
                progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                
                percent_label = ttk.Label(progress_frame, text="0%")
                percent_label.pack(side=tk.LEFT, padx=5)
            
            self.root.after(0, create_progress_bar)
            
            def update_frame_slider():
                self.frame_progress.config(from_=0, to=total_frames-1)
                self.current_frame_label.config(text=f"0/{total_frames}")
            
            self.root.after(0, update_frame_slider)
            
            UI_UPDATE_INTERVAL = max(1, int(fps / (5 * self.video_speed)))
            
            # 存储所有处理过的帧以支持帧控制功能
            self.processed_frames = []
            last_detection_log = time.time()
            
            while cap.isOpened() and not self.stop_video_processing:
                while self.pause_video_processing and not self.stop_video_processing:
                    time.sleep(0.1)
                    continue
                
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = self.detector.detect(frame)
                    
                    output_frame = self.detector.draw_results(frame, results)
                    
                    self.processed_frames.append(output_frame.copy())
                    
                    processing_times.append(results['inference_time'])
                    
                    frame_count += 1
                    progress = (frame_count / total_frames) * 100
                    
                    if frame_count % UI_UPDATE_INTERVAL == 0 or frame_count == 1 or frame_count == total_frames:
                        def update_progress():
                            progress_var.set(progress)
                            percent_label.config(text=f"{progress:.1f}%")
                            
                            self.current_frame_label.config(text=f"{frame_count}/{total_frames}")
                            self.frame_progress.set(frame_count)
                            
                            elapsed_time = time.time() - start_time
                            if frame_count > 10:
                                avg_time_per_frame = elapsed_time / frame_count
                                remaining_frames = total_frames - frame_count
                                remaining_time = avg_time_per_frame * remaining_frames
                                self.time_info_label.config(
                                    text=f"已处理: {frame_count}/{total_frames} | " +
                                        f"剩余时间: {remaining_time:.1f}秒"
                                )
                        
                        self.root.after(0, update_progress)
                    
                    # 减慢UI更新频率，使显示更清晰
                    adj_interval = max(1, int(UI_UPDATE_INTERVAL / self.video_speed))
                    if frame_count % adj_interval == 0 or frame_count == 1 or frame_count == total_frames:
                        def update_ui():
                            nonlocal last_detection_log
                            self.show_image(output_frame)
                            
                            current_time = time.time()
                            if current_time - last_detection_log > 1.0 or frame_count == 1 or frame_count == total_frames:
                                self.detection_results = results
                                self.update_result_text()
                                last_detection_log = current_time
                        
                        self.root.after(0, update_ui)
                    
                    if self.video_output_path:
                        out.write(output_frame)
                    
                    # 根据播放速度控制帧处理速度
                    if self.video_speed < 1.0:
                        target_time = 1.0 / (fps * self.video_speed)
                        elapsed = time.time() - start_time
                        frame_time = elapsed / frame_count if frame_count > 0 else 0
                        if frame_time < target_time:
                            time.sleep(target_time - frame_time)
                    
                    if frame_count % (UI_UPDATE_INTERVAL * 2) == 0:
                        self.root.update_idletasks()
                    
                    del frame
                    
                except Exception as e:
                    print(f"\n处理第 {frame_count} 帧时出错: {e}")
                    continue
                
            cap.release()
            if self.video_output_path and 'out' in locals():
                out.release()
                
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            final_msg = f"\n视频处理{'完成' if not self.stop_video_processing else '已停止'}! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s, 处理速度: {1/avg_time:.1f} FPS"
            print(final_msg)
            
            self.log_detection_result("视频", self.current_video_path, {"num_detections": "N/A", "inference_time": avg_time, "detections": []})
            
            def remove_progress():
                if progress_frame:
                    progress_frame.destroy()
            
            self.root.after(0, remove_progress)
            
            if self.stop_video_processing:
                messagebox.showinfo("处理停止", "视频处理已停止")
            else:
                messagebox.showinfo("处理完成", f"视频处理完成!\n处理了 {frame_count} 帧\n平均FPS: {1/avg_time:.1f}")
            
        except Exception as e:
            print(f"视频处理失败: {e}")
            messagebox.showerror("错误", f"视频处理失败: {e}")
        finally:
            def restore_ui():
                self.toggle_video_controls(False)
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
                
                if progress_frame:
                    progress_frame.destroy()
            
            self.root.after(0, restore_ui)
    
    def detect_from_camera(self):
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        self.stop_video_processing = False
        self.pause_video_processing = False
        
        self.toggle_video_controls(True)
        self.stop_button.config(state=tk.NORMAL)
        
        self.disable_operation_buttons()
        
        save_video = messagebox.askyesno("保存视频", "是否要保存摄像头视频?")
        
        output_path = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = filedialog.asksaveasfilename(
                title="选择保存路径",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
                initialfile=f"camera_{timestamp}.mp4"
            )
            
            if not output_path:
                self.toggle_video_controls(False)
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
                return
        
        self.video_output_path = output_path
        self.processing_thread = threading.Thread(target=self.process_camera)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_camera(self):
        progress_frame = None
        percent_label = None
        
        try:
            self.detector.conf_thres = self.conf_thres_var.get()
            self.detector.batch_size = int(self.batch_size_var.get())
            
            print("正在打开摄像头...")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("无法打开摄像头")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"摄像头信息: {width}x{height}, {fps}fps")
            
            def update_file_info():
                self.file_info_label.config(text=f"摄像头直播 | 尺寸: {width}x{height}")
            
            self.root.after(0, update_file_info)
            
            out = None
            if self.video_output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.video_output_path, fourcc, 20, (width, height))
            
            frame_count = 0
            processing_times = []
            start_time = time.time()
            
            progress_var = tk.StringVar(value="摄像头实时处理中")
            
            def create_status_bar():
                nonlocal progress_frame, percent_label
                progress_frame = ttk.Frame(self.main_frame)
                progress_frame.pack(fill=tk.X, pady=5)
                
                progress_label = ttk.Label(progress_frame, textvariable=progress_var)
                progress_label.pack(side=tk.LEFT, padx=5)
                
                percent_label = ttk.Label(progress_frame, text="计算中...")
                percent_label.pack(side=tk.RIGHT, padx=5)
            
            self.root.after(0, create_status_bar)
            
            UI_UPDATE_INTERVAL = max(1, int(5 / self.video_speed))
            self.processed_frames = []
            
            while cap.isOpened() and not self.stop_video_processing:
                while self.pause_video_processing and not self.stop_video_processing:
                    time.sleep(0.1)
                    continue
                
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = self.detector.detect(frame)
                    
                    output_frame = self.detector.draw_results(frame, results)
                    
                    self.processed_frames.append(output_frame.copy())
                    
                    processing_times.append(results['inference_time'])
                    
                    frame_count += 1
                    
                    if frame_count % UI_UPDATE_INTERVAL == 0 or frame_count == 1:
                        recent_times = processing_times[-20:]
                        avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
                        fps_processed = 1 / avg_time if avg_time > 0 else 0
                        
                        elapsed_time = time.time() - start_time
                        
                        def update_ui():
                            self.show_image(output_frame)
                            
                            percent_label.config(text=f"处理速度: {fps_processed:.1f} FPS")
                            
                            progress_var.set(f"已处理 {frame_count} 帧 | 运行时间: {elapsed_time:.1f}秒")
                            
                            if frame_count % (UI_UPDATE_INTERVAL * 3) == 0 or frame_count == 1:
                                self.detection_results = results
                                self.update_result_text()
                        
                        self.root.after(0, update_ui)
                    
                    if out:
                        out.write(output_frame)
                    
                    # 根据播放速度控制帧处理速度
                    if self.video_speed < 1.0:
                        time.sleep(0.03 / self.video_speed)
                    
                    if frame_count % (UI_UPDATE_INTERVAL * 2) == 0:
                        self.root.update_idletasks()
                    
                    del frame
                    
                except Exception as e:
                    print(f"处理摄像头帧时出错: {e}")
                    continue
            
            cap.release()
            if out:
                out.release()
                
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            final_msg = f"\n摄像头处理{'完成' if not self.stop_video_processing else '已停止'}! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s, 处理速度: {1/avg_time:.1f} FPS"
            print(final_msg)
            
            def remove_progress():
                if progress_frame:
                    progress_frame.destroy()
            
            self.root.after(0, remove_progress)
            
            if self.video_output_path and not self.stop_video_processing:
                messagebox.showinfo("保存完成", f"摄像头视频已保存到:\n{self.video_output_path}")
            
        except Exception as e:
            print(f"摄像头处理失败: {e}")
            messagebox.showerror("错误", f"摄像头处理失败: {e}")
        finally:
            def restore_ui():
                self.toggle_video_controls(False)
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
                
                if progress_frame:
                    progress_frame.destroy()
            
            self.root.after(0, restore_ui)
    
    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            print("正在停止处理...")
            self.stop_video_processing = True
    
    def update_result_text(self):
        if not self.detection_results:
            return
            
        result_text = f"检测到 {self.detection_results['num_detections']} 个缺陷:\n\n"
        
        class_counts = {}
        for det in self.detection_results['detections']:
            cls_name = det['class_name']
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
        
        result_text += "类别统计:\n"
        for cls_name, count in class_counts.items():
            result_text += f"  - {cls_name}: {count}个\n"
        
        result_text += "\n详细检测结果:\n"
        
        for i, det in enumerate(self.detection_results['detections']):
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            result_text += f"{i+1}. 类别: {det['class_name']}\n"
            result_text += f"   置信度: {det['confidence']:.2f}\n"
            result_text += f"   位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"
            result_text += f"   尺寸: {width}x{height}像素\n\n"
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
    
    def export_result_details(self):
        if not self.detection_results:
            messagebox.showerror("错误", "没有检测结果可导出")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="导出结果详细报告",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            source = self.current_image_path if self.current_image_path else (self.current_video_path if self.current_video_path else "摄像头")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"缺陷检测结果详细报告\n")
                f.write(f"生成时间: {now}\n")
                f.write(f"来源: {source}\n")
                f.write(f"推理时间: {self.detection_results['inference_time']:.3f}秒\n")
                f.write(f"检测到 {self.detection_results['num_detections']} 个缺陷\n\n")
                
                f.write("类别统计:\n")
                class_counts = {}
                for det in self.detection_results['detections']:
                    cls_name = det['class_name']
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                    else:
                        class_counts[cls_name] = 1
                        
                for cls_name, count in class_counts.items():
                    f.write(f"  - {cls_name}: {count}个\n")
                    
                f.write("\n详细检测结果:\n")
                for i, det in enumerate(self.detection_results['detections']):
                    bbox = det['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    
                    f.write(f"缺陷 #{i+1}:\n")
                    f.write(f"  类别: {det['class_name']}\n")
                    f.write(f"  置信度: {det['confidence']:.4f}\n")
                    f.write(f"  边界框: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n")
                    f.write(f"  尺寸: {width}x{height}像素\n")
                    f.write(f"  面积: {area}像素²\n")
                    f.write("\n")
            
            print(f"详细结果已导出到: {file_path}")
            messagebox.showinfo("导出成功", f"详细结果已导出到:\n{file_path}")
            
        except Exception as e:
            print(f"导出结果详情失败: {e}")
            messagebox.showerror("错误", f"导出结果详情失败: {e}")
    
    def export_log(self):
        file_path = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            log_content = self.log_text.get(1.0, tk.END)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
                
            print(f"日志已导出到: {file_path}")
            messagebox.showinfo("导出成功", f"日志已导出到:\n{file_path}")
            
        except Exception as e:
            print(f"导出日志失败: {e}")
            messagebox.showerror("错误", f"导出日志失败: {e}")
    
    def search_log(self):
        search_text = self.log_search_var.get().strip()
        if not search_text:
            return
            
        self.log_text.tag_remove("search", "1.0", tk.END)
        
        start_pos = "1.0"
        while True:
            start_pos = self.log_text.search(search_text, start_pos, tk.END)
            if not start_pos:
                break
                
            end_pos = f"{start_pos}+{len(search_text)}c"
            self.log_text.tag_add("search", start_pos, end_pos)
            start_pos = end_pos
            
        self.log_text.tag_config("search", background="yellow", foreground="black")
        
        if self.log_text.tag_ranges("search"):
            self.log_text.see(self.log_text.tag_ranges("search")[0])
        else:
            messagebox.showinfo("搜索结果", f"未找到 '{search_text}'")
    
    def clear_log(self):
        if messagebox.askyesno("清空日志", "确定要清空日志吗?"):
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            print("日志已清空")
    
    def save_results(self):
        if not self.detection_results:
            messagebox.showerror("错误", "没有检测结果可保存")
            return
            
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                title="保存结果图像",
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            try:
                result_image = self.detector.draw_results(self.current_image.copy(), self.detection_results)
                
                cv2.imwrite(file_path, result_image)
                
                json_path = os.path.splitext(file_path)[0] + '.json'
                self.detector.save_results_to_json(self.detection_results, json_path)
                
                messagebox.showinfo("保存成功", f"结果已保存到:\n{file_path}\n{json_path}")
                
            except Exception as e:
                print(f"保存结果失败: {e}")
                messagebox.showerror("错误", f"保存结果失败: {e}")
        else:
            file_path = filedialog.asksaveasfilename(
                title="保存结果JSON",
                defaultextension=".json",
                filetypes=[("JSON File", "*.json"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            try:
                self.detector.save_results_to_json(self.detection_results, file_path)
                
                messagebox.showinfo("保存成功", f"结果已保存到:\n{file_path}")
                
            except Exception as e:
                print(f"保存结果失败: {e}")
                messagebox.showerror("错误", f"保存结果失败: {e}")
    
    def show_image(self, image):
        if image is None:
            return
            
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        img_width, img_height = pil_image.size
        
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = (canvas_width - self.photo_image.width()) // 2
        y = (canvas_height - self.photo_image.height()) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
    
    def disable_operation_buttons(self):
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget("text") == "操作":
                        for button in child.winfo_children():
                            if button != self.stop_button:
                                button.config(state=tk.DISABLED)
    
    def enable_operation_buttons(self):
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget("text") == "操作":
                        for button in child.winfo_children():
                            if button != self.stop_button:
                                button.config(state=tk.NORMAL)
                                
        if self.detection_results:
            self.save_result_button.config(state=tk.NORMAL)
    
    def on_close(self):
        self.stop_processing()
        
        if self.processing_thread and self.processing_thread.is_alive():
            print("等待处理线程结束...")
            self.processing_thread.join(timeout=1.0)
        
        sys.stdout = sys.__stdout__
        
        self.save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    
    try:
        import os
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
    
    app = DetectionApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
