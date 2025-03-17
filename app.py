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

# 导入检测器类
try:
    from detect import DefectDetector
except ImportError:
    # 如果在打包后运行，可能需要调整导入路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detect import DefectDetector

class RedirectText:
    """用于重定向输出到GUI的类"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.update_timer = None
        
    def write(self, string):
        self.queue.put(string)
        # 确保只有一个更新定时器在运行
        if self.update_timer is None:
            self.update_timer = self.text_widget.after(100, self.update_text_widget)
    
    def update_text_widget(self):
        # 处理队列中的所有待处理文本
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
        
        # 重置定时器为None
        self.update_timer = None
        
    def flush(self):
        pass

class DetectionApp:
    """缺陷检测应用程序类"""
    
    def __init__(self, root):
        """初始化应用程序"""
        self.root = root
        self.root.title("以心医疗瓣叶缺陷检测")
        self.root.geometry("1280x720")
        self.root.minsize(1024, 600)
        
        # 设置应用程序图标
        try:
            # 尝试加载图标（需要将图标文件打包到应用中）
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except Exception:
            pass  # 忽略图标加载错误
        
        # 初始化变量
        self.detector = None
        self.current_image = None
        self.current_image_path = None
        self.current_video_path = None
        self.video_output_path = None
        self.detection_results = None
        self.processing_thread = None
        self.stop_video_processing = False
        
        # 设置样式
        self.setup_styles()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建面板
        self.create_control_panel()
        self.create_display_panel()
        self.create_log_panel()
        
        # 设置日志重定向
        self.redirect_stdout()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 初始化完成提示
        print("应用程序初始化完成，请选择模型文件开始使用。")
    
    def setup_styles(self):
        """设置ttk样式"""
        self.style = ttk.Style()
        
        # 配置按钮样式
        self.style.configure("Primary.TButton", font=("微软雅黑", 10, "bold"))
        self.style.configure("Secondary.TButton", font=("微软雅黑", 10))
        
        # 配置标签样式
        self.style.configure("Title.TLabel", font=("微软雅黑", 12, "bold"))
        self.style.configure("Subtitle.TLabel", font=("微软雅黑", 10))
        
        # 配置框架样式
        self.style.configure("Panel.TFrame", relief=tk.RIDGE, borderwidth=2)
    
    def create_control_panel(self):
        """创建控制面板"""
        # 主控制面板
        control_frame = ttk.Frame(self.main_frame, style="Panel.TFrame", padding=5)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 模型设置面板
        model_frame = ttk.LabelFrame(control_frame, text="模型设置", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # 模型路径输入和浏览按钮
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.grid(row=0, column=1, sticky=tk.EW, pady=2)
        
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_path_frame, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="浏览", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 设备选择
        ttk.Label(model_frame, text="推理设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # 设备选择下拉框
        self.device_var = tk.StringVar(value="auto")
        device_choices = ["auto", "cpu"]
        # 检测是否有CUDA设备
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_choices.append(f"cuda:{i}")
        
        ttk.Combobox(model_frame, textvariable=self.device_var, values=device_choices).grid(row=1, column=1, sticky=tk.EW, pady=2)
        
        # 置信度阈值设置
        ttk.Label(model_frame, text="置信度阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # 置信度阈值滑块
        self.conf_thres_var = tk.DoubleVar(value=0.25)
        ttk.Scale(model_frame, from_=0.01, to=1.0, variable=self.conf_thres_var, 
                orient=tk.HORIZONTAL, command=self.update_conf_label).grid(row=2, column=1, sticky=tk.EW, pady=2)
        
        # 置信度阈值标签
        self.conf_label = ttk.Label(model_frame, text="0.25")
        self.conf_label.grid(row=2, column=2, padx=(5, 0))
        
        # 加载模型按钮
        ttk.Button(model_frame, text="加载模型", style="Primary.TButton", 
                 command=self.load_model).grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        # 操作面板
        operation_frame = ttk.LabelFrame(control_frame, text="操作", padding=5)
        operation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 图像检测按钮
        ttk.Button(operation_frame, text="从图像中检测", 
                 command=self.detect_from_image).pack(fill=tk.X, pady=2)
        
        # 视频检测按钮
        ttk.Button(operation_frame, text="从视频中检测", 
                 command=self.detect_from_video).pack(fill=tk.X, pady=2)
        
        # 摄像头检测按钮
        ttk.Button(operation_frame, text="从摄像头检测", 
                 command=self.detect_from_camera).pack(fill=tk.X, pady=2)
        
        # 停止处理按钮
        self.stop_button = ttk.Button(operation_frame, text="停止处理", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # 保存结果按钮
        self.save_result_button = ttk.Button(operation_frame, text="保存当前结果", 
                                          command=self.save_results, state=tk.DISABLED)
        self.save_result_button.pack(fill=tk.X, pady=2)
        
        # 结果面板
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果显示文本框
        self.result_text = ScrolledText(result_frame, width=30, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
    
    def create_display_panel(self):
        """创建显示面板"""
        # 显示面板主框架
        display_frame = ttk.Frame(self.main_frame, style="Panel.TFrame", padding=5)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 显示区域
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        status_frame = ttk.Frame(display_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 文件信息标签
        self.file_info_label = ttk.Label(status_frame, text="未加载任何文件")
        self.file_info_label.pack(side=tk.LEFT)
        
        # 检测时间信息
        self.time_info_label = ttk.Label(status_frame, text="")
        self.time_info_label.pack(side=tk.RIGHT)
    
    def create_log_panel(self):
        """创建日志面板"""
        # 日志面板
        log_frame = ttk.LabelFrame(self.main_frame, text="日志", padding=5)
        log_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 日志文本框
        self.log_text = ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
    
    def redirect_stdout(self):
        """重定向标准输出到日志面板"""
        self.stdout_redirector = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirector
    
    def update_conf_label(self, event=None):
        """更新置信度标签"""
        self.conf_label.config(text=f"{self.conf_thres_var.get():.2f}")
    
    def browse_model(self):
        """浏览并选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Model Files", "*.pt *.pth *.onnx"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_var.get().strip()
        if not model_path and not os.path.exists(model_path):
            # 如果没有指定模型路径，尝试查找默认路径
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
        
        # 获取设备
        device = self.device_var.get()
        if device == "auto":
            device = None
            
        # 获取置信度阈值
        conf_thres = self.conf_thres_var.get()
        
        try:
            # 显示加载中提示
            self.root.config(cursor="wait")
            self.root.update()
            
            # 开始加载模型
            print(f"正在加载模型: {model_path}")
            self.detector = DefectDetector(
                model_path=model_path,
                conf_thres=conf_thres,
                device=device
            )
            
            # 加载完成
            print(f"模型加载成功: {model_path}")
            messagebox.showinfo("成功", "模型加载成功!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            messagebox.showerror("错误", f"模型加载失败: {e}")
            self.detector = None
        finally:
            # 恢复光标
            self.root.config(cursor="")
    
    def detect_from_image(self):
        """从图像中检测"""
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        # 浏览选择图像
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.current_image_path = file_path
        self.current_video_path = None
        
        try:
            # 显示加载中提示
            self.root.config(cursor="wait")
            self.root.update()
            
            # 执行检测
            print(f"正在检测图像: {file_path}")
            
            # 更新置信度阈值
            self.detector.conf_thres = self.conf_thres_var.get()
            
            # 执行检测
            self.detection_results = self.detector.detect(file_path)
            
            # 读取图像
            self.current_image = cv2.imread(file_path)
            
            # 绘制结果
            result_image = self.detector.draw_results(self.current_image.copy(), self.detection_results)
            
            # 显示结果图像
            self.show_image(result_image)
            
            # 更新文件信息
            file_name = os.path.basename(file_path)
            img_h, img_w = self.current_image.shape[:2]
            self.file_info_label.config(text=f"文件: {file_name} | 尺寸: {img_w}x{img_h}")
            
            # 更新检测时间信息
            self.time_info_label.config(text=f"检测时间: {self.detection_results['inference_time']:.3f}秒")
            
            # 更新检测结果
            self.update_result_text()
            
            # 启用保存结果按钮
            self.save_result_button.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"图像检测失败: {e}")
            messagebox.showerror("错误", f"图像检测失败: {e}")
        finally:
            # 恢复光标
            self.root.config(cursor="")
    
    def detect_from_video(self):
        """从视频中检测"""
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        # 浏览选择视频
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        # 询问是否保存处理后的视频
        save_video = messagebox.askyesno("保存视频", "是否要保存处理后的视频?")
        
        if save_video:
            # 选择保存路径
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
        
        # 重置停止标志
        self.stop_video_processing = False
        
        # 启用停止按钮
        self.stop_button.config(state=tk.NORMAL)
        
        # 禁用其他按钮
        self.disable_operation_buttons()
        
        # 在新线程中处理视频
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_video(self):
        """处理视频的线程函数"""
        try:
            # 更新置信度阈值
            self.detector.conf_thres = self.conf_thres_var.get()
            
            print(f"正在处理视频: {self.current_video_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {self.current_video_path}")
                
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
            
            # 更新文件信息
            file_name = os.path.basename(self.current_video_path)
            
            def update_file_info():
                self.file_info_label.config(text=f"文件: {file_name} | 尺寸: {width}x{height} | FPS: {fps:.1f}")
            
            self.root.after(0, update_file_info)
            
            # 如果需要保存视频，创建VideoWriter
            if self.video_output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.video_output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processing_times = []
            start_time = time.time()
            
            # 处理每一帧
            while cap.isOpened() and not self.stop_video_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 执行检测
                results = self.detector.detect(frame)
                
                # 绘制结果
                output_frame = self.detector.draw_results(frame, results)
                
                # 计算处理时间
                processing_times.append(results['inference_time'])
                
                # 显示进度
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                
                # 更新进度信息
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
                
                progress_msg = f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})"
                time_msg = f"已用时间: {elapsed_time:.1f}s, 预计剩余: {remaining_time:.1f}s"
                
                print(f"\r{progress_msg}, {time_msg}", end="")
                
                # 更新界面显示
                def update_ui():
                    # 显示当前帧
                    self.show_image(output_frame)
                    
                    # 更新时间信息
                    avg_time = sum(processing_times[-10:]) / min(len(processing_times), 10)
                    self.time_info_label.config(text=f"帧处理时间: {avg_time:.3f}秒 | {1/avg_time:.1f} FPS")
                    
                    # 更新检测结果
                    self.detection_results = results
                    self.update_result_text()
                
                self.root.after(0, update_ui)
                
                # 如果需要保存，写入帧
                if self.video_output_path:
                    out.write(output_frame)
                
                # 减慢显示速度，避免界面卡顿
                time.sleep(0.01)
            
            # 清理
            cap.release()
            if self.video_output_path:
                out.release()
                
            # 计算平均处理时间
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            final_msg = f"\n视频处理{'完成' if not self.stop_video_processing else '已停止'}! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s, 处理速度: {1/avg_time:.1f} FPS"
            print(final_msg)
            
            if self.stop_video_processing:
                messagebox.showinfo("处理停止", "视频处理已停止")
            else:
                messagebox.showinfo("处理完成", f"视频处理完成!\n处理了 {frame_count} 帧\n平均FPS: {1/avg_time:.1f}")
            
        except Exception as e:
            print(f"视频处理失败: {e}")
            messagebox.showerror("错误", f"视频处理失败: {e}")
        finally:
            # 恢复界面状态
            def restore_ui():
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
            
            self.root.after(0, restore_ui)
    
    def detect_from_camera(self):
        """从摄像头中检测"""
        if not self.detector:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        # 重置停止标志
        self.stop_video_processing = False
        
        # 启用停止按钮
        self.stop_button.config(state=tk.NORMAL)
        
        # 禁用其他按钮
        self.disable_operation_buttons()
        
        # 询问是否保存摄像头视频
        save_video = messagebox.askyesno("保存视频", "是否要保存摄像头视频?")
        
        output_path = None
        if save_video:
            # 选择保存路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = filedialog.asksaveasfilename(
                title="选择保存路径",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
                initialfile=f"camera_{timestamp}.mp4"
            )
            
            if not output_path:
                # 用户取消了保存，恢复界面状态
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
                return
        
        # 在新线程中处理摄像头
        self.video_output_path = output_path
        self.processing_thread = threading.Thread(target=self.process_camera)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_camera(self):
        """处理摄像头的线程函数"""
        try:
            # 更新置信度阈值
            self.detector.conf_thres = self.conf_thres_var.get()
            
            print("正在打开摄像头...")
            
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("无法打开摄像头")
                
            # 获取摄像头属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"摄像头信息: {width}x{height}, {fps}fps")
            
            # 更新文件信息
            def update_file_info():
                self.file_info_label.config(text=f"摄像头直播 | 尺寸: {width}x{height}")
            
            self.root.after(0, update_file_info)
            
            # 如果需要保存视频，创建VideoWriter
            out = None
            if self.video_output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.video_output_path, fourcc, 20, (width, height))
            
            frame_count = 0
            processing_times = []
            
            # 处理每一帧
            while cap.isOpened() and not self.stop_video_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 执行检测
                results = self.detector.detect(frame)
                
                # 绘制结果
                output_frame = self.detector.draw_results(frame, results)
                
                # 计算处理时间
                processing_times.append(results['inference_time'])
                
                # 显示进度
                frame_count += 1
                
                # 更新界面显示
                def update_ui():
                    # 显示当前帧
                    self.show_image(output_frame)
                    
                    # 更新时间信息
                    avg_time = sum(processing_times[-10:]) / min(len(processing_times), 10)
                    fps_processed = 1 / avg_time
                    self.time_info_label.config(text=f"帧处理时间: {avg_time:.3f}秒 | {fps_processed:.1f} FPS")
                    
                    # 更新检测结果
                    self.detection_results = results
                    self.update_result_text()
                
                self.root.after(0, update_ui)
                
                # 如果需要保存，写入帧
                if out:
                    out.write(output_frame)
                
                # 减慢显示速度，避免界面卡顿
                time.sleep(0.01)
            
            # 清理
            cap.release()
            if out:
                out.release()
                
            # 计算平均处理时间
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            final_msg = f"\n摄像头处理{'完成' if not self.stop_video_processing else '已停止'}! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s, 处理速度: {1/avg_time:.1f} FPS"
            print(final_msg)
            
            if self.video_output_path and not self.stop_video_processing:
                messagebox.showinfo("保存完成", f"摄像头视频已保存到:\n{self.video_output_path}")
            
        except Exception as e:
            print(f"摄像头处理失败: {e}")
            messagebox.showerror("错误", f"摄像头处理失败: {e}")
        finally:
            # 恢复界面状态
            def restore_ui():
                self.stop_button.config(state=tk.DISABLED)
                self.enable_operation_buttons()
            
            self.root.after(0, restore_ui)
    
    def stop_processing(self):
        """停止当前处理"""
        if self.processing_thread and self.processing_thread.is_alive():
            print("正在停止处理...")
            self.stop_video_processing = True
    
    def update_result_text(self):
        """更新检测结果文本"""
        if not self.detection_results:
            return
            
        # 准备结果文本
        result_text = f"检测到 {self.detection_results['num_detections']} 个缺陷:\n\n"
        
        # 按类别统计
        class_counts = {}
        for det in self.detection_results['detections']:
            cls_name = det['class_name']
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
        
        # 显示类别统计
        result_text += "类别统计:\n"
        for cls_name, count in class_counts.items():
            result_text += f"  - {cls_name}: {count}个\n"
        
        result_text += "\n详细检测结果:\n"
        
        # 显示检测详情
        for i, det in enumerate(self.detection_results['detections']):
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            result_text += f"{i+1}. 类别: {det['class_name']}\n"
            result_text += f"   置信度: {det['confidence']:.2f}\n"
            result_text += f"   位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})\n"
            result_text += f"   尺寸: {width}x{height}像素\n\n"
        
        # 更新结果文本框
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
    
    def save_results(self):
        """保存当前检测结果"""
        if not self.detection_results:
            messagebox.showerror("错误", "没有检测结果可保存")
            return
            
        # 如果当前是图像
        if self.current_image is not None:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                title="保存结果图像",
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            try:
                # 绘制结果
                result_image = self.detector.draw_results(self.current_image.copy(), self.detection_results)
                
                # 保存图像
                cv2.imwrite(file_path, result_image)
                
                # 同时保存JSON结果
                json_path = os.path.splitext(file_path)[0] + '.json'
                self.detector.save_results_to_json(self.detection_results, json_path)
                
                messagebox.showinfo("保存成功", f"结果已保存到:\n{file_path}\n{json_path}")
                
            except Exception as e:
                print(f"保存结果失败: {e}")
                messagebox.showerror("错误", f"保存结果失败: {e}")
        else:
            # 只保存JSON结果
            file_path = filedialog.asksaveasfilename(
                title="保存结果JSON",
                defaultextension=".json",
                filetypes=[("JSON File", "*.json"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            try:
                # 保存JSON结果
                self.detector.save_results_to_json(self.detection_results, file_path)
                
                messagebox.showinfo("保存成功", f"结果已保存到:\n{file_path}")
                
            except Exception as e:
                print(f"保存结果失败: {e}")
                messagebox.showerror("错误", f"保存结果失败: {e}")
    
    def show_image(self, image):
        """在画布上显示图像"""
        if image is None:
            return
            
        # 转换图像格式
        if isinstance(image, np.ndarray):
            # 如果是OpenCV图像（BGR），转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            # 如果已经是PIL图像
            pil_image = image
            
        # 获取画布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布尚未调整大小，使用初始尺寸
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # 调整图像大小以适应画布
        img_width, img_height = pil_image.size
        
        # 计算缩放比例
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        if scale < 1:
            # 只在需要缩小时缩放
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为PhotoImage
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # 清除画布
        self.canvas.delete("all")
        
        # 显示图像（居中）
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = (canvas_width - self.photo_image.width()) // 2
        y = (canvas_height - self.photo_image.height()) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
    
    def disable_operation_buttons(self):
        """禁用操作按钮"""
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget("text") == "操作":
                        for button in child.winfo_children():
                            if button != self.stop_button:
                                button.config(state=tk.DISABLED)
    
    def enable_operation_buttons(self):
        """启用操作按钮"""
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget("text") == "操作":
                        for button in child.winfo_children():
                            if button != self.stop_button:
                                button.config(state=tk.NORMAL)
                                
        # 如果有检测结果，启用保存按钮
        if self.detection_results:
            self.save_result_button.config(state=tk.NORMAL)
    
    def on_close(self):
        """关闭应用程序"""
        # 停止任何正在运行的处理
        self.stop_processing()
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            print("等待处理线程结束...")
            self.processing_thread.join(timeout=1.0)
        
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        
        # 关闭窗口
        self.root.destroy()


# 主函数
def main():
    """应用程序入口点"""
    # 创建根窗口
    root = tk.Tk()
    
    # 设置窗口图标
    try:
        # 尝试加载图标（需要将图标文件打包到应用中）
        import os
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass  # 忽略图标加载错误
    
    # 创建应用程序
    app = DetectionApp(root)
    
    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()