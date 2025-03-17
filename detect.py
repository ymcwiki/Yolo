#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import time
import json

class DefectDetector:
    """YOLOv8缺陷检测器类"""
    
    def __init__(self, model_path=None, conf_thres=0.25, iou_thres=0.45, device=None):
        """
        初始化缺陷检测器
        
        参数:
            model_path (str): 模型路径，可以是.pt或.onnx格式
            conf_thres (float): 置信度阈值
            iou_thres (float): IoU阈值
            device (str): 推理设备，例如'cpu'、'0'（表示第一个GPU）
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 如果没有指定设备，则自动选择
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 如果没有指定模型路径，尝试查找默认位置
        if model_path is None:
            # 尝试在常见路径中查找最新的模型文件
            possible_paths = [
                './runs/detect/defect_detection_v8n/weights/best.pt',
                './weights/best.pt',
                './model/best.pt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
                    
            if model_path is None:
                raise FileNotFoundError("未找到模型文件，请指定model_path参数")
        
        print(f"加载模型: {model_path} 到设备: {self.device}")
        
        # 加载YOLO模型
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
            
        # 获取类别列表
        self.class_names = self.model.names
        print(f"检测类别: {self.class_names}")
    
    def detect(self, image, size=640, augment=False):
        """
        对图像进行缺陷检测
        
        参数:
            image: 可以是图像路径、PIL图像、OpenCV图像或numpy数组
            size (int): 推理尺寸
            augment (bool): 是否使用测试时增强
            
        返回:
            results: 包含检测结果的字典
        """
        # 计时开始
        start_time = time.time()
        
        # 执行推理
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            imgsz=size,
            augment=augment,
            verbose=False
        )[0]
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        # 解析结果
        boxes = results.boxes.cpu().numpy()
        
        # 准备返回结果
        detections = []
        
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': cls_name
                })
        
        # 准备返回的完整结果
        output = {
            'num_detections': len(detections),
            'detections': detections,
            'inference_time': inference_time,
            'raw_results': results
        }
        
        return output

    def draw_results(self, image, detection_results, show_conf=True, line_thickness=2):
        """
        在图像上绘制检测结果
        
        参数:
            image: OpenCV格式的图像
            detection_results: detect()方法返回的结果
            show_conf (bool): 是否显示置信度
            line_thickness (int): 边界框线条粗细
            
        返回:
            带有检测结果的图像
        """
        # 如果输入是PIL图像，转换为OpenCV格式
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 如果输入是路径，读取图像
        if isinstance(image, str):
            image = cv2.imread(image)
            
        # 创建图像副本以避免修改原图
        img_result = image.copy()
        
        # 为每个类别分配不同的颜色
        num_classes = len(self.class_names)
        colors = {}
        for cls_id in range(num_classes):
            # 为每个类别生成唯一颜色
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors[cls_id] = color
            
        # 获取图像尺寸
        height, width = img_result.shape[:2]
        
        # 绘制检测结果
        for det in detection_results['detections']:
            # 获取边界框坐标
            x1, y1, x2, y2 = det['bbox']
            
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # 获取类别和置信度
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']
            
            # 获取颜色
            color = colors.get(cls_id, (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, line_thickness)
            
            # 准备标签文本
            if show_conf:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = cls_name
                
            # 计算文本大小
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 绘制标签背景
            cv2.rectangle(img_result, 
                        (x1, y1 - text_size[1] - 5), 
                        (x1 + text_size[0], y1),
                        color, -1)
            
            # 绘制标签文本
            cv2.putText(img_result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 在左上角显示推理时间和检测数量
        inference_time = detection_results['inference_time']
        num_detections = detection_results['num_detections']
        
        cv2.putText(img_result, f"推理时间: {inference_time:.3f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_result, f"检测到 {num_detections} 个缺陷", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_result

    def process_video(self, video_path, output_path=None, show_preview=False):
        """
        处理视频文件
        
        参数:
            video_path (str): 输入视频路径
            output_path (str): 输出视频路径，如果为None则不保存
            show_preview (bool): 是否显示预览窗口
            
        返回:
            处理的帧数
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        
        # 如果需要保存视频，创建VideoWriter
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        # 处理每一帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 计时开始
            start_time = time.time()
            
            # 执行检测
            results = self.detect(frame)
            
            # 绘制结果
            output_frame = self.draw_results(frame, results)
            
            # 计算处理时间
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # 显示进度
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"\r处理进度: {progress:.1f}% ({frame_count}/{total_frames}), 当前帧处理时间: {process_time:.3f}s", end="")
            
            # 如果需要保存，写入帧
            if output_path:
                out.write(output_frame)
                
            # 如果需要显示预览
            if show_preview:
                cv2.imshow('Detection Preview', output_frame)
                
                # 按下'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # 清理
        cap.release()
        if output_path:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
        # 计算平均处理时间
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        print(f"\n视频处理完成! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s")
        
        return frame_count
        
    def save_results_to_json(self, detection_results, output_path):
        """
        将检测结果保存为JSON文件
        
        参数:
            detection_results: detect()方法返回的结果
            output_path (str): 输出JSON文件路径
        """
        # 准备要保存的数据
        data_to_save = {
            'num_detections': detection_results['num_detections'],
            'detections': detection_results['detections'],
            'inference_time': detection_results['inference_time'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
        print(f"检测结果已保存到: {output_path}")


# 如果直接运行此脚本，执行测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8缺陷检测器测试')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--image', type=str, default=None, help='测试图像路径')
    parser.add_argument('--video', type=str, default=None, help='测试视频路径')
    parser.add_argument('--output', type=str, default=None, help='输出结果路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', type=str, default=None, help='推理设备')
    args = parser.parse_args()
    
    # 初始化检测器
    detector = DefectDetector(model_path=args.model, conf_thres=args.conf, device=args.device)
    
    # 测试图像检测
    if args.image:
        print(f"\n开始检测图像: {args.image}")
        # 执行检测
        results = detector.detect(args.image)
        
        # 打印结果
        print(f"检测到 {results['num_detections']} 个缺陷:")
        for i, det in enumerate(results['detections']):
            print(f"  {i+1}. 类别: {det['class_name']}, 置信度: {det['confidence']:.4f}, 边界框: {det['bbox']}")
        
        # 绘制结果
        img = cv2.imread(args.image)
        result_img = detector.draw_results(img, results)
        
        # 显示结果
        cv2.imshow('Detection Result', result_img)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if args.output:
            output_img_path = args.output if args.output.endswith(('.jpg', '.png', '.jpeg')) else args.output + '.jpg'
            cv2.imwrite(output_img_path, result_img)
            print(f"结果已保存到: {output_img_path}")
            
            # 同时保存JSON结果
            json_path = os.path.splitext(output_img_path)[0] + '.json'
            detector.save_results_to_json(results, json_path)
    
    # 测试视频检测
    elif args.video:
        print(f"\n开始处理视频: {args.video}")
        detector.process_video(
            args.video, 
            output_path=args.output, 
            show_preview=True
        )
    
    else:
        print("请指定--image或--video参数来测试检测器")