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
    def __init__(self, model_path=None, conf_thres=0.25, iou_thres=0.45, device=None, batch_size=1):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if model_path is None:
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
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
            
        self.class_names = self.model.names
        print(f"检测类别: {self.class_names}")
    
    def detect(self, image, size=640, augment=False):
        start_time = time.time()
        
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            imgsz=size,
            augment=augment,
            verbose=False,
            batch=self.batch_size
        )[0]
        
        inference_time = time.time() - start_time
        
        boxes = results.boxes.cpu().numpy()
        
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
        
        output = {
            'num_detections': len(detections),
            'detections': detections,
            'inference_time': inference_time,
            'raw_results': results
        }
        
        return output

    def draw_results(self, image, detection_results, show_conf=True, line_thickness=2):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if isinstance(image, str):
            image = cv2.imread(image)
            
        img_result = image.copy()
        
        num_classes = len(self.class_names)
        colors = {}
        for cls_id in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors[cls_id] = color
            
        height, width = img_result.shape[:2]
        
        for det in detection_results['detections']:
            x1, y1, x2, y2 = det['bbox']
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']
            
            color = colors.get(cls_id, (0, 255, 0))
            
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, line_thickness)
            
            if show_conf:
                label = f"{cls_name} {conf:.2f}"
            else:
                label = cls_name
                
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img_result, 
                        (x1, y1 - text_size[1] - 5), 
                        (x1 + text_size[0], y1),
                        color, -1)
            
            cv2.putText(img_result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        inference_time = detection_results['inference_time']
        num_detections = detection_results['num_detections']
        
        cv2.putText(img_result, f"推理时间: {inference_time:.3f}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_result, f"检测到 {num_detections} 个缺陷", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_result

    def process_video(self, video_path, output_path=None, show_preview=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            results = self.detect(frame)
            
            output_frame = self.draw_results(frame, results)
            
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"\r处理进度: {progress:.1f}% ({frame_count}/{total_frames}), 当前帧处理时间: {process_time:.3f}s", end="")
            
            if output_path:
                out.write(output_frame)
                
            if show_preview:
                cv2.imshow('Detection Preview', output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        print(f"\n视频处理完成! 处理了 {frame_count} 帧, 平均每帧处理时间: {avg_time:.3f}s")
        
        return frame_count
        
    def save_results_to_json(self, detection_results, output_path):
        data_to_save = {
            'num_detections': detection_results['num_detections'],
            'detections': detection_results['detections'],
            'inference_time': detection_results['inference_time'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
        print(f"检测结果已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8缺陷检测器测试')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--image', type=str, default=None, help='测试图像路径')
    parser.add_argument('--video', type=str, default=None, help='测试视频路径')
    parser.add_argument('--output', type=str, default=None, help='输出结果路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', type=str, default=None, help='推理设备')
    parser.add_argument('--batch', type=int, default=1, help='批处理大小')
    args = parser.parse_args()
    
    detector = DefectDetector(model_path=args.model, conf_thres=args.conf, device=args.device, batch_size=args.batch)
    
    if args.image:
        print(f"\n开始检测图像: {args.image}")
        results = detector.detect(args.image)
        
        print(f"检测到 {results['num_detections']} 个缺陷:")
        for i, det in enumerate(results['detections']):
            print(f"  {i+1}. 类别: {det['class_name']}, 置信度: {det['confidence']:.4f}, 边界框: {det['bbox']}")
        
        img = cv2.imread(args.image)
        result_img = detector.draw_results(img, results)
        
        cv2.imshow('Detection Result', result_img)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if args.output:
            output_img_path = args.output if args.output.endswith(('.jpg', '.png', '.jpeg')) else args.output + '.jpg'
            cv2.imwrite(output_img_path, result_img)
            print(f"结果已保存到: {output_img_path}")
            
            json_path = os.path.splitext(output_img_path)[0] + '.json'
            detector.save_results_to_json(results, json_path)
    
    elif args.video:
        print(f"\n开始处理视频: {args.video}")
        detector.process_video(
            args.video, 
            output_path=args.output, 
            show_preview=True
        )
    
    else:
        print("请指定--image或--video参数来测试检测器")
