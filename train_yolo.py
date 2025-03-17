#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
from ultralytics import YOLO
import yaml

# 设置环境变量解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8缺陷检测训练脚本')
    parser.add_argument('--data', type=str, default='D:/Temp/Yolov8/defect_detection/dataset/data.yaml', help='数据配置文件路径')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8模型大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--img_size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='设备编号')
    parser.add_argument('--output_dir', type=str, default='D:/Temp/Yolov8/defect_detection/runs', help='输出目录')
    parser.add_argument('--workers', type=int, default=2, help='数据加载器工作进程数量')
    parser.add_argument('--cache', action='store_true', help='缓存图像以提高训练速度')
    args = parser.parse_args()
    return args

def train_yolov8(args):
    """使用YOLOv8训练缺陷检测模型"""
    
    try:
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 初始化模型
        model = YOLO(f'yolov8{args.model_size}.pt')
        
        # 配置训练参数
        train_args = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.img_size,
            'device': args.device,
            'project': args.output_dir,
            'name': f'defect_detection_v8{args.model_size}',
            'exist_ok': True,
            'patience': 30,  # 早停耐心值
            'pretrained': True,  # 使用预训练权重
            'optimizer': 'SGD',  # 优化器类型
            'cos_lr': True,  # 余弦学习率调度
            'lr0': 0.01,  # 初始学习率
            'lrf': 0.01,  # 最终学习率因子
            'momentum': 0.937,  # 动量
            'weight_decay': 0.0005,  # 权重衰减
            'warmup_epochs': 3,  # 预热轮数
            'warmup_momentum': 0.8,  # 预热动量
            'warmup_bias_lr': 0.1,  # 预热偏置学习率
            'hsv_h': 0.015,  # HSV色调增强
            'hsv_s': 0.7,  # HSV饱和度增强
            'hsv_v': 0.4,  # HSV亮度增强
            'degrees': 0.0,  # 旋转增强范围（degrees±）
            'translate': 0.1,  # 平移增强范围（±fraction）
            'scale': 0.5,  # 缩放增强范围（±fraction）
            'fliplr': 0.5,  # 水平翻转概率
            'mosaic': 1.0,  # 马赛克增强概率
            'mixup': 0.1,  # 混合增强概率
            'copy_paste': 0.1,  # 复制粘贴增强概率
            'conf': 0.001,  # 测试时置信度阈值
            'iou': 0.6,  # 测试时IoU阈值
            'save': True,  # 保存模型和结果
            'save_period': -1,  # 每隔x轮保存一次，-1表示仅保存最后一轮
            'plots': True,  # 保存训练图表
            'verbose': True,  # 详细输出
            'workers': args.workers,  # 工作进程数量
            'amp': False,  # 关闭自动混合精度以提高稳定性
            'cache': args.cache,  # 缓存图像以减少磁盘IO
            'seed': 0,  # 固定随机种子
            'deterministic': True,  # 确保结果可复现
            'rect': False,  # 矩形训练（非马赛克）
            'close_mosaic': 10,  # 最后10个epoch关闭马赛克增强
        }
        
        print("\n开始训练YOLOv8缺陷检测模型...")
        print(f"- 模型: YOLOv8{args.model_size}")
        print(f"- 数据集: {args.data}")
        print(f"- 训练轮数: {args.epochs}")
        print(f"- 批量大小: {args.batch_size}")
        print(f"- 图像尺寸: {args.img_size}")
        print(f"- 设备: {args.device}")
        print(f"- 工作进程: {args.workers}")
        print(f"- 图像缓存: {'开启' if args.cache else '关闭'}")
        
        # 开始训练
        results = model.train(**train_args)
        
        # 训练完成后打印结果
        print("\n训练完成!")
        print(f"最佳模型路径: {results.best}")
        print(f"最佳mAP@0.5: {results.metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"最佳mAP@0.5:0.95: {results.metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        
        # 验证模型
        print("\n正在验证模型...")
        val_results = model.val()
        
        print("\n验证结果:")
        print(f"mAP@0.5: {val_results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"精确率: {val_results.box.mp:.4f}")
        print(f"召回率: {val_results.box.mr:.4f}")
        
        # 导出模型为ONNX格式
        try:
            print("\n正在导出ONNX模型...")
            export_path = model.export(format="onnx")
            print(f"ONNX模型已导出至: {export_path}")
        except Exception as e:
            print(f"导出ONNX模型时出错: {e}")
        
        return results
    
    except KeyboardInterrupt:
        print("\n训练被用户中断!")
        return None
    except Exception as e:
        print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    train_yolov8(args)

if __name__ == "__main__":
    main()