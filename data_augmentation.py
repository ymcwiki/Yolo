#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import random
import shutil
from pathlib import Path
import time
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='缺陷检测图像数据增强工具')
    parser.add_argument('--good_dir', type=str, default='D:/Temp/Yolov8/defect_detection/good', help='无缺陷样本目录')
    parser.add_argument('--bad_dir', type=str, default='D:/Temp/Yolov8/defect_detection/notgood', help='有缺陷样本目录')
    parser.add_argument('--output_dir', type=str, default='D:/Temp/Yolov8/defect_detection/dataset', help='输出数据集目录')
    parser.add_argument('--augment_per_image', type=int, default=10, help='每张图片的增强倍数')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    args = parser.parse_args()
    return args

def create_dataset_directories(output_dir):
    """创建数据集目录结构"""
    for subset in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, subset, folder), exist_ok=True)

def parse_xml(xml_file):
    """解析XML标注文件"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 获取所有标注对象
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'width': width,
        'height': height,
        'objects': objects
    }

def create_xml(filename, width, height, objects, folder=''):
    """创建XML标注文件"""
    root = ET.Element('annotation')
    
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'path').text = os.path.join(folder, filename)
    
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    ET.SubElement(root, 'segmented').text = '0'
    
    for obj in objects:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = obj['name']
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0'
        
        bbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(int(obj['bbox'][0]))
        ET.SubElement(bbox, 'ymin').text = str(int(obj['bbox'][1]))
        ET.SubElement(bbox, 'xmax').text = str(int(obj['bbox'][2]))
        ET.SubElement(bbox, 'ymax').text = str(int(obj['bbox'][3]))
    
    tree = ET.ElementTree(root)
    return tree

def create_yolo_txt(objects, width, height):
    """创建YOLO格式的txt标注文件"""
    lines = []
    for obj in objects:
        # 计算归一化的中心点坐标和宽高
        xmin, ymin, xmax, ymax = obj['bbox']
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        
        # 对于缺陷检测，我们使用类别ID 0（假设只有一类缺陷）
        class_id = 0
        
        # 添加到行
        lines.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
    
    return "\n".join(lines)

def apply_augmentation(img, annotation, aug_type, aug_intensity=None):
    """应用数据增强方法"""
    height, width = img.shape[:2]
    new_img = img.copy()
    new_anno = annotation.copy()
    
    # 更新新的标注对象列表
    new_objects = []
    for obj in new_anno['objects']:
        new_objects.append({
            'name': obj['name'],
            'bbox': obj['bbox'].copy()
        })
    new_anno['objects'] = new_objects
    
    # 根据不同增强类型处理图像和标注
    if aug_type == 'flip_horizontal':
        # 水平翻转
        new_img = cv2.flip(new_img, 1)
        
        # 更新边界框坐标
        for obj in new_anno['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            obj['bbox'][0] = width - xmax
            obj['bbox'][2] = width - xmin
    
    elif aug_type == 'flip_vertical':
        # 垂直翻转
        new_img = cv2.flip(new_img, 0)
        
        # 更新边界框坐标
        for obj in new_anno['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            obj['bbox'][1] = height - ymax
            obj['bbox'][3] = height - ymin
    
    elif aug_type == 'rotate':
        # 随机旋转角度
        angle = random.uniform(-30, 30) if aug_intensity is None else aug_intensity * 30
        
        # 获取旋转矩阵
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转
        new_img = cv2.warpAffine(new_img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        
        # 更新边界框
        if new_anno['objects']:
            for obj in new_anno['objects']:
                xmin, ymin, xmax, ymax = obj['bbox']
                
                # 计算四个角点
                corners = np.array([
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmin, ymax],
                    [xmax, ymax]
                ], dtype=np.float32)
                
                # 旋转角点
                corners = np.hstack((corners, np.ones((4, 1))))
                corners = np.dot(matrix, corners.T).T
                
                # 获取新边界框
                obj['bbox'][0] = max(0, min(corners[:, 0]))
                obj['bbox'][1] = max(0, min(corners[:, 1]))
                obj['bbox'][2] = min(width, max(corners[:, 0]))
                obj['bbox'][3] = min(height, max(corners[:, 1]))
    
    elif aug_type == 'brightness':
        # 亮度调整
        alpha = random.uniform(0.5, 1.5) if aug_intensity is None else 0.5 + aug_intensity
        beta = random.uniform(-30, 30) if aug_intensity is None else aug_intensity * 60 - 30
        new_img = cv2.convertScaleAbs(new_img, alpha=alpha, beta=beta)
    
    elif aug_type == 'contrast':
        # 对比度调整 - 修复版本
        alpha = random.uniform(0.5, 1.5) if aug_intensity is None else 0.5 + aug_intensity
        
        # 创建查找表 (LUT) 进行对比度调整
        # 使用CLAHE（对比度受限的自适应直方图均衡化）进行对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        l_clahe = clahe.apply(l)
        
        # 合并通道
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # 转换回BGR颜色空间
        new_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    elif aug_type == 'noise':
        # 添加噪声
        noise_level = random.uniform(5, 20) if aug_intensity is None else aug_intensity * 20
        noise = np.random.normal(0, noise_level, new_img.shape).astype(np.uint8)
        new_img = cv2.add(new_img, noise)
    
    elif aug_type == 'blur':
        # 模糊
        blur_level = random.randint(1, 5) if aug_intensity is None else int(aug_intensity * 5) + 1
        # 确保kernel尺寸为奇数
        if blur_level % 2 == 0:
            blur_level += 1
        new_img = cv2.GaussianBlur(new_img, (blur_level, blur_level), 0)
    
    elif aug_type == 'hue':
        # 色调变化
        hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        hue_shift = random.uniform(-20, 20) if aug_intensity is None else aug_intensity * 40 - 20
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif aug_type == 'saturation':
        # 饱和度变化
        hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        sat_factor = random.uniform(0.5, 1.5) if aug_intensity is None else 0.5 + aug_intensity
        hsv[:, :, 1] = hsv[:, :, 1] * sat_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif aug_type == 'perspective':
        # 透视变换
        height, width = new_img.shape[:2]
        
        # 定义变形强度
        distortion = random.uniform(0.05, 0.2) if aug_intensity is None else aug_intensity * 0.2
        
        # 源点
        src_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ])
        
        # 目标点（添加随机扰动）
        dst_points = np.float32([
            [width * random.uniform(0, distortion), height * random.uniform(0, distortion)],
            [width * (1 - random.uniform(0, distortion)), height * random.uniform(0, distortion)],
            [width * random.uniform(0, distortion), height * (1 - random.uniform(0, distortion))],
            [width * (1 - random.uniform(0, distortion)), height * (1 - random.uniform(0, distortion))]
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        new_img = cv2.warpPerspective(new_img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        
        # 对标注框应用相同变换
        if new_anno['objects']:
            for obj in new_anno['objects']:
                xmin, ymin, xmax, ymax = obj['bbox']
                
                corners = np.array([
                    [xmin, ymin, 1],
                    [xmax, ymin, 1],
                    [xmin, ymax, 1],
                    [xmax, ymax, 1]
                ])
                
                # 应用变换
                transformed_corners = np.dot(matrix, corners.T).T
                
                # 归一化
                for i in range(4):
                    transformed_corners[i, 0] /= transformed_corners[i, 2]
                    transformed_corners[i, 1] /= transformed_corners[i, 2]
                
                # 更新边界框
                transformed_corners = transformed_corners[:, :2]
                obj['bbox'][0] = max(0, min(transformed_corners[:, 0]))
                obj['bbox'][1] = max(0, min(transformed_corners[:, 1]))
                obj['bbox'][2] = min(width, max(transformed_corners[:, 0]))
                obj['bbox'][3] = min(height, max(transformed_corners[:, 1]))
    
    # 裁剪边界框以确保在图像范围内
    for obj in new_anno['objects']:
        obj['bbox'][0] = max(0, min(obj['bbox'][0], width-1))
        obj['bbox'][1] = max(0, min(obj['bbox'][1], height-1))
        obj['bbox'][2] = max(0, min(obj['bbox'][2], width-1))
        obj['bbox'][3] = max(0, min(obj['bbox'][3], height-1))
        
        # 确保边界框是有效的（宽高大于0）
        if obj['bbox'][2] <= obj['bbox'][0] or obj['bbox'][3] <= obj['bbox'][1]:
            obj['bbox'][2] = min(obj['bbox'][0] + 1, width-1)
            obj['bbox'][3] = min(obj['bbox'][1] + 1, height-1)
    
    return new_img, new_anno

def augment_images(source_dir, is_defect, augmentations, aug_count):
    """增强指定目录中的图像"""
    images_and_annotations = []
    
    # 查找所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png']
    img_files = []
    
    for ext in img_extensions:
        img_files.extend(list(Path(source_dir).glob(f'*{ext}')))
    
    print(f"在 {source_dir} 中找到 {len(img_files)} 张图像")
    
    # 处理每张图像
    for img_path in img_files:
        img_file = img_path.name
        img_stem = img_path.stem
        
        # 寻找对应的XML标注文件
        xml_file = os.path.join(source_dir, f"{img_stem}.xml")
        if not os.path.exists(xml_file):
            print(f"警告: 未找到 {img_file} 的XML标注文件")
            continue
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像 {img_file}")
            continue
        
        # 解析XML标注
        try:
            annotation = parse_xml(xml_file)
        except Exception as e:
            print(f"警告: 解析XML文件 {xml_file} 时出错: {e}")
            continue
        
        # 添加原始图像
        images_and_annotations.append((img_file, img, annotation))
        
        # 应用增强
        for i in range(aug_count):
            # 对于每张图像，应用多种随机增强
            aug_img = img.copy()
            aug_anno = annotation.copy()
            
            # 为了增加多样性，每个增强实例应用随机数量的增强方法
            selected_augs = random.sample(augmentations, k=random.randint(1, min(3, len(augmentations))))
            
            for aug_type in selected_augs:
                try:
                    aug_img, aug_anno = apply_augmentation(aug_img, aug_anno, aug_type)
                except Exception as e:
                    print(f"警告: 应用增强 {aug_type} 到 {img_file} 时出错: {e}")
                    continue
            
            # 生成增强后的文件名
            aug_filename = f"{img_stem}_aug_{i}{img_path.suffix}"
            
            # 添加增强后的图像和标注
            images_and_annotations.append((aug_filename, aug_img, aug_anno))
    
    return images_and_annotations, len(img_files)  # 返回增强后的图像和原始图像数量

def save_to_dataset(images_and_annotations, output_dir, subset, class_names):
    """保存增强后的图像和标注到数据集目录"""
    images_dir = os.path.join(output_dir, subset, 'images')
    labels_dir = os.path.join(output_dir, subset, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for filename, img, anno in images_and_annotations:
        # 保存图像
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(images_dir, filename), img)
        
        # 保存YOLO格式标注
        yolo_content = create_yolo_txt(anno['objects'], anno['width'], anno['height'])
        with open(os.path.join(labels_dir, f"{base_name}.txt"), 'w') as f:
            f.write(yolo_content)

def create_data_yaml(output_dir, class_names):
    """创建data.yaml配置文件"""
    yaml_content = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

def show_menu(options):
    """显示选择菜单"""
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    choice = input("请输入选项编号(多选请用逗号分隔，输入'a'选择全部): ")
    
    if choice.lower() == 'a':
        return list(range(len(options)))
    
    try:
        choices = [int(c.strip()) - 1 for c in choice.split(',')]
        return [c for c in choices if 0 <= c < len(options)]
    except:
        print("输入无效，请重试")
        return show_menu(options)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义类别名称
    class_names = ['defect']  # 缺陷类别名称
    
    # 创建数据集目录结构
    create_dataset_directories(args.output_dir)
    
    # 定义可用的增强方法
    augmentation_types = [
        '水平翻转(flip_horizontal)', 
        '垂直翻转(flip_vertical)',
        '旋转(rotate)', 
        '亮度调整(brightness)', 
        '对比度调整(contrast)',
        '添加噪声(noise)', 
        '模糊(blur)', 
        '色调变化(hue)', 
        '饱和度变化(saturation)',
        '透视变换(perspective)'
    ]
    
    aug_methods = [method.split('(')[1].rstrip(')') for method in augmentation_types]
    
    print("=" * 50)
    print("缺陷检测数据增强工具")
    print("=" * 50)
    
    # 选择增强方法
    print("\n请选择要应用的数据增强方法:")
    selected_indices = show_menu(augmentation_types)
    selected_methods = [aug_methods[i] for i in selected_indices]
    
    if not selected_methods:
        print("未选择任何增强方法，使用默认方法: 水平翻转, 旋转, 亮度调整")
        selected_methods = ['flip_horizontal', 'rotate', 'brightness']
    
    print(f"\n已选择的增强方法: {', '.join([augmentation_types[i] for i in selected_indices])}")
    
    # 询问每张图像的增强数量
    try:
        aug_per_image = int(input(f"\n每张图像生成多少个增强样本？(默认: {args.augment_per_image}): ") or args.augment_per_image)
    except ValueError:
        aug_per_image = args.augment_per_image
        print(f"输入无效，使用默认值: {aug_per_image}")
    
    # 增强有缺陷的样本
    print("\n正在增强有缺陷的样本...")
    defect_images, defect_orig_count = augment_images(args.bad_dir, True, selected_methods, aug_per_image)
    
    # 增强无缺陷的样本
    print("\n正在增强无缺陷的样本...")
    normal_images, normal_orig_count = augment_images(args.good_dir, False, selected_methods, aug_per_image)
    
    # 合并所有图像
    all_images = defect_images + normal_images
    random.shuffle(all_images)
    
    # 统计原始图像总数
    total_orig_count = defect_orig_count + normal_orig_count
    
    # 划分数据集
    n_images = len(all_images)
    n_train = int(n_images * args.train_ratio)
    n_val = int(n_images * args.val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    # 保存到相应的子集
    print(f"\n正在保存训练集 ({len(train_images)} 张图像)...")
    save_to_dataset(train_images, args.output_dir, 'train', class_names)
    
    print(f"正在保存验证集 ({len(val_images)} 张图像)...")
    save_to_dataset(val_images, args.output_dir, 'val', class_names)
    
    print(f"正在保存测试集 ({len(test_images)} 张图像)...")
    save_to_dataset(test_images, args.output_dir, 'test', class_names)
    
    # 创建data.yaml文件
    create_data_yaml(args.output_dir, class_names)
    
    print("\n数据增强完成!")
    print(f"总计生成 {n_images} 张图像 (原始: {total_orig_count}, 增强: {total_orig_count * aug_per_image})")
    print(f"- 训练集: {len(train_images)} 张图像")
    print(f"- 验证集: {len(val_images)} 张图像")
    print(f"- 测试集: {len(test_images)} 张图像")
    print(f"\n数据集保存至: {args.output_dir}")

if __name__ == "__main__":
    main()