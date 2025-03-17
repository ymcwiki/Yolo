#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from cx_Freeze import setup, Executable

# 检查assets目录是否存在，不存在则创建
if not os.path.exists('assets'):
    os.makedirs('assets')

# 检查是否有图标文件
icon_path = None
if os.path.exists('assets/icon.ico'):
    icon_path = 'assets/icon.ico'

# 构建选项
build_options = {
    'packages': [
        'os', 'sys', 'time', 'cv2', 'numpy', 'torch', 'PIL', 
        'tkinter', 'ultralytics', 'threading', 'queue', 'datetime', 'json'
    ],
    'excludes': [],
    'include_files': [
        # 添加资源文件
        ('assets', 'assets'),
    ],
    # 添加必要的包含项以处理 DLL 缺失警告
    'bin_includes': [
        'mkl_core.2.dll', 
        'mkl_intel_thread.2.dll',
        'mkl_avx2.2.dll'
    ],
    # 排除某些模块以减小打包体积
    'exclude_modules': [
        'tkinter.test', 
        'lib2to3',
        'unittest',
        'email',
        'html',
        'http',
        'xmlrpc',
        'pydoc_data',
    ]
}

# 基本信息
base = None
if sys.platform == 'win32':
    base = 'Win32GUI'  # 使用Windows GUI模式，不显示控制台窗口

# 可执行文件配置
executables = [
    Executable(
        'app.py',                      # 主脚本
        base=base,                     # 基础
        target_name='瓣叶缺陷检测.exe',   # 可执行文件名
        icon=icon_path,                # 图标，如果找不到则为None
        shortcut_name='瓣叶缺陷检测',     # 快捷方式名称
        copyright='YourCompany',       # 版权信息
    )
]

# 设置
setup(
    name="瓣叶缺陷检测",
    version="1.0.0",
    description="YOLOv8缺陷检测应用程序",
    options={'build_exe': build_options},
    executables=executables,
)