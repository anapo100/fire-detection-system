# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# torch/ultralytics 데이터 및 바이너리 수집
torch_datas = collect_data_files('torch')
ultralytics_datas = collect_data_files('ultralytics')
torch_binaries = collect_dynamic_libs('torch')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[] + torch_binaries,
    datas=[
        ('config', 'config'),
        ('platform-tools', 'platform-tools'),
        ('models', 'models'),
    ] + torch_datas + ultralytics_datas,
    hiddenimports=[
        'cv2',
        'numpy',
        'yaml',
        'requests',
        'PIL',
        'slack_sdk',
        'torch',
        'torchvision',
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.nn.tasks',
        'ultralytics.utils',
        'ultralytics.engine',
        'ultralytics.engine.model',
        'ultralytics.engine.predictor',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FireDetectionSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FireDetectionSystem',
)
