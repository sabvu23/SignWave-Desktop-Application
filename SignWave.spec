# -*- mode: python ; coding: utf-8 -*-

import sys
import os
import site

# --- Block to find site-packages and mediapipe path ---
try:
    site_packages_path = next(p for p in sys.path if 'site-packages' in p and os.path.isdir(p))
    print(f"[SPEC] Found site-packages: {site_packages_path}")
    mediapipe_path = os.path.join(site_packages_path, 'mediapipe')
    print(f"[SPEC] Expected MediaPipe path: {mediapipe_path}")
    if not os.path.isdir(mediapipe_path):
         print(f"[SPEC] WARNING: MediaPipe path not found at {mediapipe_path}")
         mediapipe_path = None
except StopIteration:
    print("[SPEC] WARNING: Could not automatically find site-packages directory.")
    mediapipe_path = None
except Exception as e:
    print(f"[SPEC] WARNING: Error finding MediaPipe path: {e}")
    mediapipe_path = None
# --- End block ---

# Initialize the list of data files
datas_list = [
    ('signlanguagemodel.h5', '.') # Your model file in the root
]

# --- Block to add MediaPipe data IF path was found ---
if mediapipe_path and os.path.isdir(mediapipe_path):
    mediapipe_modules_src = os.path.join(mediapipe_path, 'modules')
    if os.path.isdir(mediapipe_modules_src):
        datas_list.append((mediapipe_modules_src, 'mediapipe/modules'))
        print(f"[SPEC] Adding MediaPipe modules from: {mediapipe_modules_src}")
    else:
         print(f"[SPEC] WARNING: MediaPipe modules directory not found at {mediapipe_modules_src}")
else:
    print("[SPEC] WARNING: Skipping MediaPipe data inclusion as its path was not found.")
# --- End block ---

block_cipher = None

a = Analysis(
    ['signwave_app.py'],
    pathex=[],
    binaries=[],
    datas=datas_list, # Use the prepared list
    hiddenimports=[
        'mediapipe.python._framework_bindings',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Define the EXE object (gets built into the build/ folder)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='SignWave', # Base name of the executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Keep console=True for debugging
)

# --- ADD THE COLLECT STEP ---
# This gathers the EXE, binaries, and data into the dist/SignWave folder
coll = COLLECT(
    exe, # The EXE object defined above
    a.binaries, # Include binaries found by Analysis
    a.datas, # Include datas found by Analysis (contains model + mediapipe)
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SignWave', # Name of the output FOLDER in dist/
)
# --- END OF COLLECT STEP ---

# Note: We are NOT using BUNDLE here, so this will create a folder, not one file.