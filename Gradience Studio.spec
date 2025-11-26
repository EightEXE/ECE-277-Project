# -*- mode: python ; coding: utf-8 -*-

import os

asset_datas = []
spec_dir = os.path.dirname(os.path.abspath("Gradience Studio.spec"))
icon_root = os.path.join(spec_dir, "icons")
if os.path.isdir(icon_root):
    for root, _, files in os.walk(icon_root):
        rel_dir = os.path.relpath(root, icon_root)
        dest_dir = "icons" if rel_dir in (".", "") else os.path.join("icons", rel_dir)
        for filename in files:
            asset_datas.append((os.path.join(root, filename), dest_dir))
asset_datas += [
    ("card.png", "."),
    ("icon.ico", "."),
]

a = Analysis(
    ['Gradience Studio.py'],
    pathex=[],
    binaries=[],
    datas=asset_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Gradience Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
