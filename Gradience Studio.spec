# Gradience Studio PyInstaller spec

block_cipher = None

import os

project_root = os.path.abspath(os.path.dirname(__file__))

a = Analysis(
    ['Gradience Studio.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('icons/*', 'icons'),       # include icon folder
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='GradienceStudio',
    debug=False,
    strip=False,
    upx=False,
    console=False,        # GUI mode (no console window)
    icon='icon.ico',      # your app icon (optional)
)

