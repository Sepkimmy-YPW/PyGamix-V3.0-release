# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py', 'cdftool.py', 'desc.py', 'designer.py', 'dxftool.py', 'logic.py', 'new_module_dialog.py', 'pref_pack.py', 'stl_dialog.py', 'stltool2.py', 'threading_design_dialog.py', 'tm_window.py', 'units.py', 'utils.py', './gui/Ui_new_module.py', './gui/Ui_pref_window.py', './gui/Ui_stl_dialog.py', './gui/Ui_window.py'],
    pathex=[],
    binaries=[],
    datas=[('phys_sim25.py', '.')],
    hiddenimports=['taichi', 'taichi.math', 'ori_sim_sys'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['phys_sim25'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='./setting/icon.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
