# Changelog / 更新日志

All notable changes to PyGamiX are documented in this file.
本文件记录 PyGamiX 的所有重要更新。

---

## [3.0.2] — 2026-09-11

### Added / 新增

**English**

- **`spatialhash.py`**: spatial-hash utility for fast 3D key-point neighborhood lookup (used by `ori_sim_sys.py`). It was referenced by the code but previously missing from the repository — a fresh clone crashed with `ImportError` on startup; this release fixes that.
- **Miura-EA system-description examples**: `descriptionData/f-miura-EA.json` and `descriptionData/miura-EA.json`, matching the `miura-EA` configuration shipped in `trainer.py` / `phys_sim25.py`.
- **Complete tendon-routing data in the `bird4` example**: `descriptionData/bird4.json` now ships its full `strings` section (per-tendon type/id/reverse sequences) and `trans_units`, so the example can be simulated end-to-end as-is.

**中文**

- **`spatialhash.py`**：用于 3D 关键点快速邻域查找的空间哈希工具类（`ori_sim_sys.py` 使用）。此前代码已引用该模块但仓库中缺失——fresh clone 启动即报 `ImportError`，本次发布修复该问题。
- **Miura-EA 系统描述示例**：`descriptionData/f-miura-EA.json` 与 `descriptionData/miura-EA.json`，与 `trainer.py` / `phys_sim25.py` 中的 `miura-EA` 配套配置对应。
- **`bird4` 示例补全腱绳路径数据**：`descriptionData/bird4.json` 现包含完整 `strings` 段（每根腱绳的 type/id/reverse 序列）及 `trans_units`，示例可即开即仿真。

### Changed / 变更

**English**

- **Curated `descriptionData/`**: removed 23 experimental system-description files, keeping the documented example set; bundled examples now cover the Miura / Resch / robot-arm / box families plus the new Miura-EA designs.
- **`main.spec` (PyInstaller) updated**: `threading_design_dialog.py` is now analyzed; `phys_sim25.py` is shipped as data (so the simulator runs with Taichi intact), `taichi` / `taichi.math` / `ori_sim_sys` added as hidden imports, simulator modules excluded from analysis, and the application icon (`./setting/icon.ico`) restored.
- **Explicit Simulation menu action disabled with a notice**: since the Projective Dynamics simulator (`phys_sim_pd14.py`) is not part of this release, *Simulation → Explicit Simulation* now shows an "available in a future version" dialog instead of crashing with `ModuleNotFoundError`.

**中文**

- **整理 `descriptionData/`**：移除 23 个实验性系统描述文件，保留文档化的示例集；内置示例现覆盖 Miura / Resch / 机械臂 / 盒族折纸以及新增的 Miura-EA 设计。
- **更新 `main.spec`（PyInstaller 打包配置）**：`threading_design_dialog.py` 纳入分析；`phys_sim25.py` 以数据文件方式携带（保证仿真器在打包后 Taichi 正常运行）；`taichi` / `taichi.math` / `ori_sim_sys` 加入 hidden imports；仿真模块从分析中排除；恢复应用程序图标（`./setting/icon.ico`）。
- **Explicit Simulation 菜单禁用并弹出提示**：由于 Projective Dynamics 仿真器（`phys_sim_pd14.py`）未随本版本发布，菜单 *Simulation → Explicit Simulation* 现在弹出「将于未来版本提供」的提示框，不再因 `ModuleNotFoundError` 崩溃。

### Removed / 移除

- `packedImport/mountain-big-fix-thick.json` (superseded by other bundled examples).
- 移除 `packedImport/mountain-big-fix-thick.json`（已被其他内置示例取代）。

### Documentation / 文档

- Removed leftover Chinese text from the English README; the language switcher now links to "Chinese".
- 移除英文 README 中残留的中文文本，语言切换栏改为 "Chinese"。

---

## [3.0.1] — 2026-09-09

### Added / 新增

**English**

- **Threading-design presets**: the Threading Design dialog now saves its settings to `./setting/threading_design/<origami_name>.json` on confirm, and pre-fills the origami name plus every field from the saved preset on the next run — no re-entering from scratch.
- **Auto-loading of description data**: after feature extraction (or when reusing an existing description file), the generated `descriptionData/<name>.json` is now automatically loaded into the main window, so the simulation view is ready without manual re-import.
- **Live progress bar for threading search**: trainer log lines tagged `[TRAINER]` are parsed by the GUI in real time — the status bar shows search progress and the progress bar tracks the search (cut/step ratio), completing at 100% when the search finishes.
- **Per-unit crease export with progress feedback**: the crease-layer generation step of STL export now emits progress per unit (`calculateTriPlaneForSingleCrease` / `outputCreaseDxf`), filling the former 40%–50% "blind zone" where the progress bar froze.
- **`miura-EA` example configuration** in `trainer.py` and `phys_sim25.py` (EA system with −Z gravity).

**中文**

- **穿线设计参数预设**：穿线设计对话框确认后自动保存设置到 `./setting/threading_design/<origami_name>.json`；再次运行时自动预填折纸名称与全部参数字段，无需从头输入。
- **描述数据自动载入**：特征提取完成后（或复用已有描述文件时），生成的 `descriptionData/<name>.json` 会自动加载进主窗口，无需手动重新导入即可查看仿真。
- **穿线搜索实时进度条**：GUI 实时解析 trainer 输出的 `[TRAINER]` 日志——状态栏显示搜索进度，进度条按剪枝/步数比例推进，搜索结束时归位 100%。
- **折痕层逐单元导出进度**：STL 导出中的折痕层生成改为逐单元计算并逐步上报进度（`calculateTriPlaneForSingleCrease` / `outputCreaseDxf`），消除了原先 40%–50% 区间进度条冻结的"盲区"。
- 在 `trainer.py` 与 `phys_sim25.py` 中新增 **`miura-EA` 示例配置**（带 −Z 重力的 EA 系统）。

### Performance / 性能

**English**

- **STL export is now O(N) instead of O(N²)**: `StlMaker` internally accumulates STL text in a chunk list behind the unchanged `self.s` str interface (with a join cache so trailing `s += 'endsolid\n'` appends stay O(1)), `addInfoToStlFile` uses f-strings with an optional streaming `out=` path, and `addSpace` appends one chunk instead of looping per space. A 14,208-facet model now exports in ~0.04 s (previously ~15 s, ≈340× faster). The progress-bar stall around 33%–39% is eliminated.

**中文**

- **STL 导出从 O(N²) 优化为 O(N)**：`StlMaker` 在保持 `self.s` 字符串接口不变的前提下，内部改用 chunk 列表累积文本（并维护 join 缓存，使末尾 `s += 'endsolid\n'` 仍为 O(1) 追加）；`addInfoToStlFile` 改用 f-string 并支持 `out=` 流式输出；`addSpace` 由逐空格循环改为单次追加。14208 个 facet 的模型导出从约 15 s 降至约 0.04 s（约 340 倍加速），进度条 33%–39% 附近的卡顿彻底消除。

### Changed / 变更

**English**

- `border_nobias` border penalty is now a configurable attribute `border_nobias_penalty` (default `1e-1`, previously a hardcoded `1e-3`).
- GUI wording: "string(s)" → "tendon(s)"; "TSA candidator" → "actuation end".
- `trainer.py` console output is prefixed with `[TRAINER]` and cleaned up (noisy per-case prints removed/commented).
- `phys_sim25.py`: simulation timestep count `T = 24` extracted to a module-level constant; default `STROKE_PERCENT` 0.5 → 0.75; default GGUI window 1600×900 → 1280×720.
- `appendSimulationAngles` accepts an optional `path` argument (programmatic import without a file dialog).
- `environment.yml` / `requirement.txt`: installation instructions simplified.

**中文**

- `border_nobias` 边界惩罚改为可配置属性 `border_nobias_penalty`（默认 `1e-1`，原为硬编码 `1e-3`）。
- GUI 措辞统一："string(s)" → "tendon(s)"，"TSA candidator" → "actuation end"（驱动端）。
- `trainer.py` 控制台输出统一加 `[TRAINER]` 前缀，并清理了冗余的逐用例打印。
- `phys_sim25.py`：仿真时间步数 `T = 24` 提取为模块级常量；默认 `STROKE_PERCENT` 由 0.5 改为 0.75；默认 GGUI 窗口由 1600×900 改为 1280×720。
- `appendSimulationAngles` 支持可选 `path` 参数（无需文件对话框即可程序化导入）。
- `environment.yml` / `requirement.txt`：安装说明精简。

### Dependencies / 依赖

- Added `scipy` to `environment.yml` and `requirement.txt` (used by `logic.py` for interpolation and Gaussian smoothing of experiment data).
- 新增依赖 `scipy`（`logic.py` 用于实验数据的插值与高斯平滑）。

### Documentation / 文档

- Fixed the quick-start guide: **File ▸ Open file...** opens packed-data design files (`packedImport/*.json`) only — `descriptionData/` holds simulator system-description files that cannot be opened there.
- 修正快速开始指引：**File ▸ Open file...** 只能打开 packed-data 设计文件（`packedImport/*.json`）—— `descriptionData/` 下是仿真器系统描述文件，不能通过该入口打开。
- Runtime output directories (`cdfResult/`, `dxfResult/`, `legacy/`, `pdf/`, `physResult/`, `stlResult/`, `threadingResult/`) are now shipped via `.gitkeep` placeholders so a fresh clone has them ready (code writes results directly into them); their runtime contents remain git-ignored.
- 运行时输出目录（`cdfResult/`、`dxfResult/`、`legacy/`、`pdf/`、`physResult/`、`stlResult/`、`threadingResult/`）通过 `.gitkeep` 占位文件随仓库分发，fresh clone 后即可直接使用（代码会直接向其中写入结果）；运行期生成的内容仍被 git 忽略。
- Documented the save-and-resume workflow: **File ▸ Save result...** (`Ctrl+S`) packs the entire current design into a packed-data JSON that **File ▸ Open file...** reopens directly, so users no longer have to restart from a DXF/KL import every session. Added to Quick start (step 4), the data-formats section, and the directory table of both READMEs.
- 补充「保存并恢复设计」工作流文档：**File ▸ Save result...**（`Ctrl+S`）可将当前完整设计打包为 packed-data JSON，之后用 **File ▸ Open file...** 直接打开继续编辑，用户无需每次会话都从 DXF/KL 导入重新开始。已写入两份 README 的快速开始（第 4 步）、数据格式章节与目录表。

---

## [3.0.0] — 2026-09 (initial release / 首次发布)

- Initial open-source release of PyGamiX: integrated CAD / CAM / simulation workbench for engineering origami with automated tendon-threading design.
- PyGamiX 首次开源发布：面向工程折纸的一体化 CAD / CAM / 仿真软件平台，支持腱驱动折纸的穿线方案自动规划。
