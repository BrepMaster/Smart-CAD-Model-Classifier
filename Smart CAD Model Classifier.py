"""
CAD 模型智能分类工具 - PyQt5 图形界面版（十档文件大小分类）
支持格式：.step / .stp / .stl / .obj / .3mf
分类策略：颜色、文件大小（十档精细划分）、格式、组合策略

功能：
- 拖拽文件夹到输入框
- 日志导出为文件
- 快捷帮助与详细说明
- 界面美化，处理期间锁定设置

依赖安装：
    pip install PyQt5 pythonocc-core numpy trimesh
"""

import os
import sys
import shutil
import traceback
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QRadioButton,
    QButtonGroup, QProgressBar, QTextEdit, QGroupBox, QMessageBox,
    QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor, QDragEnterEvent, QDropEvent

# ---------- OCC 相关导入 ----------
try:
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Quantity import Quantity_Color
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import (
        XCAFDoc_DocumentTool,
        XCAFDoc_ColorSurf,
        XCAFDoc_ColorGen,
        XCAFDoc_ColorCurv
    )
    from OCC.Core.TDF import TDF_LabelSequence
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Reader
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

# 可选依赖
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# ---------- 工具函数 ----------
def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    if rgb is None:
        return "#CCCCCC"
    r = int(max(0, min(1, rgb[0])) * 255)
    g = int(max(0, min(1, rgb[1])) * 255)
    b = int(max(0, min(1, rgb[2])) * 255)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# ---------- 模型加载器 ----------
class ModelLoader:
    SUPPORTED_FORMATS = ['.step', '.stp', '.stl', '.obj', '.3mf']

    @staticmethod
    def load_shape(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.step', '.stp']:
            return ModelLoader._load_step(file_path)
        elif ext == '.stl':
            return ModelLoader._load_stl_occ(file_path)
        elif ext == '.obj':
            return ModelLoader._load_obj_occ(file_path)
        elif ext == '.3mf':
            return None
        return None

    @staticmethod
    def load_mesh(file_path: str):
        if not HAS_TRIMESH:
            return None
        try:
            return trimesh.load(file_path, force='mesh')
        except Exception:
            return None

    @staticmethod
    def _load_step(file_path: str):
        reader = STEPControl_Reader()
        if not reader.ReadFile(file_path):
            return None
        reader.TransferRoots()
        return reader.OneShape()

    @staticmethod
    def _load_stl_occ(file_path: str):
        reader = StlAPI_Reader()
        shape = reader.Read(file_path)
        return shape if not shape.IsNull() else None

    @staticmethod
    def _load_obj_occ(file_path: str):
        from OCC.Core.RWObj import RWObj_CafReader
        app = XCAFApp_Application.GetApplication()
        doc = TDocStd_Document(TCollection_ExtendedString("OBJ-Doc"))
        app.InitDocument(doc)
        reader = RWObj_CafReader()
        reader.SetDocument(doc)
        if not reader.ReadFile(file_path):
            return None
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        labels = TDF_LabelSequence()
        shape_tool.GetShapes(labels)
        if labels.Length() > 0:
            return shape_tool.GetShape(labels.Value(1))
        return None

# ---------- 分类策略 ----------
class ClassificationStrategy(ABC):
    @abstractmethod
    def get_category(self, file_path: str) -> str:
        pass
    def get_display_name(self, category: str) -> str:
        return category

class ColorStrategy(ClassificationStrategy):
    def __init__(self, use_color_names=True):
        self.use_color_names = use_color_names
        self.color_name_map = {
            "#FF0000": "红色", "#00FF00": "绿色", "#0000FF": "蓝色",
            "#FFFF00": "黄色", "#FF00FF": "品红", "#00FFFF": "青色",
            "#FFFFFF": "白色", "#000000": "黑色", "#808080": "灰色",
            "#FFA500": "橙色", "#800080": "紫色", "#A52A2A": "棕色",
            "#FFC0CB": "粉色",
        }

    def get_category(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        rgb = None
        if ext in ['.step', '.stp']:
            rgb = self._extract_step_color(file_path)
        elif ext == '.stl':
            rgb = self._extract_stl_color(file_path)
        elif ext == '.obj':
            rgb = self._extract_obj_color(file_path)
        elif ext == '.3mf':
            rgb = self._extract_3mf_color(file_path)
        if rgb is None:
            return "无颜色信息"
        hex_color = rgb_to_hex(rgb)
        if self.use_color_names:
            return self._closest_color_name(hex_color)
        return hex_color

    def _extract_step_color(self, file_path: str):
        app = XCAFApp_Application.GetApplication()
        doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        app.InitDocument(doc)
        reader = STEPCAFControl_Reader()
        reader.SetColorMode(True)
        reader.SetNameMode(True)
        reader.SetLayerMode(True)
        if not reader.ReadFile(file_path) or not reader.Transfer(doc):
            return None
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
        labels = TDF_LabelSequence()
        shape_tool.GetFreeShapes(labels)
        color_area_map = defaultdict(float)
        for i in range(1, labels.Length() + 1):
            label = labels.Value(i)
            shape = shape_tool.GetShape(label)
            if shape.IsNull():
                continue
            base_rgb = None
            if color_tool.IsSet(label, XCAFDoc_ColorSurf):
                c = Quantity_Color()
                if color_tool.GetColor(label, XCAFDoc_ColorSurf, c):
                    base_rgb = (c.Red(), c.Green(), c.Blue())
            elif color_tool.IsSet(label, XCAFDoc_ColorGen):
                c = Quantity_Color()
                if color_tool.GetColor(label, XCAFDoc_ColorGen, c):
                    base_rgb = (c.Red(), c.Green(), c.Blue())
            elif color_tool.IsSet(label, XCAFDoc_ColorCurv):
                c = Quantity_Color()
                if color_tool.GetColor(label, XCAFDoc_ColorCurv, c):
                    base_rgb = (c.Red(), c.Green(), c.Blue())
            if base_rgb is None:
                continue
            topo_exp = TopologyExplorer(shape)
            for face in topo_exp.faces():
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)
                area = props.Mass()
                color_area_map[base_rgb] += area
        if not color_area_map:
            return None
        return max(color_area_map, key=color_area_map.get)

    def _extract_stl_color(self, file_path: str):
        if not HAS_TRIMESH:
            return None
        try:
            mesh = trimesh.load(file_path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                return None
            avg_normal = np.mean(mesh.face_normals, axis=0)
            r = (avg_normal[0] + 1.0) / 2.0
            g = (avg_normal[1] + 1.0) / 2.0
            b = (avg_normal[2] + 1.0) / 2.0
            return (r, g, b)
        except:
            return None

    def _extract_obj_color(self, file_path: str):
        if not HAS_TRIMESH:
            return None
        try:
            mesh = trimesh.load(file_path, force='mesh')
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                avg = np.mean(mesh.visual.vertex_colors[:, :3], axis=0)
                return tuple(avg / 255.0)
            elif hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                mat = mesh.visual.material
                if hasattr(mat, 'main_color'):
                    return (mat.main_color[0]/255, mat.main_color[1]/255, mat.main_color[2]/255)
        except:
            pass
        return None

    def _extract_3mf_color(self, file_path: str):
        if not HAS_TRIMESH:
            return None
        try:
            mesh = trimesh.load(file_path, force='mesh')
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                avg = np.mean(mesh.visual.vertex_colors[:, :3], axis=0)
                return tuple(avg / 255.0)
        except:
            pass
        return None

    def _closest_color_name(self, hex_color: str) -> str:
        target = hex_to_rgb(hex_color)
        min_dist = float('inf')
        closest = hex_color
        for hex_val, name in self.color_name_map.items():
            c = hex_to_rgb(hex_val)
            dist = sum((a-b)**2 for a,b in zip(target, c))
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def get_display_name(self, category: str) -> str:
        if category.startswith('#'):
            return f"颜色_{category}"
        return category

class FileSizeStrategy(ClassificationStrategy):
    def __init__(self, thresholds=None):
        # 十档精细划分（单位：字节）
        self.thresholds = thresholds or {
            "微小文件_小于100KB": 100 * 1024,                     # 100 KB
            "小文件_100-500KB": 500 * 1024,                       # 500 KB
            "中小文件_500KB-1MB": 1 * 1024 * 1024,                # 1 MB
            "中文件_1-5MB": 5 * 1024 * 1024,                      # 5 MB
            "中大文件_5-10MB": 10 * 1024 * 1024,                  # 10 MB
            "较大文件_10-20MB": 20 * 1024 * 1024,                 # 20 MB
            "大文件_20-50MB": 50 * 1024 * 1024,                   # 50 MB
            "很大文件_50-100MB": 100 * 1024 * 1024,               # 100 MB
            "超大文件_100-500MB": 500 * 1024 * 1024,              # 500 MB
        }

    def get_category(self, file_path: str) -> str:
        size = os.path.getsize(file_path)
        for cat, limit in sorted(self.thresholds.items(), key=lambda x: x[1]):
            if size < limit:
                return cat
        return "巨型文件_大于500MB"

class FormatStrategy(ClassificationStrategy):
    def __init__(self):
        self.format_groups = {
            "STEP": ['.step','.stp'],
            "STL": ['.stl'],
            "OBJ": ['.obj'],
            "3MF": ['.3mf'],
        }
    def get_category(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        for name, exts in self.format_groups.items():
            if ext in exts:
                return name
        return "其他格式"

class CompositeStrategy(ClassificationStrategy):
    def __init__(self, strategies: List[ClassificationStrategy], separator=os.sep):
        self.strategies = strategies
        self.separator = separator
    def get_category(self, file_path: str) -> str:
        return self.separator.join(s.get_category(file_path) for s in self.strategies)
    def get_display_name(self, category: str) -> str:
        return category

# ---------- 工作线程 ----------
class ClassifyWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, strategy, copy_mode):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.strategy = strategy
        self.copy_mode = copy_mode
        self.supported_extensions = ['.step','.stp','.stl','.obj','.3mf']

    def run(self):
        try:
            files = self._get_files()
            total = len(files)
            self.log_signal.emit(f"找到 {total} 个支持的 CAD 文件")
            self.status_signal.emit(f"准备处理 {total} 个文件")
            if total == 0:
                self.finished_signal.emit({})
                return

            os.makedirs(self.output_dir, exist_ok=True)
            stats = defaultdict(int)

            for idx, file_path in enumerate(files):
                file_name = os.path.basename(file_path)
                self.log_signal.emit(f"处理: {file_name}")
                self.status_signal.emit(f"正在处理 {idx+1}/{total}: {file_name}")
                try:
                    category = self.strategy.get_category(file_path)
                    display = self.strategy.get_display_name(category)
                except Exception as e:
                    self.log_signal.emit(f"  -> 分类失败: {e}")
                    category = "处理失败"
                    display = "处理失败"

                target_dir = os.path.join(self.output_dir, display)
                target_path = os.path.join(target_dir, file_name)

                os.makedirs(target_dir, exist_ok=True)
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
                        counter += 1
                    target_path = os.path.join(target_dir, f"{base}_{counter}{ext}")

                try:
                    if self.copy_mode:
                        shutil.copy2(file_path, target_path)
                        self.log_signal.emit(f"  -> 复制到 {display}/")
                    else:
                        shutil.move(file_path, target_path)
                        self.log_signal.emit(f"  -> 移动到 {display}/")
                except Exception as e:
                    self.log_signal.emit(f"  -> 文件操作失败: {e}")
                    continue

                stats[display] += 1
                self.progress_signal.emit(idx + 1, total)

            self.log_signal.emit("处理完成！")
            self.status_signal.emit("处理完成")
            self.finished_signal.emit(stats)

        except Exception as e:
            self.error_signal.emit(traceback.format_exc())
            self.status_signal.emit("处理出错")

    def _get_files(self):
        files = []
        for root, _, filenames in os.walk(self.input_dir):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.supported_extensions:
                    files.append(os.path.join(root, f))
        return files

# ---------- 可拖拽的输入框 ----------
class DragDropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)

# ---------- 主窗口 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD 模型智能分类工具")
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.init_menu()
        self.init_statusbar()
        self.apply_stylesheet()
        self.setup_tooltips()
        self.worker = None

    def apply_stylesheet(self):
        style = """
        QMainWindow { background-color: #f5f7fa; }
        QWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 10pt; }
        QLabel { color: #2c3e50; font-weight: 500; }
        QLineEdit, QComboBox, QTextEdit {
            background-color: #ffffff;
            border: 1px solid #dcdfe6;
            border-radius: 6px;
            padding: 6px 8px;
            selection-background-color: #409eff;
        }
        QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border: 1px solid #409eff; }
        QLineEdit:disabled, QComboBox:disabled, QTextEdit:disabled {
            background-color: #f0f2f5;
            color: #a0a4ab;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #dcdfe6;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #606266;
            margin-right: 5px;
        }
        QComboBox:disabled::down-arrow { border-top-color: #a0a4ab; }
        QPushButton {
            background-color: #409eff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #66b1ff; }
        QPushButton:pressed { background-color: #337ecc; }
        QPushButton:disabled { background-color: #c0c4cc; }
        QRadioButton {
            spacing: 8px;
            color: #2c3e50;
        }
        QRadioButton:disabled { color: #a0a4ab; }
        QRadioButton::indicator {
            width: 16px; height: 16px;
            border-radius: 8px;
            border: 2px solid #dcdfe6;
            background-color: #ffffff;
        }
        QRadioButton::indicator:checked {
            border: 4px solid #409eff;
            background-color: #ffffff;
        }
        QRadioButton::indicator:checked:disabled {
            border-color: #a0cfff;
        }
        QProgressBar {
            border: none;
            border-radius: 10px;
            background-color: #e9ecef;
            text-align: center;
            color: #2c3e50;
            font-weight: bold;
            height: 20px;
        }
        QProgressBar::chunk { background-color: #67c23a; border-radius: 10px; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #dcdfe6;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #409eff;
        }
        QTextEdit {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: "Consolas", "Courier New", monospace;
            border: 1px solid #3c3c3c;
        }
        QStatusBar { background-color: #ecf0f1; color: #2c3e50; }
        """
        self.setStyleSheet(style)

    def setup_tooltips(self):
        self.input_edit.setToolTip("输入包含CAD文件的文件夹路径。\n支持拖拽文件夹到此处。")
        self.output_edit.setToolTip("分类后的文件将存放到此目录下。")
        self.strategy_combo.setToolTip(
            "选择分类依据：\n"
            "• 颜色：提取模型主色进行分类\n"
            "• 文件大小：按十档精细划分（100KB～500MB）\n"
            "• 文件格式：按扩展名分类\n"
            "• 组合策略：先格式再颜色"
        )
        self.copy_radio.setToolTip("复制文件到输出目录，原文件保留。")
        self.move_radio.setToolTip("移动文件到输出目录，原文件将被删除。")
        self.export_log_btn.setToolTip("将当前日志保存为文本文件。")
        self.help_btn.setToolTip("显示快速使用指南。")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 标题行（包含帮助按钮）
        title_layout = QHBoxLayout()
        title_label = QLabel("<h2>CAD 模型智能分类工具</h2>")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.help_btn = QPushButton("❓ 帮助")
        self.help_btn.setFixedWidth(100)
        self.help_btn.clicked.connect(self.show_quick_help)
        title_layout.addWidget(self.help_btn)
        main_layout.addLayout(title_layout)

        # 输入目录行
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)
        input_layout.addWidget(QLabel("输入目录:"))
        self.input_edit = DragDropLineEdit()
        self.input_edit.setPlaceholderText("选择或拖拽文件夹到此处...")
        input_layout.addWidget(self.input_edit)
        self.btn_input = QPushButton("浏览")
        self.btn_input.clicked.connect(self.browse_input)
        self.btn_input.setFixedWidth(80)
        input_layout.addWidget(self.btn_input)
        main_layout.addLayout(input_layout)

        # 输出目录行
        output_layout = QHBoxLayout()
        output_layout.setSpacing(8)
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择分类后存放的文件夹...")
        output_layout.addWidget(self.output_edit)
        self.btn_output = QPushButton("浏览")
        self.btn_output.clicked.connect(self.browse_output)
        self.btn_output.setFixedWidth(80)
        output_layout.addWidget(self.btn_output)
        main_layout.addLayout(output_layout)

        # 分类策略与操作模式
        setting_layout = QHBoxLayout()
        setting_layout.setSpacing(20)
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("分类策略:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["按颜色分类", "按文件大小分类", "按文件格式分类", "组合策略(格式→颜色)"])
        self.strategy_combo.setMinimumWidth(200)
        strategy_layout.addWidget(self.strategy_combo)
        setting_layout.addLayout(strategy_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("操作模式:"))
        self.copy_radio = QRadioButton("复制文件")
        self.move_radio = QRadioButton("移动文件")
        self.copy_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.copy_radio)
        mode_group.addButton(self.move_radio)
        mode_layout.addWidget(self.copy_radio)
        mode_layout.addWidget(self.move_radio)
        setting_layout.addLayout(mode_layout)
        setting_layout.addStretch()
        main_layout.addLayout(setting_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 日志分组框
        log_group = QGroupBox("运行日志")
        log_group_layout = QVBoxLayout(log_group)
        log_group_layout.setContentsMargins(12, 18, 12, 12)

        log_toolbar = QHBoxLayout()
        log_toolbar.addStretch()
        self.export_log_btn = QPushButton("导出日志")
        self.export_log_btn.setFixedWidth(100)
        self.export_log_btn.clicked.connect(self.export_log)
        log_toolbar.addWidget(self.export_log_btn)
        log_group_layout.addLayout(log_toolbar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        log_group_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group, 1)

        # 开始按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("开始分类")
        self.start_btn.setFixedSize(140, 36)
        self.start_btn.clicked.connect(self.start_classify)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 依赖检查提示
        if not HAS_OCC:
            self.log_text.append("<font color='#f56c6c'><b>错误: 未安装 pythonocc-core，请先安装: pip install pythonocc-core</b></font>")
            self.start_btn.setEnabled(False)
        if not HAS_NUMPY or not HAS_TRIMESH:
            self.log_text.append("<font color='#e6a23c'>提示: 未安装 numpy/trimesh，部分格式的颜色提取功能将受限</font>")

        # 保存所有需要在处理期间禁用的控件列表
        self.control_widgets = [
            self.input_edit, self.btn_input,
            self.output_edit, self.btn_output,
            self.strategy_combo,
            self.copy_radio, self.move_radio,
            self.help_btn, self.export_log_btn
        ]

    def set_controls_enabled(self, enabled):
        """启用/禁用所有输入控件"""
        for widget in self.control_widgets:
            widget.setEnabled(enabled)

    def init_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def init_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("帮助(&H)")
        help_action = help_menu.addAction("使用说明(&U)")
        help_action.triggered.connect(self.show_help)
        quick_help_action = help_menu.addAction("快速帮助(&Q)")
        quick_help_action.triggered.connect(self.show_quick_help)
        about_action = help_menu.addAction("关于(&A)")
        about_action.triggered.connect(self.show_about)

    def show_quick_help(self):
        msg = """
        <h3>📘 快速使用指南</h3>
        <ul>
            <li><b>1. 选择目录</b> —— 指定输入（待分类文件）和输出（结果存放）文件夹。</li>
            <li><b>2. 选择策略</b> —— 决定分类方式（颜色、大小、格式或组合）。</li>
            <li><b>3. 操作模式</b> —— 复制（保留原文件）或移动（移除原文件）。</li>
            <li><b>4. 开始分类</b> —— 点击按钮，等待进度完成。</li>
        </ul>
        <p><b>💡 小技巧：</b></p>
        <ul>
            <li>可直接将文件夹拖拽至输入框。</li>
            <li>日志可导出为文本文件保存。</li>
        </ul>
        <p>更多详情请查看菜单栏 <b>帮助 → 使用说明</b>。</p>
        """
        QMessageBox.information(self, "快速帮助", msg)

    def show_help(self):
        help_text = """
        <h2>CAD 模型智能分类工具</h2>
        <p><b>功能：</b>根据多种策略自动将 CAD 模型文件分类到不同文件夹。</p>

        <h3>支持的文件格式</h3>
        <ul>
            <li>.step / .stp（STEP 格式）</li>
            <li>.stl（STL 格式）</li>
            <li>.obj（OBJ 格式）</li>
            <li>.3mf（3MF 格式）</li>
        </ul>
        <p>不区分大小写，支持递归子目录。</p>

        <h3>分类策略说明</h3>
        <ul>
            <li><b>按颜色分类：</b>提取模型主色（面积加权或材质颜色），映射为中文颜色名或十六进制色码。</li>
            <li><b>按文件大小分类（十档精细版）：</b>
                <ul>
                    <li>微小文件：小于100KB</li>
                    <li>小文件：100KB～500KB</li>
                    <li>中小文件：500KB～1MB</li>
                    <li>中文件：1MB～5MB</li>
                    <li>中大文件：5MB～10MB</li>
                    <li>较大文件：10MB～20MB</li>
                    <li>大文件：20MB～50MB</li>
                    <li>很大文件：50MB～100MB</li>
                    <li>超大文件：100MB～500MB</li>
                    <li>巨型文件：大于500MB</li>
                </ul>
            </li>
            <li><b>按文件格式分类：</b>按扩展名归类为 STEP、STL、OBJ、3MF 文件夹。</li>
            <li><b>组合策略（格式→颜色）：</b>先按格式分类，再在格式文件夹内按颜色分类。</li>
        </ul>

        <h3>新增功能</h3>
        <ul>
            <li><b>拖拽支持：</b>可将文件夹直接拖入输入框。</li>
            <li><b>日志导出：</b>保存运行日志为文本文件。</li>
            <li><b>快捷帮助：</b>点击右上角“帮助”按钮或悬停查看提示。</li>
        </ul>

        <h3>依赖环境</h3>
        <ul>
            <li><b>必需：</b>pythonocc-core</li>
            <li><b>推荐：</b>numpy, trimesh（增强 STL/OBJ/3MF 颜色提取）</li>
        </ul>
        """
        QMessageBox.information(self, "使用说明", help_text)

    def show_about(self):
        about_text = """
        <h3>CAD 模型智能分类工具</h3>
        <p>版本 3.0 (十档文件大小分类)</p>
        <p>基于 PythonOCC 和 PyQt5 开发。</p>
        """
        QMessageBox.about(self, "关于", about_text)

    def browse_input(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_edit.setText(dir_path)

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_edit.setText(dir_path)

    def export_log(self):
        if not self.log_text.toPlainText().strip():
            QMessageBox.information(self, "提示", "当前日志为空，无需导出。")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存日志", "cad_classify_log.txt", "文本文件 (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "成功", f"日志已保存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存日志失败: {e}")

    def start_classify(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "警告", "请指定输入和输出目录")
            return
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", "输入目录不存在")
            return

        idx = self.strategy_combo.currentIndex()
        if idx == 0:
            strategy = ColorStrategy(use_color_names=True)
        elif idx == 1:
            strategy = FileSizeStrategy()
        elif idx == 2:
            strategy = FormatStrategy()
        elif idx == 3:
            strategy = CompositeStrategy([FormatStrategy(), ColorStrategy(use_color_names=True)])
        else:
            strategy = ColorStrategy(use_color_names=True)

        copy_mode = self.copy_radio.isChecked()

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法创建输出目录:\n{e}")
            return

        # 清空日志并准备UI
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.set_controls_enabled(False)
        self.status_bar.showMessage("正在初始化...")

        self.worker = ClassifyWorker(input_dir, output_dir, strategy, copy_mode)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.status_signal.connect(self.status_bar.showMessage)
        self.worker.start()

    def append_log(self, msg):
        if msg.startswith("  ->"):
            self.log_text.append(f"<font color='#67c23a'>{msg}</font>")
        elif "失败" in msg or "错误" in msg:
            self.log_text.append(f"<font color='#f56c6c'>{msg}</font>")
        elif "完成" in msg:
            self.log_text.append(f"<font color='#409eff'><b>{msg}</b></font>")
        else:
            self.log_text.append(msg)
        self.log_text.moveCursor(QTextCursor.End)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_finished(self, stats):
        self.start_btn.setEnabled(True)
        self.set_controls_enabled(True)
        self.log_text.append("<br><font color='#409eff'><b>统计结果:</b></font>")
        for cat, count in sorted(stats.items()):
            self.log_text.append(f"  {cat}: {count} 个文件")
        self.status_bar.showMessage("处理完成")
        QMessageBox.information(self, "完成", "分类处理完成！")

    def on_error(self, err_msg):
        self.start_btn.setEnabled(True)
        self.set_controls_enabled(True)
        self.log_text.append(f"<font color='#f56c6c'><b>发生错误:</b><br>{err_msg}</font>")
        self.status_bar.showMessage("处理出错")
        QMessageBox.critical(self, "错误", "处理过程中发生错误，详见日志")

# ---------- 入口 ----------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()