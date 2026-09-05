"""
Threading Design Settings Dialog

Custom QDialog for configuring the automated threading design process.
Organized into three groups:
  1. Design Info        - Origami name
  2. Simulation Properties - Height, control mode, ground, gravity, sim time, etc.
  3. Search Algorithm Properties - Tendon count, threads, mask crease, constraints, etc.

Conditional fields:
  - Ground Friction (K) shown only when Ground Enabled (E) = 1
  - Controller Type (L) and Actuator Stroke Percent (S) shown only when Control Mode (D) = 1
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QRadioButton, QButtonGroup, QDialogButtonBox, QLabel, QWidget
)
from PyQt5.QtCore import Qt


# Gravity direction options: index -> (display_label, string_repr, gravity_flag)
GRAVITY_OPTIONS = [
    ("0 (none)", "0",  0),   # default
    ("+x",      "+x",  1),
    ("+y",      "+y",  2),
    ("+z",      "+z",  3),
    ("-x",      "-x",  4),
    ("-y",      "-y",  5),
    ("-z",      "-z",  6),
]
DEFAULT_GRAVITY_INDEX = 0  # "0 (none)"


class ThreadingDesignDialog(QDialog):
    """Dialog for collecting threading design automation settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threading Design Settings")
        self.setMinimumWidth(520)
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # === Group 1: Design Info ===
        group_design = QGroupBox("Design Info")
        form_design = QFormLayout(group_design)
        form_design.setLabelAlignment(Qt.AlignRight)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. bird, miura-ori-16, ...")
        form_design.addRow("Origami Name:", self.edit_name)
        main_layout.addWidget(group_design)

        # === Group 2: Simulation Properties ===
        group_sim = QGroupBox("Simulation Properties")
        form_sim = QFormLayout(group_sim)
        form_sim.setLabelAlignment(Qt.AlignRight)

        # C: Structure Height (mm)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.01, 99999.0)
        self.spin_height.setValue(1.0)
        self.spin_height.setSuffix(" mm")
        self.spin_height.setDecimals(2)
        self.spin_height.setToolTip("Initial height of the origami structure")
        form_sim.addRow("Structure Height:", self.spin_height)

        # D: Control Mode
        self.combo_control = QComboBox()
        self.combo_control.addItem("0 - Structure Folding", 0)
        self.combo_control.addItem("1 - Robot Actuation", 1)
        self.combo_control.setToolTip("0 = structure folding, 1 = robot actuation")
        form_sim.addRow("Control Mode:", self.combo_control)

        # L: Controller Type (conditional: only when D=1)
        self.label_controller = QLabel("Controller Type:")
        self.combo_controller = QComboBox()
        self.combo_controller.addItem("1", 1)
        self.combo_controller.addItem("2", 2)
        self.combo_controller.setToolTip("Controller hardware type (1 or 2)")
        form_sim.addRow(self.label_controller, self.combo_controller)

        # S: Stroke Percent (conditional: only when D=1)
        self.label_stroke = QLabel("Actuator Stroke Percent:")
        self.spin_stroke = QDoubleSpinBox()
        self.spin_stroke.setRange(0.00, 1.00)
        self.spin_stroke.setValue(0.75)
        self.spin_stroke.setDecimals(2)
        self.spin_stroke.setSingleStep(0.05)
        self.spin_stroke.setToolTip("Actuator stroke as a fraction of full range (0.00 ~ 1.00)")
        form_sim.addRow(self.label_stroke, self.spin_stroke)

        # E: Ground Enabled
        self.combo_ground = QComboBox()
        self.combo_ground.addItem("0 - Disabled", 0)
        self.combo_ground.addItem("1 - Enabled", 1)
        self.combo_ground.setToolTip("Enable/disable ground collision")
        form_sim.addRow("Ground Enabled:", self.combo_ground)

        # K: Ground Friction Coefficient (conditional: only when E=1)
        self.label_friction = QLabel("Ground Friction Coeff:")
        self.spin_friction = QDoubleSpinBox()
        self.spin_friction.setRange(0.0, 5.0)
        self.spin_friction.setValue(0.3)
        self.spin_friction.setDecimals(2)
        self.spin_friction.setSingleStep(0.1)
        self.spin_friction.setToolTip("Friction coefficient between origami and ground")
        form_sim.addRow(self.label_friction, self.spin_friction)

        # F: Gravity Direction (radio buttons, 7 options)
        gravity_widget = QWidget()
        gravity_grid = QGridLayout(gravity_widget)
        gravity_grid.setContentsMargins(0, 0, 0, 0)
        gravity_grid.setSpacing(6)
        self.gravity_group = QButtonGroup(self)
        for i, (label, _str, _flag) in enumerate(GRAVITY_OPTIONS):
            rb = QRadioButton(label)
            self.gravity_group.addButton(rb, i)
            row, col = divmod(i, 4)
            gravity_grid.addWidget(rb, row, col)
        # Set default
        default_btn = self.gravity_group.button(DEFAULT_GRAVITY_INDEX)
        if default_btn:
            default_btn.setChecked(True)
        form_sim.addRow("Gravity Direction:", gravity_widget)

        # G: Simulation Time (s)
        self.spin_sim_time = QDoubleSpinBox()
        self.spin_sim_time.setRange(0.1, 9999.0)
        self.spin_sim_time.setValue(20.0)
        self.spin_sim_time.setSuffix(" s")
        self.spin_sim_time.setDecimals(1)
        self.spin_sim_time.setToolTip("Maximum simulation duration")
        form_sim.addRow("Simulation Time:", self.spin_sim_time)

        # I: Extra Tendon Length (mm)
        self.spin_extra_len = QDoubleSpinBox()
        self.spin_extra_len.setRange(0.0, 99999.0)
        self.spin_extra_len.setValue(0.0)
        self.spin_extra_len.setSuffix(" mm")
        self.spin_extra_len.setDecimals(2)
        self.spin_extra_len.setToolTip("Additional length added to each tendon")
        form_sim.addRow("Extra Tendon Length:", self.spin_extra_len)

        # J: Material Type
        self.combo_material = QComboBox()
        self.combo_material.addItem("0 - PLA+TPU", 0)
        self.combo_material.addItem("1 - PU foam+TPU", 1)
        self.combo_material.setToolTip("0 = PLA+TPU, 1 = PU foam+TPU (affects density/mass)")
        form_sim.addRow("Material Type:", self.combo_material)

        main_layout.addWidget(group_sim)

        # === Group 3: Search Algorithm Properties ===
        group_search = QGroupBox("Search Algorithm Properties")
        form_search = QFormLayout(group_search)
        form_search.setLabelAlignment(Qt.AlignRight)

        # B: Min Tendon Count
        self.spin_min_tendon = QSpinBox()
        self.spin_min_tendon.setRange(1, 99)
        self.spin_min_tendon.setValue(1)
        self.spin_min_tendon.setToolTip("Minimum number of tendons to search for")
        form_search.addRow("Min Tendon Count:", self.spin_min_tendon)

        # H: Thread Count
        self.spin_thread = QSpinBox()
        self.spin_thread.setRange(1, 16)
        self.spin_thread.setValue(4)
        self.spin_thread.setToolTip("Number of parallel processes for search")
        form_search.addRow("Thread Count:", self.spin_thread)

        # N: Mask Crease Type
        self.combo_mask_crease = QComboBox()
        self.combo_mask_crease.addItem("0 - No", 0)
        self.combo_mask_crease.addItem("1 - Yes", 1)
        self.combo_mask_crease.setToolTip("If enabled, crease types (mountain/valley) are masked during search")
        form_search.addRow("Mask Crease Type:", self.combo_mask_crease)

        # M: Search Mode
        self.combo_search_mode = QComboBox()
        self.combo_search_mode.addItem("0 - Depth-weighted (more rollouts, slower)", 0)
        self.combo_search_mode.addItem("1 - Uniform (fewer rollouts, faster)", 1)
        self.combo_search_mode.setToolTip("0 = rollout count scales with remaining depth (more rollouts, slower step-in)\n1 = uniform rollout count (fewer rollouts, faster step-in)")
        form_search.addRow("Search Mode:", self.combo_search_mode)

        # O: EA Constraint - Initial Segment
        self.combo_ea_init = QComboBox()
        self.combo_ea_init.addItem("0 - No", 0)
        self.combo_ea_init.addItem("1 - Yes", 1)
        self.combo_ea_init.setCurrentIndex(1)  # default 1
        self.combo_ea_init.setToolTip("External actuation constraint on the initial segment")
        form_search.addRow("EA Constraint: Initial Segment:", self.combo_ea_init)

        # P: EA Constraint - Final Tip
        self.combo_ea_final = QComboBox()
        self.combo_ea_final.addItem("0 - No", 0)
        self.combo_ea_final.addItem("1 - Yes", 1)
        self.combo_ea_final.setCurrentIndex(1)  # default 1
        self.combo_ea_final.setToolTip("External actuation constraint on the final tip")
        form_search.addRow("EA Constraint: Final Tip:", self.combo_ea_final)

        # Q: Enable Work Criteria
        self.combo_work = QComboBox()
        self.combo_work.addItem("0 - No", 0)
        self.combo_work.addItem("1 - Yes", 1)
        self.combo_work.setCurrentIndex(1)  # default 1
        self.combo_work.setToolTip("If enabled, work/energy criteria are used in reward evaluation")
        form_search.addRow("Enable Work Criteria:", self.combo_work)

        main_layout.addWidget(group_search)

        # === Buttons ===
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # === Conditional field signals ===
        self.combo_control.currentIndexChanged.connect(self._update_conditional_fields)
        self.combo_ground.currentIndexChanged.connect(self._update_conditional_fields)
        self._update_conditional_fields()

    # ------------------------------------------------------------------ #
    #  Conditional visibility
    # ------------------------------------------------------------------ #

    def _update_conditional_fields(self):
        """Show/hide conditional fields based on Control Mode and Ground Enabled."""
        # L (Controller Type) and S (Stroke Percent) visible only when D=1
        show_controller = self.combo_control.currentData() == 1
        self.label_controller.setVisible(show_controller)
        self.combo_controller.setVisible(show_controller)
        self.label_stroke.setVisible(show_controller)
        self.spin_stroke.setVisible(show_controller)

        # K (Ground Friction) visible only when E=1
        show_friction = self.combo_ground.currentData() == 1
        self.label_friction.setVisible(show_friction)
        self.spin_friction.setVisible(show_friction)

        # Adjust dialog size to fit content
        self.adjustSize()

    # ------------------------------------------------------------------ #
    #  Settings retrieval
    # ------------------------------------------------------------------ #

    def get_settings(self):
        """
        Return a dictionary of all settings.

        Keys map to the parameter names used in trainer.py / phys_sim25.py:
          origami_name               -> A
          min_tendon_count           -> B
          structure_height           -> C  (mm)
          control_mode               -> D  (0 or 1)
          ground_enabled             -> E  (0 or 1)
          gravity_direction          -> F  (string: "+x", "-z", "0", ...)
          gravity_flag               -> F  (int: 0-6, for internal use)
          simulation_time            -> G  (s)
          thread_count               -> H
          extra_tendon_length        -> I  (mm)
          material_type              -> J  (0=PLA+TPU, 1=PU foam+TPU)
          mask_crease_type           -> M  (0 or 1)
          ea_constraint_initial_segment -> O  (0 or 1)
          ea_constraint_final_tip       -> P  (0 or 1)
          enable_work_criteria         -> Q  (0 or 1)

        Conditional keys (present only when relevant):
          ground_friction            -> K  (float, only when E=1)
          controller_type            -> L  (1 or 2, only when D=1)
          stroke_percent             -> S  (float 0.00~1.00, only when D=1)
        """
        gravity_idx = self.gravity_group.checkedId()
        if gravity_idx < 0:
            gravity_idx = DEFAULT_GRAVITY_INDEX
        _label, gravity_str, gravity_flag = GRAVITY_OPTIONS[gravity_idx]

        settings = {
            # A: Design info
            "origami_name": self.edit_name.text().strip() or "default",

            # B: Search - min tendon count
            "min_tendon_count": self.spin_min_tendon.value(),

            # C: Simulation - structure height (mm)
            "structure_height": self.spin_height.value(),

            # D: Simulation - control mode
            "control_mode": self.combo_control.currentData(),

            # E: Simulation - ground enabled
            "ground_enabled": self.combo_ground.currentData(),

            # F: Simulation - gravity direction
            "gravity_direction": gravity_str,
            "gravity_flag": gravity_flag,

            # G: Simulation - time (s)
            "simulation_time": self.spin_sim_time.value(),

            # H: Search - thread count
            "thread_count": self.spin_thread.value(),

            # I: Simulation - extra tendon length (mm)
            "extra_tendon_length": self.spin_extra_len.value(),

            # J: Simulation - material type
            "material_type": self.combo_material.currentData(),

            # M: Search - mask crease type
            "mask_crease_type": self.combo_mask_crease.currentData(),

            # M: Search - search mode
            "search_mode": self.combo_search_mode.currentData(),

            # O: Search - EA constraint initial segment
            "ea_constraint_initial_segment": self.combo_ea_init.currentData(),

            # P: Search - EA constraint final tip
            "ea_constraint_final_tip": self.combo_ea_final.currentData(),

            # Q: Search - enable work criteria
            "enable_work_criteria": self.combo_work.currentData(),
        }

        # Conditional: K (ground friction) only when E=1
        if settings["ground_enabled"] == 1:
            settings["ground_friction"] = self.spin_friction.value()

        # Conditional: L (controller type) and S (stroke percent) only when D=1
        if settings["control_mode"] == 1:
            settings["controller_type"] = self.combo_controller.currentData()
            settings["stroke_percent"] = self.spin_stroke.value()

        return settings
