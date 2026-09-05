from PyQt5.QtWidgets import QDialog

from gui.Ui_stl_dialog import Ui_Dialog

class StlSettingDialog(Ui_Dialog, QDialog):
    def __init__(self, parent=None) -> None:
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.ok = False
        self.disableConnection()
        self.disableBoard()
        self.buttonBox.accepted.connect(self.onAccepted)
        self.buttonBox.rejected.connect(self.onRejected)
        self.radioButton_both_bias.clicked.connect(self.enableConnection)
        self.radioButton_upper_bias.clicked.connect(self.disableConnection)
        self.radioButton_symmetry.clicked.connect(self.enableConnection)
        self.pushButton_double_materials.clicked.connect(self.recommandSettingOfDM)
        self.pushButton_regular.clicked.connect(self.recommandSettingOfRegular)
        self.checkBox_true.clicked.connect(self.onChooseConnection)
        self.checkBox_board_true.clicked.connect(self.onChooseBoard)

    def onAccepted(self):
        self.ok = True

    def onRejected(self):
        self.ok = False

    def getOK(self):
        return self.ok

    def getHeight(self):
        return self.doubleSpinBox_unit_height.value()
    
    def getBias(self):
        return self.doubleSpinBox_unit_bias.value()

    def getMethod(self):
        if self.radioButton_both_bias.isChecked():
            return "both_bias"
        elif self.radioButton_upper_bias.isChecked():
            return "upper_bias"
        elif self.radioButton_symmetry.isChecked():
            return "symmetry"

    def getHoleWidth(self):
        return self.doubleSpinBox_hole_width.value()

    def getHoleLength(self):
        return self.doubleSpinBox_hole_length.value()

    def getUnitId(self):
        return self.spinBox_unit_id.value()

    def getBoardHeight(self):
        return self.doubleSpinBox_board_height.value()
    
    def onChooseConnection(self):
        if self.checkBox_true.isChecked():
            self.label_hole_width.setVisible(True)
            self.label_hole_length.setVisible(True)
            self.doubleSpinBox_hole_length.setVisible(True)
            self.doubleSpinBox_hole_width.setVisible(True)
        else:
            self.label_hole_width.setVisible(False)
            self.label_hole_length.setVisible(False)
            self.doubleSpinBox_hole_length.setVisible(False)
            self.doubleSpinBox_hole_width.setVisible(False)

    def getConnectionNeeded(self):
        return self.checkBox_true.isChecked()
    
    def enableConnection(self):
        self.label_connections.setVisible(True)
        self.checkBox_true.setVisible(True)
        self.onChooseConnection()
    
    def disableConnection(self):
        self.label_connections.setVisible(False)
        self.checkBox_true.setVisible(False)
        self.label_hole_width.setVisible(False)
        self.label_hole_length.setVisible(False)
        self.doubleSpinBox_hole_length.setVisible(False)
        self.doubleSpinBox_hole_width.setVisible(False)

    def enableBoard(self):
        self.label_board_height.setVisible(True)
        self.doubleSpinBox_board_height.setVisible(True)

    def disableBoard(self):
        self.label_board_height.setVisible(False)
        self.doubleSpinBox_board_height.setVisible(False)

    def getBoardNeeded(self):
        return self.checkBox_board_true.isChecked()
    
    def onChooseBoard(self):
        if self.checkBox_board_true.isChecked():
            self.enableBoard()
        else:
            self.disableBoard()

    def recommandSettingOfDM(self):
        self.radioButton_both_bias.setChecked(True)
        self.checkBox_true.setChecked(True)
        self.enableConnection()
        self.doubleSpinBox_unit_height.setValue(self.doubleSpinBox_unit_bias.value() / 1.732)
        self.doubleSpinBox_hole_width.setValue(0.35)
        self.doubleSpinBox_hole_length.setValue(0.80)
    
    def recommandSettingOfRegular(self):
        self.radioButton_upper_bias.setChecked(True)
        self.checkBox_true.setChecked(False)
        self.disableConnection()
        self.doubleSpinBox_unit_height.setValue(self.doubleSpinBox_unit_bias.value() / 1.732)

    def setBiasAndLock(self, bias_val):
        self.doubleSpinBox_unit_bias.setValue(bias_val)
        self.doubleSpinBox_unit_bias.setEnabled(False)