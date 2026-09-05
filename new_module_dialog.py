# import math
import numpy as np
# import json

from PyQt5.QtWidgets import QDialog
# from PyQt5.QtGui import QPainter, QColor, QPixmap, QPen, QCursor
# from PyQt5.QtCore import Qt, QTimer, QSize
# from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from gui.Ui_new_module import Ui_Dialog

from utils import LEFT_HALF, RIGHT_HALF, NO_HALF

class NewModuleDialog(Ui_Dialog, QDialog):
    def __init__(self, parent=None) -> None:
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.ok = False

        self.buttonBox.accepted.connect(self.onAccepted)
        self.buttonBox.rejected.connect(self.onRejected)

        self.checkBox_using_global_params.clicked.connect(self.changeUsingGlobalData)

        self.global_data_enabled = self.checkBox_using_global_params.isChecked()
        self.disable()

    def onAccepted(self):
        self.ok = True

    def onRejected(self):
        self.ok = False

    def getOK(self):
        return self.ok

    def getUnitWidth(self):
        return self.doubleSpinBox_unit_width.value()
    
    def getCopyTime(self):
        return self.spinBox_copy_time.value()
    
    def getEntryFlag(self):
        return self.horizontalSlider_entry_flag.value()
    
    def getEnableConnection(self):
        return self.checkBox_enable_connection.isChecked()
    
    def getConnectionLeftLength(self):
        return self.doubleSpinBox_con_left.value()
    
    def getConnectionRightLength(self):
        return self.doubleSpinBox_con_right.value()
    
    def getConnectionRadius(self):
        return self.doubleSpinBox_con_radius.value()
    
    def getUsingGlobalData(self):
        return self.global_data_enabled
    
    def getTsp(self):
        return [self.doubleSpinBox_tspx.value(), self.doubleSpinBox_tspy.value()]
    
    def getHalfFlag(self):
        if self.radioButton_left_half.isChecked():
            return LEFT_HALF
        elif self.radioButton_right_half.isChecked():
            return RIGHT_HALF
        elif self.radioButton_all.isChecked():
            return NO_HALF
    
    def getStretchLength(self):
        return self.doubleSpinBox_stretch_length.value()
    
    def changeUsingGlobalData(self):
        checked = self.checkBox_using_global_params.isChecked()
        if checked:
            self.disable()
        else:
            self.enable()

    def enable(self):
        self.global_data_enabled = False
        self.doubleSpinBox_unit_width.setEnabled(True)
        self.spinBox_copy_time.setEnabled(True)
        self.doubleSpinBox_con_left.setEnabled(True)
        self.checkBox_enable_connection.setEnabled(True)
        self.doubleSpinBox_con_right.setEnabled(True)
        self.doubleSpinBox_con_radius.setEnabled(True)
        self.horizontalSlider_entry_flag.setEnabled(True)

    def disable(self):
        self.global_data_enabled = True
        self.doubleSpinBox_unit_width.setEnabled(False)
        self.spinBox_copy_time.setEnabled(False)
        self.doubleSpinBox_con_left.setEnabled(False)
        self.checkBox_enable_connection.setEnabled(False)
        self.doubleSpinBox_con_right.setEnabled(False)
        self.doubleSpinBox_con_radius.setEnabled(False)
        self.horizontalSlider_entry_flag.setEnabled(False)
