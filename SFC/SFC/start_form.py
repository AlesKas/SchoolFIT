# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import sys

from PySide6.QtCore import Slot
from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize, Qt)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget, QLineEdit)

from write import Ui_Form as writeForm
from read import Ui_Form as readForm

class Ui_Form(QWidget):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(419, 310)
        Form.setMinimumSize(QSize(419, 310))
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.read_button = QPushButton(Form)
        self.read_button.setObjectName(u"read_button")

        self.horizontalLayout.addWidget(self.read_button)

        self.write_button = QPushButton(Form)
        self.write_button.setObjectName(u"write_button")

        self.horizontalLayout.addWidget(self.write_button)

        self.exit_button = QPushButton(Form)
        self.exit_button.setObjectName(u"exit_button")

        self.horizontalLayout.addWidget(self.exit_button)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText(QCoreApplication.translate("Form", u"Uk\u00e1zka algoritmu Sparse Distributed Memory", None))
        self.read_button.setText(QCoreApplication.translate("Form", u"\u010cten\u00ed", None))
        self.write_button.setText(QCoreApplication.translate("Form", u"Z\u00e1pis", None))
        self.exit_button.setText(QCoreApplication.translate("Form", u"Konec", None))
    # retranslateUi
        
    @Slot()
    def read_clicked(self):
        self.hide()
        self.ui = readForm()
        self.ui.setupUi(self.ui)
        for child in self.ui.findChildren(QLineEdit):
            child.setReadOnly(True)
        self.ui.init_buttons()
        self.ui.show()
    
    @Slot()
    def write_clicked(self):
        self.hide()
        self.ui = writeForm()
        self.ui.setupUi(self.ui)
        for child in self.ui.findChildren(QLineEdit):
            child.setReadOnly(True)
        self.ui.init_buttons()
        self.ui.show()

    @Slot()
    def exit_form(self):
        exit(0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.read_button.clicked.connect(self.read_clicked)
        self.write_button.clicked.connect(self.write_clicked)
        self.exit_button.clicked.connect(self.exit_form)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    widget = Ui_Form()
    widget.show()
    sys.exit(app.exec())