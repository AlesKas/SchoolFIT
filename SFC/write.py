# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'write.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_Form(QWidget):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(828, 458)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, -1, 10, -1)
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_1 = QLineEdit(Form)
        self.input_1.setObjectName(u"input_1")
        self.input_1.setMaxLength(5)
        self.input_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_1)

        self.input_2 = QLineEdit(Form)
        self.input_2.setObjectName(u"input_2")
        self.input_2.setMaxLength(5)
        self.input_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_2)

        self.input_3 = QLineEdit(Form)
        self.input_3.setObjectName(u"input_3")
        self.input_3.setMaxLength(5)
        self.input_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_3)

        self.input_4 = QLineEdit(Form)
        self.input_4.setObjectName(u"input_4")
        self.input_4.setMaxLength(5)
        self.input_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_4)

        self.input_5 = QLineEdit(Form)
        self.input_5.setObjectName(u"input_5")
        self.input_5.setMaxLength(5)
        self.input_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_5)

        self.input_6 = QLineEdit(Form)
        self.input_6.setObjectName(u"input_6")
        self.input_6.setMaxLength(5)
        self.input_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_6)

        self.input_7 = QLineEdit(Form)
        self.input_7.setObjectName(u"input_7")
        self.input_7.setMaxLength(5)
        self.input_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_7)

        self.input_8 = QLineEdit(Form)
        self.input_8.setObjectName(u"input_8")
        self.input_8.setMaxLength(5)
        self.input_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.input_8)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.horizontalLayout_25.addLayout(self.verticalLayout)

        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.label_3)

        self.radius = QLineEdit(Form)
        self.radius.setObjectName(u"radius")
        self.radius.setMaxLength(1)
        self.radius.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.radius)


        self.horizontalLayout_25.addLayout(self.verticalLayout_9)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(70, -1, -1, -1)
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_2)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.odezva_1 = QLineEdit(Form)
        self.odezva_1.setObjectName(u"odezva_1")
        self.odezva_1.setMaxLength(5)
        self.odezva_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_1)

        self.odezva_2 = QLineEdit(Form)
        self.odezva_2.setObjectName(u"odezva_2")
        self.odezva_2.setMaxLength(5)
        self.odezva_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_2)

        self.odezva_3 = QLineEdit(Form)
        self.odezva_3.setObjectName(u"odezva_3")
        self.odezva_3.setMaxLength(5)
        self.odezva_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_3)

        self.odezva_4 = QLineEdit(Form)
        self.odezva_4.setObjectName(u"odezva_4")
        self.odezva_4.setMaxLength(5)
        self.odezva_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_4)

        self.odezva_5 = QLineEdit(Form)
        self.odezva_5.setObjectName(u"odezva_5")
        self.odezva_5.setMaxLength(5)
        self.odezva_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_5)

        self.odezva_6 = QLineEdit(Form)
        self.odezva_6.setObjectName(u"odezva_6")
        self.odezva_6.setMaxLength(5)
        self.odezva_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_6)

        self.odezva_7 = QLineEdit(Form)
        self.odezva_7.setObjectName(u"odezva_7")
        self.odezva_7.setMaxLength(5)
        self.odezva_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_7)

        self.odezva_8 = QLineEdit(Form)
        self.odezva_8.setObjectName(u"odezva_8")
        self.odezva_8.setMaxLength(5)
        self.odezva_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_24.addWidget(self.odezva_8)


        self.verticalLayout_3.addLayout(self.horizontalLayout_24)


        self.horizontalLayout_25.addLayout(self.verticalLayout_3)


        self.verticalLayout_10.addLayout(self.horizontalLayout_25)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_26.addWidget(self.label_4)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_7)

        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_26.addWidget(self.label_5)

        self.label_6 = QLabel(Form)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_26.addWidget(self.label_6)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_9)

        self.label_7 = QLabel(Form)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_26.addWidget(self.label_7)


        self.verticalLayout_10.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(-1, -1, 10, -1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.addres_1_1 = QLineEdit(Form)
        self.addres_1_1.setObjectName(u"addres_1_1")
        self.addres_1_1.setMaxLength(5)
        self.addres_1_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_1)

        self.addres_1_2 = QLineEdit(Form)
        self.addres_1_2.setObjectName(u"addres_1_2")
        self.addres_1_2.setMaxLength(5)
        self.addres_1_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_2)

        self.addres_1_3 = QLineEdit(Form)
        self.addres_1_3.setObjectName(u"addres_1_3")
        self.addres_1_3.setMaxLength(5)
        self.addres_1_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_3)

        self.addres_1_4 = QLineEdit(Form)
        self.addres_1_4.setObjectName(u"addres_1_4")
        self.addres_1_4.setMaxLength(5)
        self.addres_1_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_4)

        self.addres_1_5 = QLineEdit(Form)
        self.addres_1_5.setObjectName(u"addres_1_5")
        self.addres_1_5.setMaxLength(5)
        self.addres_1_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_5)

        self.addres_1_6 = QLineEdit(Form)
        self.addres_1_6.setObjectName(u"addres_1_6")
        self.addres_1_6.setMaxLength(5)
        self.addres_1_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_6)

        self.addres_1_7 = QLineEdit(Form)
        self.addres_1_7.setObjectName(u"addres_1_7")
        self.addres_1_7.setMaxLength(5)
        self.addres_1_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_7)

        self.addres_1_8 = QLineEdit(Form)
        self.addres_1_8.setObjectName(u"addres_1_8")
        self.addres_1_8.setMaxLength(5)
        self.addres_1_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.addres_1_8)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.addres_1_9 = QLineEdit(Form)
        self.addres_1_9.setObjectName(u"addres_1_9")
        self.addres_1_9.setMaxLength(5)
        self.addres_1_9.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_9)

        self.addres_1_10 = QLineEdit(Form)
        self.addres_1_10.setObjectName(u"addres_1_10")
        self.addres_1_10.setMaxLength(5)
        self.addres_1_10.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_10)

        self.addres_1_11 = QLineEdit(Form)
        self.addres_1_11.setObjectName(u"addres_1_11")
        self.addres_1_11.setMaxLength(5)
        self.addres_1_11.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_11)

        self.addres_1_12 = QLineEdit(Form)
        self.addres_1_12.setObjectName(u"addres_1_12")
        self.addres_1_12.setMaxLength(5)
        self.addres_1_12.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_12)

        self.addres_1_13 = QLineEdit(Form)
        self.addres_1_13.setObjectName(u"addres_1_13")
        self.addres_1_13.setMaxLength(5)
        self.addres_1_13.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_13)

        self.addres_1_14 = QLineEdit(Form)
        self.addres_1_14.setObjectName(u"addres_1_14")
        self.addres_1_14.setMaxLength(5)
        self.addres_1_14.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_14)

        self.addres_1_15 = QLineEdit(Form)
        self.addres_1_15.setObjectName(u"addres_1_15")
        self.addres_1_15.setMaxLength(5)
        self.addres_1_15.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_15)

        self.addres_1_16 = QLineEdit(Form)
        self.addres_1_16.setObjectName(u"addres_1_16")
        self.addres_1_16.setMaxLength(5)
        self.addres_1_16.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(self.addres_1_16)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.addres_1_17 = QLineEdit(Form)
        self.addres_1_17.setObjectName(u"addres_1_17")
        self.addres_1_17.setMaxLength(5)
        self.addres_1_17.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_17)

        self.addres_1_18 = QLineEdit(Form)
        self.addres_1_18.setObjectName(u"addres_1_18")
        self.addres_1_18.setMaxLength(5)
        self.addres_1_18.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_18)

        self.addres_1_19 = QLineEdit(Form)
        self.addres_1_19.setObjectName(u"addres_1_19")
        self.addres_1_19.setMaxLength(5)
        self.addres_1_19.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_19)

        self.addres_1_20 = QLineEdit(Form)
        self.addres_1_20.setObjectName(u"addres_1_20")
        self.addres_1_20.setMaxLength(5)
        self.addres_1_20.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_20)

        self.addres_1_21 = QLineEdit(Form)
        self.addres_1_21.setObjectName(u"addres_1_21")
        self.addres_1_21.setMaxLength(5)
        self.addres_1_21.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_21)

        self.addres_1_22 = QLineEdit(Form)
        self.addres_1_22.setObjectName(u"addres_1_22")
        self.addres_1_22.setMaxLength(5)
        self.addres_1_22.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_22)

        self.addres_1_23 = QLineEdit(Form)
        self.addres_1_23.setObjectName(u"addres_1_23")
        self.addres_1_23.setMaxLength(5)
        self.addres_1_23.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_23)

        self.addres_1_24 = QLineEdit(Form)
        self.addres_1_24.setObjectName(u"addres_1_24")
        self.addres_1_24.setMaxLength(5)
        self.addres_1_24.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.addres_1_24)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.addres_1_25 = QLineEdit(Form)
        self.addres_1_25.setObjectName(u"addres_1_25")
        self.addres_1_25.setMaxLength(5)
        self.addres_1_25.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_25)

        self.addres_1_26 = QLineEdit(Form)
        self.addres_1_26.setObjectName(u"addres_1_26")
        self.addres_1_26.setMaxLength(5)
        self.addres_1_26.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_26)

        self.addres_1_27 = QLineEdit(Form)
        self.addres_1_27.setObjectName(u"addres_1_27")
        self.addres_1_27.setMaxLength(5)
        self.addres_1_27.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_27)

        self.addres_1_28 = QLineEdit(Form)
        self.addres_1_28.setObjectName(u"addres_1_28")
        self.addres_1_28.setMaxLength(5)
        self.addres_1_28.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_28)

        self.addres_1_29 = QLineEdit(Form)
        self.addres_1_29.setObjectName(u"addres_1_29")
        self.addres_1_29.setMaxLength(5)
        self.addres_1_29.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_29)

        self.addres_1_30 = QLineEdit(Form)
        self.addres_1_30.setObjectName(u"addres_1_30")
        self.addres_1_30.setMaxLength(5)
        self.addres_1_30.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_30)

        self.addres_1_31 = QLineEdit(Form)
        self.addres_1_31.setObjectName(u"addres_1_31")
        self.addres_1_31.setMaxLength(5)
        self.addres_1_31.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_31)

        self.addres_1_32 = QLineEdit(Form)
        self.addres_1_32.setObjectName(u"addres_1_32")
        self.addres_1_32.setMaxLength(5)
        self.addres_1_32.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.addres_1_32)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.addres_1_33 = QLineEdit(Form)
        self.addres_1_33.setObjectName(u"addres_1_33")
        self.addres_1_33.setMaxLength(5)
        self.addres_1_33.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_33)

        self.addres_1_34 = QLineEdit(Form)
        self.addres_1_34.setObjectName(u"addres_1_34")
        self.addres_1_34.setMaxLength(5)
        self.addres_1_34.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_34)

        self.addres_1_35 = QLineEdit(Form)
        self.addres_1_35.setObjectName(u"addres_1_35")
        self.addres_1_35.setMaxLength(5)
        self.addres_1_35.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_35)

        self.addres_1_36 = QLineEdit(Form)
        self.addres_1_36.setObjectName(u"addres_1_36")
        self.addres_1_36.setMaxLength(5)
        self.addres_1_36.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_36)

        self.addres_1_37 = QLineEdit(Form)
        self.addres_1_37.setObjectName(u"addres_1_37")
        self.addres_1_37.setMaxLength(5)
        self.addres_1_37.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_37)

        self.addres_1_38 = QLineEdit(Form)
        self.addres_1_38.setObjectName(u"addres_1_38")
        self.addres_1_38.setMaxLength(5)
        self.addres_1_38.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_38)

        self.addres_1_39 = QLineEdit(Form)
        self.addres_1_39.setObjectName(u"addres_1_39")
        self.addres_1_39.setMaxLength(5)
        self.addres_1_39.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_39)

        self.addres_1_40 = QLineEdit(Form)
        self.addres_1_40.setObjectName(u"addres_1_40")
        self.addres_1_40.setMaxLength(5)
        self.addres_1_40.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_6.addWidget(self.addres_1_40)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.addres_1_41 = QLineEdit(Form)
        self.addres_1_41.setObjectName(u"addres_1_41")
        self.addres_1_41.setMaxLength(5)
        self.addres_1_41.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_41)

        self.addres_1_42 = QLineEdit(Form)
        self.addres_1_42.setObjectName(u"addres_1_42")
        self.addres_1_42.setMaxLength(5)
        self.addres_1_42.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_42)

        self.addres_1_43 = QLineEdit(Form)
        self.addres_1_43.setObjectName(u"addres_1_43")
        self.addres_1_43.setMaxLength(5)
        self.addres_1_43.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_43)

        self.addres_1_44 = QLineEdit(Form)
        self.addres_1_44.setObjectName(u"addres_1_44")
        self.addres_1_44.setMaxLength(5)
        self.addres_1_44.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_44)

        self.addres_1_45 = QLineEdit(Form)
        self.addres_1_45.setObjectName(u"addres_1_45")
        self.addres_1_45.setMaxLength(5)
        self.addres_1_45.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_45)

        self.addres_1_46 = QLineEdit(Form)
        self.addres_1_46.setObjectName(u"addres_1_46")
        self.addres_1_46.setMaxLength(5)
        self.addres_1_46.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_46)

        self.addres_1_47 = QLineEdit(Form)
        self.addres_1_47.setObjectName(u"addres_1_47")
        self.addres_1_47.setMaxLength(5)
        self.addres_1_47.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_47)

        self.addres_1_48 = QLineEdit(Form)
        self.addres_1_48.setObjectName(u"addres_1_48")
        self.addres_1_48.setMaxLength(5)
        self.addres_1_48.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.addres_1_48)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.addres_1_49 = QLineEdit(Form)
        self.addres_1_49.setObjectName(u"addres_1_49")
        self.addres_1_49.setMaxLength(5)
        self.addres_1_49.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_49)

        self.addres_1_50 = QLineEdit(Form)
        self.addres_1_50.setObjectName(u"addres_1_50")
        self.addres_1_50.setMaxLength(5)
        self.addres_1_50.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_50)

        self.addres_1_51 = QLineEdit(Form)
        self.addres_1_51.setObjectName(u"addres_1_51")
        self.addres_1_51.setMaxLength(5)
        self.addres_1_51.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_51)

        self.addres_1_52 = QLineEdit(Form)
        self.addres_1_52.setObjectName(u"addres_1_52")
        self.addres_1_52.setMaxLength(5)
        self.addres_1_52.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_52)

        self.addres_1_53 = QLineEdit(Form)
        self.addres_1_53.setObjectName(u"addres_1_53")
        self.addres_1_53.setMaxLength(5)
        self.addres_1_53.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_53)

        self.addres_1_54 = QLineEdit(Form)
        self.addres_1_54.setObjectName(u"addres_1_54")
        self.addres_1_54.setMaxLength(5)
        self.addres_1_54.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_54)

        self.addres_1_55 = QLineEdit(Form)
        self.addres_1_55.setObjectName(u"addres_1_55")
        self.addres_1_55.setMaxLength(5)
        self.addres_1_55.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_55)

        self.addres_1_56 = QLineEdit(Form)
        self.addres_1_56.setObjectName(u"addres_1_56")
        self.addres_1_56.setMaxLength(5)
        self.addres_1_56.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_8.addWidget(self.addres_1_56)


        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.addres_1_57 = QLineEdit(Form)
        self.addres_1_57.setObjectName(u"addres_1_57")
        self.addres_1_57.setMaxLength(5)
        self.addres_1_57.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_57)

        self.addres_1_58 = QLineEdit(Form)
        self.addres_1_58.setObjectName(u"addres_1_58")
        self.addres_1_58.setMaxLength(5)
        self.addres_1_58.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_58)

        self.addres_1_59 = QLineEdit(Form)
        self.addres_1_59.setObjectName(u"addres_1_59")
        self.addres_1_59.setMaxLength(5)
        self.addres_1_59.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_59)

        self.addres_1_60 = QLineEdit(Form)
        self.addres_1_60.setObjectName(u"addres_1_60")
        self.addres_1_60.setMaxLength(5)
        self.addres_1_60.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_60)

        self.addres_1_61 = QLineEdit(Form)
        self.addres_1_61.setObjectName(u"addres_1_61")
        self.addres_1_61.setMaxLength(5)
        self.addres_1_61.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_61)

        self.addres_1_62 = QLineEdit(Form)
        self.addres_1_62.setObjectName(u"addres_1_62")
        self.addres_1_62.setMaxLength(5)
        self.addres_1_62.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_62)

        self.addres_1_63 = QLineEdit(Form)
        self.addres_1_63.setObjectName(u"addres_1_63")
        self.addres_1_63.setMaxLength(5)
        self.addres_1_63.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_63)

        self.addres_1_64 = QLineEdit(Form)
        self.addres_1_64.setObjectName(u"addres_1_64")
        self.addres_1_64.setMaxLength(5)
        self.addres_1_64.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.addres_1_64)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.addres_1_65 = QLineEdit(Form)
        self.addres_1_65.setObjectName(u"addres_1_65")
        self.addres_1_65.setMaxLength(5)
        self.addres_1_65.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_65)

        self.addres_1_66 = QLineEdit(Form)
        self.addres_1_66.setObjectName(u"addres_1_66")
        self.addres_1_66.setMaxLength(5)
        self.addres_1_66.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_66)

        self.addres_1_67 = QLineEdit(Form)
        self.addres_1_67.setObjectName(u"addres_1_67")
        self.addres_1_67.setMaxLength(5)
        self.addres_1_67.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_67)

        self.addres_1_68 = QLineEdit(Form)
        self.addres_1_68.setObjectName(u"addres_1_68")
        self.addres_1_68.setMaxLength(5)
        self.addres_1_68.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_68)

        self.addres_1_69 = QLineEdit(Form)
        self.addres_1_69.setObjectName(u"addres_1_69")
        self.addres_1_69.setMaxLength(5)
        self.addres_1_69.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_69)

        self.addres_1_70 = QLineEdit(Form)
        self.addres_1_70.setObjectName(u"addres_1_70")
        self.addres_1_70.setMaxLength(5)
        self.addres_1_70.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_70)

        self.addres_1_71 = QLineEdit(Form)
        self.addres_1_71.setObjectName(u"addres_1_71")
        self.addres_1_71.setMaxLength(5)
        self.addres_1_71.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_71)

        self.addres_1_72 = QLineEdit(Form)
        self.addres_1_72.setObjectName(u"addres_1_72")
        self.addres_1_72.setMaxLength(5)
        self.addres_1_72.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.addres_1_72)


        self.verticalLayout_2.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.addres_1_73 = QLineEdit(Form)
        self.addres_1_73.setObjectName(u"addres_1_73")
        self.addres_1_73.setMaxLength(5)
        self.addres_1_73.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_73)

        self.addres_1_74 = QLineEdit(Form)
        self.addres_1_74.setObjectName(u"addres_1_74")
        self.addres_1_74.setMaxLength(5)
        self.addres_1_74.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_74)

        self.addres_1_75 = QLineEdit(Form)
        self.addres_1_75.setObjectName(u"addres_1_75")
        self.addres_1_75.setMaxLength(5)
        self.addres_1_75.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_75)

        self.addres_1_76 = QLineEdit(Form)
        self.addres_1_76.setObjectName(u"addres_1_76")
        self.addres_1_76.setMaxLength(5)
        self.addres_1_76.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_76)

        self.addres_1_77 = QLineEdit(Form)
        self.addres_1_77.setObjectName(u"addres_1_77")
        self.addres_1_77.setMaxLength(5)
        self.addres_1_77.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_77)

        self.addres_1_78 = QLineEdit(Form)
        self.addres_1_78.setObjectName(u"addres_1_78")
        self.addres_1_78.setMaxLength(5)
        self.addres_1_78.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_78)

        self.addres_1_79 = QLineEdit(Form)
        self.addres_1_79.setObjectName(u"addres_1_79")
        self.addres_1_79.setMaxLength(5)
        self.addres_1_79.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_79)

        self.addres_1_80 = QLineEdit(Form)
        self.addres_1_80.setObjectName(u"addres_1_80")
        self.addres_1_80.setMaxLength(5)
        self.addres_1_80.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.addres_1_80)


        self.verticalLayout_2.addLayout(self.horizontalLayout_11)


        self.horizontalLayout_21.addLayout(self.verticalLayout_2)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_17)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.distance_1 = QLineEdit(Form)
        self.distance_1.setObjectName(u"distance_1")
        self.distance_1.setMaxLength(5)
        self.distance_1.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_1)

        self.distance_2 = QLineEdit(Form)
        self.distance_2.setObjectName(u"distance_2")
        self.distance_2.setMaxLength(5)
        self.distance_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_2)

        self.distance_3 = QLineEdit(Form)
        self.distance_3.setObjectName(u"distance_3")
        self.distance_3.setMaxLength(5)
        self.distance_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_3)

        self.distance_4 = QLineEdit(Form)
        self.distance_4.setObjectName(u"distance_4")
        self.distance_4.setMaxLength(5)
        self.distance_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_4)

        self.distance_5 = QLineEdit(Form)
        self.distance_5.setObjectName(u"distance_5")
        self.distance_5.setMaxLength(5)
        self.distance_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_5)

        self.distance_6 = QLineEdit(Form)
        self.distance_6.setObjectName(u"distance_6")
        self.distance_6.setMaxLength(5)
        self.distance_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_6)

        self.distance_7 = QLineEdit(Form)
        self.distance_7.setObjectName(u"distance_7")
        self.distance_7.setMaxLength(5)
        self.distance_7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_7)

        self.distance_8 = QLineEdit(Form)
        self.distance_8.setObjectName(u"distance_8")
        self.distance_8.setMaxLength(5)
        self.distance_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_8)

        self.distance_9 = QLineEdit(Form)
        self.distance_9.setObjectName(u"distance_9")
        self.distance_9.setMaxLength(5)
        self.distance_9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_9)

        self.distance_10 = QLineEdit(Form)
        self.distance_10.setObjectName(u"distance_10")
        self.distance_10.setMaxLength(5)
        self.distance_10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.distance_10)


        self.verticalLayout_8.addLayout(self.verticalLayout_4)


        self.horizontalLayout_12.addLayout(self.verticalLayout_8)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_2)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.select_1 = QLineEdit(Form)
        self.select_1.setObjectName(u"select_1")
        self.select_1.setMaxLength(5)
        self.select_1.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_1)

        self.select_2 = QLineEdit(Form)
        self.select_2.setObjectName(u"select_2")
        self.select_2.setMaxLength(5)
        self.select_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_2)

        self.select_3 = QLineEdit(Form)
        self.select_3.setObjectName(u"select_3")
        self.select_3.setMaxLength(5)
        self.select_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_3)

        self.select_4 = QLineEdit(Form)
        self.select_4.setObjectName(u"select_4")
        self.select_4.setMaxLength(5)
        self.select_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_4)

        self.select_5 = QLineEdit(Form)
        self.select_5.setObjectName(u"select_5")
        self.select_5.setMaxLength(5)
        self.select_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_5)

        self.select_6 = QLineEdit(Form)
        self.select_6.setObjectName(u"select_6")
        self.select_6.setMaxLength(5)
        self.select_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_6)

        self.select_7 = QLineEdit(Form)
        self.select_7.setObjectName(u"select_7")
        self.select_7.setMaxLength(5)
        self.select_7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_7)

        self.select_8 = QLineEdit(Form)
        self.select_8.setObjectName(u"select_8")
        self.select_8.setMaxLength(5)
        self.select_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_8)

        self.select_9 = QLineEdit(Form)
        self.select_9.setObjectName(u"select_9")
        self.select_9.setMaxLength(5)
        self.select_9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_9)

        self.select_10 = QLineEdit(Form)
        self.select_10.setObjectName(u"select_10")
        self.select_10.setMaxLength(5)
        self.select_10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.select_10)


        self.verticalLayout_7.addLayout(self.verticalLayout_5)


        self.horizontalLayout_12.addLayout(self.verticalLayout_7)


        self.horizontalLayout_21.addLayout(self.horizontalLayout_12)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_3)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(10, -1, 0, -1)
        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.counter_1 = QLineEdit(Form)
        self.counter_1.setObjectName(u"counter_1")
        self.counter_1.setMaxLength(5)
        self.counter_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_1)

        self.counter_2 = QLineEdit(Form)
        self.counter_2.setObjectName(u"counter_2")
        self.counter_2.setMaxLength(5)
        self.counter_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_2)

        self.counter_3 = QLineEdit(Form)
        self.counter_3.setObjectName(u"counter_3")
        self.counter_3.setMaxLength(5)
        self.counter_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_3)

        self.counter_4 = QLineEdit(Form)
        self.counter_4.setObjectName(u"counter_4")
        self.counter_4.setMaxLength(5)
        self.counter_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_4)

        self.counter_5 = QLineEdit(Form)
        self.counter_5.setObjectName(u"counter_5")
        self.counter_5.setMaxLength(5)
        self.counter_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_5)

        self.counter_6 = QLineEdit(Form)
        self.counter_6.setObjectName(u"counter_6")
        self.counter_6.setMaxLength(5)
        self.counter_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_6)

        self.counter_7 = QLineEdit(Form)
        self.counter_7.setObjectName(u"counter_7")
        self.counter_7.setMaxLength(5)
        self.counter_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_7)

        self.counter_8 = QLineEdit(Form)
        self.counter_8.setObjectName(u"counter_8")
        self.counter_8.setMaxLength(5)
        self.counter_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_13.addWidget(self.counter_8)


        self.verticalLayout_6.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.counter_9 = QLineEdit(Form)
        self.counter_9.setObjectName(u"counter_9")
        self.counter_9.setMaxLength(5)
        self.counter_9.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_9)

        self.counter_10 = QLineEdit(Form)
        self.counter_10.setObjectName(u"counter_10")
        self.counter_10.setMaxLength(5)
        self.counter_10.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_10)

        self.counter_11 = QLineEdit(Form)
        self.counter_11.setObjectName(u"counter_11")
        self.counter_11.setMaxLength(5)
        self.counter_11.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_11)

        self.counter_12 = QLineEdit(Form)
        self.counter_12.setObjectName(u"counter_12")
        self.counter_12.setMaxLength(5)
        self.counter_12.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_12)

        self.counter_13 = QLineEdit(Form)
        self.counter_13.setObjectName(u"counter_13")
        self.counter_13.setMaxLength(5)
        self.counter_13.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_13)

        self.counter_14 = QLineEdit(Form)
        self.counter_14.setObjectName(u"counter_14")
        self.counter_14.setMaxLength(5)
        self.counter_14.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_14)

        self.counter_15 = QLineEdit(Form)
        self.counter_15.setObjectName(u"counter_15")
        self.counter_15.setMaxLength(5)
        self.counter_15.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_15)

        self.counter_16 = QLineEdit(Form)
        self.counter_16.setObjectName(u"counter_16")
        self.counter_16.setMaxLength(5)
        self.counter_16.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_14.addWidget(self.counter_16)


        self.verticalLayout_6.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.counter_17 = QLineEdit(Form)
        self.counter_17.setObjectName(u"counter_17")
        self.counter_17.setMaxLength(5)
        self.counter_17.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_17)

        self.counter_18 = QLineEdit(Form)
        self.counter_18.setObjectName(u"counter_18")
        self.counter_18.setMaxLength(5)
        self.counter_18.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_18)

        self.counter_19 = QLineEdit(Form)
        self.counter_19.setObjectName(u"counter_19")
        self.counter_19.setMaxLength(5)
        self.counter_19.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_19)

        self.counter_20 = QLineEdit(Form)
        self.counter_20.setObjectName(u"counter_20")
        self.counter_20.setMaxLength(5)
        self.counter_20.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_20)

        self.counter_21 = QLineEdit(Form)
        self.counter_21.setObjectName(u"counter_21")
        self.counter_21.setMaxLength(5)
        self.counter_21.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_21)

        self.counter_22 = QLineEdit(Form)
        self.counter_22.setObjectName(u"counter_22")
        self.counter_22.setMaxLength(5)
        self.counter_22.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_22)

        self.counter_23 = QLineEdit(Form)
        self.counter_23.setObjectName(u"counter_23")
        self.counter_23.setMaxLength(5)
        self.counter_23.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_23)

        self.counter_24 = QLineEdit(Form)
        self.counter_24.setObjectName(u"counter_24")
        self.counter_24.setMaxLength(5)
        self.counter_24.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.counter_24)


        self.verticalLayout_6.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.counter_25 = QLineEdit(Form)
        self.counter_25.setObjectName(u"counter_25")
        self.counter_25.setMaxLength(5)
        self.counter_25.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_25)

        self.counter_26 = QLineEdit(Form)
        self.counter_26.setObjectName(u"counter_26")
        self.counter_26.setMaxLength(5)
        self.counter_26.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_26)

        self.counter_27 = QLineEdit(Form)
        self.counter_27.setObjectName(u"counter_27")
        self.counter_27.setMaxLength(5)
        self.counter_27.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_27)

        self.counter_28 = QLineEdit(Form)
        self.counter_28.setObjectName(u"counter_28")
        self.counter_28.setMaxLength(5)
        self.counter_28.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_28)

        self.counter_29 = QLineEdit(Form)
        self.counter_29.setObjectName(u"counter_29")
        self.counter_29.setMaxLength(5)
        self.counter_29.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_29)

        self.counter_30 = QLineEdit(Form)
        self.counter_30.setObjectName(u"counter_30")
        self.counter_30.setMaxLength(5)
        self.counter_30.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_30)

        self.counter_31 = QLineEdit(Form)
        self.counter_31.setObjectName(u"counter_31")
        self.counter_31.setMaxLength(5)
        self.counter_31.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_31)

        self.counter_32 = QLineEdit(Form)
        self.counter_32.setObjectName(u"counter_32")
        self.counter_32.setMaxLength(5)
        self.counter_32.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.counter_32)


        self.verticalLayout_6.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.counter_33 = QLineEdit(Form)
        self.counter_33.setObjectName(u"counter_33")
        self.counter_33.setMaxLength(5)
        self.counter_33.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_33)

        self.counter_34 = QLineEdit(Form)
        self.counter_34.setObjectName(u"counter_34")
        self.counter_34.setMaxLength(5)
        self.counter_34.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_34)

        self.counter_35 = QLineEdit(Form)
        self.counter_35.setObjectName(u"counter_35")
        self.counter_35.setMaxLength(5)
        self.counter_35.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_35)

        self.counter_36 = QLineEdit(Form)
        self.counter_36.setObjectName(u"counter_36")
        self.counter_36.setMaxLength(5)
        self.counter_36.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_36)

        self.counter_37 = QLineEdit(Form)
        self.counter_37.setObjectName(u"counter_37")
        self.counter_37.setMaxLength(5)
        self.counter_37.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_37)

        self.counter_38 = QLineEdit(Form)
        self.counter_38.setObjectName(u"counter_38")
        self.counter_38.setMaxLength(5)
        self.counter_38.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_38)

        self.counter_39 = QLineEdit(Form)
        self.counter_39.setObjectName(u"counter_39")
        self.counter_39.setMaxLength(5)
        self.counter_39.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_39)

        self.counter_40 = QLineEdit(Form)
        self.counter_40.setObjectName(u"counter_40")
        self.counter_40.setMaxLength(5)
        self.counter_40.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.counter_40)


        self.verticalLayout_6.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.counter_41 = QLineEdit(Form)
        self.counter_41.setObjectName(u"counter_41")
        self.counter_41.setMaxLength(5)
        self.counter_41.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_41)

        self.counter_42 = QLineEdit(Form)
        self.counter_42.setObjectName(u"counter_42")
        self.counter_42.setMaxLength(5)
        self.counter_42.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_42)

        self.counter_43 = QLineEdit(Form)
        self.counter_43.setObjectName(u"counter_43")
        self.counter_43.setMaxLength(5)
        self.counter_43.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_43)

        self.counter_44 = QLineEdit(Form)
        self.counter_44.setObjectName(u"counter_44")
        self.counter_44.setMaxLength(5)
        self.counter_44.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_44)

        self.counter_45 = QLineEdit(Form)
        self.counter_45.setObjectName(u"counter_45")
        self.counter_45.setMaxLength(5)
        self.counter_45.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_45)

        self.counter_46 = QLineEdit(Form)
        self.counter_46.setObjectName(u"counter_46")
        self.counter_46.setMaxLength(5)
        self.counter_46.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_46)

        self.counter_47 = QLineEdit(Form)
        self.counter_47.setObjectName(u"counter_47")
        self.counter_47.setMaxLength(5)
        self.counter_47.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_47)

        self.counter_48 = QLineEdit(Form)
        self.counter_48.setObjectName(u"counter_48")
        self.counter_48.setMaxLength(5)
        self.counter_48.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.counter_48)


        self.verticalLayout_6.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.counter_49 = QLineEdit(Form)
        self.counter_49.setObjectName(u"counter_49")
        self.counter_49.setMaxLength(5)
        self.counter_49.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_49)

        self.counter_50 = QLineEdit(Form)
        self.counter_50.setObjectName(u"counter_50")
        self.counter_50.setMaxLength(5)
        self.counter_50.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_50)

        self.counter_51 = QLineEdit(Form)
        self.counter_51.setObjectName(u"counter_51")
        self.counter_51.setMaxLength(5)
        self.counter_51.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_51)

        self.counter_52 = QLineEdit(Form)
        self.counter_52.setObjectName(u"counter_52")
        self.counter_52.setMaxLength(5)
        self.counter_52.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_52)

        self.counter_53 = QLineEdit(Form)
        self.counter_53.setObjectName(u"counter_53")
        self.counter_53.setMaxLength(5)
        self.counter_53.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_53)

        self.counter_54 = QLineEdit(Form)
        self.counter_54.setObjectName(u"counter_54")
        self.counter_54.setMaxLength(5)
        self.counter_54.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_54)

        self.counter_55 = QLineEdit(Form)
        self.counter_55.setObjectName(u"counter_55")
        self.counter_55.setMaxLength(5)
        self.counter_55.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_55)

        self.counter_56 = QLineEdit(Form)
        self.counter_56.setObjectName(u"counter_56")
        self.counter_56.setMaxLength(5)
        self.counter_56.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.counter_56)


        self.verticalLayout_6.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.counter_57 = QLineEdit(Form)
        self.counter_57.setObjectName(u"counter_57")
        self.counter_57.setMaxLength(5)
        self.counter_57.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_57)

        self.counter_58 = QLineEdit(Form)
        self.counter_58.setObjectName(u"counter_58")
        self.counter_58.setMaxLength(5)
        self.counter_58.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_58)

        self.counter_59 = QLineEdit(Form)
        self.counter_59.setObjectName(u"counter_59")
        self.counter_59.setMaxLength(5)
        self.counter_59.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_59)

        self.counter_60 = QLineEdit(Form)
        self.counter_60.setObjectName(u"counter_60")
        self.counter_60.setMaxLength(5)
        self.counter_60.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_60)

        self.counter_61 = QLineEdit(Form)
        self.counter_61.setObjectName(u"counter_61")
        self.counter_61.setMaxLength(5)
        self.counter_61.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_61)

        self.counter_62 = QLineEdit(Form)
        self.counter_62.setObjectName(u"counter_62")
        self.counter_62.setMaxLength(5)
        self.counter_62.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_62)

        self.counter_63 = QLineEdit(Form)
        self.counter_63.setObjectName(u"counter_63")
        self.counter_63.setMaxLength(5)
        self.counter_63.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_63)

        self.counter_64 = QLineEdit(Form)
        self.counter_64.setObjectName(u"counter_64")
        self.counter_64.setMaxLength(5)
        self.counter_64.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_20.addWidget(self.counter_64)


        self.verticalLayout_6.addLayout(self.horizontalLayout_20)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.counter_65 = QLineEdit(Form)
        self.counter_65.setObjectName(u"counter_65")
        self.counter_65.setMaxLength(5)
        self.counter_65.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_65)

        self.counter_66 = QLineEdit(Form)
        self.counter_66.setObjectName(u"counter_66")
        self.counter_66.setMaxLength(5)
        self.counter_66.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_66)

        self.counter_67 = QLineEdit(Form)
        self.counter_67.setObjectName(u"counter_67")
        self.counter_67.setMaxLength(5)
        self.counter_67.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_67)

        self.counter_68 = QLineEdit(Form)
        self.counter_68.setObjectName(u"counter_68")
        self.counter_68.setMaxLength(5)
        self.counter_68.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_68)

        self.counter_69 = QLineEdit(Form)
        self.counter_69.setObjectName(u"counter_69")
        self.counter_69.setMaxLength(5)
        self.counter_69.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_69)

        self.counter_70 = QLineEdit(Form)
        self.counter_70.setObjectName(u"counter_70")
        self.counter_70.setMaxLength(5)
        self.counter_70.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_70)

        self.counter_71 = QLineEdit(Form)
        self.counter_71.setObjectName(u"counter_71")
        self.counter_71.setMaxLength(5)
        self.counter_71.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_71)

        self.counter_72 = QLineEdit(Form)
        self.counter_72.setObjectName(u"counter_72")
        self.counter_72.setMaxLength(5)
        self.counter_72.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.counter_72)


        self.verticalLayout_6.addLayout(self.horizontalLayout_22)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.counter_73 = QLineEdit(Form)
        self.counter_73.setObjectName(u"counter_73")
        self.counter_73.setMaxLength(5)
        self.counter_73.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_73)

        self.counter_74 = QLineEdit(Form)
        self.counter_74.setObjectName(u"counter_74")
        self.counter_74.setMaxLength(5)
        self.counter_74.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_74)

        self.counter_75 = QLineEdit(Form)
        self.counter_75.setObjectName(u"counter_75")
        self.counter_75.setMaxLength(5)
        self.counter_75.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_75)

        self.counter_76 = QLineEdit(Form)
        self.counter_76.setObjectName(u"counter_76")
        self.counter_76.setMaxLength(5)
        self.counter_76.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_76)

        self.counter_77 = QLineEdit(Form)
        self.counter_77.setObjectName(u"counter_77")
        self.counter_77.setMaxLength(5)
        self.counter_77.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_77)

        self.counter_78 = QLineEdit(Form)
        self.counter_78.setObjectName(u"counter_78")
        self.counter_78.setMaxLength(5)
        self.counter_78.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_78)

        self.counter_79 = QLineEdit(Form)
        self.counter_79.setObjectName(u"counter_79")
        self.counter_79.setMaxLength(5)
        self.counter_79.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_79)

        self.counter_80 = QLineEdit(Form)
        self.counter_80.setObjectName(u"counter_80")
        self.counter_80.setMaxLength(5)
        self.counter_80.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.counter_80)


        self.verticalLayout_6.addLayout(self.horizontalLayout_23)


        self.horizontalLayout_21.addLayout(self.verticalLayout_6)


        self.verticalLayout_10.addLayout(self.horizontalLayout_21)


        self.verticalLayout_11.addLayout(self.verticalLayout_10)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_27.addItem(self.horizontalSpacer_10)

        self.init_button = QPushButton(Form)
        self.init_button.setObjectName(u"init_button")

        self.horizontalLayout_27.addWidget(self.init_button)

        self.step_button = QPushButton(Form)
        self.step_button.setObjectName(u"step_button")

        self.horizontalLayout_27.addWidget(self.step_button)

        self.exit_button = QPushButton(Form)
        self.exit_button.setObjectName(u"exit_button")

        self.horizontalLayout_27.addWidget(self.exit_button)


        self.verticalLayout_11.addLayout(self.horizontalLayout_27)


        self.gridLayout.addLayout(self.verticalLayout_11, 0, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"SDM", None))
        self.label.setText(QCoreApplication.translate("Form", u"Vstupn\u00ed vektor", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Polom\u011br", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Po\u017eadovan\u00e1 odezva", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"Pole adres", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Vzd\u00e1lenost", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"V\u00fdb\u011br", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"Pole \u010d\u00edta\u010d\u016f", None))
        self.init_button.setText(QCoreApplication.translate("Form", u"Init", None))
        self.step_button.setText(QCoreApplication.translate("Form", u"Krok", None))
        self.exit_button.setText(QCoreApplication.translate("Form", u"Konec", None))
    # retranslateUi

    def fill_input_response(self):
        widgetName = 'input_'
        for num in range(0, 8):
            widget = widgetName + str(num+1)
            self.findChild(QLineEdit, widget).setText(str(self.input_vector[0][num]))
        widgetName = 'odezva_'
        for num in range(0, 8):
            widget = widgetName + str(num+1)
            self.findChild(QLineEdit, widget).setText(str(self.input_vector[0][num]))

    @Slot()
    def step(self):
        if self.init_counter == 0:
            return
        if self.states[self.state] == 'init':
            self.input_vector = np.random.randint(2, size=(1, 8), dtype=np.int8)
            self.fill_input_response()
            self.state += 1
            return
        if self.states[self.state] == 'distance':
            widgetName = 'distance_'
            for iter in range(0, 10):
                widget = widgetName + str(iter + 1)
                distance = np.logical_xor(self.input_vector, self.sdm.sparse_addresses[iter]).sum(axis=1)
                self.findChild(QLineEdit, widget).setText(str(distance[0]))
            self.state += 1
            return
        if self.states[self.state] == 'select':
            self.selected_states = []
            widgetNameSelect = 'select_'
            widgetNameDistance = 'distance_'
            found = 0
            for iter in range(1, 11):
                widgetDistance = widgetNameDistance + str(iter)
                widgetSelect = widgetNameSelect + str(iter)
                distance = self.findChild(QLineEdit, widgetDistance).text()
                if int(distance) <= 2:
                    self.findChild(QLineEdit, widgetSelect).setText(str(1))
                    found += 1
                else:
                    self.findChild(QLineEdit, widgetSelect).setText(str(0))
            self.state += 1
            if found == 0:
                self.state = 0
            return
        if self.states[self.state] == 'select_selected':
            widgetNameSelect = 'select_'
            for iter1 in range(1, 11):
                widgetSelect = widgetNameSelect + str(iter1)
                if self.findChild(QLineEdit, widgetSelect).text() == '1':
                    addressName = 'addres_1_'
                    distanceName = 'distance_'
                    counterName = 'counter_'
                    self.findChild(QLineEdit, distanceName + str(iter1)).setStyleSheet("QLineEdit {background : green;}")
                    self.findChild(QLineEdit, widgetNameSelect + str(iter1)).setStyleSheet("QLineEdit {background : green;}")
                    for iter2 in range(1, 9):
                        self.findChild(QLineEdit, addressName + str(((iter1 - 1) * 8 + (iter2 - 1)) + 1)).setStyleSheet("QLineEdit {background : green;}")
                        self.findChild(QLineEdit, counterName + str(((iter1 - 1) * 8 + (iter2 - 1)) + 1)).setStyleSheet("QLineEdit {background : green;}")
                        self.selected_states.append(self.findChild(QLineEdit, counterName + str(((iter1 - 1) * 8 + (iter2 - 1)) + 1)))
            self.state += 1
            return
        if self.states[self.state] == 'check_response':
            self.positive = []
            self.negative = []
            outputName = 'odezva_'
            odezva = 1
            for selected in self.selected_states:
                if self.findChild(QLineEdit, outputName + str(odezva)).text() == '1':
                    selected.setStyleSheet("QLineEdit {background : red;}")
                    odezva += 1
                    self.positive.append(selected)
                else:
                    selected.setStyleSheet("QLineEdit {background : blue;}")
                    odezva += 1
                    self.negative.append(selected)
                if odezva == 9:
                    odezva = 1
            self.state += 1
            return
        if self.states[self.state] == 'update_values':
            for pos in self.positive:
                val = pos.text()
                pos.setText(str(int(val) + 1))
            for pos in self.negative:
                val = pos.text()
                pos.setText(str(int(val) - 1))
            self.state += 1
            return
        if self.states[self.state] == 'clear_colours':
            for child in self.findChildren(QLineEdit):
                child.setStyleSheet("QLineEdit {background : white;}")
            widgetNameSelect = 'select_'
            widgetNameDistance = 'distance_'
            found = False
            for iter in range(1, 11):
                widgetDistance = widgetNameDistance + str(iter)
                widgetSelect = widgetNameSelect + str(iter)
                self.findChild(QLineEdit, widgetDistance).setText(str(0))
                self.findChild(QLineEdit, widgetSelect).setText(str(0))
            self.state = 0
            return

    @Slot()
    def init(self):
        if self.init_counter == 0:
            self.sdm = SDM(8, 10, 2, 8)
            self.radius.setText('2')
            self.input_vector = np.random.randint(2, size=(1, 8), dtype=np.int8)
            self.fill_input_response()
            widgetName = 'addres_1_'
            for iter1 in range(0, 10):
                for iter2 in range(0, 8):
                    widget = widgetName + str((iter1 * 8 + iter2) + 1)
                    self.findChild(QLineEdit, widget).setText(str(self.sdm.sparse_addresses[iter1][iter2]))
            widgetName = 'counter_'
            for num in range(1, 81):
                widget = widgetName + str(num)
                self.findChild(QLineEdit, widget).setText(str(0))
            self.init_counter += 1

    @Slot()
    def exit_form(self):
        import os
        os._exit(0)

    def init_buttons(self):
        self.states = ['init', 'distance', 'select', 'select_selected', 'check_response', 'update_values', 'clear_colours']
        self.state = 1
        self.init_counter = 0
        self.init_button.clicked.connect(self.init)
        self.step_button.clicked.connect(self.step)
        self.exit_button.clicked.connect(self.exit_form)

class SDM():
    def __init__(self, addr_length, num_addresses, diametr, counter_length) -> None:
        self.addr_length = addr_length
        self.num_addresses = num_addresses
        self.diameter = diametr
        self.counter_length = counter_length

        self.sparse_addresses = np.random.randint(2, size=(self.num_addresses, self.addr_length))
        self.weights = np.zeros((self.num_addresses, self.counter_length), dtype=np.int16)

    def write(self, input_data):
        # Calculate hamming distance, vectors are different on place i when x_i != w_i
        # https://www.fit.vutbr.cz/~grebenic/Publikace/mosis2000.pdf
        distance = np.logical_xor(input_data, self.sparse_addresses).sum(axis=1)
        # Get indexes of weights whose distance is lower than on equal to diameter
        indexes = np.where(distance <= self.diameter)
        # v_jk = v_jk + (2d_pj - 1)
        self.weights[indexes] += 2 * input_data - 1

    def read(self, data):
        # Calculate distance again
        distance = np.logical_xor(data, self.sparse_addresses).sum(axis=1)
        # Get indexes
        indexes = np.where(distance <= self.diameter)
        # Sum all neurons
        sum_neurons = self.weights[indexes].sum(axis=0)
        # Calculate threshold, sum all weights and devide the, by 2
        threshold = self.weights.sum(axis=0)
        threshold = threshold / 2
        threshold = threshold.astype(np.int16)
        # Output[i] = (sum_neurons[i] >= threshold[i]) ? 1 : 0
        output = (sum_neurons >= threshold).astype(np.int8)
        return output