# This Python file uses the following encoding: utf-8
import sys

import numpy as np

from PySide6 import QtCore
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget

class SDM():
    def __init__(self, addr_length, num_addresses, diametr, counter_length, qtWidget) -> None:
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

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        @Slot()
        def say_hello(self):
            print("Button clicked, Hello!")

        for child in self.findChildren(QTextEdit):
            child.setReadOnly(True)
        self.ui.initButton.clicked.connect(say_hello)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
