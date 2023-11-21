# This Python file uses the following encoding: utf-8
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QWidget, QGridLayout, QVBoxLayout, \
    QPushButton, QDialog
from PyQt5.uic import loadUi
from hopfild import Hopfild
from cosco import Cosco
from PyQt5.QtCore import qDebug


class MainWindow(QDialog):
    def __init__(self):
        QMainWindow.__init__(self)
        super().__init__()
        loadUi('untitled.ui', self)
        self.pushButton_2.clicked.connect(self.showInputWidget)
        self.pushButton.clicked.connect(self.neurons)
        # Сколько строк и столбцов в поле ввода
        self.hopfild = Hopfild(100, 100000000)
        self.cosco  = Cosco(10, 100000)
        self.inputSize = 10
        self.iterations = 1000
        self.inputtedLetter = [0 for i in range(self.inputSize ** 2)]

    def neurons(self):
        if self.radioButton.isChecked():
            self.hopfild_neurons()
        elif self.radioButton_2.isChecked():
            self.kosko_neurons()

    def showInputWidget(self):
        self.input = QWidget()
        grid = QGridLayout()
        self.input.setFixedSize(40 * self.inputSize, 40 * self.inputSize)
        mainLayout = QVBoxLayout()
        self.input.saveButton = QPushButton('Сохранить')
        self.input.saveButton.clicked.connect(self.getLetter)
        mainLayout.addLayout(grid)
        mainLayout.addWidget(self.input.saveButton)
        self.input.setLayout(mainLayout)
        self.labels = []
        for i in range(self.inputSize):
            for j in range(self.inputSize):
                tmp = Cell(self.input, i, j)
                grid.addWidget(tmp, i, j)
                self.labels.append(tmp)
        self.input.saveButton.clicked.connect(self.getLetter)
        self.input.show()

    def getLetter(self):
        pos = 0
        a = np.array([])
        for label in self.labels:
            if label.isBlack():
                a = np.append(a, 1)
            else:
                a = np.append(a, -1)
            pos += 1
        print(a)
        if self.radioButton.isChecked():
            result, bol = self.hopfild.get_letter(a)

            print(result)
            self.showLog(result)

        elif self.radioButton_2.isChecked():
            vec, index, bol = self.cosco.get_letter(a)
            print(index)
            if bol:
                self.showLog(index)
            else:
                self.showLog("Не удалось распознать образ!")


    def showLog(self, result):
        self.textEdit.setText('')
        self.textEdit.setText(result)

    def showWarning(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Ну вот не надо!")
        msg.setText("Сначала введите параметры!")
        msg.exec_()

    def hopfild_neurons(self):
        print("Hopfild")
        self.hopfild.teach_neurons()

    def kosko_neurons(self):
        print("Kosko")
        self.cosco.teach_neurons()


class Cell(QLabel):
    def __init__(self, parent, i, j):
        QLabel.__init__(self, parent)
        self.resize(30, 30)
        self.x = i
        self.y = j
        self.mousePressEvent = self.changeColor
        self.color = 'white'
        self.setStyleSheet("QLabel { background-color : white;  }")

    def changeColor(self, args):
        if self.color == 'black':
            self.color = 'white'
            self.setStyleSheet("QLabel { background-color : white;  }")
        else:
            self.color = 'black'
            self.setStyleSheet("QLabel { background-color : black;  }")

    def isBlack(self):
        return self.color == 'black'

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())