# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainaud.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import time
import numpy as np
import numpy as np
import cv2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(568, 377)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input = QtWidgets.QLabel(self.centralwidget)
        self.input.setGeometry(QtCore.QRect(50, 60, 200, 200))
        self.input.setObjectName("input")
        self.output = QtWidgets.QLabel(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(320, 60, 200, 200))
        self.output.setObjectName("output")
        self.clickButton = QtWidgets.QPushButton(self.centralwidget)
        self.clickButton.setGeometry(QtCore.QRect(250, 20, 75, 23))
        self.clickButton.setObjectName("clickButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 568, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.clickButton.clicked.connect(self.loadImage)

    def loadImage(self):
        camera = True
        if camera:
            vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            img, self.image = vid.read()
            self.image  = imutils.resize(self.image,height = 800)
            image = imutils.resize(self.image,width=800)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            gray = QImage(frame_gray, frame_gray.shape[1],frame_gray.shape[0],frame_gray.strides[0],QImage.Format_RGB888)
            self.input.setPixmap(QtGui.QPixmap.fromImage(image))
            self.output.setPixmap(QtGui.QPixmap.fromImage(gray))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input.setText(_translate("MainWindow", "TextLabel"))
        self.output.setText(_translate("MainWindow", "TextLabel"))
        self.clickButton.setText(_translate("MainWindow", "click me"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
