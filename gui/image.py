# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtCore import QString
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import time
import requests



class AnotherWindow(QMainWindow):

	# constructor
	def __init__(self):
		super().__init__()

		# setting geometry
		self.setGeometry(100, 100,
						800, 600)

		# setting style sheet
		self.setStyleSheet("background : lightgrey;")

		# getting available cameras
		self.available_cameras = QCameraInfo.availableCameras()

		# if no camera found
		if not self.available_cameras:
			# exit the code
			sys.exit()

		# creating a status bar
		self.status = QStatusBar()

		# setting style sheet to the status bar
		self.status.setStyleSheet("background : white;")

		# adding status bar to the main window
		# self.setStatusBar(self.status)

		# path to save
		self.save_path = ""

		# creating a QCameraViewfinder object
		self.viewfinder = QCameraViewfinder()

		# showing this viewfinder
		self.viewfinder.show()

		# making it central widget of main window
		self.setCentralWidget(self.viewfinder)

		# Set the default camera.
		self.select_camera(0)

		# creating a tool bar
		toolbar = QToolBar("Camera Tool Bar")

		# adding tool bar to main window
		self.addToolBar(toolbar)

		# creating a photo action to take photo
		click_action = QAction("Click photo", self)

		# adding status tip to the photo action
		click_action.setStatusTip("This will capture picture")

		# adding tool tip
		click_action.setToolTip("Capture picture")


		# adding action to it
		# calling take_photo method
		click_action.triggered.connect(self.click_photo)

		# adding this to the tool bar
		toolbar.addAction(click_action)

		# similarly creating action for changing save folder
		change_folder_action = QAction("Change save location",
									self)

		# adding status tip
		change_folder_action.setStatusTip("Change folder where picture will be saved saved.")

		# adding tool tip to it
		change_folder_action.setToolTip("Change save location")

		# setting calling method to the change folder action
		# when triggered signal is emitted
		change_folder_action.triggered.connect(self.change_folder)

		# adding this to the tool bar
		toolbar.addAction(change_folder_action)


		# creating a combo box for selecting camera
		camera_selector = QComboBox()

		# adding status tip to it
		camera_selector.setStatusTip("Choose camera to take pictures")

		# adding tool tip to it
		camera_selector.setToolTip("Select Camera")
		camera_selector.setToolTipDuration(2500)

		# adding items to the combo box
		camera_selector.addItems([camera.description()
								for camera in self.available_cameras])

		# adding action to the combo box
		# calling the select camera method
		camera_selector.currentIndexChanged.connect(self.select_camera)

		# adding this to tool bar
		toolbar.addWidget(camera_selector)

		# setting tool bar stylesheet
		toolbar.setStyleSheet("background : white;")



		# setting window title
		self.setWindowTitle("PyQt5 Cam")

		# showing the main window
		self.show()

	# method to select camera
	def select_camera(self, i):

		# getting the selected camera
		self.camera = QCamera(self.available_cameras[i])

		# setting view finder to the camera
		self.camera.setViewfinder(self.viewfinder)

		# setting capture mode to the camera
		self.camera.setCaptureMode(QCamera.CaptureStillImage)

		# if any error occur show the alert
		self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

		# start the camera
		self.camera.start()

		# creating a QCameraImageCapture object
		self.capture = QCameraImageCapture(self.camera)

		# showing alert if error occur
		self.capture.error.connect(lambda error_msg, error,
								msg: self.alert(msg))

		# when image captured showing message
		self.capture.imageCaptured.connect(lambda d,
										i: self.status.showMessage("Image captured : "
																	+ str(self.save_seq)))

		# getting current camera name
		self.current_camera_name = self.available_cameras[i].description()

		# inital save sequence
		self.save_seq = 0

	# method to take photo
	def click_photo(self):

		# time stamp
		timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")

		# capture the image and save it on the save path
		self.capture.capture("P:\\6thSense\\gui\\image.jpg")

		# increment the sequence
		self.save_seq += 1

	# change folder method
	def change_folder(self):

		# open the dialog to select path
		path = QFileDialog.getExistingDirectory(self,
												"Picture Location", "")

		# if path is selected
		if path:

			# update the path
			self.save_path = path

			# update the sequence
			self.save_seq = 0

	# method for alerts
	def alert(self, msg):

		# error message
		error = QErrorMessage(self)

		# setting text to the error message
		error.showMessage(msg)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(529, 396)
        self.captionString = ""
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.inputImage = QtWidgets.QLabel(self.centralwidget)
        self.inputImage.setGeometry(QtCore.QRect(10, 90, 241, 261))
        self.inputImage.setObjectName("inputImage")
        self.uploadButton = QtWidgets.QPushButton(self.centralwidget)
        self.uploadButton.setGeometry(QtCore.QRect(340, 40, 91, 23))
        self.uploadButton.setObjectName("uploadButton")
        self.imageLocation = QtWidgets.QTextBrowser(self.centralwidget)
        self.imageLocation.setGeometry(QtCore.QRect(10, 40, 321, 21))
        self.imageLocation.setObjectName("imageLocation")
        self.caption = QtWidgets.QTextBrowser(self.centralwidget)
        self.caption.setGeometry(QtCore.QRect(270, 90, 241, 261))
        self.caption.setObjectName("caption")
        self.camera = QtWidgets.QPushButton(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(440, 40, 75, 23))
        self.camera.setObjectName("camera")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 529, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.uploadButton.clicked.connect(self.upload)
        self.camera.clicked.connect(self.cam)

    def upload(self):
        self.openFileNameDialog()

    def openFileNameDialog(self):
        self.caption.setPlainText(self.captionString)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Image files (*.jpg *.gif)", options=options)
        if fileName:
            self.imageLocation.setPlainText(fileName)
        # r = requests.post(
        #     "https://api.deepai.org/api/densecap",
        #     files={
        #         'image': open(fileName,'rb'),
        #     },
        #     headers={'api-key': 'd1d4ec9c-ac2b-4077-820b-177732187475'}
        # )
        # # text = QString.fromStdString(str)
        # # self.caption.setPlainText()
        # out = r.json()
        # print(out)
        # print(out['output'])
        # for i in out['output']['captions']:
        #     # print(i['caption'])
        #     self.captionString += i['caption']+'\n'
        # print(self.captionString)
        r = requests.post(
        "https://api.deepai.org/api/neuraltalk",
        files={
            'image': open(fileName,'rb'),
        },
        headers={'api-key': 'd1d4ec9c-ac2b-4077-820b-177732187475'}
        )
        self.captionString = r.json()['output']
        self.caption.setPlainText(self.captionString)
        self.captionString = ""

    def cam(self):
        self.w = AnotherWindow()
        self.w.show()
        self.caption.setPlainText(self.captionString)
        fileName = "P:\\6thSense\\gui\\image.jpg" 
        r = requests.post(
        "https://api.deepai.org/api/neuraltalk",
        files={
            'image': open(fileName,'rb'),
        },
        headers={'api-key': 'd1d4ec9c-ac2b-4077-820b-177732187475'}
        )
        self.captionString = r.json()['output']
        self.caption.setPlainText(self.captionString)
        self.captionString = ""


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.inputImage.setText(_translate("MainWindow", "TextLabel"))
        self.uploadButton.setText(_translate("MainWindow", "Upload"))
        self.camera.setText(_translate("MainWindow", "Camera"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
