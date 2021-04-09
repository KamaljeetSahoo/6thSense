# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(903, 794)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.dashboard_navigator = QtWidgets.QTabWidget(self.centralwidget)
        self.dashboard_navigator.setGeometry(QtCore.QRect(10, 0, 1001, 761))
        self.dashboard_navigator.setObjectName("dashboard_navigator")
        self.hand_symbol_recognition = QtWidgets.QWidget()
        self.hand_symbol_recognition.setObjectName("hand_symbol_recognition")
        self.input_feed = QtWidgets.QLabel(self.hand_symbol_recognition)
        self.input_feed.setGeometry(QtCore.QRect(20, 20, 400, 300))
        self.input_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.input_feed.setObjectName("input_feed")
        self.pose_estimation_feed = QtWidgets.QLabel(self.hand_symbol_recognition)
        self.pose_estimation_feed.setGeometry(QtCore.QRect(470, 20, 400, 300))
        self.pose_estimation_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.pose_estimation_feed.setObjectName("pose_estimation_feed")
        self.roi_feed = QtWidgets.QLabel(self.hand_symbol_recognition)
        self.roi_feed.setGeometry(QtCore.QRect(20, 350, 400, 300))
        self.roi_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.roi_feed.setObjectName("roi_feed")
        self.final_feed = QtWidgets.QLabel(self.hand_symbol_recognition)
        self.final_feed.setGeometry(QtCore.QRect(470, 350, 341, 301))
        self.final_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.final_feed.setObjectName("final_feed")
        self.start_hsr = QtWidgets.QPushButton(self.hand_symbol_recognition)
        self.start_hsr.setGeometry(QtCore.QRect(170, 680, 111, 41))
        self.start_hsr.setObjectName("start_hsr")
        self.out_symbol = QtWidgets.QLabel(self.hand_symbol_recognition)
        self.out_symbol.setGeometry(QtCore.QRect(450, 680, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.out_symbol.setFont(font)
        self.out_symbol.setAlignment(QtCore.Qt.AlignCenter)
        self.out_symbol.setObjectName("out_symbol")
        self.dashboard_navigator.addTab(self.hand_symbol_recognition, "")
        self.ocr = QtWidgets.QWidget()
        self.ocr.setObjectName("ocr")
        self.ocr_upload = QtWidgets.QPushButton(self.ocr)
        self.ocr_upload.setGeometry(QtCore.QRect(350, 50, 171, 71))
        self.ocr_upload.setObjectName("ocr_upload")
        self.ocr_image = QtWidgets.QLabel(self.ocr)
        self.ocr_image.setGeometry(QtCore.QRect(60, 180, 761, 381))
        self.ocr_image.setAlignment(QtCore.Qt.AlignCenter)
        self.ocr_image.setObjectName("ocr_image")
        self.ocr_output = QtWidgets.QTextBrowser(self.ocr)
        self.ocr_output.setGeometry(QtCore.QRect(50, 580, 791, 141))
        self.ocr_output.setObjectName("ocr_output")
        self.dashboard_navigator.addTab(self.ocr, "")
        self.image_captioning = QtWidgets.QWidget()
        self.image_captioning.setObjectName("image_captioning")
        self.ic_upload = QtWidgets.QPushButton(self.image_captioning)
        self.ic_upload.setGeometry(QtCore.QRect(100, 50, 171, 71))
        self.ic_upload.setObjectName("ic_upload")
        self.ic_camera = QtWidgets.QPushButton(self.image_captioning)
        self.ic_camera.setGeometry(QtCore.QRect(620, 50, 161, 71))
        self.ic_camera.setObjectName("ic_camera")
        self.ic_inp_image = QtWidgets.QLabel(self.image_captioning)
        self.ic_inp_image.setGeometry(QtCore.QRect(60, 170, 741, 351))
        self.ic_inp_image.setAlignment(QtCore.Qt.AlignCenter)
        self.ic_inp_image.setObjectName("ic_inp_image")
        self.ic_image_caption = QtWidgets.QTextBrowser(self.image_captioning)
        self.ic_image_caption.setGeometry(QtCore.QRect(40, 570, 821, 141))
        self.ic_image_caption.setObjectName("ic_image_caption")
        self.dashboard_navigator.addTab(self.image_captioning, "")
        self.text_to_speech = QtWidgets.QWidget()
        self.text_to_speech.setObjectName("text_to_speech")
        self.tts_input = QtWidgets.QTextEdit(self.text_to_speech)
        self.tts_input.setGeometry(QtCore.QRect(20, 60, 841, 471))
        self.tts_input.setObjectName("tts_input")
        self.tts_button = QtWidgets.QPushButton(self.text_to_speech)
        self.tts_button.setGeometry(QtCore.QRect(30, 610, 821, 41))
        self.tts_button.setObjectName("tts_button")
        self.dashboard_navigator.addTab(self.text_to_speech, "")
        self.speech_to_text = QtWidgets.QWidget()
        self.speech_to_text.setObjectName("speech_to_text")
        self.dashboard_navigator.addTab(self.speech_to_text, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionhello = QtWidgets.QAction(MainWindow)
        self.actionhello.setObjectName("actionhello")

        self.retranslateUi(MainWindow)
        self.dashboard_navigator.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_feed.setText(_translate("MainWindow", "Input Feed"))
        self.pose_estimation_feed.setText(_translate("MainWindow", "Pose Estimation"))
        self.roi_feed.setText(_translate("MainWindow", "ROI"))
        self.final_feed.setText(_translate("MainWindow", "Final Feed"))
        self.start_hsr.setText(_translate("MainWindow", "Start Here"))
        self.out_symbol.setText(_translate("MainWindow", "Output"))
        self.dashboard_navigator.setTabText(self.dashboard_navigator.indexOf(self.hand_symbol_recognition), _translate("MainWindow", "Hand Symbol Recognition"))
        self.ocr_upload.setText(_translate("MainWindow", "Upload Image"))
        self.ocr_image.setText(_translate("MainWindow", "Input Image"))
        self.ocr_output.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt;\">Output Result</span></p></body></html>"))
        self.dashboard_navigator.setTabText(self.dashboard_navigator.indexOf(self.ocr), _translate("MainWindow", "OCR"))
        self.ic_upload.setText(_translate("MainWindow", "Upload an Image"))
        self.ic_camera.setText(_translate("MainWindow", "Click a Picture"))
        self.ic_inp_image.setText(_translate("MainWindow", "Input Image"))
        self.dashboard_navigator.setTabText(self.dashboard_navigator.indexOf(self.image_captioning), _translate("MainWindow", "Image Captioning"))
        self.tts_button.setText(_translate("MainWindow", "Play Generated Speech"))
        self.dashboard_navigator.setTabText(self.dashboard_navigator.indexOf(self.text_to_speech), _translate("MainWindow", "Text To Speech"))
        self.dashboard_navigator.setTabText(self.dashboard_navigator.indexOf(self.speech_to_text), _translate("MainWindow", "Speech To Text"))
        self.actionhello.setText(_translate("MainWindow", "hello"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
