from layout import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage
import cv2, imutils

class DashBoard(Ui_MainWindow):
    def __init__(self, MainWindow):
        super(DashBoard, self).__init__(MainWindow)
        self.start_hsr.clicked.connect(self.loadImage)
    
    def clicked(self):
        print("clicked")

    def loadImage(self):
        camera = True
        if camera:
            vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            img, image = vid.read()
            #self.image  = imutils.resize(self.image,height = 800)
            #image = imutils.resize(self.image,width=800)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            gray = QImage(frame_gray, frame_gray.shape[1],frame_gray.shape[0],frame_gray.strides[0],QImage.Format_RGB888)
            self.final_feed.setPixmap(QtGui.QPixmap.fromImage(image))
            self.input_feed.setPixmap(QtGui.QPixmap.fromImage(gray))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = DashBoard(MainWindow)
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())