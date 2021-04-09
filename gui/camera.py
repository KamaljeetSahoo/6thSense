
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
cap = cv2.VideoCapture(0)

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):

        while (True):
            ret, frame = cap.read()

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)



class camera(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Camera'
        self.left = 0
        self.top = 0
        self.width = 640
        self.height = 480


        self.cameraUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def cameraUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(100, 120)
        self.label.resize(640, 480)

        camera_button = QPushButton("camera_click", self)
        camera_button.move(50, 50)
        camera_button.clicked.connect(self.click_picture)




        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()


        self.show()



    def click_picture(self):
        while (True):
            _, frame= cap.read()
            img_name = "out.png"
            cv2.imwrite(img_name,frame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = camera()
    ex.show()
    sys.exit(app.exec_())