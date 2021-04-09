import os

import sys

from PyQt5 import QtCore, QtMultimedia

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():

    filename = os.path.join(CURRENT_DIR, "beal.wav")

    app = QtCore.QCoreApplication(sys.argv)

    QtMultimedia.QSound.play(filename)

    # end in 5 seconds:

    # QtCore.QTimer.singleShot(5 * 1000, app.quit)

    sys.exit(app.exec_())

if __name__ == "__main__":

    main()
# import sys

# from PyQt5 import QtCore, QtWidgets, QtMultimedia

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     filename = 'beal.wav'
#     fullpath = QtCore.QDir.current().absoluteFilePath(filename) 
#     url = QtCore.QUrl.fromLocalFile(fullpath)
#     content = QtMultimedia.QMediaContent(url)
#     player = QtMultimedia.QMediaPlayer()
#     player.setMedia(content)
#     player.play()
#     sys.exit(app.exec_())