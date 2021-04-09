from layout import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import requests
from PyQt5.QtGui import QImage
import cv2, imutils
from utils.util import peaks, connection, merge, draw_bodypose, padRightDownCorner, roi
from model import PoseEstimationWithMobileNet
import torch
import numpy as np
import sys
import time

def Net_Prediction(model, image, device, backbone = 'Mobilenet'):
    scale_search = [1]
    stride = 8
    padValue = 128
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
    paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
    
    for m in range(len(scale_search)):
        scale = scale_search[m]
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        # pad right and down corner to make sure image size is divisible by 8
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)
   
        with torch.no_grad():
            if backbone == 'CMU':
                Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)
                _paf = Mconv7_stage6_L1.cpu().numpy()
                _heatmap = Mconv7_stage6_L2.cpu().numpy()
            elif backbone == 'Mobilenet':
                stages_output = model(data)
                _paf = stages_output[-1].cpu().numpy()
                _heatmap = stages_output[-2].cpu().numpy()  
            
        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print(heatmap.shape)
        
        paf = np.transpose(np.squeeze(_paf), (1, 2, 0))  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        #print(paf.shape)
        heatmap_avg += heatmap / len(scale_search)
        paf_avg += paf / len(scale_search)
        
    return heatmap_avg, paf_avg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PoseEstimationWithMobileNet().to(device)
model.load_state_dict(torch.load('weights/MobileNet_bodypose_model', map_location=lambda storage, loc: storage))
model.eval()

class AnotherWindow(QMainWindow):

	# constructor
	def __init__(self):
		super().__init__()

		self.setGeometry(100, 100, 800, 600)
		self.setStyleSheet("background : lightgrey;")
		self.available_cameras = QCameraInfo.availableCameras()
		if not self.available_cameras:
			sys.exit()

		self.status = QStatusBar()
		self.status.setStyleSheet("background : white;")
		self.save_path = ""

		self.viewfinder = QCameraViewfinder()
		self.viewfinder.show()
		self.setCentralWidget(self.viewfinder)
		self.select_camera(0)
		toolbar = QToolBar("Camera Tool Bar")
		self.addToolBar(toolbar)
		click_action = QAction("Click photo", self)
		click_action.setStatusTip("This will capture picture")
		click_action.setToolTip("Capture picture")
		click_action.triggered.connect(self.click_photo)
		toolbar.addAction(click_action)
		change_folder_action = QAction("Change save location",
									self)
		change_folder_action.setStatusTip("Change folder where picture will be saved saved.")
		change_folder_action.setToolTip("Change save location")
		change_folder_action.triggered.connect(self.change_folder)
		toolbar.addAction(change_folder_action)
		camera_selector = QComboBox()
		camera_selector.setStatusTip("Choose camera to take pictures")
		camera_selector.setToolTip("Select Camera")
		camera_selector.setToolTipDuration(2500)

		camera_selector.addItems([camera.description() for camera in self.available_cameras])
		camera_selector.currentIndexChanged.connect(self.select_camera)
		toolbar.addWidget(camera_selector)
		toolbar.setStyleSheet("background : white;")
		self.setWindowTitle("PyQt5 Cam")
		self.show()

	def select_camera(self, i):
		self.camera = QCamera(self.available_cameras[i])
		self.camera.setViewfinder(self.viewfinder)
		self.camera.setCaptureMode(QCamera.CaptureStillImage)
		self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
		self.camera.start()
		self.capture = QCameraImageCapture(self.camera)
		self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg))
		self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image captured : " + str(self.save_seq)))

		self.current_camera_name = self.available_cameras[i].description()

		self.save_seq = 0

	def click_photo(self):

		timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
		self.capture.capture("P:\\6thSense\\gui\\image.jpg")
		self.save_seq += 1

	def change_folder(self):
		path = QFileDialog.getExistingDirectory(self, "Picture Location", "")

		if path:
			self.save_path = path
			self.save_seq = 0

	def alert(self, msg):
		error = QErrorMessage(self)
		error.showMessage(msg)



class DashBoard(Ui_MainWindow):
    def __init__(self, MainWindow):
        super(DashBoard, self).__init__(MainWindow)
        self.start_hsr.clicked.connect(self.loadImage)
        self.ic_upload.clicked.connect(self.upload)
        self.ic_camera.clicked.connect(self.cam)
        self.captionString = ""
        

    
    def clicked(self):
        print("clicked")

    def upload(self):
        self.openFileNameDialog()

    def openFileNameDialog(self):
        self.ic_image_caption.setPlainText(self.captionString)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Image files (*.jpg *.gif)", options=options)
        
        r = requests.post(
        "https://api.deepai.org/api/neuraltalk",
        files={
            'image': open(fileName,'rb'),
        },
        headers={'api-key': 'd1d4ec9c-ac2b-4077-820b-177732187475'}
        )
        self.captionString = r.json()['output']
        self.ic_image_caption.setPlainText(self.captionString)
        self.captionString = ""
    
    def cam(self):
        self.w = AnotherWindow()
        self.w.show()
        self.ic_image_caption.setPlainText(self.captionString)
        fileName = "P:\\6thSense\\gui\\image.jpg" 
        r = requests.post(
        "https://api.deepai.org/api/neuraltalk",
        files={
            'image': open(fileName,'rb'),
        },
        headers={'api-key': 'd1d4ec9c-ac2b-4077-820b-177732187475'}
        )
        self.captionString = r.json()['output']
        self.ic_image_caption.setPlainText(self.captionString)
        self.captionString = ""

    def loadImage(self):
        camera = True
        if camera:
            vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            ret, image = vid.read()
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imageToTest = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = 'Mobilenet')
            #self.image  = imutils.resize(self.image,height = 800)
            #image = imutils.resize(self.image,width=800)
            all_peaks = peaks(heatmap, 0.1)
            connection_all, special_k = connection(all_peaks, paf, imageToTest)
            candidate, subset = merge(all_peaks, connection_all, special_k)
            canvas = draw_bodypose(image, candidate, subset, 0.3)
            #frame, frame_grey
            frame = canvas[:, :, [2, 1, 0]]
            frame = np.require(frame, np.uint8, 'C')

            image = QImage(frame, frame.shape[1], frame.shape[0], frame.shape[2]*frame.shape[1],QImage.Format_RGB888)
            gray = QImage(img, img.shape[1],img.shape[0],img.strides[0],QImage.Format_RGB888)
            self.final_feed.setPixmap(QtGui.QPixmap.fromImage(image))
            self.input_feed.setPixmap(QtGui.QPixmap.fromImage(gray))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                vid.release()
                break


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = DashBoard(MainWindow)
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())