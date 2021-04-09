import cv2
from model import PoseEstimationWithMobileNet
from utils.util import peaks, connection, merge, draw_bodypose, padRightDownCorner, roi
import torch
import numpy as np

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




def r2b(image):
    return image[:, :, [2, 1, 0]]






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PoseEstimationWithMobileNet().to(device)
model.load_state_dict(torch.load('weights/MobileNet_bodypose_model', map_location=lambda storage, loc: storage))

model.eval()

vid = cv2.VideoCapture(0)








while(True):
    ret, image = vid.read()
    imageToTest = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = 'Mobilenet')
    all_peaks = peaks(heatmap, 0.1)
    connection_all, special_k = connection(all_peaks, paf, imageToTest)
    candidate, subset = merge(all_peaks, connection_all, special_k)
    canvas = draw_bodypose(image.copy(), candidate, subset, 0.3)
    canvas1 ,crop= roi(image.copy(), candidate, subset, 0.3)
    cv2.imshow('frame', r2b(canvas))
    for i,j in enumerate(crop):
         cv2.imshow('frame_'+str(i), r2b(j))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
