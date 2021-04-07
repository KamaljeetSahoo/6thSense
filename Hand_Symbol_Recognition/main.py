import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import cv2
import numpy as np
from torchvision.transforms.transforms import Resize

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model = torchvision.models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(in_features=1024, out_features=len(classes), bias=True)
model.load_state_dict(torch.load('epoch.pt'))

def transformation(image):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
    image = trans(image)
    return image
    
vid = cv2.VideoCapture(0)
with torch.no_grad():
    while(True):
        ret, frame = vid.read()
        #frame = cv2.resize(frame, (224,224))
        roi = frame[50:300, 50:300]
        cv2.rectangle(frame, (50, 50), (300,300), (0,0,200), 2)
        input = transformation(roi)
        output = model(input.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        cv2.putText(frame, classes[predicted], (80,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.imshow('roi', roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
vid.release()
cv2.destroyAllWindows()
print(frame.shape)