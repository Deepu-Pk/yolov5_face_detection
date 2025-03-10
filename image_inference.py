# Load third-party dependencies
import cv2 
import numpy as np 
import torch 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image",type = str ,help = "input face image",required = True)
args = parser.parse_args()


# Load the image 
img = cv2.imread(args.image)


# Load yolov5s face detection model 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yoylov5/runs/train/exp4/weights/best.pt', force_reload=True)

# Model predictions 
results = model(img)

# Save the model predictions 
output_img = np.squeeze(results.render())
cv2.imwrite("output_img.jpg",output_img)