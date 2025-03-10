# Load third-party dependencies
import cv2 
import numpy as np 
import torch

# Open the default camera
cap = cv2.VideoCapture(0)


if(not cap.isOpened()):
    raise Exception("Can not open video file or camera module")
# Load yolov5s face detection model 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yoylov5/runs/train/exp4/weights/best.pt', force_reload=True)

while True:
    ret, frame = cap.read()
    if(not ret):
        print(f"[INFO] Video completed")
        break

    # Filp the mirror image 
    frame = cv2.flip(frame,1)
    
    # Display the captured frame
    results = model(frame)

    # Save the model predictions 
    output_frame = np.squeeze(results.render())

    cv2.imshow("Yolov5s face detection",output_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        print(f"[INFO] Video is stopped")
        break

# Release the capture and writer objects
cap.release()
cv2.destroyAllWindows()