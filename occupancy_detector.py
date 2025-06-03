import torch
import cv2
import numpy as np

# Load YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using small model for faster inference

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    # box format [x1, y1, x2, y2]
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # calculate intersection area
    inter_area = max(0, min(x2, x2_2) - max(x1, x1_2)) * max(0, min(y2, y2_2) - max(y1, y1_2))
    # calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    # calculate IoU
    iou = inter_area / union_area
    return iou

# Load an image or video to detect
img = cv2.imread('"C:\Users\Bhavya Oza\Seat Detection - Ai\sample videos\istockphoto-628741708-640_adpp_is.mp4"')  # Replace with your image/video path

# Perform inference with YOLOv5 model
results = model(img)

# Get detections (boxes, labels, and confidences)
boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
labels = results.names  # YOLO class labels
confidences = results.conf[0].cpu().numpy()

# Loop through all detected objects
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    label = int(results.pred[0][i][5])  # Get object label
    confidence = confidences[i]
    
    # Check if the object is a "person" or "chair" (labels in YOLOv5 are indexed from 0)
    if labels[label] == 'person':
        person_box = box  # Save the person bounding box
    if labels[label] == 'chair':
        chair_box = box  # Save the chair bounding box

# Calculate IoU to check if a person is on a chair
iou = calculate_iou(person_box, chair_box)
if iou > 0.5:  # Set a threshold for occupancy
    print("The chair is occupied!")
else:
    print("The chair is not occupied.")
