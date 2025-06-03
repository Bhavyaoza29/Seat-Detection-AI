import torch
import cv2
import matplotlib.pyplot as plt

# Load the custom trained YOLOv5 model from local path
model = torch.hub.load('C:/Users/Bhavya Oza/yolov5', 'custom', path='C:/Users/Bhavya Oza/Seat Detection - Ai/best.pt', source='local')

# Open webcam
cap = cv2.VideoCapture(0)  # Use the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference (detection) on the frame
    results = model(frame)

    # Render and annotate the frame with detected objects
    annotated_frame = results.render()[0]  # Render boxes and labels

    # Display the frame with annotations using Matplotlib
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show(block=False)  # Non-blocking to keep updating frames
    plt.pause(0.001)  # Small pause to avoid freezing the display

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()






