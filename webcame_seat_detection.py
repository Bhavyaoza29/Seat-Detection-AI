import torch
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Load the custom trained YOLOv5 model from local path
model = torch.hub.load('C:/Users/Bhavya Oza/yolov5', 'custom', path='C:/Users/Bhavya Oza/Seat Detection - Ai/best.pt', source='local') # Change your 'path' for best.pt here

# Open webcam
cap = cv2.VideoCapture(0)  # Use the default webcam

# Create a Tkinter window
root = tk.Tk()
root.title("YOLOv5 Object Detection")

# Create a label widget for displaying the frames
label = Label(root)
label.pack()

def update_frame():
    # Capture the next frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    # Perform inference (detection) on the frame
    results = model(frame)

    # Render and annotate the frame with detected objects
    annotated_frame = results.render()[0]  # Render boxes and labels

    # Convert the frame (BGR to RGB) for displaying with Tkinter
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to Image format using PIL
    pil_image = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=pil_image)

    # Update the image on the Tkinter label
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Call this function again after 10ms
    root.after(10, update_frame)

# Start the frame update
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam
cap.release()
