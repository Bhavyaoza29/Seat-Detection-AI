# 🪑 Vacant Seat Detection: Making Campus Life Easier

## 👥 Team Members
Bhavya Oza, Yashvi T, Freya Trivedi, Jiya Patel

## University
UIT, Karnavati University

---

## 🚀 Project Overview
Finding a vacant seat during peak hours in a university canteen or library is harder than it should be. Our project tackles this everyday problem by detecting vacant seats in real-time, using computer vision and deep learning.

Built with **YOLOv5**, **OpenCV**, and **Tkinter**, this system helps students quickly locate available seats, saving valuable time and reducing daily frustration.

---

## 🎯 Why We Built This
- **Save time:** Help students avoid wandering around searching for free seats.  
- **Reduce stress:** Ease peak-time crowd management in shared spaces.  
- **Leverage AI:** Build smarter campus facilities through real-time seat detection.

---

## 🛠️ Tech Stack

| Purpose           | Tools                      |
|-------------------|----------------------------|
| Object Detection  | YOLOv5                     |
| Image Processing  | OpenCV                     |
| GUI (Desktop App) | Tkinter                    |
| Backend (Planned) | Flask                      |
| Frontend (Planned)| HTML, CSS, JavaScript      |
| Data Annotation   | Roboflow                   |
| Model Training Environment | Anaconda + YOLOv5     |

---

## 🧠 How It Works

### Data Collection
- Captured canteen images ourselves  
- Used Roboflow datasets for libraries

### Annotation and Model Training
- Labeled data and trained a custom YOLOv5 model (`best.pt`)

### Real-Time Detection
- Webcam captures live feed  
- YOLOv5 detects seats and occupancy  
- Tkinter displays real-time annotated frames

### Next Step
- Extend it into a web app using Flask for real-time monitoring

---

## 📸 What You’ll See
- Your webcam will open automatically  
- The system will detect and highlight vacant and occupied seats in real-time  
- The annotated frames will be displayed inside a simple Tkinter window

---
