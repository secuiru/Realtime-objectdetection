# Object Detection App

This Python application uses computer vision and a pretrained machine learning model for real-time object detection through a webcam feed. It utilizes the torchvision library for object detection, easyocr for text extraction, and cvzone for face swapping. The application also supports translation and blur effect. It was developed as part of a group project in the Product Design and Implementation course.
![objectdetection1](https://github.com/secuiru/Realtime-objectdetection/assets/98741682/20e06dd9-989a-482b-b544-d36d03884e59)

![image](https://github.com/secuiru/Realtime-objectdetection/assets/98741682/b9b9e8bf-68b4-4a7c-8d75-fed1e6d4cd83)

## Installation
Before running the application, make sure to install the required libraries:
```
pip install opencv-python torch torchvision Pillow easyocr translate cvzone mediapipe
```

## Features
1. Video Stream Mode: Real-time object detection using Faster R-CNN ResNet-50 model.
2. Text Mode: Detect and translate text in the webcam feed using OCR (Optical Character Recognition).
3. Filter Mode: Apply face-swapping filters using pre-loaded images (e.g., Trump, Biden, Will Smith).
4. Mesh Mode: Utilize face mesh detection to add facial landmarks on the user's face.
## Controls
- Toggle Mode Button: Switch between different modes (Video Stream, Text, Filter, Mesh).
- Blur Button: Toggle between different blur effects for object detection.
- Translate Button: Toggle translation on/off for text mode.
- Language Selector: Choose the language for translation from the dropdown menu.
- Face Swap Selector: Choose a face-swapping filter from the dropdown menu.
- Input Device Selector: Choose the webcam input device from the dropdown menu.
## Additional Information
- FPS Display: Shows the frames per second (FPS) of the webcam feed.
- Text Box: Displays detected text and translations.
- Current Mode Label: Indicates the current mode (Video Stream, Text, Filter, Mesh).
