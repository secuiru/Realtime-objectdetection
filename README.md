# Object Detection App
This Python application uses computer vision for real-time object detection and interaction through a webcam feed. It utilizes the torchvision library for object detection, easyocr for text extraction, and cvzone for face swapping. The application also supports translation and blur effect.
![objectdetection1](https://github.com/secuiru/Realtime-objectdetection/assets/98741682/20e06dd9-989a-482b-b544-d36d03884e59)


## Installation
Before running the application, make sure to install the required libraries:

bash
Copy code
pip install opencv-python torch torchvision Pillow easyocr translate cvzone mediapipe
## Usage
Run the application using the following command:

bash
Copy code
python your_script_name.py
Make sure to replace your_script_name.py with the actual name of your Python script.

## Features
1. Video Stream Mode: Real-time object detection using Faster R-CNN ResNet-50.
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
