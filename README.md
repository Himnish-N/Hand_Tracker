# Hand_Tracker
using mediapipe palm recognition and finger recognition

## Overview
This Python script utilizes the MediaPipe library to detect and track hand landmarks in real-time from a webcam feed. It draws landmarks on the detected hands and prints the coordinates of specific landmarks.

## Requirements
- Python 3.x
- OpenCV (cv2) library
- MediaPipe library (`mediapipe`)

## Installation
1. Install Python
2. Install OpenCV library using pip:
    ```
    pip install opencv-python
    ```
3. Install MediaPipe library using pip:
    ```
    pip install mediapipe
    ```

## Usage
1. Connect your webcam to your computer.
2. Run the script `hand_landmark_detection.py`.
3. The script will open a window displaying the webcam feed with hand landmarks detected.
4. The script will print the coordinates of specific landmarks of the detected hand in the terminal.
5. Press `Esc` key to exit the script.

## Additional Notes
- You can change the maximum number of hands to be detected by modifying the `maxHands` parameter in the `handDetector` class constructor.
- You can adjust the model complexity by modifying the `modelComplexity` parameter in the `handDetector` class constructor. Higher complexity may provide better accuracy but may require more computational resources.
- You can adjust the minimum detection and tracking confidence thresholds by modifying the `min_detection_confidence` and `min_tracking_confidence` parameters in the `handDetector` class constructor.
- Landmarks are indexed from 0 to 20 for each hand. You can access specific landmarks' coordinates by modifying the index in the `lmlist` variable.
