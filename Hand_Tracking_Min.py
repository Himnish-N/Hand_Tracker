import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands() #initialize mediapipe hand object
''' parameters are static_image_module,
max_num_hands,
min_detection_confidence,
min_tracking_confidence'''
mpDraw = mp.solutions.drawing_utils

while True: # run the camera
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb
    results = hands.process(imgRGB) #process automatically processes

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draws landmarks on hands


    cv2.imshow("Image", img)
    cv2.waitKey(1)

