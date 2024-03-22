import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() #initialize mediapipe hand object
''' parameters are static_image_module,
max_num_hands,
min_detection_confidence,
min_tracking_confidence'''
mpDraw = mp.solutions.drawing_utils

prevtime = 0
curtime = 0

while True: # run the camera
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb
    results = hands.process(imgRGB) #process automatically processes

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy) #gives us the id and the position of the landmark in integer

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draws landmarks on hands


    curtime = time.time()
    fps = 1/(curtime-prevtime)
    prevtime = curtime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),3) # gives us the fps of the image

    cv2.imshow("Image", img)
    cv2.waitKey(1)

