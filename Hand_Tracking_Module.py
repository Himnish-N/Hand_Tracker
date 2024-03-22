import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.min_detection_confidence, self.min_tracking_confidence) #initialize mediapipe hand object
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to rgb
        self.results = self.hands.process(imgRGB) #process automatically processes

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # draws landmarks on hands
        return img


                # for id, lm in enumerate(handLms.landmark):
                #     #print(id, lm)
                #     h, w, c = img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     print(id, cx, cy) #gives us the id and the position of the landmark in integer
                    

def main():
    prevtime = 0
    curtime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True: # run the camera
        success, img = cap.read()
        img = detector.find_hands(img)


        curtime = time.time()
        fps = 1/(curtime-prevtime)
        prevtime = curtime

        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),3) # gives us the fps of the image

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()