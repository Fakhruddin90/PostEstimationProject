import cv2
import mediapipe as mp
import time
import PoseModule as pm

userInput = 0

userInput = int(input("0: Video File \t\t 1: Camera \n"))

if userInput == 0:
    cap = cv2.VideoCapture('PoseVideos/4.mp4')  # this is load from video
elif userInput == 1:
    cap = cv2.VideoCapture(0)  # this is load from camera

pTime = 0
detector = pm.poseDetector()

if not cap.isOpened():
    print("Cannot play video")
    exit()


while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)
    print(lmList)
    # if len(lmList) != 0:
    #     print(lmList[14]) # print the id 14 of landmark
    # cv2.circle(img, (lmList[14][1], lmList[14][2]), 20, (0, 0, 255), cv2.FILLED) # draw id 14 landmarks

    # calculate frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # display text on screen
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, "Press q to quit the video", (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()