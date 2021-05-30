import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('PoseVideos/4.mp4') # this is load from video
# cap = cv2.VideoCapture(0) # this is load from camera

if not cap.isOpened():
    print("Cannot play video")
    exit()

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

while True:
    success, img = cap.read()
    # since mediapipe lib only detect RGB, and our image is in BGR, then we need to convert it BGR->RGB using
    # cvtColor() method
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image to RGB
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

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
