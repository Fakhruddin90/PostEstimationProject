import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):

        # since mediapipe lib only detect RGB, and our image is in BGR, then we need to convert it BGR->RGB using
        # cvtColor() method
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert image to RGB
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList



def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')  # this is load from video
    # cap = cv2.VideoCapture(0) # this is load from camera
    pTime = 0
    detector = poseDetector()

    if not cap.isOpened():
        print("Cannot play video")
        exit()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        if len(lmList) != 0:
            print(lmList[14])

        cv2.circle(img, (lmList[14][1], lmList[14][2]), 20, (0, 0, 255), cv2.FILLED)

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

if __name__ == "__main__":
    main()