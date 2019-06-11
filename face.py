import argparse
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import numpy as np
import time
import dlib
parser = argparse.ArgumentParser()
parser.add_argument("-predictor",required = True,help = "path to predictor")
args = parser.parse_args()

print(args.predictor)

vs = VideoStream().start()
time.sleep(1.5)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)

eyelayer = np.zeros((450, 800, 3),dtype='uint8')
eye_mask = eyelayer.copy()
eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_BGR2GRAY)
while True:
    frame = vs.read()
    frame = resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    eyelayer.fill(0)
    eye_mask.fill(0)
    for rect in rects:
        #face detector rect
        #x, y ,w, h = face_utils.rect_to_bb(rect)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,125,0),2)
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        #print(frame.shape)
        
        mouth= shape[60:69]
        mouth_c = shape[48:60]
        leftEye = shape[36:42]
        rightEye = shape[42:48]
    
        cv2.fillPoly(eye_mask,[mouth],255)
        cv2.fillPoly(eye_mask,[mouth_c],255)
        cv2.fillPoly(eye_mask, [leftEye], 255)
        cv2.fillPoly(eye_mask, [rightEye], 255)

        eyelayer = cv2.bitwise_and(frame, frame, mask = eye_mask)
        #location points
        #for points in shape:
            #cv2.circle(frame,tuple(points),2,(0,0,255))
    cv2.imshow("My face",eyelayer)
    key = cv2.waitKey(1) & 0XFF

    if key==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()