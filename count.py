import cv2
import glob
import os

from matplotlib.pyplot import box
from glue_detect import GlueDetect
from functools import reduce

lst_videos = glob.glob("videos/glue right 3L 02.mp4")

MODEL_PATH = 'models/glue-yolov5.pt'
glue_detect = GlueDetect(MODEL_PATH)

for v in lst_videos:
    vid = cv2.VideoCapture(v)
    while True:
        ret, frame = vid.read()
        if ret:
            bboxes = glue_detect.detect_from_cv_mat(frame, 0.6)
            if bboxes != None and len(bboxes) > 4:
                count = glue_detect.count(bboxes, 'left')
                print(count)
                glue_detect.draw_boxes(frame, bboxes)
                cv2.imshow('Frame', frame)
                # cv2.waitKey(1000)
            else:
                cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break