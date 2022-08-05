import cv2
import glob
import os

from matplotlib.pyplot import box
from glue_detect import GlueDetect
from functools import reduce

input = "videos/glue right 5L 01.mp4"
out_video = 'output.mp4'
MODEL_PATH = 'models/glue-yolov5.pt'
glue_detect = GlueDetect(MODEL_PATH)

vid = cv2.VideoCapture(input)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
size = (frame_width, frame_height)
output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
while True:
    ret, frame = vid.read()
    if ret:
        bboxes = glue_detect.detect_from_cv_mat(frame, 0.6)
        if bboxes != None and len(bboxes) > 4:
            count = glue_detect.count(bboxes, 'left')
            count_str = reduce(lambda a,b: str(a)+str(b), count)
            glue_detect.draw_boxes(frame, bboxes)
            frame = cv2.putText(frame, count_str, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            output.write(frame)
            # cv2.waitKey(1000)
        else:
            cv2.imshow('Frame', frame)
            output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
output.release()
cv2.destroyAllWindows()