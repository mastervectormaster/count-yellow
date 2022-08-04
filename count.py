import cv2
import glob
import os
from glue_detect import GlueDetect
lst_videos = glob.glob("videos/*.mp4")

i = 0
MODEL_PATH = 'models/best.pt'
qrdetect = GlueDetect(MODEL_PATH)

for v in lst_videos:
    vid = cv2.VideoCapture(v)
    basename = os.path.basename(v)
    basename = basename.replace(".mp4", "")
    while True:
        ret, frame = vid.read()
        if ret:
            box = qrdetect.detect_from_cv_mat(frame, 0.7, True)
            if box == None:
                cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            save_idx = 0
            break

# import cv2
# import numpy as np
# qrdetect = GlueDetect(MODEL_PATH)

# # Create a VideoCapture object and read from input file
# # If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture('videos/glue left 3L 01.mp4')

# # Check if camera opened successfully
# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")

# # Read until video is completed
# while(cap.isOpened()):
#   # Capture frame-by-frame
#   ret, frame = cap.read()
#   if ret == True:

#     # Display the resulting frame
#     qrdetect.detect_from_cv_mat(frame, 0.7, True)
#     cv2.imshow('Frame',frame)

#     # Press Q on keyboard to  exit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       break

#   # Break the loop
#   else: 
#     break

# # When everything done, release the video capture object
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()