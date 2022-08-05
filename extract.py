import cv2
import glob
import os

lst_videos = glob.glob("videos/*.mp4")

i = 0
save_idx = 0
for v in lst_videos:
    vid = cv2.VideoCapture(v)
    basename = os.path.basename(v)
    print(basename)
    basename = basename.replace(".mp4", "")
    while True:
        ret, frame = vid.read()
        if ret:
            if i == 2:
                save_path = "label_img/{}_{}.jpg".format(basename, save_idx)
                print("     Save ", save_path)
                cv2.imwrite(save_path, frame)
                save_idx += 1
                i = 0
            else:
                i += 1
        else:
            save_idx = 0
            break
