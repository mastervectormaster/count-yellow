"""
    GLUE Code Detection Module based on Yolov5
    2022.8.2
    @vectorgeek
"""
from os.path import exists
from matplotlib.pyplot import box
import torch
import cv2

X_THRESH = 100
RIGHT_SEQ = [[3, 0], [4, 1], [5, 2], [8, 6], [9, 7]]
LEFT_SEQ = [[3, 0], [4, 1], [5, 2], [8, 6], [9, 7]]


def get_ymin(box):
    return box['ymin']

class GlueDetect:
    """
        Glue Detection Class based on Yolov5
    """
    def __init__(self, model_path='models/glue-yolov5.pt'):
        if not exists(model_path):
            print("Can not find the model.")
            return None
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.initialized = True
        print("Model initialized.")
    
    def detect_from_cv_mat(self, cv_mat, threshold, need_draw=False):

        """
            Detect Glue (yellow) from specified image.
            @param cv_mat: image containing (or not) glue.
            @return box: box coordinates (xmin, ymin, xmax, ymax) if detected, None otherwise
        """

        if not self.initialized:
            print("Model is not initialized.")
            return None
        image = cv_mat.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        detect_results = self.model(image)
        sorted_by_confidence_results = detect_results.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
        boxes = []
        if len(sorted_by_confidence_results) == 0:
            # print("No QR Code detected")
            return None
        else:
            boxes = [sorted_by_confidence_results.iloc[0]]         # always use one with the highest confidence
            for i in range(1, len(sorted_by_confidence_results)):  # use boxes which have higher confidence than THRESHOLD
                box = sorted_by_confidence_results.iloc[i]
                if box['confidence'] > threshold:
                    boxes.append(box)
                else:
                    break
        # most_confident_box = sorted_by_confidence_results.iloc[0]
        if need_draw:
            self.draw_box(cv_mat, boxes)
        return boxes

    def x_split(self, bbox_centers):
        result = {}
        seed_left = -1
        seed_right = -1

        length = len(bbox_centers)
        for i in range(length):
            if bbox_centers[0][0] - bbox_centers[i][0] > X_THRESH:
                seed_left = i
                seed_right = 0
                break
            elif bbox_centers[i][0] - bbox_centers[0][0] > X_THRESH:
                seed_right = i
                seed_left = 0
                break

        for i in range(length):
            if abs(bbox_centers[i][0] - bbox_centers[seed_left][0]) <= abs(bbox_centers[i][0] - bbox_centers[seed_right][0]):
                result[i] = 0
            else:
                result[i] = 1

        return result, bbox_centers[seed_right][0] - bbox_centers[seed_left][0]

    def count(self, bboxes, orientation):
        count = [0] * 10
        if orientation == 'right':
            bboxes.sort(key=get_ymin, reverse=True)
            bbox_centers = list(map(lambda box: (int((box['xmin'] + box['xmax']) / 2), int((box['ymin'] + box['ymax']) / 2)), bboxes))
            row = 0
            left_or_right, x_thresh = self.x_split(bbox_centers)
            length = len(bbox_centers)
            i = 0
            y_thresh = x_thresh / 4
            if x_thresh > 450:
                y_thresh = x_thresh / 5
            while i < length - 1:
                if bbox_centers[i][1] - bbox_centers[i + 1][1] <= y_thresh:
                    count[RIGHT_SEQ[row][0]] = 1
                    count[RIGHT_SEQ[row][1]] = 1
                    i += 2
                else:
                    count[RIGHT_SEQ[row][left_or_right[i]]] = 1
                    i += 1
                row += 1
            if bbox_centers[length - 2][1] - bbox_centers[length - 1][1] > y_thresh:
                count[RIGHT_SEQ[row][left_or_right[length - 1]]] = 1
        else:
            bboxes.sort(key=get_ymin)
            bbox_centers = list(map(lambda box: (int((box['xmin'] + box['xmax']) / 2), int((box['ymin'] + box['ymax']) / 2)), bboxes))
            row = 0
            left_or_right, x_thresh = self.x_split(bbox_centers)
            length = len(bbox_centers)
            i = 0
            y_thresh = x_thresh / 4
            if x_thresh > 450:
                y_thresh = x_thresh / 5
            while i < length - 1:
                if bbox_centers[i + 1][1] - bbox_centers[i][1] <= y_thresh:
                    count[LEFT_SEQ[row][0]] = 1
                    count[LEFT_SEQ[row][1]] = 1
                    i += 2
                else:
                    count[LEFT_SEQ[row][left_or_right[i]]] = 1
                    i += 1
                row += 1
            if bbox_centers[length - 1][1] - bbox_centers[length - 2][1] > y_thresh:
                count[LEFT_SEQ[row][left_or_right[length - 1]]] = 1
        return count


    def draw_boxes(self, image, boxes):
        """
            Draw bounding box on the image.
            @param image_path: image containing (or not) QR Code.
            @param box: box coordinates (xmin, ymin, xmax, ymax)
            @output: new image with bounding box drawn on the original one
        """
        if boxes is None:
            print("No box provided")
            return
        
        for i, box in enumerate(boxes):
            start_point = (int(box['xmin']), int(box['ymin']))
            end_point = (int(box['xmax']), int(box['ymax']))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        cv2.imshow('Frame',image)
