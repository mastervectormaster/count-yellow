"""
    GLUE Code Detection Module based on Yolov5
    2022.8.2
    @vectorgeek
"""
from os.path import exists
from matplotlib.pyplot import box
import torch
import cv2

class GlueDetect:
    """
        Glue Detection Class based on Yolov5
    """
    def __init__(self, model_path='glue-yolov5.pt'):
        if not exists(model_path):
            print("Can not find the model.")
            return None
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.initialized = True
        print("Model initialized.")
    
    def detect(self, image_path, need_draw=False):
        """
            Detect Glue (yellow) from specified image.
            @param image_path: image containing (or not) glue.
            @return box: box coordinates (xmin, ymin, xmax, ymax) if detected, None otherwise
        """

        if not self.initialized:
            print("Model is not initialized.")
            return None
        if not exists(image_path):
            print("Can not find the image.")
            return None
        
        detect_results = self.model(image_path)
        sorted_by_confidence_results = detect_results.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
        if len(sorted_by_confidence_results) == 0:
            # print("No QR Code detected")
            return None
        most_confident_box = sorted_by_confidence_results.iloc[0]
        if need_draw:
            image = cv2.imread(image_path)
            self.draw_box(image, most_confident_box)
        return most_confident_box
    
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

    def draw_box(self, image, boxes):
        """
            Draw bounding box on the image.
            @param image_path: image containing (or not) QR Code.
            @param box: box coordinates (xmin, ymin, xmax, ymax)
            @output: new image with bounding box drawn on the original one
        """
        if boxes is None:
            print("No box provided")
            return
        
        for box in boxes:
            start_point = (int(box['xmin']), int(box['ymin']))
            end_point = (int(box['xmax']), int(box['ymax']))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        cv2.imshow('Frame',image)

        # cv2.imwrite('output.jpg', image)