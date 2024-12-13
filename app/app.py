import os
import cv2
import numpy as np
from keras import metrics
from keras import models
from ultralytics import YOLO

video_path = '/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/assets/v5.mp4'
segmentation_model = YOLO("/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/app/lane_segmentation/runs/segment/train/weights/best.pt")
detection_model = YOLO("/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/app/road_detection/runs/detect/train/weights/best.pt")
steering_model = models.load_model("/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/app/steering_angle_prediction/best.h5", custom_objects={'mse': metrics.MeanSquaredError()})
steering_image_path = '/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/assets/steer-wheel.png'
classes_names = {
    0: 'Traffic Light Signal', 1: 'Stop Signal', 2: 'Speedlimit Signal', 3: 'Crosswalk Signal',
    4: 'Crosswalk', 5: 'Pedestrian', 6: 'Bus', 7: 'Car', 8: 'Truck'
}

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

def rotate_steering_wheel(steering_image_path, steering_angle):
    steering_image = cv2.imread(steering_image_path, cv2.IMREAD_UNCHANGED)  # Includes transparency (RGBA)
    (h, w) = steering_image.shape[:2]
    center = (w // 2, h // 2)
    scaled_angle = -steering_angle * 180
    rotation_matrix = cv2.getRotationMatrix2D(center, scaled_angle, 1.0)
    rotated_image = cv2.warpAffine(steering_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGRA2BGR)
    
    return rotated_image

frame_count = 0
frame_skip = 3

while (True):
    ret, frame = cap.read()
    
    if not ret:
        break  

    if frame_count % frame_skip == 0:
        segmentation_results = segmentation_model.predict(frame, imgsz=(320, 320), conf=0.25, verbose=False)

        if segmentation_results[0].masks:
            mask = segmentation_results[0].masks.data[0].cpu().numpy() 
            mask = (mask * 255).astype('uint8')
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])) 

            blue_overlay = np.zeros_like(frame)
            blue_overlay[:, :, 2] = mask_resized 

            colored_frame = cv2.bitwise_and(blue_overlay, blue_overlay, mask=mask_resized)

            blended = cv2.addWeighted(frame, 1.0, colored_frame, 0.3, 0)
        else:
            blended = frame

        detection_results = detection_model.predict(frame, imgsz=(320, 320), conf=0.25, verbose=False)
        boxes = detection_results[0].boxes
        boxes_xyxy = boxes.xyxy  
        confidences = boxes.conf  
        class_ids = boxes.cls 

        for box, conf, class_id in zip(boxes_xyxy, confidences, class_ids):
            class_name = classes_names.get(int(class_id), 'Unknown')
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(blended, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (100, 100))
        input_data = np.expand_dims(resized_frame, axis=(0, -1))
        prediction = steering_model.predict(input_data, verbose=0)
        steering_angle = prediction[0][0]
        rotated_steering_wheel = rotate_steering_wheel(steering_image_path, steering_angle)

        cv2.imshow("STEERING WHEEL", rotated_steering_wheel)
        cv2.imshow('VIDEO SEGMENTATION AND DETECTION', blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()