import os
import cv2
import numpy as np
from keras import metrics
from keras import models
from ultralytics import YOLO

video_path = '/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/assets/v3.mp4'
model = YOLO("/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/app/lane_segmentation/runs/segment/train/weights/best.pt")

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  

    results = model.predict(frame, imgsz=(640, 640), conf=0.25, verbose=False)
    print(results[0].masks)
    if results[0].masks:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask * 255).astype('uint8')
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        blue_mask = np.zeros_like(frame)
        blue_mask[:, :, 2] = mask_resized
        blended = cv2.addWeighted(frame, 0.7, blue_mask, 0.3, 0)
    else:
        blended = frame
    
    cv2.imshow('VIDEO SEGMENTATION FRAME', blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


classes_names = {
    0: 'Traffic Light Signal', 1: 'Stop Signal', 2: 'Speedlimit Signal', 3: 'Crosswalk Signal',
    4: 'Crosswalk', 5: 'Pedestrian', 6: 'Bus', 7: 'Car', 8: 'Truck'
}

def predict_image(image):
    results = model.predict(image, conf=0.1, show=True)
    return results

def draw_road_detection_bounding_boxes(image):
    results = predict_image(image)
    boxes = results[0].boxes
    boxes_xyxy = boxes.xyxy  
    confidences = boxes.conf  
    class_ids = boxes.cls 

    for box, conf, class_id in zip(boxes_xyxy, confidences, class_ids):
        class_name = classes_names.get(int(class_id), 'Unknown')
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return results


# model = models.load_model("/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/app/steering_angle_prediction/best.h5", custom_objects={'mse': metrics.MeanSquaredError()})
# video_path = '/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/assets/v5.mp4'
# steering_image_path = '/Users/princekhunt/Documents/Portfolio/Self-Drive-Car/assets/steer-wheel.png'
# cap = cv2.VideoCapture(video_path)

# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# def rotate_steering_wheel(steering_image_path, steering_angle):
#     steering_image = cv2.imread(steering_image_path, cv2.IMREAD_UNCHANGED)
#     (h, w) = steering_image.shape[:2]
#     center = (w // 2, h // 2)
#     scaled_angle = -steering_angle * 100
#     rotation_matrix = cv2.getRotationMatrix2D(center, scaled_angle, 1.0) 
#     rotated_image = cv2.warpAffine(steering_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
#     return rotated_image

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break  

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     resized_frame = cv2.resize(gray_frame, (100, 100)) 
#     input_data = np.expand_dims(resized_frame, axis=(0, -1))  
#     prediction = model.predict(np.expand_dims(resized_frame, axis=0), verbose=0)
#     steering_angle = prediction[0][0]  
#     print(steering_angle)
#     rotated_steering_wheel = rotate_steering_wheel(steering_image_path, steering_angle)
#     cv2.imshow("STEERING WHEEL", rotated_steering_wheel)
#     cv2.putText(frame, f"STEERING ANGLE: {steering_angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('STEERING ANGLE PREDICTION', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()