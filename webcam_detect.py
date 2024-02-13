import cv2
import numpy as np
import os
import wget

# Function to download YOLO model files if not already downloaded
def download_yolo_files():
    model_folder = 'yolo_model'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    files = ['yolov3.weights', 'yolov3.cfg', 'coco.names']
    urls = ['https://pjreddie.com/media/files/yolov3.weights',
            'https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg',
            'https://github.com/pjreddie/darknet/raw/master/data/coco.names']
    
    for file, url in zip(files, urls):
        file_path = os.path.join(model_folder, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            wget.download(url, out=model_folder)
            print(" Done!")

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet('yolo_model/yolov3.weights', 'yolo_model/yolov3.cfg')
    classes = []
    with open('yolo_model/coco.names', 'r') as f:
        classes = f.read().splitlines()
    return net, classes

# Detect objects using YOLO
def detect_objects(frame, net, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        object_count = {}
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            # Counting objects
            if label not in object_count:
                object_count[label] = 1
            else:
                object_count[label] += 1

        # Display object count
        text = ", ".join([f"{label}: {count}" for label, count in object_count.items()])
        cv2.rectangle(frame, (width - 300, height - 30), (width, height), (255, 255, 255), -1)
        cv2.putText(frame, text, (width - 290, height - 10), font, 1, (0, 165, 255), 2)

# Main function
def main():
    download_yolo_files()
    net, classes = load_yolo()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        detect_objects(frame, net, classes)
        cv2.imshow("Object Detection", frame)
        key = cv2.waitKey(1)
        if key == 27: # Press ESC to exit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
