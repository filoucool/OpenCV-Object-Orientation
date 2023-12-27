import cv2
import numpy as np

# Load YOLO
config_path = 'Model/yolov3-tiny.cfg'
weights_path = 'Model/yolov3-tiny.weights'
classes_file = "Model/coco.names"

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()

    if output_layers.ndim > 1:
        output_layers = output_layers.flatten()

    return [layer_names[i - 1] for i in output_layers]

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Unable to open webcam")

while True:
    ret, frame = cap.read()
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Flatten the indices array if necessary
    if len(indices) > 0 and isinstance(indices[0], (list, np.ndarray)):
        indices = [i[0] for i in indices]

    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    cv2.imshow("OpenCV - Object Orientation Detection", frame)

    c = cv2.waitKey(1)
    if c == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
