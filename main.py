import cv2
import numpy as np
from Classes.tracker import EuclideanDistTracker
from imutils.video import VideoStream

# Load Model and Tracker
tracker = EuclideanDistTracker()
ObjectClass = 'Caffe/object_detection_classes_pascal_voc.txt'
CLASSES = [line.strip() for line in open(ObjectClass)]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe('Caffe/MobileNetSSD_deploy.prototxt.txt', 'Caffe/MobileNetSSD_deploy.caffemodel')

def get_centroid_coord(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return cx, cy

def Object_detection(frame):
    object_name = None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    rect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            if str(CLASSES[class_id]).lower() in ['person']:
                rect.append([startX, startY, endX, endY])
                object_width = endX - startX
                label = "{}: {:.2f}%".format(
                    CLASSES[class_id], confidence * 100)
                object_name = CLASSES[class_id]

    return rect, object_name


def Tracker(rect):
    objects = tracker.update(rect)
    return objects

VideoPath = "Walking.mp4"
# video = VideoStream(scr=VideoPath).start()
video = cv2.VideoCapture(VideoPath)
while True:
    # frame = video.read()
    ret, frame = video.read()

    rect, object_name = Object_detection(frame)

    objects = Tracker(rect)
    for StartX, StartY, endX, endY, objectID in objects:
        if object_name != None:
            object_center_x, object_center_y = get_centroid_coord(StartX, StartY, endX, endY)
            cv2.rectangle(frame, (StartX, StartY),(endX, endY), (0,255,0), 2)
            cv2.circle(frame, (object_center_x,object_center_y), 2, (0, 255, 145), 3)
            cv2.putText(frame, str(object_name), (object_center_x - 10, object_center_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()