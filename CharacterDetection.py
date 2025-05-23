import cv2 as cv
import mediapipe as mp #a top-level module
import argparse
import math
from ultralytics import YOLO
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
from .deep_sort.nn_matching import NearestNeighborDistanceMetric
from .deep_sort.tools import generate_detections as gdet
import os
import time
# Test OpenCV

'''
print("OpenCV version:", cv.__version__)

# Test MediaPipe
mp_face_detection = mp.solutions.face_detection
print("MediaPipe import successful")

encoder = gdet.create_box_encoder('deep_sort/model_data/mars-small128.pb', batch_size=1)

print("hello world")
'''

class Frame_Summary:
    def __init__(self, frame, trackArray):
        self.frame = frame
        self.trackArray = trackArray
class Tracking_Object:
    def __init__(self, id, xCenter, yCenter):
        self.id = id
        self.xCenter = xCenter
        self.yCenter = yCenter


def process_image(img, face_detection):


   img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   out = face_detection.process(img_rgb)
   # relative keypoints -- face landmarks
   # bounding box -- where is the face
   H, W, _ = img.shape
   if out.detections is not None:  # in case no face detected
       for detection in out.detections:
           location = detection.location_data
           bbox = location.relative_bounding_box
           x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height




           #calculate the relative distance between two eyes
           #if the distance is big, then the face is facing the camera
           #if the distance is small, then the face is looking at the side
           face_score = faceCameraRate(location.relative_keypoints[1], location.relative_keypoints[0], w)




           # since x1, y1 are all relative points, convert to actual size
           x1 = max(0, int(x1 * W))
           y1 = max(0, int(y1 * H))
           w = int(w * W)
           h = int(h * H)
           x2 = min(W, x1 + w)
           y2 = min(H, y1 + h)


           if x2 > x1 and y2 > y1:
               roi = img[y1: y2, x1: x2, :]
               if roi.size != 0:
           # do some manipulation to the face
                   img[y1: y2, x1: x2, :] = cv.blur(roi,(30, 30))  # this function generates a new image with blur
           #img[y1: y1 + h, x1: x1 + w, :] = addInfo(face_score, img[y1: y1 + h, x1: x1 + w, :], (255, 255, 255))
   return img




def addInfo(score, img, fontColor):
   color = (255, 0, 0)
   thickness = 4
   H, W, _ = img.shape
   cv.rectangle(img, (0, 0), (W, H), color, thickness)


   font = cv.FONT_HERSHEY_SIMPLEX
   font_scale = 1
   thickness = 2
   position = (10, 20)


   cv.putText(img, f'{score:.2f}', position, font, font_scale, fontColor, thickness, cv.LINE_AA)


   return img
#calculate the score of whether facing the camera dynamically
#set a reasonable ideal eye distance first
#if current eye distance is greater than the ideal one, modify the ideal eye distance
#Then, calculate the score
#If Looking at the side, score will be lower
#If Looking at camera, score will be higher
def faceCameraRate(left_eye_posit, right_eye_posit, width):
   x_distance = abs(right_eye_posit.x - left_eye_posit.x)
   y_distance = abs(right_eye_posit.y - left_eye_posit.y)


   relativeDistance = math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))


   currentRatio = relativeDistance / width
   global max_eye_distance_ratio
   max_eye_distance_ratio = max(currentRatio, max_eye_distance_ratio)


   return currentRatio / max_eye_distance_ratio


def process_posture(model, tracker, encoder, frame, face_detection):

   start_time = time.time()
   result = model(frame, conf = 0.6, classes = [0])[0]
   end_time = time.time()
   print("Time taken for detection: ", end_time - start_time)
   
   H, W, _ = frame.shape
   detections = []


   bboxes = []
   scores = []
   for box in result.boxes:
       x1, y1, _, _ = box.xyxy.squeeze().tolist()
       x1, y1 = int(x1), int(y1)
       _, _, w, h = box.xywh.squeeze().tolist()
       w, h = int(w), int(h)


       bboxes.append([x1, y1, w, h])
       scores.append(box.conf[0])


   feature = encoder(frame, bboxes)

   for i in range(len(bboxes)):
       bbox = bboxes[i]
       score = scores[i]
       Personfeature = feature[i]
       detections.append(Detection(bbox, score, Personfeature))

   start_time = time.time()
   tracker.predict()

   #update the tracker
   tracker.update(detections)
   end_time = time.time()
   print("Time taken for tracking: ", end_time - start_time)
   tracking_Objects = tracker.tracks

   id_center = [] #used to record the center of the bounding box and its id
   for track in tracking_Objects:
       
       if track.time_since_update > 0:
           continue
    
       bbox = track.to_tlwh()
       x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
       id = track.track_id

       x1 = max(0, x1)
       y1= max(0, y1)
       x2 = min(W, x1 + w)
       y2 = min(H, y1 + h)
       if x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H:
           pos = frame[y1: y2, x1: x2,:]
           prcessed = addInfo(id, pos, (0, 0, 0))
           start_time = time.time()
           frame[y1: y2, x1: x2 , :] = process_image(prcessed, face_detection)
           end_time = time.time()
           print("Time taken for face detection: ", end_time - start_time)   

       tracking_object = Tracking_Object(id, (x1 + x2) / 2, (y1 + y2) / 2)   
       id_center.append(tracking_object)


   return Frame_Summary(frame, id_center)



args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filepath', default="./None.jpg")


args = args.parse_args()


max_eye_distance_ratio = 0.05
#Gaze: whether people are looking at tframehe camera
#Skeleton: Posture, orientation, depth camera
mp_face_detector = mp.solutions.face_detection #mp.solutions is an attribute of mp, and also a module, face_detection is also a module
model = YOLO("yolov8n.pt")
metric = NearestNeighborDistanceMetric("cosine", 0.5, None) #determine what metric space we use to track the objects
tracker = Tracker(metric, 0.9, 30, 10) #initialize the tracker
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "deep_sort", "model_data", "mars-small128.pb")
encoder = gdet.create_box_encoder(model_path, batch_size=1)
def process(cv_image):
    with mp_face_detector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    #FaceDetection is an instance or an object or a model used to detect the faces in an image based on the parameters we passed in
        frame_sum = process_posture(model, tracker, encoder,cv_image, face_detection)
    return frame_sum
