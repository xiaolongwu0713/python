import cv2
import mediapipe as mp
import cvzone

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
width=640
height=480

def obj_data(img):
    image_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(image_input)
    if not results.detections:
        cvzone.putTextRect(img,f'NO-Person',(30,60),scale=2,thickness=2,offset=5)    
    else:
         count=0
         for detection in results.detections:
             bbox = detection.location_data.relative_bounding_box
             x, y, w, h = int(bbox.xmin*width), int(bbox.ymin * height), int(bbox.width*width),int(bbox.height*height)
             count+=1
             b=count
           
             cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)    
         cvzone.putTextRect(img,f'person{b}',(30,60),scale=2,thickness=2,offset=5)        
                
        

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
#        self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
#        image=cv2.resize(image,(840,640))
        obj_data(image)
        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()