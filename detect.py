#in this code, i used the webcam, you could use the picam too
#make the changes accordingly for picam
#ps you just have to uncomment a few lines

import cv2
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from twilio.rest import Client
import utils
#from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

account_sid = '<Twilio Account SID>'  # Replace with your Twilio Account SID
auth_token = '<Twilio Auth Token>'  # Replace with your Twilio Auth Token
client = Client(account_sid, auth_token)

# Function to send WhatsApp message
def send_whatsapp_message(to, message):
    from_whatsapp = 'whatsapp:+14155238886'  # Twilio Sandbox number
    client.messages.create(
        body=message,
        from_=from_whatsapp,
        to=to
    )
    print(f"WhatsApp message sent to {to}: {message}")


model = 'efficientdet_lite0.tflite'
num_threads = 4
 
dispW=1280
dispH = 720
#import cv2

cam = cv2.VideoCapture(0)  # Open the first webcam
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
#cam.set(cv2.CAP_PROP_FRAME_FPS,30)

pos = (20,60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor=(255,0,0)

boxColor = (0,0,0)
boxWeight = 2
fps = 0

labelHeight= 1.5
labelColor = (255,0,0)
labelWeight = 2

base_options = core.BaseOptions(file_name=model,use_coral=False,num_threads= num_threads)
detection_options = processor.DetectionOptions(max_results=5,score_threshold=.3)
options = vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
book_detected_flag = False  # To avoid duplicate notifications
tstart = time.time()
while True:
    ret, im = cam.read()
    cv2.putText(im,str(int(fps)),pos,font,height,myColor,weight)
    imRGB =cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imTensor = vision.TensorImage.create_from_array(imRGB)
    myDetects = detector.detect(imTensor)
    #you could directly use the below command to get the default 
    #image = utils.visualize(im,myDetects)
    # for customization use the below code
    
    book_found = False
    
    for myDetect in myDetects.detections:
        UpperLeft = (myDetect.bounding_box.origin_x,myDetect.bounding_box.origin_y)
        LowerRight=(myDetect.bounding_box.origin_x+myDetect.bounding_box.width,myDetect.bounding_box.origin_y+myDetect.bounding_box.height)
        #print((UpperLeft,LowerRight))
        im = cv2.rectangle(im,UpperLeft,LowerRight,boxColor,boxWeight)
        objName = myDetect.categories[0].category_name
        cv2.putText(im,objName,UpperLeft,font,labelHeight,labelColor,labelWeight)
    #im = picam2.capture_arry()
    
    if objName.lower() == "book":
            book_found = True
            
    if book_found and not book_detected_flag:
            send_whatsapp_message('whatsapp:+91XXXXXXXXXX', "A person has been detected!") 
            book_detected_flag = True  # Prevent repeated notification
    if not book_found:
        book_detected_flag = False
    if not ret:
        break
    cv2.imshow('Webcam', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    tend = time.time()
    looptime =tend-tstart
    if looptime > 0:  # Avoid division by zero
        current_fps = 1 / looptime
        fps = 0.9 * fps + 0.1 * current_fps  # Smoothened FPS
    tstart = tend  # Update start time for the next frame
    cv2.putText(im,str(int(fps)),pos,font,height,myColor,weight)
    #print(fps)
cam.release()
cv2.destroyAllWindows()

