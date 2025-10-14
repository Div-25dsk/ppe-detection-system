from flask import Flask,Request,Response,request
import cv2
import pyttsx3
from simple_facerec import SimpleFacerec
from roboflow import Roboflow
import threading
import time
import os
from dotenv import load_dotenv

app = Flask(__name__)


load_dotenv(r'C:\ppeproject\.env')
os.getenv('API_KEY')
#initiaise sfr for calling thw class for face recognition

sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\ppeproject\images")

#loading my model

api = 'API_KEY' 
rf = Roboflow(api_key=api)
project = rf.workspace("sentilnet").project("my-first-project-zm3uf")
model = project.version(3).model

#voice engine 
engine = pyttsx3.init()
volume = engine.getProperty('volume')
engine.setProperty('volume ', 1.0)
engine.setProperty('rate',165)

# Cooldown timer
last_speak_time = 0
cooldown = 3  # seconds

speak_lock = threading.Lock()
def speak(text):
    with speak_lock:
        engine.say(text)
        engine.runAndWait()
#to capture from local machine frame by farme
vcap = cv2.VideoCapture(0)  
#class names for detection
classnames = ['shoes','no_mask','goggles','no_goggles','no_vest','no_gloves',	'no_helmet','mask',	'gloves','vest','helmet']

def gen_frames():
    global last_speak_time

    while True:
        ret, frame = vcap.read()
        if not ret:
            break

        #roboflow loop where the result is confidence scores like x,y ,w,h in json format
        result = model.predict(frame, confidence=40, overlap=30).json()

        #this extracts the class from predictions obtained from results  
        pred_cls = [obj['class'] for obj in result['predictions']]

        #takes class startswith no_
        ppe_violation = any(label.startswith('no_') for label in pred_cls)

        #and replaces with empty string soring it in  missing_ppe
        missing_ppe = [label.replace('no_', "") for label in pred_cls if label.startswith('no_')]

        #converting the values into req format 
        for obj in result["predictions"]:
            x, y = int(obj["x"]), int(obj["y"])
            w, h = int(obj['width']), int(obj['height'])
            cls = obj['class']
            con = round(obj['confidence'], 2)

            #colors for diff classess
            color = (0, 255, 0) if cls in ['shoes', 'mask', 'goggles', 'vest', 'helmet'] else (0, 0, 255)

            #draw the bounding boxes 
            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
            cv2.putText(frame, f"{cls} {con}", (x - w // 2, y - h // 2 - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

        #to call the method for detecting the faces from the face recognition file 
        face_locations, face_names = sfr.detect_known_faces(frame)



        #voie output
        current_time = time.time()
        if ppe_violation and current_time - last_speak_time > cooldown:
            for name in face_names:
                if name != 'Unknown':
                    t = threading.Thread(target=speak, args=(f"{name} PPE violation detected. Missing {', '.join(missing_ppe)}",))
                else:
                    t = threading.Thread(target=speak, args=("Unknown person PPE violation detected",))
                t.start()
            last_speak_time = current_time

        #takes the franes from cv2 and converts to jpg image format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#chels for nain fn executes if only  
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
