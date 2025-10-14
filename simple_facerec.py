import cv2
import face_recognition
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        #list to store encodings and names 
        self.known_face_encodings = []
        self.known_face_names = []
        
#to load the known faces 
    def load_encoding_images(self, images_path):
        #loop through the images in the folder
        for filename in os.listdir(images_path):
            #finds the parh and loads it 
            img = face_recognition.load_image_file(os.path.join(images_path, filename))
            #used to get the encodings 
            encoding = face_recognition.face_encodings(img)[0]
            #appends the encodings and names found
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(os.path.splitext(filename)[0])
#to detect the faces
    def detect_known_faces(self, frame):
        #gets loactions from the frame
        face_locations = face_recognition.face_locations(frame)
        #encodes the output from the frame 
        face_encodings = face_recognition.face_encodings(frame, face_locations)
#empty list to store the names
        face_names = []
        #loops through each enoding for finding matches by comparing it from knwon encodings and the camera feed
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            #if match is found returns the index and gets the name with taht index
            if True in matches:
                index = matches.index(True)
                name = self.known_face_names[index]
            face_names.append(name)

        return face_locations, face_names
