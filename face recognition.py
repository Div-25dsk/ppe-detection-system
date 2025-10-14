import cv2 
from simple_facerec import SimpleFacerec





#loading the simplefacerec file
sfr =  SimpleFacerec()
sfr.load_encoding_images(r"C:\ppeproject\images")


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    #to get face loaction and face name from known_faces method
    face_locations,face_names = sfr.detect_known_faces(frame)
    #loop to get the result
    for face_loc,names in zip(face_locations,face_names):
       # coordinates of the face detected
       top,right,bottom,left = face_loc[0],face_loc[1],face_loc[2],face_loc[3]

       cv2.rectangle(frame,(left,top), (bottom,right),(255,255,255),2)
       cv2.putText(frame,names,(left,top-10),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)


       
    cv2.imshow("face",frame)

    
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break
cap.release()
cv2.destroyAllWindows()
