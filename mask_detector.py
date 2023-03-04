#FINAL CLASSIFIER

from tensorflow.keras.models import Model, load_model
import cv2
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np 

model = load_model('mask_detection_model_final.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
video = cv2.VideoCapture(0);

while True:
    check, frame = video.read();
    faces = face_cascade.detectMultiScale(frame, minNeighbors=3, minSize=(100,100));
    
    for x,y,w,h in faces:
        face = frame[y:y + h , x:x + w] 
        cv2.imwrite('faces_detected.jpg', face)
        face_img=imread('faces_detected.jpg')
        resized=cv2.resize(face_img,(128,128))
        normalized=resized
        reshaped=np.reshape(normalized,(1,128,128,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        print(result[0][0])

        if result[0][0] > 0.5:
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2);
            cv2.putText(frame, "No Mask", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2);
            cv2.putText(frame, "Mask on", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('Face Detector', frame);

    key = cv2.waitKey(1);

    if key == ord('q'):
        break;

video.release();
cv2.destroyAllWindows();

