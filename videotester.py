import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
# from keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("EmotionNet_200Epochs.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5) #  uses haars & returns all the faces and their coordinates. 

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7) # draws a blue reactangle from the point given for the face
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48)) # resizes to the shape required by the model which in our case is 48X48
        img_pixels = img_to_array(roi_gray) # converts the image to a numpy array
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels) #predicts the class
        
        # find max indexed array
        max_index = np.argmax(predictions[0])
    
        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) #puts the predicted emotion against the rectangle box drawn

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows