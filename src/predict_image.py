import cv2
import cv2.cv2 as cv2
import numpy as np
from keras.models import load_model

model = load_model('../models/broadcast_200_epochs.model')

img_path = '../data/google_imgs/test_jumpshot/6.649728386.0.jpg'
 
img = cv2.imread(img_path)
# cv2.imshow('dunk', img) #<-this crashes the script for some god forsaken reason
cv2.waitKey()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (240, 240)).astype("float32")
img -= mean

pred = model.predict(np.expand_dims(img, axis=0))[0]

print(np.round(pred, 1))
