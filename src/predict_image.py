from keras.models import load_model
import numpy as np
import cv2

model = load_model('dunk_v_shot_15_epochs.model')

img = cv2.imread('/home/lucas/Galvanize/Projects/Capstone-3/src/data/test_dunk/2.derrick-jones-jr-dunk-ahs.jpg')
# cv2.imshow('dunk', img)
# cv2.waitKey()

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype("float32")
img -= mean

# pred = model.predict(np.expand_dims(img, axis=0))[0]

# print(pred)