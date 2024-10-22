from collections import deque
import cv2.cv2 as cv2
import numpy as np
from keras.models import load_model


def predict_img(img_path, model):
    '''
    Predicts class of an image

    Arguments:
        img_path: path to image file
        model: keras model

    Returns:
        None
    '''
    img = cv2.imread(img_path)
    cv2.waitKey()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240)).astype("float32")

    pred = model.predict(np.expand_dims(img, axis=0))[0]

    print('[0,1] = dunk, [1,0] = jumpshot')
    print(np.round(pred, 2))

if __name__ == '__main__':
    pass
