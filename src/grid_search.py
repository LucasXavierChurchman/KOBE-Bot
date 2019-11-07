import random

import cv2
import cv2.cv2 as cv2  # extra import gets rid of error warnings
import numpy as np
from imutils import paths
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def load_images_and_labels(target_labels):
    '''
    Loads all images from directories with names in target_labels
    a label and image array

    TO DO: Make this work with np.load() and skimage (if possible)
    '''
    images = []
    labels = []

    for label in target_labels:
        image_dir = '../data/google_imgs/{}'.format(label)
        print('Loading from {}'.format(image_dir))
        image_paths = list(paths.list_images(image_dir))

        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (240, 240))
            paths.list_images('../data/google_imgs/{}'.format(label))

            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def build_CNN(labels, optimizer = 'Adam'):

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)


                                                    
    ImageNet_mean = np.array([ 123.68, 116.779, 103.939 ])

    train_transformations = ImageDataGenerator(
                        rotation_range=45,
                        zoom_range=0.25,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode="wrap") #constant, nearest, reflect or wrap

    validation_transformations = ImageDataGenerator(ImageNet_mean)  

    train_transformations.mean = ImageNet_mean
    validation_transformations.mean = ImageNet_mean   

    #load transferred learning model
    transferred_model = ResNet50(weights = 'imagenet',
                                include_top = False,
                                input_tensor= Input(shape=(240, 240, 3)))

    #build head model
    head_model = transferred_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

    model = Model(inputs=transferred_model.input, outputs=head_model)

    #freeze layers from transferred model
    for layer in transferred_model.layers:
	    layer.trainable = False

    model.compile(loss="binary_crossentropy", 
                optimizer= optimizer, 
                metrics=["accuracy"])

    return model

target_labels = ['dunk', 'jumpshot']
images, labels = load_images_and_labels(target_labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

random.seed(17)
train_transformations = ImageDataGenerator(
                rotation_range=45,
                zoom_range=0.25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="wrap")

X_train, X_test, y_train, y_test = train_test_split(images, labels, 
                                                test_size = 0.20,
                                                stratify = labels,
                                                random_state = 17)

#need 2 column array for target
y_train = np.hstack((y_train, 1-y_train))
y_test = np.hstack((y_test, 1-y_test))

model = KerasClassifier(build_fn=build_CNN(labels = labels), verbose = 0)

batch_size = [10]#, 20, 40, 60, 80, 100]
epochs = [10]#, 25, 50, 100]
optimizer = ['SGD']#, 'Adam', 'RMSprop']
param_grid = dict(batch_size = batch_size, epochs = epochs)#, optimizer = optimizer)

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, scoring = 'accuracy')

grid_result = grid.fit(X_train, y_train, validation_data = (X_test, y_test))

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("{} ({}) with: {}".format(mean, stdev, param))

