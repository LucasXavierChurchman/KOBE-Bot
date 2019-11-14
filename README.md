# KOBE!Bot

### Lucas Churchman

## Table of Contents
1. [Background](#background)
2. [Objectives](#objectives)
3. [Data and EDA](#data-and-eda)
4. [Model](#analysis)
5. [Results](#results)
6. [Future Work](#future-work)
7. [References](#references)
8. [Acknowledgements](#acknowledgements)

## Background

I wanted to explore if computer vision and machine learning techniques can be used to distinguish between images of dunks and jumpshots. If so, videos could hypotheticall be classfied as well since they are simply a sequence of images. These methods could assist in automated box score statistic recording, shot chart tracking, and beyond.

## Objectives

* Classify images as a dunk or jumpshot in:</br>
a) Photos from Google Images</br>
b) Frames extracted from clips from the broadcast camera angle</br>
* Identify the unique challenges of differentiating between a dunk and jumpshot for each of these types of images.
* Classify a highlight clip by taking the majority classification prediction of its individual frames

## Data and EDA

Google images were collected using [this package](https://pypi.org/project/google_images_download/). 

For the broadcast images, I downloaded videos from [3ball.io](https://3ball.io/plays) where you can filter highlights by play type, home team, period, etc. Once downloaded I wrote a function that uses OpenCV to separate the video frame by frame and save them to a temporary directory. For the images that would actually be used for training, I decided to use frames that were as similar as possible to the Google images despite the very different camera perspective; when the player was in the shooting or dunking motion. To this end I found the frame when the player started their jumping/shooting motion and copied the next 1 second worth of frames (30 or 60 depending on the clip's framerate) into the training image directory.

Further more, due to the scope of this project and several inconsistencies in camera angles and sponsor logos between arenas only plays on the right side of the court at the Pepsi Center were trained on and validated with. However, testing on images from other arenas is showing promising results.

All images were resized to a 240x240 resolution during exploration, modeling, and prediction.

![image examples](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/each_type_and_class_example.png)


### EDA
Although they are far from consistent (especially the Google Images) we can make some generalities about key similarities and differences for both types and classes of images by looking at individual pictures:

|              | Google Images                                                                                                            | Broadcast Frames                                                                     |
|--------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Similarities | Arms extended, ball held above head                                                                                     | Paint on court and crowd creates dark vs light boundary of playing space            |
| Differences  | Backboard often visible, exaggerated body position for dunks. Another body (defender) in frame more common in jumpshots | Players are more spread out for jumpshots. Players are clumped near basket for dunk |

#### Average Image
Some of these these distinctions can be see in the following plots of each set's average image (along with pixel intensities)

##### Google Images Averages:
![Avg google images](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/google_image_avgs.png)

##### Broadcast Angle Averages:
![Avg broadcast images](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/denver_image_avgs.png)

#### NMF

Eigenfaces are a method used in facial recognition technology that uses Principal Component Analysis (PCA) to visually illustrate the most distinguishing features of a collection of headshots images.

![Eigen_Faces](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/eigenfaces.png)

I wanted to use this method, but the values of PCA generated eigenvectors can be difficult to interpret. I used Non-Negative Matrix Factorizaion (NMF) instead. This way, vector values were restricted between 0 and 1, and makes it more apparent that lighter pixels (closer to a value of 1) have a higher loading on their latent feature. This also gave the reconstructed images higher 'contrast' than using PCA.

##### Google Image NMF
![google_nmf](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/google_nmf.png)

##### Broadcast Angle NMF
![broadcast_nmf](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/denver_nmf.png)

## Model

Keras' pre-loaded ImageNet architechtures are the industry standard for image classification models. I decided to use transfer learning from one of these architectures, ultimately deciding on ResNet-50 since it gave the best results of any tested. Five additional layers were added to to the architecture. 

![network_diagram](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/cnn_diagram.png)

The same network structure was used on both sets of images giving models with the following results:

![model_results](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/model_results.png)

I was overall satisfied with the valdiation accuracies for both models, though signs of overfitting are present. Very limited time was available for hyperparameter and layer tuning so this problem could easily be fixed.

## Results

### Image Prediction Results

![image_predictions](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/plots%2Bimages/prediction_examples.png)

All of the images in the broadcast column were captured within 30 frames (a half second) of each other. It’s interesting that one was misclassified considering how similar they appear to the human eye

### Video Prediction example

The yellow text displays the prediction of the current frame being displayed while the green text displays the overall prediction of the clip which gets updated every frame

![demo gif](https://github.com/LucasXavierChurchman/KOBE-Bot/blob/master/demo/compiled_for_demo.gif)

## Future Work

* Further improve training data and methodology
* Tune CNN hyperparameters and layers to further improve model accuracy and fit
* Multiclass classfier with more types of plays (pass, block, steal, free-throw, etc.)
* Utilize object detection or other deep learning models

## References 

Mallick, Satya. “Eigenface Using OpenCV (C /Python).” Learn OpenCV, 18 Jan. 2018, 
https://www.learnopencv.com/eigenface-using-opencv-c-python/.

PyImageSearch, 11 Nov. 2019, https://www.pyimagesearch.com/.

“ResNet-50.” Applications - Keras Documentation
https://keras.io/applications/#resnet.

Rosebrock, Adrian. “ImageNet: VGGNet, ResNet, Inception, and Xception with Keras.” 
PyImageSearch, 5 Feb. 2019, https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/.

Rosebrock, Adrian. “Video Classification with Keras and Deep Learning.” 
PyImageSearch, 12 July 2019,
https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/.

“Searchable NBA Video.” 3 Ball, https://3ball.io/plays.

## Acknowledgements:

* My friends and family for their love and encouragement
* The Galvanize DSI instructors, Kayla Thomas, Frank Burkholder, and Nick Jocobsohn for their teaching and encouragement
* My DSI cohort-mates for their support and collaborative learning



