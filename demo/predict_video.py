import argparse
import pickle
from collections import deque

import cv2
import cv2.cv2 as cv2
import numpy as np
from keras.models import load_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-name", "--n", required=True,
	help="clip name in folder to predict")
ap.add_argument("-o", "--output", required=True,
	help="path to output")
args = vars(ap.parse_args())

# load the trained model
print("Loading Model")
model = load_model('../models/broadcast_200_epochs_90_acc.model')

# initialize prediction predictions queue
Q = deque()
overall_pred = np.array([0,0])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture('{}.mp4'.format(args['n']))
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
frame_n = 0
jump_frames = 0
dunk_frames = 0
while True:
	(grabbed, frame) = vs.read()


	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (240, 240)).astype("float32")

	pred_prob = model.predict(np.expand_dims(frame, axis=0))[0]
	# Q.append(frame_pred)
	overall_pred = overall_pred + pred_prob

	
	results = np.array(Q).mean(axis=0)
	if pred_prob[0] > pred_prob[1]:
		pred = 'jumpshot'
		jump_frames += 1
	else:
		pred = 'dunk'
		dunk_frames += 1

	print(frame_n, ':', pred_prob, pred)
	frame_n += 1

	rolling_pred = overall_pred/frame_n

	text1 = 'Current Frame Prediction: {}'.format(pred)
	text2 = 'Rolling Prediciton Probabilty [jumpshot, dunk]: {}'.format(np.round(rolling_pred,2))
	cv2.putText(img = output, 
				text = text1, 
				org = (35, 50), 
				fontFace = cv2.FONT_HERSHEY_DUPLEX,
				fontScale = 1, 
				color = (0, 255, 0),
				thickness = 3)
	cv2.putText(output, 
				text2, 
				(35, 100), 
				cv2.FONT_HERSHEY_DUPLEX,
				fontScale = 1, 
				color = (0, 255, 0),
				thickness = 3)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
			
	writer.write(output)

	#show
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	#end if q key is pressed
	if key == ord("q"):
		break

print('Frames classified as jumpshot: ', jump_frames, '\nFrames classified as dunk: ', dunk_frames)
print('Prediction Probability [jumpshot, dunk]: ',rolling_pred)
if rolling_pred[0] > rolling_pred[1]:
	print ('KOBE!')
else:
	print('SLAM DUNK!')

writer.release()
vs.release()
