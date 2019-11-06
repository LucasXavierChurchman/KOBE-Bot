import cv2
import cv2.cv2 as cv2
f = '../data/clips/dunk/dunk_20.mp4'
vidcap = cv2.VideoCapture(f)
success,image = vidcap.read()
count = 0
success = True
while success:
  print('sup')
  success,image = vidcap.read()
  cv2.imwrite("{}-frame-{}.jpg".format(f[:-4], count), image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1