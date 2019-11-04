import cv2
f = 'dunks-20.mp4'
vidcap = cv2.VideoCapture(f)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("dunks_frames/{}-frame-{}.jpg".format(f[:-4], count), image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1