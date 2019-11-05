import cv2

def extract_frames(type, clip_number):
  '''
  Splits a video into its individual frames. Writes the frames as images outside 
  this project directory since it generates tons of images, most of which we 
  dont actually use.

  The images we do care about get copied manually into /data/broadcast_images
  '''
  # clip = '../data/clips/{}/{}_{}.mp4'.format(type, type, clip_number)
  clip = '../data/clips/{}/{}_{}.mp4'.format(type, type, clip_number)

  print(clip)
  vidcap = cv2.VideoCapture(clip)
  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    success, image = vidcap.read()
    save_dir = "/home/lucas/Pictures/{}_frames/{}_{}_frame_{}.jpg"
    cv2.imwrite(save_dir.format(type, type, clip_number, count), image)
    if cv2.waitKey(10) == 27:  # exit if Escape is hit
        break
    count += 1

if __name__ == '__main__':
  pass