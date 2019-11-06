import cv2
import os
import shutil

def extract_frames(type, clip_number):
  '''
  Writes individual frames of clips into /data/broadcast_imgs/temp_frames'

  Erases current contents of /data/broadcast_imgs/temp_frames' each time the function
  is executed
  '''

  #erase current contents of temp folder
  frame_folder = '../data/broadcast_imgs/temp_frames'
  for the_file in os.listdir(frame_folder):
    file_path = os.path.join(frame_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

  #extract frames of clip
  clip = '../data/clips/{}/{}_{}.mp4'.format(type, type, clip_number)
  vidcap = cv2.VideoCapture(clip)
  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    success, image = vidcap.read()
    save_dir = '/home/lucas/Pictures/{}_frames/{}_{}_frame_{}.jpg'
    save_path = frame_folder + '/{}_{}_frame_{}.jpg'
    cv2.imwrite(save_path.format(type, clip_number, count), image)
    if cv2.waitKey(10) == 27:  # exit if Escape is hit
        break
    count += 1

if __name__ == '__main__':
  pass