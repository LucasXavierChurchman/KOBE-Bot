import json
import os
import time
from urllib import request
import cv2
import cv2.cv2 as cv2
import nltk
import numpy as np
import pandas as pd
import requests


def make_video_data_csv(type):
    '''
    Creates a CSV with data from highlight page links in /data/clips

    type = dunk, three, denver_three, denver_dunk

    TODO: put links in external dictionary
    '''

    if type == 'dunk':
        link = 'https://3ball.io/query?pageIndex={}&eventmsgtype[]=1&eventmsgactiontype[]=7&eventmsgactiontype[]=9&eventmsgactiontype[]=48&eventmsgactiontype[]=49&eventmsgactiontype[]=50&eventmsgactiontype[]=51&eventmsgactiontype[]=52&eventmsgactiontype[]=87&eventmsgactiontype[]=106&eventmsgactiontype[]=107&eventmsgactiontype[]=108&eventmsgactiontype[]=109&eventmsgactiontype[]=110'
    elif type == 'three':
        link = 'https://3ball.io/query?pageIndex={}&eventmsgtype[]=1&description=3PT&eventmsgactiontype[]=0&eventmsgactiontype[]=1&eventmsgactiontype[]=2&eventmsgactiontype[]=45&eventmsgactiontype[]=46&eventmsgactiontype[]=47&eventmsgactiontype[]=56&eventmsgactiontype[]=63&eventmsgactiontype[]=66&eventmsgactiontype[]=77&eventmsgactiontype[]=79&eventmsgactiontype[]=80&eventmsgactiontype[]=81&eventmsgactiontype[]=82&eventmsgactiontype[]=83&eventmsgactiontype[]=85&eventmsgactiontype[]=86&eventmsgactiontype[]=103&eventmsgactiontype[]=104&eventmsgactiontype[]=105'
    elif type == 'denver_three':
        link = 'https://3ball.io/query?pageIndex={}&eventmsgtype[]=1&description=3PT&home_team_id=1610612743'
    elif type == 'denver_dunk':
        link = 'https://3ball.io/query?pageIndex={}&eventmsgtype[]=1&eventmsgactiontype[]=7&eventmsgactiontype[]=9&eventmsgactiontype[]=48&eventmsgactiontype[]=49&eventmsgactiontype[]=50&eventmsgactiontype[]=51&eventmsgactiontype[]=52&eventmsgactiontype[]=87&eventmsgactiontype[]=106&eventmsgactiontype[]=107&eventmsgactiontype[]=108&eventmsgactiontype[]=109&eventmsgactiontype[]=110&home_team_id=1610612743'
    else:
        print('clip type not supported yet')

    n_pages = 20

    all_data = []
    for i in range(0,n_pages):
        url = link.format(i)
        print('Fetching page index: ', i)
        html = request.urlopen(url).read()
        data = json.loads(html)
        data = pd.DataFrame(data)
        all_data.append(data)

    df = pd.concat(all_data)
    df.to_csv('../data/clips/{}/{}_{}.csv'.format(type, type, n_pages*50)) #there are 50 clips per page

def download_clips(type, n_clips):
    '''
    Reads csv produced make_video_data and downloads the specificied number of clips
    to data/clips
    '''
    df = pd.read_csv('../data/clips/{}/{}_1000.csv'.format(type, type))

    links = list(df['video_url'])

    start = time.time()
    for i, link in enumerate(links[0:n_clips]):  

        local_filename = '{}_{}.mp4'.format(type, i+1)
        r = requests.get(link, stream=True)
        with open('../data/clips/{}/{}'.format(type, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)
    end = time.time()

    dl_time = np.round((end-start), 2)

    print('Time to download {} clips : {} seconds'.format(n_clips, str(dl_time)))

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
    save_path = frame_folder + '/{}_{}_frame_{}.jpg'
    cv2.imwrite(save_path.format(type, clip_number, count), image)
    if cv2.waitKey(10) == 27:  # exit if Escape is hit
        break
    count += 1

if __name__ == '__main__':
    pass
