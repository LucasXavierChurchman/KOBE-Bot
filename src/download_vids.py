import json
import time
from urllib import request

import nltk
import numpy as np
import pandas as pd
import requests


def make_video_data_csv(type):
    start_page_index = 0

    # need to put these links in an external dictionary
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

if __name__ == '__main__':
    pass
