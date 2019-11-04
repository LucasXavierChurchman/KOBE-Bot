import requests
import time
import pandas as pd

Download_n = 100

df = pd.read_csv('threes_1000.csv')

links = list(df['video_url'])

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

start = time.time()
i = 1
for link in links[0:Download_n]:  
    download_file(link)
    end = time.time()
    print('Clip #: {} Cumlative time: {}'.format(i, str(end-start)))
    i += 1

