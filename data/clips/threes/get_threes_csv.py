import nltk
import json
import pandas as pd
from urllib import request

i = 0
threes = 'https://3ball.io/query?pageIndex={}&eventmsgtype[]=1&description=3PT&eventmsgactiontype[]=0&eventmsgactiontype[]=1&eventmsgactiontype[]=2&eventmsgactiontype[]=45&eventmsgactiontype[]=46&eventmsgactiontype[]=47&eventmsgactiontype[]=56&eventmsgactiontype[]=63&eventmsgactiontype[]=66&eventmsgactiontype[]=77&eventmsgactiontype[]=79&eventmsgactiontype[]=80&eventmsgactiontype[]=81&eventmsgactiontype[]=82&eventmsgactiontype[]=83&eventmsgactiontype[]=85&eventmsgactiontype[]=86&eventmsgactiontype[]=103&eventmsgactiontype[]=104&eventmsgactiontype[]=105'.format(i)
n_pages = 20

all_data = []
for i in range(1,n_pages+1):
    url = threes
    html = request.urlopen(url).read()
    data = json.loads(html)
    data = pd.DataFrame(data)
    all_data.append(data)

df = pd.concat(all_data)
df.to_csv('threes/threes_{}.csv'.format(n_pages*50))
# df.reset_index(inplace = True)
# df.to_json('json_files/threes_{}.json'.format(n_pages))