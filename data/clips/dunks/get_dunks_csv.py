import nltk
import json
import pandas as pd
from urllib import request

i = 0
dunks = 'https://3ball.io/query?pageIndex=0&eventmsgtype[]=1&eventmsgactiontype[]=7&eventmsgactiontype[]=9&eventmsgactiontype[]=48&eventmsgactiontype[]=49&eventmsgactiontype[]=50&eventmsgactiontype[]=51&eventmsgactiontype[]=52&eventmsgactiontype[]=87&eventmsgactiontype[]=106&eventmsgactiontype[]=107&eventmsgactiontype[]=108&eventmsgactiontype[]=109&eventmsgactiontype[]=110'.format(i)
n_pages = 20

all_data = []
for i in range(1,n_pages+1):
    url = dunks
    html = request.urlopen(url).read()
    data = json.loads(html)
    data = pd.DataFrame(data)
    all_data.append(data)

df = pd.concat(all_data)
df.to_csv('dunks/dunks_{}.csv'.format(n_pages*50))
