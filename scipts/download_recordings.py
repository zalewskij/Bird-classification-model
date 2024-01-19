import pandas as pd
import numpy as np
import requests
import time

'''
Script downloading data from xeno canto
'''

df = pd.read_csv('../data/xeno_canto_recordings.csv')
df = df[['id', 'file']]

n = len(df.index)
chunk_size = 50000
chunk = np.arange(0,n,chunk_size)
chunk = np.append(chunk, n)

j = 0
start = time.time()
for i in range(chunk[j], chunk[j+1]):
    if i%1000 == 0:
        print(i, end = ' ')
    name, url = df.loc[i, "id"], df.loc[i, "file"]
    r = requests.get(url, timeout=3600)
    with open(f'/media/jacek/19adee4b-f27e-440b-a945-08c624c05165/recordings/{str(name)}.mp3', 'wb') as f:
        f.write(r.content)
end = time.time()
print(end - start)
