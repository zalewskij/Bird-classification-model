import shutil
import numpy as np
from preprocessing.filtering import filter_recordings_30

# prepare a list of subset o 30 birds, get recordings of 30 bird species from list
recordings_30 = filter_recordings_30()

# split the data into chunks
n = len(recordings_30.index)
chunk_size = 20000
chunk = np.arange(0, n, chunk_size)
chunk = np.append(chunk, n)

# copy selected files
PATH = "/media/jacek/E753-A120/xeno-canto/"
DST = '/media/jacek/E753-A120/recordings_30/'
j = 2
for i in range(chunk[j], chunk[j + 1]):
    if i % 1000 == 0:
        print(i)
    filepath = f"{PATH}{recordings_30.iloc[i]['id']}.mp3"
    shutil.copy2(filepath, DST)
