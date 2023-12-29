import shutil
import numpy as np
from birdclassification.preprocessing.filtering import filter_recordings_287

# prepare a list of subset o 30 birds, get recordings of 30 bird species from list
recordings_to_remove = filter_recordings_287(on_list=0)

# split the data into chunks
n = len(recordings_to_remove.index)
chunk_size = 10000
chunk = np.arange(0, n, chunk_size)
chunk = np.append(chunk, n)

# copy selected files
PATH = "/media/jacek/E753-A120/xeno-canto/"
DST = '/media/jacek/E753-A120/xeno-canto-removed/'

j = 2
for i in range(chunk[j], chunk[j + 1]):
    if i % 100 == 0:
        print(i)
    filepath = f"{PATH}{recordings_to_remove.iloc[i]['id']}.mp3"
    print(filepath)
    shutil.move(filepath, DST)
