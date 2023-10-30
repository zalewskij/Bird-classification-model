import os
from birdclassification.preprocessing.filtering import filter_recordings_30
import pandas as pd

print(os.getcwd())
DIR_PATH = '/Users/zosia/Desktop/recordings_30_'
DEST_PATH = "/Users/zosia/Desktop/test/"

# list files to move
files = os.listdir(DIR_PATH)

# list file details
recording_details = filter_recordings_30()

# List of top 30 bird species
bird_list = pd.read_csv("../data/bird-list-extended.csv", delimiter=";")
bird_list = bird_list[bird_list["Top 30"] == 1]["Latin name"]

# create directories
for items in bird_list:
    if not os.path.isdir(f"{DEST_PATH}{items}"):
        os.mkdir(f"{DEST_PATH}{items}")

# Move the files
for file in files:
    bird_id = file.split(".")[0]
    folder_name = recording_details[recording_details['id'] == int(bird_id)]['Latin name'].values[0]
    os.rename(f'{DIR_PATH}/{file}', f"{DEST_PATH}{folder_name}/{file}")