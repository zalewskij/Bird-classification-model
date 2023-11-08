import os
from birdclassification.preprocessing.filtering import filter_recordings_30
import pandas as pd
import shutil

print(os.getcwd())
DIR_PATH = '/media/jacek/E753-A120/recordings_30/'
DEST_PATH = "/media/jacek/E753-A120/recordings_30_v2/"

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
    if bird_id != '':
        folder_name = recording_details[recording_details['id'] == int(bird_id)]['Latin name'].values[0]
        shutil.move(f'{DIR_PATH}{file}', f"{DEST_PATH}{folder_name}/{file}")