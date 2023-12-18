import os
from birdclassification.preprocessing.filtering import filter_recordings_287
import pandas as pd
import shutil

DIR_PATH = '/media/jacek/E753-A120/xeno-canto-splitted-converted/'
DEST_PATH = "/media/jacek/E753-A120/xeno-canto-splitted-converted/"

# list files to move
files = os.listdir(DIR_PATH)
print(len(files))

# list file details
recording_details = filter_recordings_287()

# List of top 30 bird species
bird_list = pd.read_csv("../data/bird-list-extended.csv", delimiter=";")
bird_list = bird_list[bird_list["Chosen"] == 1]["Latin name"]

# create directories
for items in bird_list:
    if not os.path.isdir(f"{DEST_PATH}{items}"):
        os.mkdir(f"{DEST_PATH}{items}")


#Move the files
# for file in files:
#     bird_id = file.split(".")[0]
#     if bird_id != '':
#         folder_name = recording_details[recording_details['id'] == int(bird_id)]['Latin name'].values[0]
#         # print(f'{DIR_PATH}{file}')
#         # print(f"{DEST_PATH}{folder_name}/{file}")
#         # print("-----------------------------------------------")
#         shutil.move(f'{DIR_PATH}{file}', f"{DEST_PATH}{folder_name}/{file}")