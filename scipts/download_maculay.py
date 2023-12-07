"""
Script to download all species from BIRD_LIST[i].csv to DESTINATION DIR
"""

import pandas as pd
import requests
import os


BIRD_LIST  = ['Ciconia ciconia', 'Tetrao urogallus', 'Columba livia', 'Phalacrocorax carbo']
API_ENDPOINT = 'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/'
DESTINATION_DIR = '/media/jacek/E753-A120/macaulay_library/'


for bird in BIRD_LIST:
    print(bird)
    os.mkdir(DESTINATION_DIR + bird)
    df = pd.read_csv(f'../data/{bird}.csv')
    for ml_id in df["ML Catalog Number"]:
        response = requests.get(API_ENDPOINT + str(ml_id), timeout=3600)
        with open(f'{DESTINATION_DIR}{bird}/{str(ml_id)}.mp3', 'wb') as f:
            f.write(response.content)

