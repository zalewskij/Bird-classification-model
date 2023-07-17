import requests
import pandas as pd
from requests.exceptions import HTTPError

birds_list = pd.read_csv('data/bird_species_in_poland_31.12.2022.csv')
df = pd.DataFrame(columns=['id',
                           'gen',
                           'sp',
                           'ssp',
                           'group',
                           'en',
                           'rec',
                           'cnt',
                           'loc',
                           'lat',
                           'lng',
                           'alt',
                           'type',
                           'sex',
                           'stage',
                           'method',
                           'url',
                           'file',
                           'file-name',
                           'sono-small',
                           'sono-med',
                           'sono-large',
                           'sono-full',
                           'osci-small',
                           'osci-med',
                           'osci-large',
                           'lic',
                           'q',
                           'length',
                           'time',
                           'date',
                           'uploaded',
                           'also',
                           'rmk',
                           'bird-seen',
                           'animal-seen',
                           'playback-used',
                           'temp',
                           'regnr',
                           'auto',
                           'dvc',
                           'mic',
                           'smp',
                           ])

def request(key: str):
    print(f'Latin name: {species}')
    response = requests.get(key)
    response.raise_for_status()
    return response.json()



for species in birds_list['Latin name'][162:163]:
    try:
        key = f'https://xeno-canto.org/api/2/recordings?query={species}'
        json_response = request(key)

        #species data
        num_recordings = json_response['numRecordings']
        num_species = json_response['numSpecies']
        num_pages = json_response['numPages']
        print(f'recordings: {num_recordings} species:{num_species}, pages: {num_pages}')
        print(json_response)

        #iterate over pages
        for i in range(1,  num_pages+1):
            key = f'https://xeno-canto.org/api/2/recordings?query={species}&page={i}'
            json_response = request(key)

            for rec in json_response['recordings']:
                print(rec)
                print("\n")
                #parse recording dict
                #put into pandas dataframe


    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
