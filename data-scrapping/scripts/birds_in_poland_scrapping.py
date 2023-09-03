import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.DataFrame(columns=['Latin name',
                           'Polish name',
                           'Category',
                           'Status'
                           ])

# parse a list of bird species in Poland
URL = 'https://komisjafaunistyczna.pl/lista/'
# soup = BeautifulSoup(requests.get(URL).content, features="lxml")
html = requests.get('https://komisjafaunistyczna.pl/lista/', verify=False)
soup = BeautifulSoup(html.text, "html.parser")

for row in soup.find_all('tr'):
    if row.find_all('td'):
        latin_name = row.find_all('td')[1].contents[0]
        polish_name = row.find_all('td')[2].contents[0]
        category = row.find_all('td')[3].contents[0]
        status = row.find_all('td')[4].contents[0]
        df.loc[len(df.index)] = [latin_name, polish_name, category, status]


df.to_csv("../data/bird_species_in_poland_31.12.2022.csv")