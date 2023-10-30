import pandas as pd


def initial_filter():
    """
    Filtering based on: https://komisjafaunistyczna.pl/objasnienia/ \n
    Category: A, B, C \n
    Status: (l) P, L, l P, P

    Returns
    -------
    filtered_recordings: pandas.DataFrame
        Filtered list of recordings
    """
    bird_species = pd.read_csv('../data/bird_species_in_poland_31.12.2022.csv')
    df = pd.read_csv('../data/xeno_canto_recordings.csv')
    filtered_bird_species = bird_species[bird_species['Status'].isin(('(l) P', 'L', 'l P', 'P'))]
    df['Latin name'] = df.gen + " " + df.sp
    filtered_recordings = df[df['Latin name'].isin(filtered_bird_species['Latin name'])]
    return filtered_recordings


def filter_by_species(df, gen, sp):
    return df[(df['gen'].str.lower() == gen.lower()) & (df['sp'].str.lower() == sp.lower())]


def filter_by_gen(df, gen):
    return df[df['gen'].str.lower() == gen.lower()]


def filter_recordings_30():
    """
    To test the models there is no need to use all species.
    Based on data-scrapping/data/bird-list-extended a function filter a subset of 30 species

    Returns
    -------
    recordings_30: pandas.DataFrame
        Recordings of chosen 30 bird species
    """
    df = pd.read_csv("../data/bird-list-extended.csv", delimiter=";")
    species_30 = df[df["Top 30"] == 1]
    recordings = pd.read_csv("../data/xeno_canto_recordings.csv")
    recordings['Latin name'] = recordings.gen + " " + recordings.sp
    recordings_30 = recordings[recordings["Latin name"].isin(species_30["Latin name"])].reset_index()
    return recordings_30


