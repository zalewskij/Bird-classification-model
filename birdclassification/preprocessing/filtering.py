import pandas as pd


def initial_filter(filepath_recordings='../data/xeno_canto_recordings.csv',
                   filepath_bird_list='../data/bird_species_in_poland_31.12.2022.csv'):
    """
    Filtering based on: https://komisjafaunistyczna.pl/objasnienia/ \n
    Category: A, B, C \n
    Status: (l) P, L, l P, P

    Returns
    -------
    filtered_recordings: pandas.DataFrame
        Filtered list of recordings
    """
    recordings_to_remove = [818118]
    bird_species = pd.read_csv(filepath_bird_list)
    df = pd.read_csv(filepath_recordings)
    filtered_bird_species = bird_species[bird_species['Status'].isin(('(l) P', 'L', 'l P', 'P'))]
    df['Latin name'] = df.gen + " " + df.sp
    filtered_recordings = df[df['Latin name'].isin(filtered_bird_species['Latin name'])].reset_index()
    filtered_recordings = filtered_recordings[~filtered_recordings['id'].isin(recordings_to_remove)].reset_index()
    return filtered_recordings


def filter_by_species(df, gen, sp):
    """
    Parameters
    ----------
    df: pd.DataFrame
        Recordings with gen column
    gen: str
        Filtering key
    sp: str
        Filtering key
    Returns
    -------
    A dataframe of species filtered by gen and sp
    """
    return df[(df['gen'].str.lower() == gen.lower()) & (df['sp'].str.lower() == sp.lower())]


def filter_by_gen(df, gen):
    """
    Parameters
    ----------
    df: pd.DataFrame
        Recordings with gen column
    gen: str
        Filtering key
    Returns
    -------
    A dataframe of species filtered by gen
    """
    return df[df['gen'].str.lower() == gen.lower()]


def filter_recordings_30(filepath_recordings='../data/xeno_canto_recordings.csv',
                         filepath_bird_list='../data/bird-list-extended.csv'):
    """
    To test the models there is no need to use all species.
    Based on scipts/data/bird-list-extended a function filter a subset of 30 species

    Returns
    -------
    recordings_30: pandas.DataFrame
        Recordings of chosen 30 bird species
    """
    recordings_to_remove = [818118]
    df = pd.read_csv(filepath_bird_list, delimiter=";")
    species_30 = df[df["Top 30"] == 1]
    recordings = pd.read_csv(filepath_recordings)
    recordings['Latin name'] = recordings.gen + " " + recordings.sp
    recordings_30 = recordings[recordings["Latin name"].isin(species_30["Latin name"])].reset_index()
    recordings_30 = recordings_30[~recordings_30['id'].isin(recordings_to_remove)].reset_index()
    return recordings_30


def filter_recordings_287(filepath_recordings='../data/xeno_canto_recordings.csv',
                         filepath_bird_list='../data/bird-list-extended.csv', on_list = 1):
    """
    Based on scipts/data/bird-list-extended a function filter a subset of 287 species (all species of interest in Poland)

    Returns
    -------
    recordings_287: pandas.DataFrame
        Recordings of chosen 287 bird species
    """
    recordings_to_remove = [470775, 798805, 684749, 644502, 357351, 809967, 624256, 809054, 807141, 162553, 817521,
                            714657, 798808, 554614, 796603,798427, 257407, 257405, 787319, 791718, 507301, 80610,
                            162683, 816529, 817344, 180914, 818891, 796288, 471258, 367690,818855, 818773, 600181,
                            257406, 809965, 441473, 102788, 798809, 483205, 516953, 507302, 420376, 800153, 644503,
                            802401, 798807, 562671, 357350, 806787, 162686, 798806, 600451, 397457, 567174, 791068,
                            819062, 809966, 707378, 385433, 806174, 162612]

    df = pd.read_csv(filepath_bird_list, delimiter=";")
    species_287 = df[df["Chosen"] == on_list]
    print(len(species_287.index))
    recordings = pd.read_csv(filepath_recordings)
    recordings['Latin name'] = recordings.gen + " " + recordings.sp
    recordings_287 = recordings[recordings["Latin name"].isin(species_287["Latin name"])].reset_index()
    recordings_287 = recordings_287[~recordings_287['id'].isin(recordings_to_remove)].reset_index()
    return recordings_287
