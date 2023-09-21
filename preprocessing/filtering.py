import pandas
import pandas as pd


class Filtering:
    """
    Class implementing a filtering of xeno_canto_recordings.csv
    """

    @staticmethod
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
        bird_species = pd.read_csv('data-scrapping/data/bird_species_in_poland_31.12.2022.csv')
        df = pd.read_csv('data-scrapping/data/xeno_canto_recordings.csv')
        filtered_bird_species = bird_species[bird_species['Status'].isin(('(l) P', 'L', 'l P', 'P'))]
        df['Latin name'] = df.gen + " " + df.sp
        filtered_recordings = df[df['Latin name'].isin(filtered_bird_species['Latin name'])]
        return filtered_recordings

    @staticmethod
    def filter_by_species(df, gen, sp):
        return df[(df['gen'] == gen) & (df['sp'] == sp)]

    @staticmethod
    def filter_by_gen(df, gen):
        return df[df['gen'] == gen]
