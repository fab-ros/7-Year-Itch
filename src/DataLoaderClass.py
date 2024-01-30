# code to load the dataset

import pandas as pd
import numpy as np
import re
import os


''' Class to load the data from the dataset and define methods to preprocess it '''
class DataLoader:

    # initialize the class
    def __init__(self):
        self.data_array = np.array([])
        self.csv_df = pd.DataFrame()
        self.rel_data_dir = ""
        self.durations = []
        self.years = []
        self.kids = []
        self.year_index_offset = 1997

        # set the relative directory path to the dataset
        current_dir = re.split(r'[\\/]', os.getcwd())
        rel_data_dir = ""
        while current_dir[-1] != "DataLiteracy":
            rel_data_dir = rel_data_dir + "../"
            current_dir = current_dir[:-1]
        self.rel_data_dir = rel_data_dir + "dat/dataset.csv"

    # method to convert the given duration string in the dataset to its array index
    @staticmethod
    def convert_duration_to_index(duration_string, offset=1):
        strings = duration_string.split(" ")
        if len(strings) == 4:
            assert strings[1] == "unter", "error in parsing marriage duration <= 2"
            return 0
        else:
            try:
                return int(strings[1]) - offset
            except:
                raise Exception("error in parsing marriage duration > 2")

    # convert the data from the dataframe into a three-dimensional array
    def setup_dataset_array(self):
        self.year_index_offset = int(self.csv_df["year"][0])

        num_durations = (self.csv_df['year'] == str(self.year_index_offset)).sum()
        num_years = int(len(self.csv_df["year"]) / num_durations)
        num_kid_categories = len(self.csv_df.columns) - 2
        self.durations = np.array([d for d in range(1, num_durations + 1)])
        self.years = np.array([y for y in range(self.year_index_offset, self.year_index_offset + num_years)])
        self.kids = [k for k in range(0, num_kid_categories)]
        self.data_array = np.zeros((num_years, num_durations, num_kid_categories))

        for line in self.csv_df.iterrows():
            year_index = int(line[1]["year"]) - self.year_index_offset
            duration_index = self.convert_duration_to_index(line[1]["duration"])
            self.data_array[year_index][duration_index][0] = int(line[1]["0 kids"])
            self.data_array[year_index][duration_index][1] = int(line[1]["1 kid"])
            self.data_array[year_index][duration_index][2] = int(line[1]["2 kids"])
            self.data_array[year_index][duration_index][3] = int(line[1]["3+ kids"])

    # drop the given elements from data
    def drop_data(self, data, years_to_drop, durations_to_drop, kids_to_drop):
        years = self.years
        durations = self.durations
        kids = self.kids
        if years_to_drop is not None:
            assert all(elem in self.years for elem in years_to_drop)
            indexes = [y_i - self.year_index_offset for y_i in years_to_drop]
            data = np.delete(data, axis=0, obj=indexes)
            years = [y for y in years if y not in years_to_drop]
        if durations_to_drop is not None:
            assert all(elem in self.durations for elem in durations_to_drop)
            indexes = [d_i - 1 for d_i in durations_to_drop]
            data = np.delete(data, axis=1, obj=indexes)
            durations = [y for y in durations if y not in durations_to_drop]
        if kids_to_drop is not None:
            assert all(elem in self.kids for elem in kids_to_drop)
            indexes = [k_i for k_i in kids_to_drop]
            data = np.delete(data, axis=2, obj=indexes)
            kids = [k for k in kids if k not in kids_to_drop]

        return data, years, durations, kids

    # load the dataset and return it in form of a three-dimensional array as well as the respective axes
    def load_data(self, years_to_drop=None, durations_to_drop=None, kids_to_drop=None):
        if self.csv_df.size == 0:
            self.csv_df = pd.read_csv(self.rel_data_dir,
                                      delimiter='\t',
                                      names=["year", "duration", "0 kids", "1 kid", "2 kids", "3+ kids"],
                                      skiprows=8)[:-4]

        if np.shape(self.data_array) == np.shape(np.array([])):
            self.setup_dataset_array()

        if years_to_drop is not None or durations_to_drop is not None or years_to_drop is not None:
            result_data_array, years, durations, kids = self.drop_data(self.data_array, years_to_drop, durations_to_drop, kids_to_drop)
        else:
            result_data_array = self.data_array
            years = self.years
            durations = self.durations
            kids = self.kids

        return result_data_array, years, durations, kids

    # return the dataset as a dataframe
    def getDF(self):
        if self.csv_df.size == 0:
            self.csv_df = pd.read_csv(self.rel_data_dir,
                                      delimiter='\t',
                                      names=["year", "duration", "0 kids", "1 kid", "2 kids", "3+ kids"],
                                      skiprows=8)[:-4]

        return self.csv_df
