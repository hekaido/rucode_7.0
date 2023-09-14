import pandas as pd


def make_chunks(dataframe):
    chunks = [f'chunk_{i + 1}' for i in range(7)]
    dataframe[chunks] = dataframe['address'].str.split(';', expand=True)

