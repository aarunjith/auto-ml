import pandas as pd

def get_extention(filepath):
    return filepath.split(".")[-1]

def is_categorical(series, threshold=0.1):
    '''Identify if a column is categorical from the given values'''
    unique_values = series.nunique()
    length = len(series)
    if unique_values <= threshold * length:
        return True
    