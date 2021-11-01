import pandas as pd
import numpy as np
import secrets
from sklearn.feature_selection import SelectFromModel
from utils.utils import get_extention, is_categorical
from loguru import logger
from utils.exceptions import *
from utils.h2o_utils import train, validate

class AutoTrainer:
    '''Trainer to import, preprocess and make data model ready based on parameters, and also initiate training'''
    def __init__(self, datapath, task) -> None:
        if get_extention(datapath) == 'csv':
            self.data = pd.read_csv(datapath)
        self.columns = self.data.columns
        self.label = None
        self.features = []
        self.categorical_features = []
        self.task = task
        self.label_map = {}

    def set_label(self, colname):
        '''Method to set the training label to the input column name'''
        if self.label:
            logger.info(f'Label already exists : {self.label}')
            if colname in self.columns:
                logger.info(f'Label overwriting to : {colname}')
                self.label = colname
            else:
                raise ColumnNotFound(f'Column {colname} not found in the dataframe')
        else:
            if colname in self.columns:
                logger.info(f'Setting label to : {colname}')
                self.label = colname
            else:
                raise ColumnNotFound(f'Column {colname} not found in the dataframe')
    
    def clean_data(self, dropcol_threshold=0.8):
        '''Clean up the input data for the model training
            Inputs:
                dropcol_threshold : Threshold for missing value percentage beyond which the column is dropped from the data
        '''
        if not self.label:
            raise LabelNotSet('Label column is not set. Please use .set_label to set a column as the label')
        else:
            logger.info('Dropping missing labels')
            self.data = self.data.dropna(subset=[self.label])
            if self.task == 'classify':
                logger.info('Encoding categorical values for H2O')
                label_values = self.data[self.label].value_counts().index
                for value in label_values:
                    self.label_map[value] = secrets.token_hex(nbytes=8)
                self.data[self.label] = self.data[self.label].map(lambda x: self.label_map[x])

            for col in self.columns:
                if col == self.label:
                    continue
                else:
                    series = self.data[col]
                    if series.isna().sum() >= dropcol_threshold * len(self.data):
                        continue
                    else:
                        cat = is_categorical(series)
                        if cat:
                            mode = series.value_counts().index[0]
                            na_flag = (series.isna()).astype(int)
                            self.data[col] = self.data[col].fillna(mode)
                            self.data[col+'_imputed'] = na_flag
                            self.categorical_features.append(col)
                            self.features.append(col)
                        else:
                            mean = np.mean(series)
                            na_flag = (series.isna()).astype(int)
                            self.data[col] = self.data[col].fillna(mean)
                            self.data[col+'_imputed'] = na_flag
                            self.features.append(col)

            logger.info('Shortlisting selected features')
            self.data = self.data[self.features]

            logger.info('Imputed and flagged missing data')
            logger.info(f'Categorical Columns Identified : {self.categorical_features}')
            logger.info(f'Finalise and set categorical columns using .set_categorical if you wanna change this list')
    
    def set_categorical(self, columns):
        '''Set categorical features manually'''
        for col in columns:
            if col not in self.columns:
                raise ColumnNotFound(f'Column {col} not found in the dataframe')
        self.categorical_features = columns

    def initiate(self, runtime=60, index_column=None):
        '''Initialise the training and deployment processes
            Inputs:
                runtime -> Maximum runtime of model training in seconds
                index_column -> ID Column within the dataframe if there is any
        '''
        model_path, metrics = train(self.data, self.label, id_column=index_column, max_runtime=runtime)
        return model_path, metrics    
    




            
